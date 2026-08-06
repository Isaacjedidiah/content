"""Monitoring data layers.

PipelineMonitor (custom track) persists run metrics to local JSONL; the
native variant writes to a Unity Catalog Delta table. Both expose the same
interface so dashboards are track-agnostic.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .config import CONFIG


@dataclass
class RunMetrics:
    run_id: str
    documents: int = 0
    cost: float = 0.0
    needs_review: int = 0
    errors: int = 0
    quarantined: int = 0
    duration_s: float = 0.0
    team: str | None = None


class PipelineMonitor:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(CONFIG.storage.local_root, "monitor.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def record(self, m: RunMetrics) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(asdict(m)) + "\n")

    def load_runs(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def summary(self) -> dict:
        runs = self.load_runs()
        return {
            "runs": len(runs),
            "documents": sum(r.get("documents", 0) for r in runs),
            "cost": round(sum(r.get("cost", 0.0) for r in runs), 4),
        }


class NativePipelineMonitor:
    """Same interface, backed by a Unity Catalog Delta table."""

    def __init__(self, spark=None, table: str | None = None):
        self.spark = spark
        self.table = table or CONFIG.storage.run_metrics_table
        self._buffer: list[dict] = []

    def record(self, m: RunMetrics) -> None:
        row = asdict(m)
        if self.spark is not None:
            df = self.spark.createDataFrame([row])
            df.write.format("delta").mode("append").saveAsTable(self.table)
        else:
            self._buffer.append(row)

    def load_runs(self) -> list[dict]:
        if self.spark is not None:
            return [r.asDict() for r in self.spark.table(self.table).collect()]
        return self._buffer

    def summary(self) -> dict:
        runs = self.load_runs()
        return {"runs": len(runs),
                "documents": sum(r.get("documents", 0) for r in runs)}


def supervisor_feedback_summary(spark=None) -> dict:
    """Aggregate the supervisor query log for the ops dashboard: volume of
    questions, how many were rated, and the up/down split. This turns the
    feedback capture into a visible quality signal alongside ingestion KPIs."""
    from .supervisor_session import load_query_log

    rows = load_query_log(spark)
    # A turn can appear as an 'answer' row and later a 'feedback' update row;
    # dedup on turn_id, preferring the row that carries a rating.
    by_turn: dict = {}
    feedback_only = 0
    for r in rows:
        if r.get("log_kind") == "feedback_only":
            feedback_only += 1
            continue
        tid = r.get("turn_id")
        if not tid:
            continue
        prev = by_turn.get(tid)
        if prev is None or (r.get("feedback_rating") and not prev.get("feedback_rating")):
            by_turn[tid] = r
    turns = list(by_turn.values())
    rated = [t for t in turns if t.get("feedback_rating")]
    up = sum(1 for t in rated if t.get("feedback_rating") == "up")
    down = sum(1 for t in rated if t.get("feedback_rating") == "down")
    return {
        "questions": len(turns),
        "rated": len(rated) + feedback_only,
        "thumbs_up": up,
        "thumbs_down": down,
        "unhelpful_rate": round(down / max(len(rated), 1), 2),
        "sessions": len({t.get("session_id") for t in turns if t.get("session_id")}),
    }


def eval_accuracy_summary(spark=None) -> dict:
    """Surface the automatic-eval results on the monitoring dashboard:
    accuracy on graded rows and the human-override error rate. Delegates to the
    evaluator's own aggregator so there's one definition of the numbers."""
    from ..custom.evaluator import eval_summary
    return eval_summary(spark)
