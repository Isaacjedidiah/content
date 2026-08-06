"""Evaluator — automatic, honest accuracy measurement during the pipeline.

Grades the pipeline against ground truth that is INDEPENDENT of its own output:

  * Registered expected values (eval_registry) — a team's known-correct answers.
  * Human-reviewed values — a reviewer confirmed/overrode a flagged value, so
    the human is the truth source for that row.

It does NOT grade values against the Gold table (circular — the pipeline made
Gold), and it does NOT invent answers. A value with neither a registered
expected nor a human review is simply *not scored* — reported as "ungraded",
never counted as correct. That honesty is the whole point.

Selection is deterministic sampling: every Nth document (by a stable hash of
its id), so a fixed, predictable fraction is evaluated regardless of volume —
the at-scale strategy of continuous eval on a sample.

Reports, per run:
  * accuracy on graded rows (correct / graded)
  * override rate (human overrides / human-reviewed)  — the live error signal
  * counts by ground-truth source and by metric (stratified)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import Claim, ReviewTier
from ..shared.eval_registry import expected_for_document


def is_sampled(document_id: str, every_n: int) -> bool:
    """Deterministic 1-in-N selection by stable hash — same document is always
    in or out, and a predictable ~1/N fraction is sampled at any volume."""
    if every_n <= 1:
        return True
    h = int(hashlib.sha256(document_id.encode()).hexdigest(), 16)
    return (h % every_n) == 0


@dataclass
class EvalRow:
    metric: str
    entity_ref: Optional[str]
    extracted_value: str
    truth_value: str
    source: str                        # "registered" | "human_review"
    correct: bool


@dataclass
class EvalResult:
    document_id: str
    graded: int = 0
    correct: int = 0
    ungraded: int = 0
    human_reviewed: int = 0
    human_overrides: int = 0
    rows: list = field(default_factory=list)
    ts: str = ""

    @property
    def accuracy(self) -> Optional[float]:
        return round(self.correct / self.graded, 4) if self.graded else None

    @property
    def override_rate(self) -> Optional[float]:
        if not self.human_reviewed:
            return None
        return round(self.human_overrides / self.human_reviewed, 4)

    def summary(self) -> dict:
        return {
            "document_id": self.document_id,
            "graded": self.graded, "correct": self.correct,
            "accuracy": self.accuracy,
            "ungraded": self.ungraded,
            "human_reviewed": self.human_reviewed,
            "human_overrides": self.human_overrides,
            "override_rate": self.override_rate,
            "ts": self.ts,
        }


def evaluate_document(document_id: str, claims: list[Claim],
                      team: str, report_type: str,
                      entity_ref: Optional[str],
                      spark=None) -> EvalResult:
    """Score one document's claims against independent ground truth."""
    result = EvalResult(document_id=document_id,
                        ts=datetime.now(timezone.utc).isoformat())

    # ground-truth source 1: registered expected values for this document
    expected = expected_for_document(team, report_type, entity_ref or "",
                                     document_id, spark)
    exp_by_metric = {e.metric: e for e in expected}

    for c in claims:
        metric = c.canonical_metric or c.field_name

        # source 2: human review is the strongest truth — a person decided.
        if c.review_tier in (ReviewTier.HUMAN_CONFIRMED,
                             ReviewTier.HUMAN_OVERRIDE):
            result.human_reviewed += 1
            overridden = c.review_tier == ReviewTier.HUMAN_OVERRIDE
            if overridden:
                result.human_overrides += 1
            # A CONFIRMED value means the pipeline's value was right; an
            # OVERRIDE means it was wrong (the human changed it). Either way
            # this row is graded, with the human as truth.
            result.graded += 1
            if not overridden:
                result.correct += 1
            result.rows.append(EvalRow(
                metric=metric, entity_ref=c.entity_ref,
                extracted_value=c.value,
                truth_value=("<human-confirmed>" if not overridden
                             else "<human-overrode>"),
                source="human_review", correct=not overridden))
            continue

        # otherwise, grade against a registered expected value if one exists
        exp = exp_by_metric.get(metric)
        if exp is not None:
            ok = exp.matches(c.value)
            result.graded += 1
            if ok:
                result.correct += 1
            result.rows.append(EvalRow(
                metric=metric, entity_ref=c.entity_ref,
                extracted_value=c.value, truth_value=exp.expected_value,
                source="registered", correct=ok))
        else:
            # no independent truth for this value — NOT scored, not assumed right
            result.ungraded += 1

    return result


def maybe_evaluate(document_id: str, claims: list[Claim], team: str,
                   report_type: str, entity_ref: Optional[str],
                   every_n: int = 10, spark=None) -> Optional[EvalResult]:
    """Called from the pipeline per document. Evaluates only the ~1/N sample,
    returns None otherwise so the pipeline pays nothing on skipped documents."""
    if not is_sampled(document_id, every_n):
        return None
    result = evaluate_document(document_id, claims, team, report_type,
                               entity_ref, spark)
    _persist(result, spark)
    return result


def _persist(result: EvalResult, spark=None) -> None:
    """Best-effort write of the eval result; never blocks the pipeline."""
    try:
        row = result.summary()
        if spark is not None:
            spark.createDataFrame([row]).write.format("delta").mode(
                "append").saveAsTable(CONFIG.storage.eval_results_table)
        else:
            import json
            import os
            root = (os.environ.get("DOCEXTRACT_EVAL_REGISTRY_DIR")
                    or CONFIG.storage.local_root)
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, "eval_results.jsonl"), "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


def eval_summary(spark=None) -> dict:
    """Aggregate eval results across runs, for the monitoring dashboard."""
    import json
    import os
    rows = []
    if spark is not None:
        try:
            rows = [r.asDict() for r in
                    spark.table(CONFIG.storage.eval_results_table).collect()]
        except Exception:
            rows = []
    else:
        p = os.path.join(CONFIG.storage.local_root, "eval_results.jsonl")
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p) if l.strip()]

    graded = sum(r.get("graded", 0) for r in rows)
    correct = sum(r.get("correct", 0) for r in rows)
    reviewed = sum(r.get("human_reviewed", 0) for r in rows)
    overrides = sum(r.get("human_overrides", 0) for r in rows)
    return {
        "documents_evaluated": len(rows),
        "graded_values": graded,
        "accuracy": round(correct / graded, 4) if graded else None,
        "human_reviewed": reviewed,
        "override_rate": round(overrides / reviewed, 4) if reviewed else None,
    }
