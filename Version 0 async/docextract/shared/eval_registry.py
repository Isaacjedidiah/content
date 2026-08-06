"""Eval registry — expected values as DATA, so the pipeline is graded against
an INDEPENDENT answer key rather than its own output.

The circularity trap this avoids: you cannot evaluate a run against the Gold
table it produced — that just measures the pipeline agreeing with itself. Real
evaluation needs ground truth created *outside* the pipeline. Two honest
sources of that exist:

  1. Team-registered expected values (this module) — a team records what the
     correct value IS for a given (entity, metric) on a known document, with a
     tolerance. Independent of the pipeline, so a legitimate answer key.
  2. Human-reviewed values (from the review workflow) — when a reviewer
     confirms or overrides a flagged value, the human is the truth source for
     that row. Handled in ``evaluator.py``.

Design mirrors prompt_registry exactly: entries appended to JSONL locally / a
Delta table on Databricks; latest entry per key wins; ``list_missing`` shows
gaps; every write records who set it. Env override PRA_EVAL_REGISTRY_DIR (kept
name-compatible with the other registries' override convention) or
DOCEXTRACT_EVAL_REGISTRY_DIR for local/testing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG
from .schema import parse_numeric


@dataclass
class ExpectedValue:
    """One known-correct answer, registered by a team as ground truth."""
    team: str
    report_type: str
    entity_ref: str
    metric: str                       # canonical metric key
    expected_value: str               # the correct value, as a string
    # match rule: "numeric" compares within tolerance; "exact" string-matches
    match: str = "numeric"
    tolerance: float = 0.01           # numeric abs tolerance (e.g. 0.01 = ±0.01)
    document_id: Optional[str] = None  # optional: pin to a specific document
    updated_by: str = "system"
    updated_at: str = ""

    def key(self) -> tuple:
        return (self.team, self.report_type, self.entity_ref, self.metric,
                self.document_id or "*")

    def matches(self, actual_value: str) -> bool:
        """Does an extracted value satisfy this expected value?"""
        if self.match == "exact":
            return str(actual_value).strip() == str(self.expected_value).strip()
        a = parse_numeric(actual_value)
        e = parse_numeric(self.expected_value)
        if a is None or e is None:
            return False
        return abs(a - e) <= self.tolerance


def _local_path() -> str:
    root = (os.environ.get("DOCEXTRACT_EVAL_REGISTRY_DIR")
            or os.environ.get("PRA_EVAL_REGISTRY_DIR")
            or CONFIG.storage.local_root)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "eval_registry.jsonl")


def _read_all(spark=None) -> list[dict]:
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.eval_registry_table).collect()]
        except Exception:
            return []
    p = _local_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def register_expected(entry: ExpectedValue, updated_by: str,
                      spark=None) -> None:
    """Self-serve write: append an expected value (latest per key wins)."""
    payload = {
        "team": entry.team, "report_type": entry.report_type,
        "entity_ref": entry.entity_ref, "metric": entry.metric,
        "expected_value": entry.expected_value, "match": entry.match,
        "tolerance": entry.tolerance, "document_id": entry.document_id,
        "updated_by": updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if spark is not None:
        spark.createDataFrame([payload]).write.format("delta").mode(
            "append").saveAsTable(CONFIG.storage.eval_registry_table)
        return
    with open(_local_path(), "a") as fh:
        fh.write(json.dumps(payload) + "\n")


def _to_expected(d: dict) -> ExpectedValue:
    return ExpectedValue(
        team=d["team"], report_type=d["report_type"],
        entity_ref=d["entity_ref"], metric=d["metric"],
        expected_value=d["expected_value"], match=d.get("match", "numeric"),
        tolerance=float(d.get("tolerance", 0.01)),
        document_id=d.get("document_id"),
        updated_by=d.get("updated_by", "system"),
        updated_at=d.get("updated_at", ""))


def get_expected(team: str, report_type: str, entity_ref: str, metric: str,
                 document_id: Optional[str] = None,
                 spark=None) -> Optional[ExpectedValue]:
    """Return the registered expected value for a key, or None if a team
    hasn't registered one (that metric simply isn't graded — no default seed,
    because an invented 'correct answer' would be worse than none)."""
    matches = []
    for d in _read_all(spark):
        if (d["team"] == team and d["report_type"] == report_type
                and d["entity_ref"] == entity_ref and d["metric"] == metric):
            doc = d.get("document_id")
            if doc in (None, "*") or document_id is None or doc == document_id:
                matches.append(d)
    return _to_expected(matches[-1]) if matches else None


def expected_for_document(team: str, report_type: str, entity_ref: str,
                          document_id: Optional[str] = None,
                          spark=None) -> list[ExpectedValue]:
    """All registered expected values applicable to one document."""
    out = []
    for d in _read_all(spark):
        if (d["team"] == team and d["report_type"] == report_type
                and d["entity_ref"] == entity_ref):
            doc = d.get("document_id")
            if doc in (None, "*") or document_id is None or doc == document_id:
                out.append(_to_expected(d))
    return out


def list_missing(pairs: list[tuple[str, str]], spark=None) -> list[tuple]:
    """(team, report_type) combinations with NO registered expected values —
    i.e. teams whose extractions can't yet be ground-truth evaluated."""
    have = {(d["team"], d["report_type"]) for d in _read_all(spark)}
    return [(t, rt) for t, rt in pairs if (t, rt) not in have]
