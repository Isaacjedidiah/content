"""Threshold registry — per-metric review bars as GOVERNED data.

Unlike the prompt / eval / dictionary registries (which teams self-serve), this
one is **written centrally only**. Teams do not set review thresholds — they
don't own that calibration, and it exists for the operator's benefit (tuning how
much scrutiny each metric gets). The registry is the *storage + audit* mechanism
(append-table, latest-per-key wins, ``updated_by`` recorded, live without a
redeploy), but writes are a governance action, not a team action.

Storage mirrors the other registries: JSONL locally / Delta on Databricks. The
global floor lives in ``config`` (``human_review``); a registered per-metric
value can only RAISE the bar above that floor, never lower it — enforced in
``config.Thresholds.review_threshold``.

Because ``review_threshold`` is called per-claim (hot path), the resolved map is
loaded once and cached; call ``invalidate_cache()`` after registering to pick up
changes in a long-lived process.

Env override DOCEXTRACT_THRESHOLD_REGISTRY_DIR (or PRA_THRESHOLD_REGISTRY_DIR).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG


@dataclass
class MetricThreshold:
    """A governed per-metric review bar. ``team``/``report_type`` optional —
    leave them None for a metric-wide bar, or set them to scope the bar to one
    team's report type."""
    metric: str
    threshold: float
    team: Optional[str] = None
    report_type: Optional[str] = None
    updated_by: str = "governance"
    updated_at: str = ""

    def key(self) -> tuple:
        return (self.team or "*", self.report_type or "*", self.metric)


_CACHE: Optional[dict] = None


def _local_path() -> str:
    root = (os.environ.get("DOCEXTRACT_THRESHOLD_REGISTRY_DIR")
            or os.environ.get("PRA_THRESHOLD_REGISTRY_DIR")
            or CONFIG.storage.local_root)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "threshold_registry.jsonl")


def _read_all(spark=None) -> list[dict]:
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.threshold_registry_table).collect()]
        except Exception:
            return []
    p = _local_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def register_threshold(entry: MetricThreshold, updated_by: str,
                       spark=None) -> None:
    """Governed write: append a per-metric review bar (latest per key wins).

    This is an operator action, not a team action. ``updated_by`` records who
    set it. Invalidates the in-process cache so the new value takes effect.
    """
    payload = {
        "metric": entry.metric, "threshold": float(entry.threshold),
        "team": entry.team, "report_type": entry.report_type,
        "updated_by": updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if spark is not None:
        spark.createDataFrame([payload]).write.format("delta").mode(
            "append").saveAsTable(CONFIG.storage.threshold_registry_table)
    else:
        with open(_local_path(), "a") as fh:
            fh.write(json.dumps(payload) + "\n")
    invalidate_cache()


def _resolved_map(spark=None) -> dict:
    """{(team|*, report_type|*, metric): threshold}, latest per key wins."""
    out: dict = {}
    for d in _read_all(spark):
        key = (d.get("team") or "*", d.get("report_type") or "*", d["metric"])
        out[key] = float(d["threshold"])
    return out


def get_threshold(metric: str, team: Optional[str] = None,
                  report_type: Optional[str] = None,
                  spark=None) -> Optional[float]:
    """Return the registered per-metric threshold, or None if none is set.

    Resolution is most-specific-first: an entry scoped to this team+report_type
    beats a metric-wide entry. Cached after first load (hot path); call
    ``invalidate_cache()`` to refresh.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _resolved_map(spark)
    # most specific match first
    for k in ((team or "*", report_type or "*", metric),
              ("*", "*", metric)):
        if k in _CACHE:
            return _CACHE[k]
    return None


def invalidate_cache() -> None:
    """Drop the cached map so the next lookup reloads (after a register, or in
    a long-lived process that should pick up governance changes)."""
    global _CACHE
    _CACHE = None


def list_registered(spark=None) -> list[tuple]:
    """All registered (team, report_type, metric, threshold) — for visibility
    into what's been governed."""
    return sorted((k[0], k[1], k[2], v) for k, v in _resolved_map(spark).items())
