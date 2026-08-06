"""Reconciliation and human-review gating.

Magnitude reconciliation flags pairs of claims sharing a canonical metric
whose values differ by more than the configured ratio. This is numerical
PLAUSIBILITY (distinct from the normaliser's structural check). The
validator flags, never rejects; UNKNOWN classifications get benefit of the
doubt.

Fixes: reconciliation now groups by a structural signature (metric +
reporting_basis + scale + as_at_date) so a consolidated figure is not
compared against a solo figure and mis-flagged; and values are parsed with
the shared parser so '14.2%' / '1,234' participate instead of being dropped.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import Claim, parse_numeric


@dataclass
class Conflict:
    metric: str
    a: float
    b: float
    ratio: float
    basis: Optional[str] = None
    scale: Optional[str] = None
    as_at_date: Optional[str] = None


def _signature(c: Claim) -> tuple:
    """Only claims that are structurally like-for-like should be compared for
    magnitude. Comparing consolidated vs solo (or millions vs ratio) is a
    structural concern handled by the normaliser, not a magnitude conflict."""
    return (
        c.canonical_metric or c.field_name,
        c.reporting_basis,
        c.scale,
        c.as_at_date,
    )


def reconcile(claims: list[Claim]) -> list[Conflict]:
    ratio_th = CONFIG.thresholds.magnitude_conflict_ratio
    by_sig: dict[tuple, list[Claim]] = defaultdict(list)
    for c in claims:
        by_sig[_signature(c)].append(c)

    conflicts: list[Conflict] = []
    for sig, group in by_sig.items():
        metric, basis, scale, as_at = sig
        numeric: list[tuple[float, Claim]] = []
        for c in group:
            v = parse_numeric(c.value)
            if v is not None:
                numeric.append((v, c))
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                hi = max(numeric[i][0], numeric[j][0])
                lo = min(numeric[i][0], numeric[j][0])
                if lo > 0 and hi / lo > ratio_th:
                    conflicts.append(Conflict(
                        metric=metric, a=numeric[i][0], b=numeric[j][0],
                        ratio=round(hi / lo, 2), basis=basis, scale=scale,
                        as_at_date=as_at))
    return conflicts


def gate_for_review(claims: list[Claim],
                    conflicts: list[Conflict]) -> dict[str, list[Claim]]:
    conflicted = {c.metric for c in conflicts}
    review_th = CONFIG.thresholds.human_review
    clean: list[Claim] = []
    flagged: list[Claim] = []
    for c in claims:
        low_conf = c.confidence < review_th
        in_conflict = (c.canonical_metric or c.field_name) in conflicted
        # UNKNOWN alone does not flag: benefit of the doubt.
        if low_conf or in_conflict:
            c.needs_review = True
            flagged.append(c)
        else:
            clean.append(c)
    # Nothing is rejected here; flagged items go to a human.
    return {"clean": clean, "needs_review": flagged}
