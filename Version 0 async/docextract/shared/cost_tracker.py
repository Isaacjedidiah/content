"""Cross-run cost accounting.

Rolls per-document costs into per-team / per-tier / total views and can
project the cost of a tiering choice at scale — the analysis that justified
the two-tier cascade over an Opus pre-human gate.
"""
from __future__ import annotations

from collections import defaultdict


class CostTracker:
    def __init__(self) -> None:
        self._by_team: dict[str, float] = defaultdict(float)
        self._by_tier: dict[str, float] = defaultdict(float)

    def record(self, team: str, tier: str, usd: float) -> None:
        self._by_team[team] += usd
        self._by_tier[tier] += usd

    def totals(self) -> dict:
        return {
            "by_team": dict(self._by_team),
            "by_tier": dict(self._by_tier),
            "grand_total": round(sum(self._by_team.values()), 6),
        }

    @staticmethod
    def project(pages: int, tier1_rate: float, tier2_share: float,
                tier2_rate: float, hard_share: float = 0.0) -> float:
        """Projected cascade cost.

        Three populations of pages:
          * ``hard_share`` — content-routed straight to Tier 2 (tables,
            over-long, flagged-dense). Pays Tier 2 ONLY; the Tier-1 pass is
            skipped, which is the whole point of content-aware routing.
          * of the remaining (reactive) pages, ``tier2_share`` escalate and
            pay BOTH a Tier-1 and a Tier-2 pass (the extractor bills both).
          * the rest pay Tier 1 only.

        With ``hard_share=0`` this reduces exactly to the previous model, so
        existing callers are unaffected.
        """
        hard_share = min(max(hard_share, 0.0), 1.0)
        reactive = pages * (1.0 - hard_share)

        hard_cost = pages * hard_share * tier2_rate          # Tier 2 only
        t1 = reactive * tier1_rate                           # reactive pages
        t2 = reactive * tier2_share * tier2_rate             # escalated, additive
        return round(hard_cost + t1 + t2, 2)
