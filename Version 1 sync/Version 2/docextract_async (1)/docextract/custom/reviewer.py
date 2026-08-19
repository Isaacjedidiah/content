"""Bounding-box reviewer with manual override and citation tiers.

Shows a flagged claim highlighted on its source page (via the preserved
bbox), lets a reviewer accept or override, and records a citation tier.

``apply_decision`` closes the loop that was previously missing: it writes the
reviewer's outcome back onto the Claim (final value + citation tier +
clears needs_review) so a reviewed claim can be promoted into Gold, and
returns a row for the review-decisions sink and an audit event.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ..shared.schema import Claim, CitationTier, ReviewTier


@dataclass
class ReviewDecision:
    element_id: str
    original_value: str
    final_value: str
    review_tier: ReviewTier
    reviewer: str
    bbox: Optional[tuple]
    overridden: bool = False


def review(element_id: str, original_value: str, bbox, reviewer: str,
           override_value: Optional[str] = None) -> ReviewDecision:
    if override_value is None or override_value == original_value:
        review_tier, final, overridden = ReviewTier.HUMAN_CONFIRMED, original_value, False
    else:
        review_tier, final, overridden = ReviewTier.HUMAN_OVERRIDE, override_value, True
    return ReviewDecision(element_id, original_value, final, review_tier,
                          reviewer, bbox, overridden)


def apply_decision(claim: Claim, decision: ReviewDecision) -> Claim:
    """Fold a review outcome into the claim so it is Gold-eligible.

    Confirmed -> keep value; Override -> replace value. The review lifecycle
    tier is advanced, and an overridden value's provenance becomes MANUAL
    (a human typed it) — matching the uploaded codebase's manual-override
    semantics. needs_review is cleared so the promoted claim reflects human
    sign-off rather than raw model output.
    """
    claim.value = decision.final_value
    claim.review_tier = decision.review_tier
    if decision.overridden:
        claim.citation_tier = CitationTier.MANUAL
    claim.needs_review = False
    return claim


def decision_row(decision: ReviewDecision) -> dict:
    row = asdict(decision)
    row["review_tier"] = decision.review_tier.value
    return row
