"""Per-team prompt registry.

A base extraction prompt inherited by all teams, with per-team and
per-report-type overrides layered on top. Unmapped teams fall back to the
base prompt — prompts never block extraction.
"""
from __future__ import annotations

# The base prompt now always asks for the structural fields the downstream
# reconciler/normaliser/Gold layer depend on (unit, reporting basis, netting,
# scale, as-at date). Previously only capital/liquidity overrides mentioned
# basis, so unmapped teams produced claims that systematically failed
# structural comparison.
BASE_EXTRACTION_PROMPT = (
    "You are extracting regulatory metrics from a firm submission. "
    "Return only structured claims via the provided tool. Preserve the "
    "exact field names and values as written in the document — do not infer, "
    "convert, or round values that are not present. For every claim, also "
    "capture, when stated in the document: the unit (e.g. %, bps, GBP), the "
    "reporting basis (consolidated vs solo), whether the figure is net or "
    "gross of deductions, the scale (millions/thousands/ratio), and the "
    "as-at/reporting date. Leave a field empty rather than guessing. Set "
    "confidence lower when a value is ambiguous, footnoted, or hard to read."
)

TEAM_PROMPT_OVERRIDES: dict[str, dict[str, str]] = {
    "liquidity": {
        "_default": (
            "Pay special attention to coverage and NSFR, including the reporting "
            "basis (consolidated vs solo)."
        ),
        "coverage_report": "This is an coverage report; expect a 30-day stress horizon.",
    },
    "capital": {
        "_default": (
            "Pay special attention to the primary metric, Tier 1 and leverage ratios, and "
            "whether figures are net or gross of deductions."
        ),
    },
}


def build_prompt(team: str, report_type: str) -> str:
    """Build the extraction prompt for (team, report_type).

    Consults the stored prompt registry first — if the team has registered an
    extraction entry, its preamble + few-shots are used. Otherwise falls back
    to the built-in base prompt plus any hard-coded team/report overrides, so
    nothing regresses before a team registers its own.
    """
    try:
        from .prompt_registry import get_prompt

        entry = get_prompt(team, report_type, "extraction")
        if entry.team != "default" or entry.fewshots:
            return entry.render("Now extract from the content that follows.")
    except Exception:
        pass  # registry unavailable — fall back to built-in below

    parts = [BASE_EXTRACTION_PROMPT]
    over = TEAM_PROMPT_OVERRIDES.get(team, {})
    if "_default" in over:
        parts.append(over["_default"])
    if report_type in over:
        parts.append(over[report_type])
    return "\n\n".join(parts)
