"""Content-domain tagger — an ADVISORY tag per element, via a cheap→expensive
cascade that mirrors the extraction cascade.

A document mixes domains (risk / financial / tax / ...). Each element gets a
domain tag so teams can filter to their components. The tag is advisory:
nothing is withheld based on it, so a wrong tag is a findability miss, not a
data-integrity failure.

The cascade tags each element by the FIRST rung that resolves, so most elements
are tagged for free and the model is only called for genuinely ambiguous ones:

  1. HEADING   — the element's heading/caption matches a registered domain
                 keyword. Free, high-precision.
  2. METRIC    — the canonical metrics in the element imply a domain via the
                 registry's metric->domain map. Free, deterministic.
  3. LLM       — neither fired; ask the model to pick from the REGISTERED
                 vocabulary (constrained, not free-form), with a confidence.
  4. UNKNOWN   — LLM unsure (below threshold) or declined -> ``unknown``.

Every tag records which rung produced it (``tag_source``) so tag quality can be
audited and low-confidence LLM tags can be reviewed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from ..shared.schema import Element, Claim
from ..shared.tag_registry import (
    UNKNOWN_TAG, valid_domains, metric_domain_map, heading_domain_map, _norm)

# LLM tags below this confidence fall through to ``unknown`` rather than being
# trusted — the advisory equivalent of "flag, never guess".
_LLM_TAG_MIN_CONFIDENCE = 0.60

_TAG_TOOL = {
    "name": "tag_domain",
    "description": "Classify the content domain of a document section.",
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string",
                       "description": "One of the allowed domains, or 'unknown'."},
            "confidence": {"type": "number"},
        },
        "required": ["domain", "confidence"],
    },
}


@dataclass
class TagResult:
    domain: str
    source: str          # "heading" | "metric" | "llm" | "unknown"
    confidence: float


def _tag_from_heading(element: Element, headings: dict[str, str]) -> Optional[str]:
    """Rung 1: does the element's heading/caption contain a registered keyword?"""
    text = _norm(f"{element.description or ''} {element.content[:120]}")
    for keyword, domain in headings.items():
        if keyword and keyword in text:
            return domain
    return None


def _tag_from_metrics(claims: list[Claim], m2d: dict[str, str]) -> Optional[str]:
    """Rung 2: do the element's canonical metrics imply a single domain?

    If the metrics point at exactly one domain, use it. If they point at
    several (a genuinely mixed section), we don't guess here — fall through so
    the LLM (or unknown) decides, rather than arbitrarily picking one.
    """
    domains = {m2d[c.canonical_metric] for c in claims
               if c.canonical_metric and c.canonical_metric in m2d}
    return domains.pop() if len(domains) == 1 else None


async def tag_element(element: Element, claims: list[Claim], llm_client=None,
                      spark=None) -> TagResult:
    """Tag one element via the cascade. ``claims`` are the claims already
    extracted from this element (used by the metric rung)."""
    allowed = valid_domains(spark)
    headings = heading_domain_map(spark)
    m2d = metric_domain_map(spark)

    # 1. heading
    h = _tag_from_heading(element, headings)
    if h and h in allowed:
        return TagResult(h, "heading", 1.0)

    # 2. metrics
    m = _tag_from_metrics(claims, m2d)
    if m and m in allowed:
        return TagResult(m, "metric", 1.0)

    # 3. LLM — only for elements the cheap rungs couldn't resolve
    if llm_client is not None:
        options = sorted(allowed)
        prompt = (
            "Classify the content domain of this section. Choose exactly one "
            f"from: {', '.join(options)}. Use 'unknown' if unsure — do not "
            "guess. Return via the tool."
        )
        try:
            resp = await llm_client.extract(
                "tier1", prompt, element.content[:2000], _TAG_TOOL)
            args = resp.tool_arguments or {}
            domain = _norm(str(args.get("domain", UNKNOWN_TAG)))
            conf = float(args.get("confidence", 0.0))
            if domain in allowed and domain != UNKNOWN_TAG \
                    and conf >= _LLM_TAG_MIN_CONFIDENCE:
                return TagResult(domain, "llm", conf)
        except Exception:
            pass  # any failure falls through to unknown — never blocks

    # 4. unknown
    return TagResult(UNKNOWN_TAG, "unknown", 0.0)


async def tag_elements(pairs: list[tuple[Element, list[Claim]]], llm_client=None,
                       spark=None) -> list[TagResult]:
    """Tag many (element, claims) pairs concurrently (async, like extraction)."""
    tasks = [tag_element(el, claims, llm_client, spark) for el, claims in pairs]
    return list(await asyncio.gather(*tasks))
