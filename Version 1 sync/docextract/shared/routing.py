"""Content-aware starting-tier routing.

Decides which tier an element STARTS on from cheap, deterministic content
signals — modality, length, report type — before any model call is made.
Known-hard content (tables, over-long or flagged-dense elements) starts on
Tier 2 directly, skipping the wasted Tier 1 pass. Everything else starts on
Tier 1 and escalates reactively on low confidence, exactly as before.

This is deliberately a pure function over an ``Element`` plus policy: no I/O,
no model call, no hidden state. That keeps it cheap (it runs for every
element), testable, and auditable — each decision returns the tier AND a
human-readable reason that the orchestrator records in the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG, RoutingPolicy
from .schema import Element, Modality


@dataclass(frozen=True)
class RoutingDecision:
    tier: str            # "tier1" or "tier2"
    reason: str          # why this tier, for the audit trail
    content_aware: bool  # True if a content rule fired; False = default path


def decide_start_tier(el: Element,
                      policy: RoutingPolicy | None = None) -> RoutingDecision:
    """Return the starting tier for ``el``.

    Rules are evaluated cheapest-first and the first match wins. When routing
    is disabled or nothing matches, the element starts on Tier 1 (the existing
    reactive cascade is unchanged for that element).
    """
    policy = policy or CONFIG.routing

    if not policy.enabled:
        return RoutingDecision("tier1", "routing disabled", content_aware=False)

    # Empty / flagged-scanned elements: a paid Tier 2 pass buys nothing. Keep
    # them on Tier 1, which returns nothing and routes to review/quarantine.
    is_empty = not (el.content or "").strip()
    if is_empty and policy.skip_hard_routing_when_empty:
        return RoutingDecision(
            "tier1", "empty element: no Tier 2 benefit", content_aware=True)

    # Modality rule: tables (and any other configured hard modalities) start
    # on Tier 2 — highest-value skip, since table misreads are costly.
    modality = el.modality.value if isinstance(el.modality, Modality) else str(el.modality)
    if modality in policy.hard_modalities:
        return RoutingDecision(
            "tier2", f"hard modality: {modality}", content_aware=True)

    # Report-type rule: deployments can mark specific dense report types hard.
    if el.report_type and el.report_type in policy.hard_report_types:
        return RoutingDecision(
            "tier2", f"hard report_type: {el.report_type}", content_aware=True)

    # Length rule: long context degrades the cheap model, so over-long
    # elements start on Tier 2.
    if policy.max_tier1_chars and len(el.content or "") > policy.max_tier1_chars:
        return RoutingDecision(
            "tier2",
            f"length {len(el.content)} > {policy.max_tier1_chars}",
            content_aware=True)

    return RoutingDecision("tier1", "default", content_aware=False)
