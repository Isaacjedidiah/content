"""Two-tier extraction cascade with content-aware routing and cost tracking.

Routing is now two-stage. A cheap, deterministic content check
(``shared.routing.decide_start_tier``) picks each element's STARTING tier
before any model call: known-hard content (tables, over-long or
flagged-dense elements) starts on Tier 2 (Sonnet) directly, skipping the
wasted Tier 1 pass. Everything else starts on Tier 1 (GPT-5 mini) and
escalates reactively when confidence is low. Still-low confidence after
Tier 2 flags for human review. The system flags but never rejects. Token
cost accumulates per document, and each routing decision is recorded for the
audit trail.

Fixes over the earlier version:
  * The tool schema now returns scale/netting/as_at_date, and those are
    threaded into every Claim (previously always None).
  * entity_ref is propagated from the source Element into each Claim
    (previously nothing populated it, so Gold was unattributable).
  * Malformed model rows are quarantined, not fatal (flag-never-reject).
"""
from __future__ import annotations

import asyncio

from dataclasses import dataclass, field
from typing import Optional

from ..shared.config import CONFIG
from ..shared.dictionary import canonical_metric
from ..shared.llm_client import EXTRACTION_TOOL_SCHEMA, LLMClient
from ..shared.metric_normaliser import classify_metric
from ..shared.prompts import build_prompt
from ..shared.routing import RoutingDecision, decide_start_tier
from ..shared.schema import (Claim, CitationTier, Element, Modality, ReviewTier,
                             build_claim)
from ..search.figure_preprocessor import (encode_image_base64, is_chart_like,
                                          prepare_figure_image)


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    quarantine: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    # Per-element routing decisions, keyed by element_id, for the audit trail.
    routing: dict[str, RoutingDecision] = field(default_factory=dict)
    cropped_figures: int = 0

    def track_cost(self, model_key: str, in_tok: int, out_tok: int) -> None:
        spec = CONFIG.model(model_key)
        self.total_cost_usd += (in_tok / 1e6) * spec.input_cost_per_m
        self.total_cost_usd += (out_tok / 1e6) * spec.output_cost_per_m


class Extractor:
    def __init__(self, client: Optional[LLMClient] = None):
        self._client = client or LLMClient()
        self._th = CONFIG.thresholds

    async def extract(self, elements: list[Element],
                      document_bytes: Optional[bytes] = None,
                      filename: Optional[str] = None,
                      page_image_provider=None) -> ExtractionResult:
        """Extract claims from every element, concurrently.

        Each element is processed as its own task; ``asyncio.gather`` runs them
        overlapping, and the client's semaphore caps how many model calls are
        actually in flight (respecting the endpoint rate limit). This is the
        async payoff: 1000 elements no longer wait in a 1000-long line — they
        run in bounded concurrent waves.

        When ``document_bytes``, ``filename`` and ``page_image_provider`` are
        supplied, chart-like FIGURE elements are cropped-and-zoomed and read
        multimodally (see search.figure_preprocessor). Text-only extraction
        is unchanged when they are absent — crop-and-zoom is fully optional
        and degrades to text-only per element, never crashing a document.
        """
        result = ExtractionResult()
        tasks = []
        for el in elements:
            prompt = build_prompt(el.team or "", el.report_type or "")
            tasks.append(self._extract_element(
                el, prompt, result, document_bytes, filename,
                page_image_provider))
        # return_exceptions=True so one element's failure can't cancel the
        # whole gather; a raised element is quarantined below.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for el, outcome in zip(elements, outcomes):
            if isinstance(outcome, Exception):
                result.quarantine.append(
                    {"kind": "element", "source_element_id": el.element_id,
                     "error": f"{type(outcome).__name__}: {outcome}"})
        return result

    def _figure_image(self, el: Element, document_bytes, filename,
                      page_image_provider) -> Optional[str]:
        """Return a base64 cropped-and-zoomed image for a chart-like figure,
        or None (→ text-only). Any failure degrades to None, never raises."""
        if not (document_bytes and filename and page_image_provider):
            return None
        if not is_chart_like(el):
            return None
        try:
            cropped = prepare_figure_image(
                page_image_provider, document_bytes, filename, el.page, el.bbox)
        except Exception:
            return None
        if cropped is None:
            return None
        try:
            return encode_image_base64(cropped)
        except Exception:
            return None

    async def _extract_element(self, el: Element, prompt: str,
                         result: ExtractionResult,
                         document_bytes: Optional[bytes] = None,
                         filename: Optional[str] = None,
                         page_image_provider=None) -> None:
        # Content-aware routing decides the STARTING tier before any model
        # call. Hard content (tables, over-long, flagged-dense) starts on
        # Tier 2, skipping the wasted Tier 1 pass.
        decision = decide_start_tier(el)
        result.routing[el.element_id] = decision

        # Crop-and-zoom: compute once for a chart-like figure, reuse across
        # tier1/tier2 calls for this element. None => text-only, unchanged.
        image_b64 = self._figure_image(el, document_bytes, filename,
                                       page_image_provider)
        if image_b64 is not None:
            result.cropped_figures += 1
            prompt = prompt + ("\n\n(The attached image is this figure, "
                               "cropped and zoomed for legibility.)")

        if decision.tier == "tier2":
            # Start on Tier 2 directly: one pass, no Tier 1 spend. A
            # low-confidence result here has no higher tier to escalate to,
            # so it flows to human review via the needs_review flag in
            # _build_claim (same as a Tier-2 result on the reactive path).
            resp = await self._client.extract("tier2", prompt, el.content,
                                        EXTRACTION_TOOL_SCHEMA, image_b64)
            result.track_cost("tier2", resp.input_tokens, resp.output_tokens)
            model_used = "tier2"
            raw_claims = (resp.tool_arguments or {}).get("claims", [])
        else:
            # Reactive cascade: Tier 1 first, escalate on low confidence.
            resp = await self._client.extract("tier1", prompt, el.content,
                                        EXTRACTION_TOOL_SCHEMA, image_b64)
            result.track_cost("tier1", resp.input_tokens, resp.output_tokens)
            model_used = "tier1"
            raw_claims = (resp.tool_arguments or {}).get("claims", [])

            # Escalate if nothing came back, or the lowest-confidence claim is
            # below the escalate threshold. The whole element is re-run on
            # Tier 2 (and both passes are billed — see CostTracker.project).
            if _min_conf(raw_claims) < self._th.tier1_escalate:
                resp = await self._client.extract("tier2", prompt, el.content,
                                            EXTRACTION_TOOL_SCHEMA, image_b64)
                result.track_cost("tier2", resp.input_tokens,
                                  resp.output_tokens)
                model_used = "tier2"
                raw_claims = (resp.tool_arguments or {}).get("claims", [])

        for rc in raw_claims:
            claim = self._build_claim(rc, el, model_used, result)
            if claim is not None:
                result.claims.append(claim)

    def _build_claim(self, rc: dict, el: Element, model_used: str,
                     result: ExtractionResult) -> Optional[Claim]:
        """Assemble a Claim from a raw model row, quarantining bad rows.

        Required keys (field_name, value, confidence) missing => quarantine,
        never a KeyError that kills the element.
        """
        if not isinstance(rc, dict) or "field_name" not in rc or "value" not in rc:
            result.quarantine.append(
                {"kind": "claim_row", "raw": rc, "error": "missing field_name/value",
                 "source_element_id": el.element_id})
            return None

        conf = _as_float(rc.get("confidence"), default=0.0)
        canon, _mapped = canonical_metric(str(rc["field_name"]))
        # Provenance: a value read from a chart/figure is LLM_ESTIMATED (lower
        # real-world reliability at equal confidence); parsed text/tables are
        # PARSED. This does NOT force review on its own — a chart-derived
        # value only goes to a human if its own confidence is low, matching
        # the uploaded codebase's deliberate choice to keep review volume
        # manageable.
        is_figure = el.modality == Modality.FIGURE
        citation = CitationTier.LLM_ESTIMATED if is_figure else CitationTier.PARSED
        # Structural classification (measure shape + period) from the raw name.
        cls = classify_metric(str(rc["field_name"]), rc.get("value"))
        payload = {
            "field_name": str(rc["field_name"]),
            "canonical_metric": canon,
            "value": str(rc["value"]),
            "unit": _clean(rc.get("unit")),
            "reporting_basis": _clean(rc.get("reporting_basis")),
            "netting": _clean(rc.get("netting")),
            "scale": _clean(rc.get("scale")),
            "as_at_date": _clean(rc.get("as_at_date")),
            "confidence": conf,
            "source_element_id": el.element_id,
            "entity_ref": el.entity_ref,
            "model_used": model_used,
            "needs_review": conf < self._th.review_threshold(canon),
            "citation_tier": citation,
            "review_tier": ReviewTier.MODEL_AUTO,
            "measure_type": cls.measure_type.value,
            "period": cls.period.value,
            "page": el.page,
            "bbox": el.bbox,
        }
        return build_claim(payload, result.quarantine)


def _clean(v) -> Optional[str]:
    """Empty strings from the model become None so structural comparison and
    Gold columns are null rather than ''."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _min_conf(raw_claims: list) -> float:
    if not raw_claims:
        return 0.0  # nothing extracted -> escalate
    return min(_as_float(c.get("confidence"), 0.0)
               for c in raw_claims if isinstance(c, dict))
