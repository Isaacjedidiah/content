"""Link a narrative claim to the quant metric that supports it.

A regulatory document says things like "Primary metric strengthened over the quarter";
this module finds the specific extracted metric that backs that claim, so a
supervisor can ask "what's the evidence for this" and get the number.

Two-stage, matching the uploaded design: cheap lexical retrieval of the top-N
candidate metrics, then one LLM confirmation call (tier1) to pick the best —
or none. A claim that doesn't clear the confidence threshold is left
UNLINKED rather than force-matched to the nearest number, so a spurious link
is never manufactured.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..shared.config import CONFIG, CLASSIFIER_MODEL
from ..shared.llm_client import LLMClient
from ..shared.schema import Claim

_CLAIM_KEYWORDS = re.compile(
    r"\b(increased|decreased|improved|declined|strengthened|weakened|remained|"
    r"grew|fell|rose|maintained|exceeded|breached|below|above)\b",
    re.IGNORECASE,
)


def looks_like_claim(text: str) -> bool:
    """Cheap gate: only sentences with change/comparison language are worth
    trying to link, so we don't spend an LLM call on every paragraph."""
    return bool(_CLAIM_KEYWORDS.search(text))


@dataclass
class ClaimQuantLink:
    claim_text: str
    matched_metric: Optional[str]      # canonical_metric of the supporting claim
    confidence: float
    candidates_considered: int


@dataclass
class _Candidate:
    metric: str
    claim: Claim
    lexical_score: float


def _retrieve_candidates(claim_text: str, claims: list[Claim],
                         top_n: int) -> list[_Candidate]:
    claim_terms = set(re.findall(r"[a-z0-9]+", claim_text.lower()))
    out: list[_Candidate] = []
    seen: set[str] = set()
    for c in claims:
        metric = c.canonical_metric or c.field_name
        # score against both the raw and canonical names
        name_terms = set(re.findall(r"[a-z0-9]+",
                                    f"{c.field_name} {metric}".lower()))
        overlap = len(claim_terms & name_terms)
        if overlap > 0 and metric not in seen:
            seen.add(metric)
            out.append(_Candidate(metric, c,
                                  overlap / max(len(claim_terms), 1)))
    out.sort(key=lambda x: x.lexical_score, reverse=True)
    return out[:top_n]


_CONFIRM_PROMPT = (
    "Match a narrative claim from a regulatory document to the quantitative "
    "metric it refers to, if any. Respond with JSON only: "
    '{"best_candidate_index": int or null, "confidence": float}. '
    "Use null if none plausibly support the claim — do not force a match."
)


async def _confirm(claim_text: str, candidates: list[_Candidate],
             client: LLMClient) -> tuple[Optional[str], float]:  # async
    if not candidates:
        return None, 0.0
    block = "\n".join(
        f"{i+1}. {c.claim.field_name} = {c.claim.value}"
        for i, c in enumerate(candidates))
    content = (f'Claim: "{claim_text}"\n\nCandidate metrics:\n{block}')
    resp = await client.complete(CLASSIFIER_MODEL, _CONFIRM_PROMPT, content)
    try:
        parsed = json.loads(resp.text)
        idx = parsed.get("best_candidate_index")
        conf = float(parsed.get("confidence", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return None, 0.0
    if idx is None or not isinstance(idx, int) or not (0 <= idx < len(candidates)):
        return None, conf
    return candidates[idx].metric, conf


async def match_claim(claim_text: str, claims: list[Claim],
                client: LLMClient) -> ClaimQuantLink:
    candidates = _retrieve_candidates(
        claim_text, claims, CONFIG.claim_quant.top_n_candidates)
    metric, conf = await _confirm(claim_text, candidates, client)
    if conf < CONFIG.claim_quant.confidence_threshold:
        metric = None  # below the bar: leave unlinked, never force a match
    return ClaimQuantLink(claim_text=claim_text, matched_metric=metric,
                          confidence=conf,
                          candidates_considered=len(candidates))


async def match_all(narrative_texts: list[str], claims: list[Claim],
                    client: LLMClient) -> list[ClaimQuantLink]:
    import asyncio
    texts = [t for t in narrative_texts if looks_like_claim(t)]
    return list(await asyncio.gather(
        *(match_claim(t, claims, client) for t in texts)))
