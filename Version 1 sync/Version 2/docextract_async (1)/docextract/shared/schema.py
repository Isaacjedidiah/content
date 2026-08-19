"""Shared data contracts used across every module.

Uses Pydantic v2. Validation failures route to a quarantine table rather
than crash a run (see ``build_chunk`` / ``build_claim``). The ``metrics``
field is deliberately a free ``dict`` so document schemas that vary by
version are absorbed rather than rejected.
"""
from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"
    KEYVAL = "keyval"


class ChunkType(str, Enum):
    RAW_TEXT = "raw_text"
    STRUCTURED_NL = "structured_nl"
    NORMALISATION_CONFLICT = "normalisation_conflict"


class CitationTier(str, Enum):
    """How a value was obtained — its source provenance, independent of the
    model's confidence or the review lifecycle. A deterministically parsed
    table cell and a value read from a chart image carry different real-world
    reliability even at the same confidence. (Adopted from the uploaded
    codebase's four-way tier.)"""
    PARSED = "parsed"                # deterministic structural output (table/text)
    LLM_ESTIMATED = "llm_estimated"  # read from interpreting a figure/chart image
    DERIVED = "derived"              # computed/matched downstream (e.g. claim link)
    MANUAL = "manual"                # entered/corrected by a human reviewer


class ReviewTier(str, Enum):
    """Where a value sits in the human-review lifecycle — a separate axis from
    CitationTier (provenance). A parsed value can still be human-confirmed;
    a chart-read value can still be auto-accepted."""
    MODEL_AUTO = "review_auto"
    HUMAN_CONFIRMED = "review_confirmed"
    HUMAN_OVERRIDE = "review_override"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- value parsing --------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_numeric(raw: str) -> Optional[float]:
    """Best-effort numeric parse of a regulatory value string.

    Handles '14.2%', '1,234', '£5m', ' 12.3 bps'. Returns ``None`` when no
    number can be recovered (caller decides how to treat that). Suffixes like
    'm'/'bn' are NOT scaled here — scale is a first-class structural field and
    scaling belongs in normalisation, not parsing.
    """
    if raw is None:
        return None
    m = _NUM_RE.search(str(raw).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


class Element(BaseModel):
    element_id: str
    modality: Modality
    content: str
    source_document: str
    page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    team: Optional[str] = None
    report_type: Optional[str] = None
    entity_ref: Optional[str] = None
    content_hash: str = ""
    # Structural signals from the parser, used by complexity routing and the
    # crop-and-zoom path. description carries the parser's caption for a
    # figure (used to detect chart-like figures); has_merged_cells marks
    # structurally complex tables.
    description: Optional[str] = None
    has_merged_cells: bool = False

    @field_validator("content_hash", mode="before")
    @classmethod
    def _fill_hash(cls, v: str, info: Any) -> str:
        return v or content_hash(info.data.get("content", ""))


class Claim(BaseModel):
    field_name: str                     # raw name as found
    canonical_metric: Optional[str] = None
    value: str
    unit: Optional[str] = None
    reporting_basis: Optional[str] = None
    netting: Optional[str] = None
    scale: Optional[str] = None
    as_at_date: Optional[str] = None
    confidence: float = 1.0
    source_element_id: str
    entity_ref: Optional[str] = None
    model_used: Optional[str] = None
    needs_review: bool = False
    # Provenance (how obtained) and review lifecycle are separate axes.
    citation_tier: CitationTier = CitationTier.PARSED
    review_tier: ReviewTier = ReviewTier.MODEL_AUTO
    # Structural classification (metric_normaliser): measure shape + period,
    # independent of the metric's name. reporting_basis above doubles as the
    # normaliser's "basis" axis.
    measure_type: Optional[str] = None
    period: Optional[str] = None
    # Figure locality: page + bbox let a chart-derived claim be cropped/zoomed
    # for re-reading and highlighted for a human reviewer.
    page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    # Advisory content-domain tag (risk / financial / tax / ... / unknown) so
    # teams can filter to their components. Advisory only — nothing is gated on
    # it. tag_source records which cascade rung produced it (heading/metric/
    # llm/unknown) for tag-quality auditing.
    domain_tag: Optional[str] = None
    tag_source: Optional[str] = None


class RegulatoryChunk(BaseModel):
    """A chunk pushed to Azure AI Search.

    We own chunking and push ``content`` as plain text; AI Search embeds it.
    ``chunk_id`` must be a valid AI Search document key (letters, digits,
    underscore, dash, equals) — ``content_hash`` satisfies this.
    """
    chunk_id: str
    chunk_type: ChunkType = ChunkType.RAW_TEXT
    content: str
    entity_ref: str = Field(..., description="the pipeline firm reference number or LEI")
    source_document_id: str
    content_hash: str
    schema_version: str = "1.0"
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_ref")
    @classmethod
    def _entity_ref_required(cls, v: str) -> str:
        if not v:
            raise ValueError("entity_ref required (the pipeline ref or LEI)")
        return v


class GoldMetric(BaseModel):
    entity_ref: str
    canonical_metric: str
    value: str
    unit: Optional[str] = None
    reporting_basis: Optional[str] = None
    scale: Optional[str] = None
    as_at_date: Optional[str] = None
    netting: Optional[str] = None
    measure_type: Optional[str] = None
    period: Optional[str] = None
    citation_tier: str = CitationTier.PARSED.value
    review_tier: str = ReviewTier.MODEL_AUTO.value
    # Advisory content-domain tag, carried into Gold so teams can filter to
    # their components when querying the curated table.
    domain_tag: Optional[str] = None


def build_chunk(raw: dict, quarantine: list[dict]) -> Optional[RegulatoryChunk]:
    """Validate a raw dict into a chunk; route failures to quarantine."""
    try:
        return RegulatoryChunk(**raw)
    except Exception as exc:  # pydantic.ValidationError or TypeError
        quarantine.append({"kind": "chunk", "raw": raw, "error": str(exc)})
        return None


def build_claim(raw: dict, quarantine: list[dict]) -> Optional[Claim]:
    """Validate a raw dict into a Claim; route failures to quarantine.

    Mirrors ``build_chunk`` so a malformed model row flags-not-crashes,
    consistent with the pipeline's flag-never-reject principle.
    """
    try:
        return Claim(**raw)
    except Exception as exc:
        quarantine.append({"kind": "claim", "raw": raw, "error": str(exc)})
        return None
