"""Batch orchestrator for the custom track.

Wires preprocess -> extract -> reconcile/gate -> store -> index, per
submission, with audit throughout. Run-level counters are threaded from
return values.

End-to-end responsibilities now fully wired:
  * entity_ref flows from the submission descriptor onto every Element/Claim.
  * Malformed rows and empty (scanned) pages go to quarantine, not silence.
  * Silver holds all claims; Gold holds only clean (auto or human-signed).
  * Structural conflicts are indexed to Azure AI Search as searchable chunks.
  * Every stage emits audit events, persisted to the audit sink.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..shared.audit_log import AuditLog
from ..shared.metric_normaliser import check_compatible, to_conflict_chunk
from ..shared.schema import ChunkType, Modality, RegulatoryChunk, content_hash
from .claim_quant_matcher import match_all
from .extractor import Extractor
from .preprocessor import preprocess, _footnote_markers
from .evaluator import maybe_evaluate
from .storage import DeltaLakeStorage
from .validator import gate_for_review, reconcile


@dataclass
class BatchResult:
    documents: int = 0
    gold_metrics_total: int = 0
    total_cost_usd: float = 0.0
    needs_review: int = 0
    conflicts: int = 0
    quarantined: int = 0
    cropped_figures: int = 0
    claim_links: int = 0
    evaluated_documents: int = 0
    eval_graded: int = 0
    eval_correct: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


async def process_batch(submissions: list[dict],
                  extractor: Optional[Extractor] = None,
                  storage: Optional[DeltaLakeStorage] = None,
                  search_store=None,
                  audit: Optional[AuditLog] = None,
                  page_image_provider=None,
                  link_claims: bool = False,
                  eval_every_n: int = 0) -> BatchResult:
    """submissions: [{path, team, report_type, entity_ref}, ...].

    When ``page_image_provider`` is supplied, chart-like figure elements are
    cropped-and-zoomed and read multimodally. Left None, extraction is
    text-only (crop-and-zoom skipped) — a safe default, not a data risk.
    """
    extractor = extractor or Extractor()
    storage = storage or DeltaLakeStorage()
    audit = audit or AuditLog()
    result = BatchResult()

    for sub in submissions:
        result.documents += 1
        entity_ref = sub.get("entity_ref")

        elements = preprocess(sub["path"], sub["team"], sub["report_type"],
                              entity_ref=entity_ref)
        audit.log(result.run_id, "preprocess", sub["path"],
                  elements=len(elements), entity_ref=entity_ref)

        # Read the source bytes once so the crop-and-zoom provider can
        # rasterise figure pages. If unreadable, extraction proceeds
        # text-only (provider will get None and degrade per element).
        doc_bytes = None
        if page_image_provider is not None:
            try:
                with open(sub["path"], "rb") as fh:
                    doc_bytes = fh.read()
            except OSError:
                doc_bytes = None

        extraction = await extractor.extract(
            elements, document_bytes=doc_bytes,
            filename=sub["path"], page_image_provider=page_image_provider)
        result.total_cost_usd += extraction.total_cost_usd
        result.quarantined += len(extraction.quarantine)
        result.cropped_figures += extraction.cropped_figures
        if extraction.quarantine:
            storage.write_quarantine(extraction.quarantine)
        # Record the content-aware routing decision for each element.
        for element_id, decision in extraction.routing.items():
            audit.log(result.run_id, "route", element_id,
                      start_tier=decision.tier, reason=decision.reason,
                      content_aware=decision.content_aware)
        for c in extraction.claims:
            audit.log(result.run_id, "extract", c.source_element_id,
                      metric=c.canonical_metric, confidence=c.confidence,
                      model=c.model_used, entity_ref=c.entity_ref)

        # Claim↔quant linking: connect narrative sentences to the metric that
        # supports them. Optional (needs an LLM client) and best-effort — a
        # failure here never blocks the medallion write.
        if link_claims and extraction.claims:
            narrative = [e.content for e in elements
                         if e.modality == Modality.TEXT and e.content.strip()]
            try:
                links = await match_all(narrative, extraction.claims,
                                  extractor._client)
            except Exception:
                links = []
            resolved = [l for l in links if l.matched_metric]
            result.claim_links += len(resolved)
            for l in resolved:
                audit.log(result.run_id, "claim_link", sub["path"],
                          claim=l.claim_text[:120], metric=l.matched_metric,
                          confidence=l.confidence)
            if search_store is not None and resolved:
                _index_claim_links(resolved, sub, search_store, entity_ref)

        conflicts = reconcile(extraction.claims)
        result.conflicts += len(conflicts)

        gated = gate_for_review(extraction.claims, conflicts)
        result.needs_review += len(gated["needs_review"])

        storage.write_silver(extraction.claims, team=sub["team"],
                             report_type=sub["report_type"])
        result.gold_metrics_total += storage.write_gold_metrics(
            gated["clean"], team=sub["team"], report_type=sub["report_type"])

        # Structural conflicts become searchable chunks (not Gold columns).
        if search_store is not None:
            n = _index_conflicts(extraction.claims, sub, search_store, entity_ref)
            audit.log(result.run_id, "index_conflicts", sub["path"], indexed=n)

            # Index the raw narrative + footnote text so the chat can retrieve
            # context (e.g. a footnote qualifying a table figure). Without this
            # path, text/footnotes never reach the vector store at all.
            t = _index_text_elements(elements, sub, search_store, entity_ref)
            audit.log(result.run_id, "index_text", sub["path"], indexed=t)

        # Automatic eval on a deterministic 1-in-N sample. Scores this
        # document's claims against INDEPENDENT ground truth (team-registered
        # expected values + human-reviewed decisions) — never against the Gold
        # table it just produced. eval_every_n=0 disables it.
        if eval_every_n:
            ev = maybe_evaluate(sub["path"], extraction.claims, sub["team"],
                                sub["report_type"], entity_ref,
                                every_n=eval_every_n)
            if ev is not None:
                result.evaluated_documents += 1
                result.eval_graded += ev.graded
                result.eval_correct += ev.correct
                audit.log(result.run_id, "eval", sub["path"],
                          graded=ev.graded, correct=ev.correct,
                          accuracy=ev.accuracy)

    storage.write_audit(audit.to_rows())
    return result


def _index_conflicts(claims, sub, search_store, entity_ref) -> int:
    by_metric: dict[str, list] = {}
    for c in claims:
        by_metric.setdefault(c.canonical_metric or c.field_name, []).append(c)
    chunks = []
    for metric, group in by_metric.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                r = check_compatible(group[i], group[j])
                if not r.compatible:
                    chunks.append(to_conflict_chunk(
                        metric, group[i], group[j], r,
                        entity_ref=entity_ref or "UNKNOWN_ENTITY_REF",
                        source_document_id=sub["path"]))
    if chunks:
        search_store.index_many(chunks)
    return len(chunks)


def _index_claim_links(links, sub, search_store, entity_ref) -> int:
    """Push resolved claim→metric links to search as their own chunk type, so
    'what's the evidence for this claim' has something to retrieve."""
    chunks = []
    for l in links:
        text = (f'Claim: "{l.claim_text}" is supported by metric '
                f'{l.matched_metric}.')
        chunks.append(RegulatoryChunk(
            chunk_id=content_hash(text),
            chunk_type=ChunkType.STRUCTURED_NL,
            content=text,
            entity_ref=entity_ref or "UNKNOWN_ENTITY_REF",
            source_document_id=sub["path"],
            content_hash=content_hash(text)))
    if chunks:
        search_store.index_many(chunks)
    return len(chunks)


def _index_text_elements(elements, sub, search_store, entity_ref) -> int:
    """Index raw narrative + footnote text into the vector store.

    This is the path that lets the chat retrieve context that never becomes a
    Gold column — most importantly a footnote that qualifies a table figure
    (e.g. "SLR of which 3.0% is the minimum plus a 2.0% buffer").

    Each chunk carries page + bbox + any footnote-marker ids in its ``metrics``
    metadata bag, so:
      * a query can be ranked/located by page and position, and
      * a footnote chunk and the table chunk that references the same marker
        share ``footnote_markers`` — an explicit link that survives even when
        the footnote and its table separate across a page break.
    """
    chunks = []
    for e in elements:
        if e.modality not in (Modality.TEXT, Modality.TABLE):
            continue
        content = (e.content or "").strip()
        if not content:
            continue
        markers = _footnote_markers(content)
        meta = {
            "page": e.page,
            "bbox": list(e.bbox) if e.bbox else None,
            "modality": e.modality.value,
            "source_element_id": e.element_id,
        }
        if markers:
            meta["footnote_markers"] = markers
        chunks.append(RegulatoryChunk(
            chunk_id=content_hash(content + (e.element_id or "")),
            chunk_type=ChunkType.RAW_TEXT,
            content=content,
            entity_ref=entity_ref or "UNKNOWN_ENTITY_REF",
            source_document_id=sub["path"],
            content_hash=content_hash(content),
            metrics=meta))
    if chunks:
        search_store.index_many(chunks)
    return len(chunks)
