"""Delta -> Azure AI Search bridge for the native track.

Reads newly-written narrative / conflict rows from a Delta table and upserts
them into the Azure AI Search index. Incremental and idempotent via
content_hash (== chunk_id, the index key), so re-running never duplicates.

Rows missing a entity_ref are routed to quarantine rather than indexed under a
catch-all sentinel — firm attribution is the point of the product.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..shared.schema import ChunkType, RegulatoryChunk, build_chunk


class VectorSync:
    def __init__(self, search_store, watermark: Optional[set[str]] = None):
        self.search_store = search_store
        # In production the watermark is a Delta table of synced hashes.
        self._synced: set[str] = set(watermark or set())

    def sync(self, delta_rows: Iterable[dict],
             quarantine: Optional[list[dict]] = None) -> int:
        quarantine = quarantine if quarantine is not None else []
        batch: list[RegulatoryChunk] = []
        for row in delta_rows:
            h = row["content_hash"]
            if h in self._synced:
                continue
            if not row.get("entity_ref"):
                quarantine.append({"kind": "chunk_no_firm", "raw": row,
                                   "error": "missing entity_ref"})
                continue
            chunk = build_chunk({
                "chunk_id": h,
                "chunk_type": ChunkType(row.get("chunk_type", "raw_text")).value,
                "content": row["text"],
                "entity_ref": row["entity_ref"],
                "source_document_id": row.get("source_document_id", ""),
                "content_hash": h,
            }, quarantine)
            if chunk is not None:
                batch.append(chunk)
                self._synced.add(h)
        return self.search_store.index_many(batch) if batch else 0
