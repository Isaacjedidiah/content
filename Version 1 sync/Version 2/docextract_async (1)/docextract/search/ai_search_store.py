"""Databricks AI Search vector store (Delta Sync).

Formerly backed by Azure AI Search; rewritten to use Databricks AI Search
(formerly Databricks Vector Search) so the vector store lives inside the same
governed Unity Catalog schema as the rest of the pipeline — no separate Azure
service to provision or govern.

Division of labour (unchanged in spirit):
  * WE own chunking (upstream, in the pipeline) and APPEND chunk rows to a
    governed Delta table (``ops_search_chunks``) — like every other table.
  * Databricks AI Search owns embedding + indexing: a Delta Sync Index mirrors
    the chunks table; a triggered ``sync()`` after each batch embeds the new
    rows (via the managed embedding model) and makes them searchable.

The class name ``AzureAISearchStore`` is retained so callers don't change; the
implementation underneath is Databricks AI Search.

Verified for STRUCTURE + INTERFACE against the databricks-ai-search SDK docs.
The real AISearchClient calls, embedding, and sync behaviour are proven only on
a live workspace — confirm on the first real run.
"""
from __future__ import annotations

from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import RegulatoryChunk

# Columns in the source chunks table. content is the embedded text column.
_COL_ID = "chunk_id"
_COL_DOC = "content"


class AzureAISearchStore:
    """Delta Sync vector store over a governed chunks table.

    Interface preserved from the Azure implementation: ensure_index(),
    index(chunk), index_many(chunks), search(...). Callers are unchanged.
    """

    def __init__(self, index_name: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_key: Optional[str] = None):
        # Signature kept for call-site compatibility; endpoint/api_key now
        # resolve to Databricks AI Search settings.
        self._index_name = index_name or CONFIG.storage.search_index
        self._source_table = CONFIG.storage.search_chunks_table
        self._endpoint_name = endpoint or CONFIG.search.endpoint_name
        self._client = None
        self._index = None

    # -- client ------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from databricks.ai_search.client import AISearchClient
            self._client = AISearchClient()
        return self._client

    def _spark(self):
        # Databricks injects `spark` at runtime; import lazily so the module
        # imports off-cluster (tests use the stub, not this class).
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()

    # -- index provisioning -----------------------------------------------

    def ensure_index(self) -> None:
        """Ensure the source chunks table and the Delta Sync Index both exist.

        Idempotent: creates the chunks Delta table (with Change Data Feed, which
        Delta Sync requires on standard endpoints) if absent, then creates the
        Delta Sync Index pointing at it if absent. Safe to call every run.
        """
        spark = self._spark()
        # 1. source chunks table — governed Delta table, CDF on for Delta Sync.
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._source_table} (
                {_COL_ID}           STRING,
                {_COL_DOC}          STRING,
                chunk_type          STRING,
                entity_ref          STRING,
                source_document_id  STRING,
                content_hash        STRING
            ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        # 2. Delta Sync Index with Databricks-managed embedding.
        client = self._get_client()
        sc = CONFIG.search
        try:
            self._index = client.get_index(index_name=self._index_name)
        except Exception:
            self._index = client.create_delta_sync_index(
                endpoint_name=self._endpoint_name,
                source_table_name=self._source_table,
                index_name=self._index_name,
                pipeline_type=sc.sync_mode,           # TRIGGERED
                primary_key=sc.primary_key,
                embedding_source_column=sc.embedding_source_column,
                embedding_model_endpoint_name=sc.embedding_model,
            )

    # -- indexing ----------------------------------------------------------

    def index(self, chunk: RegulatoryChunk) -> None:
        """Append one chunk and sync. Idempotent on chunk_id (== content_hash)."""
        self.index_many([chunk])

    def index_many(self, chunks: list[RegulatoryChunk]) -> int:
        """Append chunks to the source Delta table, then trigger a sync so the
        index embeds + includes them.

        Append-across-runs, like every other table. Re-appending the same
        chunk_id is deduplicated at query time by the index's primary key.
        """
        if not chunks:
            return 0
        spark = self._spark()
        rows = [{
            _COL_ID: c.chunk_id,
            _COL_DOC: c.content,
            "chunk_type": c.chunk_type.value,
            "entity_ref": c.entity_ref,
            "source_document_id": c.source_document_id,
            "content_hash": c.content_hash,
        } for c in chunks]
        spark.createDataFrame(rows).write.format("delta").mode(
            "append").saveAsTable(self._source_table)
        # trigger the index to pick up the newly appended rows
        if self._index is None:
            self._index = self._get_client().get_index(index_name=self._index_name)
        self._index.sync()
        return len(rows)

    # -- query -------------------------------------------------------------

    def search(self, query: str, top_k: int = 5,
               team_filter: Optional[str] = None,
               entity_ref: Optional[str] = None) -> list[dict]:
        """Similarity search. Query text is embedded by the managed model.

        Returns the same shape as the Azure implementation:
        [{text, metadata:{...}, score}], so callers are unchanged.
        """
        if self._index is None:
            self._index = self._get_client().get_index(index_name=self._index_name)

        filters = {}
        if entity_ref:
            filters["entity_ref"] = entity_ref

        resp = self._index.similarity_search(
            query_text=query,
            columns=[_COL_ID, _COL_DOC, "chunk_type", "entity_ref",
                     "source_document_id", "content_hash"],
            num_results=top_k,
            filters=filters or None,
        )
        return _parse_results(resp)


def _parse_results(resp) -> list[dict]:
    """Normalise a similarity_search response into [{text, metadata, score}].

    The SDK returns results under result.data_array with a column manifest;
    map columns by name so ordering changes don't break parsing.
    """
    try:
        result = resp.get("result", {})
        manifest = resp.get("manifest", {}).get("columns", [])
        names = [c.get("name") for c in manifest]
        rows = result.get("data_array", []) or []
    except AttributeError:
        return []

    out: list[dict] = []
    for row in rows:
        rec = dict(zip(names, row))
        out.append({
            "text": rec.get(_COL_DOC, ""),
            "metadata": {
                "chunk_type": rec.get("chunk_type"),
                "entity_ref": rec.get("entity_ref"),
                "source_document_id": rec.get("source_document_id"),
                "content_hash": rec.get("content_hash"),
            },
            "score": rec.get("score"),
        })
    return out
