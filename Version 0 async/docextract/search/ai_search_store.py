"""Azure AI Search vector store (replaces the portable ChromaDB store).

Division of labour, per the deployment decision:
  * WE own chunking (upstream, in the pipeline).
  * Azure AI Search owns embedding and indexing. We push chunk *text*; an
    indexer/skillset with an AzureOpenAIEmbedding skill vectorises pushed
    content, and a vectorizer bound to the vector field vectorises the query
    text at search time. Our code never calls an embedding API.

Why this over ChromaDB-on-a-Volume: ChromaDB's PersistentClient uses
memory-mapped SQLite, which is unreliable on FUSE-mounted Unity Catalog
Volumes under concurrent writes. Azure AI Search is a managed service with
no such constraint and gives hybrid + semantic ranking for free.

Auth: keyless (Entra ID / managed identity) is preferred in production; an
admin key is supported for local/dev via the ``search-api-key`` secret.
"""
from __future__ import annotations

from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import RegulatoryChunk

# Metadata fields we filter/scope on. ``entity_ref`` powers supervisor team
# scoping (via a firm->team mapping) and per-firm retrieval.
_FIELDS_DOC = "content"
_FIELDS_VECTOR = "content_vector"


class AzureAISearchStore:
    """Push-based vector store over an Azure AI Search index."""

    def __init__(self, index_name: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._index_name = index_name or CONFIG.search.index_name
        self._endpoint = endpoint or CONFIG.azure.search_endpoint
        self._api_key = api_key or CONFIG.search_api_key
        self._search_client = None
        self._index_client = None

    # -- credentials -------------------------------------------------------

    def _credential(self):
        """Prefer a key when supplied; else fall back to Entra ID identity."""
        if self._api_key:
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(self._api_key)
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

    def _require_endpoint(self) -> str:
        if not self._endpoint:
            raise RuntimeError(
                "AZURE_SEARCH_ENDPOINT is not set; refusing to fall back to "
                "a stub. Configure the Azure AI Search endpoint."
            )
        return self._endpoint

    def _client(self):
        if self._search_client is None:
            from azure.search.documents import SearchClient

            self._search_client = SearchClient(
                endpoint=self._require_endpoint(),
                index_name=self._index_name,
                credential=self._credential(),
            )
        return self._search_client

    # -- index provisioning -----------------------------------------------

    def ensure_index(self) -> None:
        """Create the index with integrated vectorization if it is absent.

        The vector field is bound to a profile whose vectorizer points at an
        Azure OpenAI embedding deployment, so both push-time embedding (via
        skillset) and query-time embedding are handled by the service.
        """
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchIndex, SearchField, SearchFieldDataType, SimpleField,
            SearchableField, VectorSearch, VectorSearchProfile,
            HnswAlgorithmConfiguration, AzureOpenAIVectorizer,
            AzureOpenAIVectorizerParameters, SemanticConfiguration,
            SemanticPrioritizedFields, SemanticField, SemanticSearch,
        )

        sc = CONFIG.search
        index_client = SearchIndexClient(
            endpoint=self._require_endpoint(), credential=self._credential())

        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String,
                        key=True, filterable=True),
            SearchableField(name=_FIELDS_DOC, type=SearchFieldDataType.String),
            SearchField(
                name=_FIELDS_VECTOR,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=sc.embedding_dimensions,
                vector_search_profile_name=sc.vector_profile),
            SimpleField(name="chunk_type", type=SearchFieldDataType.String,
                        filterable=True),
            SimpleField(name="entity_ref", type=SearchFieldDataType.String,
                        filterable=True),
            SimpleField(name="source_document_id",
                        type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="content_hash", type=SearchFieldDataType.String,
                        filterable=True),
        ]

        vectorizer = AzureOpenAIVectorizer(
            vectorizer_name=sc.vectorizer_name,
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=CONFIG.azure.aoai_endpoint,
                deployment_name=sc.embedding_deployment,
                model_name=sc.embedding_deployment,
                # api_key omitted => managed identity used by the service.
                api_key=CONFIG.aoai_api_key,
            ),
        )
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="reg-hnsw")],
            profiles=[VectorSearchProfile(
                name=sc.vector_profile, algorithm_configuration_name="reg-hnsw",
                vectorizer_name=sc.vectorizer_name)],
            vectorizers=[vectorizer],
        )
        semantic = SemanticSearch(configurations=[SemanticConfiguration(
            name=sc.semantic_config,
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name=_FIELDS_DOC)]))])

        index = SearchIndex(name=self._index_name, fields=fields,
                            vector_search=vector_search, semantic_search=semantic)
        index_client.create_or_update_index(index)

    # -- indexing ----------------------------------------------------------

    def index(self, chunk: RegulatoryChunk) -> None:
        """Upsert one chunk. Idempotent on ``chunk_id`` (== content_hash).

        We upload the TEXT only. Embedding of pushed content is performed by
        the index's skillset / integrated vectorization, not here.
        """
        self.index_many([chunk])

    def index_many(self, chunks: list[RegulatoryChunk]) -> int:
        if not chunks:
            return 0
        docs = [{
            "chunk_id": c.chunk_id,
            _FIELDS_DOC: c.content,
            "chunk_type": c.chunk_type.value,
            "entity_ref": c.entity_ref,
            "source_document_id": c.source_document_id,
            "content_hash": c.content_hash,
        } for c in chunks]
        # mergeOrUpload = upsert: re-pushing the same chunk_id is a no-op
        # update, giving idempotency on re-runs.
        self._client().merge_or_upload_documents(documents=docs)
        return len(docs)

    # -- query -------------------------------------------------------------

    def search(self, query: str, top_k: int = 5,
               team_filter: Optional[str] = None,
               entity_ref: Optional[str] = None) -> list[dict]:
        """Vector search. Query text is vectorised by the service vectorizer.

        ``team_filter`` is resolved to firms by the caller normally; here we
        accept an explicit ``entity_ref`` filter for direct scoping.
        """
        from azure.search.documents.models import VectorizableTextQuery

        filters = []
        if entity_ref:
            filters.append(f"entity_ref eq '{entity_ref}'")
        filter_expr = " and ".join(filters) if filters else None

        vq = VectorizableTextQuery(
            text=query, k_nearest_neighbors=top_k, fields=_FIELDS_VECTOR)

        results = self._client().search(
            search_text=query,               # hybrid: keyword + vector
            vector_queries=[vq],
            filter=filter_expr,
            top=top_k,
        )
        out: list[dict] = []
        for r in results:
            out.append({
                "text": r.get(_FIELDS_DOC, ""),
                "metadata": {
                    "chunk_type": r.get("chunk_type"),
                    "entity_ref": r.get("entity_ref"),
                    "source_document_id": r.get("source_document_id"),
                    "content_hash": r.get("content_hash"),
                },
                "score": r.get("@search.score"),
            })
        return out
