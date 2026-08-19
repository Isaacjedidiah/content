# the pipeline Regulatory Data Product

End-to-end content-extraction pipeline for document submissions, targeting **Azure Databricks** with **Azure AI Search** as the vector store.

Two interchangeable tracks share one set of data contracts (`docextract/shared`):

- **Custom track** (`docextract/custom`): a content-aware two-tier LLM cascade (GPT-5 mini → Claude Sonnet → human) with cost tracking, reconciliation, human review, and governance gates. A cheap deterministic routing check sends known-hard content (tables, over-long or flagged-dense elements) straight to Tier 2, skipping the wasted Tier 1 pass; everything else starts on Tier 1 and escalates reactively. This is the primary extraction path.
- **Native track** (`docextract/native`): a Lakeflow Declarative Pipeline using Databricks `ai_parse_document` / `ai_extract` / `ai_classify`, producing the same Gold shape.

## Vector store: Azure AI Search

Division of labour, per the deployment decision — **you chunk, AI Search embeds**:

- The pipeline owns chunking and pushes chunk **text** to the index (`AzureAISearchStore.index_many`).
- Azure AI Search owns embedding: a skillset with an `AzureOpenAIEmbedding` skill embeds pushed content (`resources/ai_search_skillset.json`), and a **vectorizer** bound to the vector field embeds the query text at search time. Application code never calls an embedding API.
- Model: `text-embedding-3-large` @ 3072 dims (configurable in `SearchConfig`).

This replaces the previous ChromaDB store, which relied on memory-mapped SQLite on a Unity Catalog Volume — unreliable under concurrent writes on FUSE-mounted storage.

## Data flow (custom track)

```
preprocess ─▶ route ─▶ extract (cascade) ─▶ reconcile ─▶ gate_for_review ─▶ Silver (all) + Gold (clean)
   │            │              │                  │              │                      │
 entity_ref   start tier    quarantine         magnitude       needs_review        structural conflicts
 on Element from content  bad rows/pages     conflicts        → human            → Azure AI Search
            (modality/                                        → apply_decision → promote to Gold
             length/type)
```

Routing (`shared/routing.py`) picks each element's starting tier before any
model call: tables, over-long, and flagged-dense elements start on Tier 2
directly (no wasted Tier 1 pass); everything else starts on Tier 1 and
escalates reactively on low confidence. The policy is data-driven
(`RoutingPolicy` in `shared/config.py`) and every decision is audited.

Every stage emits structured audit events, persisted to the audit sink.

## Install & test

```bash
pip install -e ".[apps,dev]"
pytest -q
```

## Deploy to Azure Databricks

1. **Secrets** — create scope `docextract` with: `anthropic-api-key`, `openai-api-key`, `search-api-key`, `aoai-api-key`. (Prefer managed identity / Entra ID for AI Search + AOAI where possible; keys are the dev fallback.)
2. **Environment** — set `AZURE_SEARCH_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`.
3. **AI Search** — deploy the index (`AzureAISearchStore.ensure_index()`), then create the skillset from `resources/ai_search_skillset.json` and an indexer to embed pushed content.
4. **Bundle** — `databricks bundle deploy -t prod` (see `resources/databricks.yml`).
5. **Landing layout** — `<volume_root>/<team>/<report_type>/<file>`. The orchestration scans every team folder, then every report_type folder under it. `entity_ref` is resolved from the filename convention (`entityID_entityName_documentTitle_reportDate.ext`) and stays a column in the data; Bronze/Silver/Gold are written per `<team>_<report_type>` (e.g. `silver.fx_exchange_report`).

## Azure preflight checklist (verify BEFORE first run)

These are environment assumptions the code cannot self-check:

- [ ] Serving endpoints exist in your workspace/region: `gpt-5-mini` (External Model), `databricks-claude-sonnet`, `databricks-ai-parse` (used by the native track and by the custom track's scanned-page OCR fallback), and (native track) `databricks-meta-llama`.
- [ ] `ai_parse_document` / `ai_extract` / `ai_classify` are GA in your Azure region and DBR version.
- [ ] Azure OpenAI has a `text-embedding-3-large` deployment reachable by the AI Search service identity.
- [ ] AI Search service identity has `Cognitive Services OpenAI User` on the AOAI resource (for keyless vectorizer).
- [ ] The Databricks job/pipeline identity can read the landing Volume and write the target catalog/schemas.

## Key modules

| Area | Module |
|------|--------|
| Config, teams, thresholds, Azure/Search settings | `shared/config.py` |
| Data contracts (Element, Claim, Chunk, Gold) | `shared/schema.py` |
| Content-aware starting-tier routing | `shared/routing.py` |
| Document preprocessing (pdfplumber + OCR fallback) | `custom/preprocessor.py` |
| Scanned-page OCR via ai_parse_document | `custom/ai_parse_client.py` |
| Crop-and-zoom for chart figures | `search/figure_preprocessor.py` |
| Claim↔quant linking | `custom/claim_quant_matcher.py` |
| Structural metric classification | `shared/metric_normaliser.py` |
| Extraction cascade | `custom/extractor.py` |
| Reconciliation + review gating | `custom/validator.py` |
| Human review loop → Gold promotion | `custom/reviewer.py` |
| Storage (Delta/JSONL, idempotent) | `custom/storage.py` |
| Orchestrator | `custom/production.py` |
| Azure AI Search store | `search/ai_search_store.py` |
| Governance gates | `shared/gates.py` |
| Stored prompt registry (team/report_type few-shots) | `shared/prompt_registry.py` |
| Native Lakeflow pipeline | `native/native_pipeline.py` |
