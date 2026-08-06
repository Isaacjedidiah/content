# DocExtract — technical reference

A module-by-module reference for engineers: what each module does, the key
technical principles behind it, the libraries it uses, why it matters, and how
it connects to the rest of the system. Grouped by subpackage.

The package has four layers, and dependencies flow one way:

```
shared/   ← foundations: contracts, config, clients, cross-cutting concerns
custom/   ← the per-document extraction pipeline (depends on shared, search)
search/   ← Azure AI Search + figure preprocessing (depends on shared)
native/   ← the alternative Databricks-native track (depends on shared)
```

`shared/` never imports from the other three — that one-way rule is what keeps
the foundations reusable by both extraction tracks. Two companion diagrams
show the flow visually: `module_flow.svg` (one document's path) and
`system_map.svg` (all flows at a higher level).

---

## shared/ — foundations

### schema.py
**What it does.** Defines the data contracts every other module speaks:
`Element` (a parsed piece of a document — text, table, or figure), `Claim` (an
extracted value with its metadata), `RegulatoryChunk` (a searchable text unit),
and `GoldMetric` (the final trusted record). Also the two enums that encode the
trust model — `CitationTier` (provenance: parsed / llm_estimated / derived /
manual) and `ReviewTier` (lifecycle: auto / confirmed / override) — plus
helpers like `parse_numeric`, `content_hash`, `build_claim`, `build_chunk`.

**Key principles.** Two-axis trust modelling (provenance separate from review
state). Content-hash identity for idempotent dedup. A single source of truth
for shapes, so no module invents its own.

**Libraries.** `hashlib` (content hashing), `re` (numeric parsing), `enum`,
`typing`. Pydantic in production (a shim stands in during sandbox tests).

**Importance & connections.** The spine of the whole system — imported by
almost every other module. If you change a contract here, everything that
produces or consumes that shape is affected, so treat it as the most
load-bearing file in the package.

### config.py
**What it does.** Central configuration: the `Team` enum (canonical teams),
model-tier registry (which serving endpoint is tier1 vs tier2), thresholds
(escalation, review, per-metric review bars), routing policy, search config
(embedding model + dimensions), and `StoragePaths` — including the
`slug()`/`layer_table()` helpers that name per-`team_reporttype` tables and the
secret-scope resolution.

**Key principles.** One place for every tunable. Environment detection
(local vs Databricks) so the same code runs in both. Secrets resolved at
runtime, never hard-coded. Frozen dataclasses so config is read-only.

**Libraries.** `dataclasses`, `enum`, `os`, `typing`; `pyspark` only when
resolving Databricks secrets.

**Importance & connections.** Imported nearly everywhere. The single knob-board
— endpoints, thresholds, table names all live here, so it's the first file to
edit when wiring to a real environment.

### llm_client.py
**What it does.** A unified client over Databricks serving endpoints
(OpenAI-compatible). Exposes `complete()` for text and `extract()` for
forced-JSON tool calls, with an optional `image_base64` parameter for
multimodal (chart) reads. Holds `EXTRACTION_TOOL_SCHEMA`, the tool definition
that forces structured claim output.

**Key principles.** Forced-JSON via tool-calling (the model must return valid
structured output, not prose to be regex-parsed). One client abstraction over
all model tiers, so callers don't care which endpoint they hit. Multimodal is
an optional argument, not a separate path.

**Libraries.** `openai` (the OpenAI-compatible SDK pointed at Databricks
serving), `json`, `dataclasses`, `os`, `typing`.

**Importance & connections.** The single gateway to every model call. Used by
`extractor`, `claim_quant_matcher`, and `llm_router`. Depends only on `config`.

### dictionary.py
**What it does.** Canonical-name resolution — maps the many ways a metric or
document type is written ("Primary ratio", "Common Equity Tier 1 %") to one
canonical key (`primary_ratio`).

**Key principles.** Deterministic normalisation before any downstream grouping;
a cheap rule-based step, not a model call.

**Libraries.** `re`.

**Importance & connections.** Used by `extractor` to canonicalise each claim.
Feeds reconciliation and storage, which group by canonical metric. Pure, no
internal dependencies.

### metric_normaliser.py
**What it does.** Two jobs: (1) classify a metric's *structural shape* —
`measure_type` (level/rate/change/ratio) and `period` (point-in-time / YoY /
QoQ) from its name; (2) `check_compatible()` — decide whether two values are
genuinely like-for-like, and `to_conflict_chunk()` to turn an incompatibility
into a searchable record.

**Key principles.** Structural compatibility *before* magnitude comparison —
don't compare a YoY change with a point-in-time level. UNKNOWN is treated as
compatible (benefit of the doubt) to avoid false positives.

**Libraries.** `re`, `enum`, `dataclasses`, `typing`; imports `schema`.

**Importance & connections.** Called by `extractor` (to tag each claim) and by
`production`/`validator` (for structural reconciliation). Depends on `schema`.

### routing.py
**What it does.** `decide_start_tier()` — picks the starting model tier for an
element from cheap structural signals (modality, table complexity, text length,
report type) *before* any API call.

**Key principles.** Model selection as a cheap classification problem — no LLM
call to decide which LLM to call. Known-hard content (charts, merged-cell
tables, long text) starts on the stronger tier; the easy majority starts cheap.

**Libraries.** `dataclasses`; imports `config` (RoutingPolicy) and `schema`.

**Importance & connections.** Used by `extractor` at the top of each element's
processing. Its decision shapes cost. Depends on `config`, `schema`.

### prompts.py
**What it does.** Builds the extraction prompt for a `(team, report_type)`.
Consults the stored `prompt_registry` first, then falls back to a base prompt
plus any built-in team overrides.

**Key principles.** Registry-first with a safe fallback, so a team that hasn't
registered prompts still gets a working one.

**Libraries.** none (pure Python); lazily imports `prompt_registry`.

**Importance & connections.** Used by `extractor`. Bridges to `prompt_registry`.

### prompt_registry.py
**What it does.** A stored database of few-shot prompts keyed by `(team,
report_type, kind)` where kind is `extraction` or `query`. JSONL locally, Delta
on Databricks. `get_prompt()` returns a registered entry or a default seed;
`register_prompt()` appends (latest wins); `list_missing()` surfaces gaps.

**Key principles.** Prompts as *data, not code* — onboarding a team is a data
change. Default-seed fallback so nothing ever blocks. Every write records who
changed it (audit posture).

**Libraries.** `json`, `os`, `dataclasses`, `datetime`, `typing`; imports
`config`. Env override `PRA_PROMPT_REGISTRY_DIR` for testability.

**Importance & connections.** Feeds both `prompts` (extraction) and `llm_router`
(query synthesis). The single store behind both extraction and the chat's
domain grounding.

### gates.py
**What it does.** Responsible-AI governance gates — `gate_external_release` and
`gate_production_deployment` enforce human sign-off (with ISO-8601 timestamps)
before an output leaves the building or a model goes to production.

**Key principles.** Governance as an explicit checkpoint, separate from the
data pipeline. Sign-off is recorded, not implied.

**Libraries.** `dataclasses`, `datetime`, `enum`.

**Importance & connections.** A cross-cutting concern, not on the per-document
path — it guards *release and deployment*. Standalone; no internal deps.

### audit_log.py
**What it does.** Structured, traceable audit logging — every pipeline stage
emits an event (run_id, stage, element, metric, confidence, model, entity_ref);
`to_rows()` renders them for persistence.

**Key principles.** Traceability by construction — any final number traces back
through model, confidence, and stage to its source. Append-only.

**Libraries.** `dataclasses`, `json`, `time`.

**Importance & connections.** Written by `production` throughout the run,
persisted via `storage`. The ingestion-side sibling of `supervisor_session`
(which logs the query side).

### cost_tracker.py
**What it does.** Cross-run cost accounting — `project()` estimates spend at
scale from per-tier token costs, with a `hard_share` for the fraction needing
the expensive model.

**Key principles.** Cost is projected and reconciled, not guessed. Separates
the offline projection tool from the live per-run tracking (which lives inline
in the extractor).

**Libraries.** `collections`.

**Importance & connections.** Cross-cutting analysis utility beside the
pipeline; complements the live cost captured in `extractor`. Standalone.

### monitoring.py
**What it does.** Aggregate monitoring — `PipelineMonitor` loads run metrics
and summarises throughput, cost, review backlog, quarantine and error rates.
`supervisor_feedback_summary()` aggregates the chat query log (volume, up/down,
unhelpful-rate).

**Key principles.** Aggregate health across *all* runs (vs. per-document
inspection). A dashboard, not a microscope.

**Libraries.** `json`, `os`, `dataclasses`; imports `config` and lazily
`supervisor_session`.

**Importance & connections.** Read by the monitoring app. Consumes what
`production` and `supervisor_session` write. Depends on `config`.

### supervisor_session.py
**What it does.** The supervisor chat's per-session store doing three jobs at
once: **memory** (recent turns + key figures, for follow-up resolution),
**audit** (every Q&A persisted to `supervisor_query_log`), and **feedback**
(thumbs up/down attached to a turn).

**Key principles.** One record, three jobs — memory, auditability, and a
quality signal fall out of the same per-session data. Best-effort persistence
that never blocks an answer. Per-session scope for memory; permanent for audit.

**Libraries.** `dataclasses`, `datetime`, `json`, `os`, `uuid`, `typing`;
imports `config`. Env override `PRA_SUPERVISOR_LOG_DIR`.

**Importance & connections.** Threaded through `llm_router.answer()`; surfaced
by `monitoring`. The query-side counterpart to `audit_log`.

### llm_router.py
**What it does.** The supervisor chat brain. `answer()` classifies a question
(SQL / narrative / both), generates read-only SQL against Gold, retrieves
narrative evidence from search, and synthesises a grounded answer using the
team's query few-shots. `resolve_references()` rewrites a follow-up into a
self-contained question using session memory.

**Key principles.** Reference-resolution by *rewrite*, not history-dumping.
Read-only SQL guard (SELECT-only). Registry-grounded synthesis (answers use the
domain's own examples, never invented figures).

**Libraries.** `re`, `enum`, `typing`; imports `config`, `llm_client`, and
lazily `prompt_registry` + `supervisor_session`.

**Importance & connections.** The consumption-side entry point. Uses
`llm_client`, `ai_search_store` (injected), `prompt_registry`,
`supervisor_session`. Not on the ingestion path.

---

## custom/ — the extraction pipeline

### preprocessor.py
**What it does.** Format-agnostic parsing — turns a PDF / Word / Excel file
into `Element` objects (text, tables, figures with bounding boxes). Falls back
to `ai_parse_client` (OCR) for scanned pages.

**Key principles.** One `Element` contract regardless of source format. Degrade
gracefully — a page that can't be parsed goes to OCR, then to quarantine, never
lost.

**Libraries.** `pdfplumber` (PDF text/tables/figure bboxes), `openpyxl`
(Excel), `os`, `typing`; imports `schema`, `ai_parse_client`.

**Importance & connections.** Stage 1 of the pipeline — everything downstream
works on its `Element` output. Feeds `extractor`.

### ai_parse_client.py
**What it does.** OCR fallback via the `databricks-ai-parse` serving endpoint —
extracts text from scanned/image-only pages the normal parser can't read.

**Key principles.** A targeted fallback, not the default path (OCR is slower and
costlier). Degrade-to-quarantine if OCR also fails.

**Libraries.** `openai` (serving client), `dataclasses`, `os`, `typing`;
imports `native.ai_functions_config` (endpoint config) and `config`.

**Importance & connections.** Called by `preprocessor` when a page has no text
layer. Bridges the custom track to the native track's endpoint config.

### figure_preprocessor.py *(lives in search/ but used by extractor)*
See under **search/** below — logically part of extraction.

### extractor.py
**What it does.** The two-tier extraction cascade. For each element: route to a
starting tier, crop-and-zoom chart figures for multimodal reads, build the
team/report prompt, call the model with forced-JSON, escalate low-confidence
tier1 output to tier2, canonicalise and structurally classify each claim, tag
provenance, and track cost live. Quarantines malformed output.

**Key principles.** Cost-shaped cascade (spend follows difficulty). Content-aware
routing. Crop-and-zoom for charts. Two-axis provenance tagging. Flag/quarantine,
never crash.

**Libraries.** `dataclasses`, `typing` (logic module — model access via
`llm_client`).

**Importance & connections.** The heart of the pipeline. Depends on `config`,
`dictionary`, `llm_client`, `metric_normaliser`, `prompts`, `routing`, `schema`,
and `search.figure_preprocessor`. Orchestrated by `production`.

### claim_quant_matcher.py
**What it does.** Links a narrative claim ("Primary metric strengthened") to the metric
that supports it — cheap lexical shortlist, then one tier1 confirm call; leaves
a claim UNLINKED below a confidence threshold rather than force-matching.

**Key principles.** Retrieve-then-confirm. Never force a match — a spurious link
is worse than none.

**Libraries.** `json`, `re`, `dataclasses`, `typing`; imports `config`,
`llm_client`, `schema`.

**Importance & connections.** Optional stage in `production` (`link_claims`
flag). Uses `llm_client`; produces links indexed to search.

### validator.py
**What it does.** `reconcile()` groups claims by structural signature and flags
magnitude conflicts and domain impossibilities (the primary metric ≤ total capital);
`gate_for_review()` splits claims into clean vs needs-review using per-metric
thresholds.

**Key principles.** Reconcile like-for-like, then flag. Per-metric review bars
(a misread ratio ≠ a misread date). Flag, never reject.

**Libraries.** `collections`, `dataclasses`, `typing`; imports `config`,
`schema`.

**Importance & connections.** Runs after extraction in `production`; its
gating decides what reaches Gold vs review. Depends on `config`, `schema`.

### reviewer.py
**What it does.** Applies a human review decision — `apply_decision()` folds an
approve/override into the claim, setting MANUAL provenance + HUMAN_OVERRIDE and
promoting the reviewed value toward Gold. Renders the bbox for the reviewer.

**Key principles.** Human decisions are first-class and audited. An override
updates provenance, so the trail shows a human changed it.

**Libraries.** `dataclasses`, `typing`; imports `schema`.

**Importance & connections.** Async, triggered by a person — not on the
automated path. `validator.gate_for_review` sends a claim to review; `reviewer`
acts on it. Depends on `schema`.

### storage.py
**What it does.** Medallion storage — writes Bronze / Silver / Gold to
per-`team_reporttype` tables (with `entity_ref` as a column), plus the shared ops
tables (quarantine, reviews, audit). Idempotent content-hash dedup. JSONL
locally, Delta on Databricks.

**Key principles.** Partition by domain (team × report type). Entity attribution
as a column, not a folder. Idempotent writes (re-runs are safe). Build-then-
persist separation.

**Libraries.** `json`, `os`, `typing`; imports `config`, `schema`.

**Importance & connections.** The persistence layer for `production` (and via
`vector_sync` for the native track). Depends on `config`, `schema`.

### production.py
**What it does.** The batch orchestrator — `process_batch()` runs the full
per-document pipeline: preprocess → extract → (optional) claim-link → reconcile
→ gate → write Silver/Gold → detect structural conflicts → index to search,
emitting audit events throughout.

**Key principles.** Explicit staged pipeline, each stage independently testable.
Audit at every stage. Best-effort side-channels (linking, indexing) never block
the medallion write.

**Libraries.** `dataclasses`, `typing`, `uuid`; imports `audit_log`,
`metric_normaliser`, `schema`, `claim_quant_matcher`, `extractor`,
`preprocessor`, `storage`, `validator`.

**Importance & connections.** The conductor — wires nearly every custom + shared
module together. Called by `run_ingestion_job` and `workflow_tasks`.

### run_ingestion_job.py
**What it does.** Batch entry point — `discover_submissions()` scans the volume
`<team>/<report_type>/<file>`, resolves `entity_ref` from the filename convention,
and hands submissions to `process_batch`.

**Key principles.** Scan every team/report folder; entity_ref from filename (not
path); a bad filename still processes (routed to review), never crashes.

**Libraries.** `os`, `re`; imports `ai_search_store`, `config`, `production`.

**Importance & connections.** The CLI/`main()` entry into the custom pipeline.
Calls `production`; wrapped by `workflow_tasks`.

### workflow_tasks.py
**What it does.** The three Databricks Job task entry points — `ingest_task`,
`index_task`, `healthcheck_task` — mirroring the left-to-right Jobs graph.
Ingest processes documents; index pushes chunks to AI Search as its own
re-runnable task; healthcheck fails red on an abnormal run.

**Key principles.** Split at boundaries where failure and recovery differ.
State handoff via storage (tasks don't share memory). Fail loudly (non-zero
exit) so the graph shows red.

**Libraries.** `datetime`, `json`, `os`; imports `ai_search_store`, `config`,
`production`, `run_ingestion_job`.

**Importance & connections.** The wrapper *around* the ingestion path — the box
that contains the whole `module_flow.svg`. Lane 1 of `system_map.svg`.

---

## search/ — vector search & figure preprocessing

### ai_search_store.py
**What it does.** The Azure AI Search vector store — `ensure_index()`
(provisions an HNSW index with an embedding skillset + query vectorizer +
semantic config), `index_many()` (push chunks), and hybrid `search()` with team
scoping. Replaces the portable ChromaDB store from the original design.

**Key principles.** Integrated vectorization (the index embeds at write and
query time). Push-only indexing. Hybrid (vector + keyword + semantic) retrieval.
Team-scoped queries.

**Libraries.** `azure-search-documents`, `azure-identity`, `azure-core`;
imports `config`, `schema`.

**Importance & connections.** The retrieval backend for `llm_router` (narrative
path) and the index target for `production`, `workflow_tasks`, and
`vector_sync`. Depends on `config`, `schema`.

### figure_preprocessor.py
**What it does.** Crop-and-zoom for chart extraction — crops a figure to its
bounding box (+2% padding), upscales the smaller dimension toward 1024px
(capped 2000), and provides a working `PdfPlumberPageImageProvider` that
rasterises the source page. Also `is_chart_like()` and `encode_image_base64()`.

**Key principles.** Turn "find and read a tiny chart" into "read this chart".
Pad the crop so axis labels survive. The model gets a *clean* crop (no reviewer
box drawn on it). Ships a real page-image source (the piece many open designs
leave stubbed).

**Libraries.** `PIL` (Pillow — crop/resize), `pdfplumber` (page rasterisation),
`re`, `typing`.

**Importance & connections.** Used by `extractor` for multimodal figure reads.
Standalone (no internal deps) so it's reusable.

---

## native/ — the alternative Databricks-native track

### ai_functions_config.py
**What it does.** Configuration for the native track — endpoint names and the
extract-schema for Databricks AI functions (`ai_parse`, `ai_extract`,
`ai_classify`), including `extract_fields_array`.

**Key principles.** Declarative extraction config, separate from the custom
track's imperative cascade.

**Libraries.** `dataclasses`.

**Importance & connections.** Used by `native_pipeline` and (for the endpoint)
by `ai_parse_client`. Standalone config.

### native_pipeline.py
**What it does.** A Lakeflow Declarative Pipeline that does Bronze → Silver →
Gold using Databricks AI functions instead of the custom Python cascade — same
Gold shape, different engine. Gold filters on confidence.

**Key principles.** Platform-native, managed simplicity — one output contract,
two interchangeable engines, so the org isn't locked in.

**Libraries.** Databricks/Lakeflow runtime (declarative); imports `config`,
`ai_functions_config`.

**Importance & connections.** The parallel track to the whole `custom/`
pipeline. Depends on `config`, `ai_functions_config`.

### vector_sync.py
**What it does.** Bridges native-track Delta Gold to Azure AI Search — reads
newly-approved Gold rows, builds `RegulatoryChunk`s, and pushes them to the
index.

**Key principles.** Keep the search index in sync with the native track's Gold
without re-processing.

**Libraries.** `typing`; imports `schema` (and `ai_search_store` at call time).

**Importance & connections.** Connects `native_pipeline`'s output to
`ai_search_store`. Depends on `schema`.

---

## The one-way dependency rule, restated

`shared/` is imported by `custom/`, `search/`, and `native/` — never the
reverse. `custom/` may use `search/` (for figures and indexing). `native/` is
independent of `custom/` (it's an alternative, not a dependency). This is what
lets the two extraction tracks coexist behind one `schema` and one `config`,
and it's the rule to preserve when extending the system: new foundations go in
`shared/`, new pipeline stages in `custom/`, and neither should make `shared/`
depend on them.
