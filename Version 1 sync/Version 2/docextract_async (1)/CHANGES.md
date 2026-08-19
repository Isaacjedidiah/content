# Changes vs. the uploaded version

Each change maps to an issue flagged during review. Behaviour verified by
`tests/test_pipeline.py` (all passing).

## Supervisor chat: memory, audit logging, and feedback (added on request)

- **Per-session conversation memory** (`shared/supervisor_session.py`). A
  `SupervisorSession` keeps recent turns (question + the key figures returned)
  so follow-ups resolve against prior context. Scope is per session; memory
  resets when the session ends.
- **Reference-resolution rewrite** (`llm_router.resolve_references`). A
  follow-up like "will firm A survive from the figure?" is rewritten into a
  self-contained question ("will firm A survive given its the primary metric of 8.2%?")
  using session memory before it hits classification / SQL / synthesis. First
  turns and stateless calls pass through unchanged; a rewrite failure falls
  back to the original question.
- **Audit logging of every Q&A** — each turn (raw question, resolved question,
  route, SQL, answer, sources, prompt source) is persisted to
  `ops.supervisor_query_log` (JSONL locally, Delta on Databricks). Best-effort:
  a logging failure never denies the supervisor an answer.
- **Feedback capture** — thumbs up/down (+ optional note) attaches to the turn
  it concerns; feedback on an expired turn still logs a standalone event. The
  supervisor app renders the controls per answer.
- **Monitoring view** (`monitoring.supervisor_feedback_summary`) — question
  volume, rated count, up/down split, unhelpful-rate, and session count, so the
  feedback signal shows up in the ops dashboard alongside ingestion KPIs.

The three jobs share one per-session store: the same record that gives the
chat memory is the audit trail and the anchor for feedback.

## Multi-team / multi-report support (added on request)

- **Per-(team, report_type) storage** (`config.StoragePaths.slug` /
  `layer_table` / `layer_local`). Bronze/Silver/Gold now write to a table (or
  local file) named `<team>_<report_type>` — e.g. an FX team's exchange report
  lands in `docextract.silver.fx_exchange_report`. `entity_ref` stays a **column
  inside** each table, not part of the name. Ops tables (quarantine, reviews,
  audit) remain shared, being cross-cutting. Writers fall back to a single
  shared table when no team/report_type is given, so existing callers/tests
  are unaffected.
- **Three-level orchestration discovery** (`run_ingestion_job`). The volume
  layout is `<team>/<report_type>/<file>`; the job scans every team folder,
  then every report_type folder under it, and processes every document found.
  `entity_ref` is resolved from the filename convention
  (`entityID_entityName_documentTitle_reportDate.ext`) and left None (routed via
  the existing quarantine/UNKNOWN handling) when a name doesn't parse — a bad
  filename never crashes the run.
- **Stored prompt registry** (`shared/prompt_registry.py`). Few-shots as data,
  keyed by `(team, report_type, kind)` where kind is `extraction` or `query`,
  backed by a JSONL file locally or a Delta table on Databricks. The extractor
  pulls the `extraction` set via `build_prompt`; the supervisor chat pulls the
  `query` set — the router now identifies team + report_type and synthesises
  the answer from that specific few-shot set (`llm_router.answer(...,
  report_type=...)`). Unregistered `(team, report_type)` fall back to a
  built-in default seed, so nothing is ever blocked; `list_missing` surfaces
  the backlog. Every write records who changed it.

## Techniques adopted from the uploaded codebase (kept on our substrate)

Substrate is unchanged: Azure AI Search + GPT-5-mini→Sonnet cascade. The
uploads were mined for techniques, not adopted wholesale (ChromaDB, Voyage,
Haiku/Sonnet-4.5 and local-disk PipelineMode were deliberately not taken).

- **Crop-and-zoom for chart figures** (`search/figure_preprocessor.py`). Crop
  to a figure's bbox + 2% padding, upscale the smaller dimension to ≥1024px
  (capped 2000px), and read the crop multimodally so the model READS rather
  than has to LOCATE-and-read a small chart. The uploaded codebase left its
  page-image source unimplemented (so crop-and-zoom never actually ran); here
  it is backed by `PdfPlumberPageImageProvider`, reusing the same
  `page.to_image()` rasterisation the OCR fallback already uses — so it runs
  for real on PDFs. Fully optional and gated: enabled per run
  (`--crop-zoom` / `page_image_provider`), only fires for chart-like FIGURE
  elements with a bbox, and degrades to text-only (never crashes) if the
  provider is absent, returns nothing, or errors.
- **Two-axis tiering**. Split provenance from lifecycle: `CitationTier`
  (parsed / llm_estimated / derived / manual) records HOW a value was
  obtained; `ReviewTier` (auto / confirmed / override) records where it sits
  in human review. A chart-read value is `LLM_ESTIMATED` but still only goes
  to a human if its own confidence is low — matching the uploads' deliberate
  choice not to force-review every chart-derived field.
- **Structural metric classification** (`metric_normaliser`). Rule-based
  `measure_type` (level/rate/change/ratio) and `period` (YoY/QoQ/…) from the
  raw name, carried on every claim and into Gold, and used by the
  compatibility check (UNKNOWN = benefit of the doubt).
- **Claim↔quant linking** (`custom/claim_quant_matcher.py`). Links a narrative
  claim to the metric that supports it: cheap lexical top-N retrieval then one
  tier1 confirm call; below-threshold claims left UNLINKED, never
  force-matched. Resolved links are indexed to Azure AI Search. Opt-in via
  `process_batch(link_claims=True)`.
- **Per-metric review thresholds** (`config.Thresholds.review_threshold`). A
  misread Primary ratio (0.95) and a misread date don't share one global bar.

## Content-aware routing (added after review)

**Known-hard content skips the wasted Tier 1 pass.** A new pure function
`shared/routing.decide_start_tier` picks each element's starting tier from
cheap, deterministic content signals before any model call:

- **Modality** — tables start on Tier 2 (where extraction is hardest and
  misreads costliest). Configurable via `RoutingPolicy.hard_modalities`.
- **Length** — elements longer than `max_tier1_chars` (default 6000) start on
  Tier 2, since long context degrades the cheap model.
- **Report type** — deployments can mark specific dense report types hard
  (`hard_report_types`, opt-in/empty by default).
- **Empty/flagged elements** stay on Tier 1 (a paid Tier 2 pass buys nothing);
  they return nothing and route to review/quarantine cheaply.

Content routed to Tier 2 does **one** Tier 2 pass — no Tier 1 waste. The
reactive cascade (Tier 1 → escalate on low confidence) is unchanged for
everything else. The policy is data-driven (`RoutingPolicy` in
`shared/config.py`) so rules tune without code changes, and each decision is
recorded as a `route` audit event with its reason. `CostTracker.project`
gained a `hard_share` parameter so cost projections reflect the skipped Tier 1
passes (defaults to 0, so existing callers are unaffected).

Not added (deliberately): an *LLM* pre-classifier. A model call to decide
routing would reintroduce the very cost the routing avoids; the deterministic
rules capture the high-value cases (tables, length) without it. A learned
router can layer on later if confidence calibration data justifies it.

## Root-cause alignment fixes

1. **Tool schema ↔ Claim fields.** `EXTRACTION_TOOL_SCHEMA` now requests
   `scale`, `netting`, `as_at_date` (plus unit/basis). `Extractor._build_claim`
   threads all of them into every `Claim`. Previously these were always
   `None`, making the normaliser's `scale` axis dead and Gold columns null.

2. **`entity_ref` now real.** `Element` and `Claim` carry `entity_ref`;
   `preprocess(..., entity_ref=)` stamps it onto every element; the extractor
   propagates it to every claim; `storage._entity_ref` reads it from the claim.
   Ingestion discovers it from the `<entity_ref>/<team>/<report_type>/<file>`
   path. Previously every Gold row was `"UNKNOWN_ENTITY_REF"`.

3. **Native Silver→Gold bridge.** Added `silver_claims_exploded` (explode
   per-claim array + canonicalise metric + cast confidence). The native
   `ai_extract` schema now returns per-claim structural fields, so Gold selects
   real columns and the two tracks are genuinely swappable.

## Robustness / correctness

4. **Quarantine, don't crash.** `build_claim` mirrors `build_chunk`; malformed
   model rows and missing required keys route to `ExtractionResult.quarantine`.
   `_safe_tool_args` tolerates non-JSON / missing tool calls (relevant to the
   tier-2 FMAPI forced-tool-choice concern).

5. **Reconciliation grouping.** `reconcile` groups by
   `(metric, reporting_basis, scale, as_at_date)` so consolidated vs solo isn't
   mis-flagged. Values parsed via shared `parse_numeric` (handles `14.2%`,
   `1,234`, `£5m`) instead of bare `float()`.

6. **SELECT-only guard.** `is_safe_select` uses `sqlparse` (single statement,
   type == SELECT), with a word-boundary regex fallback so `created_returns` /
   `delete_flag` are no longer false-positives.

7. **Idempotency.** `StorageManager.append_jsonl` dedups on content hash;
   AI Search uses `merge_or_upload` (upsert). Re-runs no longer duplicate.

## Completeness (previously missing)

8. **NL→SQL generation.** `QueryRouter.generate_sql` + `_apply_team_scope`;
   the supervisor UI now produces a query from a plain question (the SQL branch
   previously never fired). Team scope enforced by the caller, not the model.

9. **Review loop → Gold.** `reviewer.apply_decision` folds the outcome into the
   claim (final value, citation tier, clears `needs_review`) so reviewed claims
   are Gold-eligible; `decision_row` + storage sinks persist decisions.

10. **Audit persistence.** `AuditLog.to_rows`; orchestrator persists to the
    audit sink. Was in-memory only.

11. **Governance timestamps.** `SignOff` / `SecondLineValidation` validate
    ISO-8601 in `__post_init__`.

12. **Canonical teams.** `Team` enum in config; all three UIs and prompt lookup
    resolve against it (was three divergent hard-coded lists).

13. **Preprocessor.** `entity_ref` param; ragged tables padded so markdown
    columns don't desync. Scanned / image-only PDF pages (no text layer, so
    pdfplumber returns nothing) now fall back to `ai_parse_document` for OCR
    via the same `databricks-ai-parse` endpoint the native track uses; if the
    endpoint is unavailable or returns nothing, the page keeps its
    flagged-empty element and goes to quarantine (no silent data loss).

14. **Package + deploy.** `__init__.py` throughout; `pyproject.toml` with a
    wheel entry point; `resources/databricks.yml` (job + Lakeflow pipeline);
    `resources/ai_search_skillset.json`; `requirements.txt`; test stubs.

15. **Cost projection.** `CostTracker.project` now adds the Tier-1 pass on
    escalated pages (the cascade bills both), matching actual extractor billing.

## Store change

16. **ChromaDB → Azure AI Search** (`search/ai_search_store.py`): push chunk
    text; service performs embedding (skillset) and query-time vectorization
    (vectorizer). Avoids memory-mapped SQLite on FUSE-mounted UC Volumes.

## Not changed / still needs your input

- Real serving-endpoint + AI-function availability in your Azure region
  (see README preflight) — cannot be verified from code. This now includes
  `databricks-ai-parse`, which the custom track's OCR fallback also calls.
- The `firm_team` mapping table referenced by supervisor team-scoping SQL must
  exist (or swap the scope subquery for your firm↔team source).
