# Integration notes (uploads → current package)

## Target substrate (FIXED — do not change)
- Vector store: **Azure AI Search** (not ChromaDB).
- Extraction cascade: **GPT-5-mini (tier1) → Claude Sonnet (tier2) → human** (not Haiku/Sonnet-4.5).
- Everything already built (content-aware routing, OCR fallback, quarantine,
  idempotency, gates, audit, entity_ref propagation) stays.

## Rule for uploads
Uploaded modules are a SOURCE OF TECHNIQUES, not a replacement codebase.
Where an upload assumes ChromaDB or Haiku/Sonnet-4.5, keep our substrate and
take only the idea. Map their model IDs: HAIKU_MODEL_ID→tier1, SONNET_MODEL_ID→tier2.

## Batch 1 — techniques identified (integration pending full set)
- complexity_router.py: per-ELEMENT routing on ai_parse structural signals
  (table count, merged cells, chart-figure detection) + doc-level score.
  Richer than our routing.py (modality/length). Chart-derived fields escalate
  to tier2 but are NOT force-sent to human review. → MERGE into routing.py.
- claim_quant_matcher.py: claim↔metric linking (lexical top-N → LLM confirm,
  low-confidence left UNLINKED). NEW capability we lack. → ADD.
- bronze_ingest.py: parser-dispatch boundary; xlsx + eml fallbacks; explicit
  ai_parse_document native-format list. → MERGE into preprocessor/parse layer.
- bbox_viewer.py: reviewer bbox highlight rendering. → ADD to reviewer/apps.
- config.py: per-field-type confidence thresholds, per-stage concurrency caps,
  pricing table, PipelineMode (custom/databricks/hybrid), team/report_type as
  folders-not-enums. → SELECTIVELY merge (per-field thresholds, concurrency).
- app.py: analyst view surfacing citation tier, claim links, conflicts. → MERGE
  into analyst_app.py.

## CROP-AND-ZOOM (the user's tracked item)
Lives in **figure_preprocessor.py** (referenced by complexity_router.py's
ParsedElement.bbox comment) — NOT yet uploaded. bbox_viewer.py only *draws* a
bbox for review; it does not crop/zoom for extraction. Read figure_preprocessor.py
closely when it arrives.

## Open questions to resolve during integration
- ParsedElement (upload) vs Element (ours): reconcile into one contract, keeping
  bbox + has_merged_cells + element_type from upload, entity_ref from ours.
- Model IDs: uploads use claude-haiku-4-5 / claude-sonnet-5; ours uses tier1/tier2
  keys. Keep our config.MODELS registry; do not introduce their raw IDs.

## Batch 2 — techniques identified
- figure_preprocessor.py: **CROP-AND-ZOOM** (the tracked item), full design:
  crop to bbox + 2% padding, upscale smaller dim to >=1024px cap 2000px,
  PageImageProvider abstraction (Databricks impl left NotImplementedError),
  clean image for model (NO highlight box — opposite of bbox_viewer). Invoked
  only for chart-like figures, degrades to text-only if bbox/page image
  missing or errors. → ADD as new module + multimodal path in extractor.
- extractor.py: multimodal image_base64 path; per-element routing consumption;
  citation tier tagging at extraction; escalation. Maps to our cascade
  (their Haiku→tier1, Sonnet→tier2). We already have cascade+routing; take the
  MULTIMODAL crop-zoom call path + citation-tier-at-extraction.
- metric_normaliser.py: structural classifier (measure_type/period/basis) with
  UNKNOWN=benefit-of-doubt compatibility check. Richer than ours (we only had
  unit/basis/scale). → MERGE: add period + measure_type classification.
- dictionary.py: canonical terms + pending-terms queue for unmapped (graceful
  degrade, human review). Richer than ours. → MERGE pending-terms queue.
- human_review_queue.py: second-review lifecycle, file-move-with-queue-entry,
  manual_override tagging CitationTier.MANUAL. → fold into our reviewer path.
- llm_router.py: query intent (SQL/NARRATIVE/BOTH); pure-SQL returns rows with
  NO synthesis LLM call. We have NL2SQL already; take the "no synthesis for
  pure SQL" cost optimisation + BOTH synthesis.
- cost_tracker.py: per-call logging + projection variance. → MERGE with ours.

## CROP-AND-ZOOM — confirmed present in uploads, ABSENT in our package.
Decision pending from full set, but this is a clear ADD. Requires:
  (1) new search/figure_preprocessor.py equivalent (crop_and_zoom + provider),
  (2) multimodal image path in llm_client.extract (base64 image block),
  (3) chart-figure detection in preprocessor/routing to trigger it,
  (4) a page-image source (PDF rasteriser via pdfplumber page.to_image we
      already use for OCR — reuse that! we have Pillow already).
Note: our OCR fallback already renders full pages via page.to_image(res=200);
crop-and-zoom is the same rasterise step + a crop/upscale on the bbox region.

## Batch 3 — contracts + orchestration seen
- schema_registry.py: DocumentRecord/ExtractedField/CitationTier(PARSED/
  LLM_ESTIMATED/DERIVED/MANUAL)/ClaimQuantLink/ModelTier. ExtractedField
  already carries bbox + measure_type/period/basis + citation_tier. This is
  their canonical contract; richer than our Claim. → reconcile: add citation
  tiers (4-way), measure_type/period, claim links, bbox to our schema.
- pipeline.py: full orchestration incl. crop-and-zoom wiring (page_image_provider
  threaded run_pipeline→extract_document→extract_element; cropped_count in
  lineage). Confirms crop-zoom is OPTIONAL + gated + degrades to text-only.
- prompt_registry.py + prompts.py: prompts-as-data with (team,report_type)
  few-shot, default seed fallback. → OPTIONAL later; not core to crop-zoom.
- monitoring.py/_app.py: KPIs, confidence drift, review queue w/ bbox side-by-side
  (Change 5), cost vs projection. Richer than ours. → MERGE later.
- register_team_config.py: one-off team onboarding (standardisation + prompts).
- run_ingestion_job.py: real entry point; page_image_provider=None until
  DatabricksPageImageProvider implemented. KEY: they leave crop-zoom OFF in
  prod because their page-image source is unimplemented. WE CAN IMPLEMENT IT
  (pdfplumber page.to_image, already used for OCR) — so we can ship crop-zoom
  actually working locally, not stubbed.

## Still not seen (referenced): silver_transform, storage, volumes, vector_store,
## sql_query_tool, validator, team_standardiser.  (batch 4 expected)

## CROP-AND-ZOOM full call-chain now confirmed:
run_pipeline(page_image_provider) → extract_document → extract_element:
  if is_chart_like_figure(el) and bbox and provider:
     prepare_figure_image → crop_and_zoom(bbox,+2% pad, upscale≥1024 cap2000)
     → encode_image_base64 → llm_client.call(model, prompt, image_base64)
  citation_tier = LLM_ESTIMATED for figures else PARSED
Provider is the only missing piece in THEIR code; we supply it via pdfplumber.

## Batch 4 — final; remaining referenced modules seen
- silver_transform.py: merge per-element extractions; normalisation-conflict
  detection on canonical-key collision (compatibility check); idempotent upsert
  by document_id; apply_manual_overrides tags CitationTier.MANUAL.
- sql_query_tool.py: NL→SQL with LAG/window fns; TWO-LAYER safety (readonly
  regex guard + ephemeral in-memory copy); per-team pivot table; NO_QUERY_POSSIBLE.
- storage.py: two-path (Gold + vector); build_gold_from_silver governance gate
  (only approved reach Gold/index); build_vector_documents 3 chunk types
  (structured/narrative/claim_link) w/ chart caveat inline.
- validator.py: per-field-type thresholds; minimum-extraction check;
  reconciliation (the primary metric<=total capital); never auto-REJECT (only PENDING/APPROVE).
- team_standardiser.py: declarative 2nd Gold layer (rename/select/simple ratio).
- volumes.py: lifecycle (inbound/accepted/rejected), path-as-identity.
- supervisor_app.py + supervisor_memory_logging.py: chat + audit log of turns.
- vector_store.py: ChromaDB/Voyage (IGNORE per user — we use AI Search).

## FINAL DECISION — what to ADD to our package (user: "add all new things")
Substrate stays: Azure AI Search + GPT-5-mini→Sonnet. Adopt techniques:
 1. CROP-AND-ZOOM (figure_preprocessor) — the headline item. + multimodal path.
 2. ParsedElement structural signals + per-element complexity routing (merge
    into our routing.py): table/merged-cells/chart-figure detection.
 3. Chart-figure → tier2 but NOT forced human review (confidence-gated only).
 4. CitationTier 4-way (parsed/llm_estimated/derived/manual) on our schema.
 5. metric_normaliser: measure_type + period + basis structural classify.
 6. claim_quant_matcher: narrative claim → quant field linking (lexical→LLM).
 7. Per-field-type confidence thresholds (validator).
 8. Reconciliation sanity check (the primary metric<=total capital) + minimum-extraction.
 9. NL→SQL with window functions + two-layer readonly safety (upgrade ours).
10. Dictionary pending-terms queue (unmapped → human review, never blocked).
11. Manual override → CitationTier.MANUAL → Gold promotion (reviewer).
12. Vector chunk types: structured / narrative / claim_link (into AI Search).
IGNORE: ChromaDB/Voyage, Haiku/Sonnet-4.5 IDs, PipelineMode local-disk.

## Integration priority for THIS pass (highest value, self-contained):
A. figure_preprocessor (crop-and-zoom) + multimodal extract path + provider
   backed by pdfplumber page.to_image (we already have it).  ← user's ask
B. complexity/structural routing merge into routing.py.
C. schema: CitationTier + measure_type/period/basis + bbox.
D. metric_normaliser period/measure_type.
E. claim_quant_matcher.
Others (SQL upgrade, per-field thresholds, pending-terms, vector chunk types)
are valuable but lower-risk to add incrementally after A–E verified.
