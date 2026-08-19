# Getting all the scripts into a Databricks Workflow — design & guide

This explains how the pipeline's modules become a Databricks Job (Workflow)
like the graph you shared, where each box is a task, arrows are dependencies,
and a failed task shows red so you can re-run just that one.

## The design (three tasks, left to right)

```
┌────────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ ingest_documents   │──▶ │ index_to_search   │──▶ │ run_health_check   │
│                    │    │                   │    │                    │
│ scan volume →      │    │ push conflict /   │    │ read run summary;  │
│ process each doc   │    │ claim chunks to   │    │ FAIL (red) if      │
│ (sequential loop): │    │ Azure AI Search   │    │ quarantine/review  │
│ parse→route→extract│    │ (own task: talks  │    │ rate abnormal      │
│ →reconcile→gate→   │    │ to a different    │    │                    │
│ Bronze/Silver/Gold │    │ service, re-runs  │    │                    │
│ per team_reporttype│    │ independently)    │    │                    │
└────────────────────┘    └───────────────────┘    └────────────────────┘
```

This mirrors your reference graph: a first processing task, a downstream task
that can fail on its own (like `Aggregate_processed_data` did), and a final
task. The review-gate is *inside* ingestion (per-document logic, not a batch
stage), so it isn't its own box.

### Why this split and not others
- **One processing task, not one-per-team.** You chose a sequential loop, so
  `ingest_documents` walks `<team>/<report_type>/<file>` and processes each in
  turn. Simple, and the whole loop is one re-runnable unit.
- **Indexing is its own task.** It depends on approved Gold, talks to Azure
  (not the LLM), and fails for different reasons. When it breaks you re-run
  *just* it — no re-extraction. This is the highest-value split.
- **Health check as a task.** So the Job goes red on an abnormal run instead
  of a green "success" that hides a spike in quarantine or review volume.

## How a module becomes a task

Databricks runs a task by calling an **entry point** — a named function in the
installed wheel. Three are registered in `pyproject.toml`:

| Task (graph box)   | Entry point        | Function                                   |
|--------------------|--------------------|--------------------------------------------|
| `ingest_documents` | `ingest_documents` | `custom/workflow_tasks.py:ingest_task`     |
| `index_to_search`  | `index_to_search`  | `custom/workflow_tasks.py:index_task`      |
| `run_health_check` | `run_health_check` | `custom/workflow_tasks.py:healthcheck_task`|

Each is a thin wrapper that calls the same library functions your tests
exercise (`discover_submissions`, `process_batch`, `AzureAISearchStore`), so
the tasks and the tested code are the same code — no fork.

### State handoff between tasks
Databricks tasks **don't share memory** — each runs on its own. They hand off
via storage, not variables:
- `ingest_documents` writes Bronze/Silver/Gold tables and a small run-summary
  row (JSONL locally / an ops table on Databricks).
- `index_to_search` reads approved Gold / detected conflicts and pushes chunks
  to AI Search.
- `run_health_check` reads the run summary and decides pass/fail.

The real artifacts (Gold tables, the search index) are the handoff; the
summary row is just for the health check.

## Build it — step by step

### 1. Build the wheel
From the package root:
```bash
pip install build
python -m build --wheel        # produces dist/docextract-*.whl
```
The bundle references `../dist/*.whl`.

### 2. Point the bundle at your workspace
Edit `resources/databricks.yml`:
- `targets.dev.workspace.host` / `targets.prod.workspace.host` → your workspace URL.
- `variables.catalog` and `variables.volume_root` → your Unity Catalog + Volume.
- `email_notifications.on_failure` → a real recipient (so a red task reaches someone).

### 3. Set secrets & endpoints (one-time)
The tasks need the same environment the package needs. In the job cluster's
config (or a cluster policy), set:
- secret scope `docextract` with `search-api-key`, `aoai-api-key`, and your LLM key;
- env vars `AZURE_SEARCH_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`, and `DATABRICKS_HOST`.

See the main README's **Azure preflight checklist** — the job will only run as
well as those endpoints actually exist in your region.

### 4. Deploy and run
```bash
databricks bundle validate -t dev      # checks the YAML against your workspace
databricks bundle deploy   -t dev      # creates the Job + tasks
databricks bundle run pra_ingestion -t dev
```
Open **Workflows → DocExtract Ingestion** and you'll see the three-box graph.
It also runs on the schedule (06:00 Europe/London by default).

### 5. Read the graph like your reference image
- Green box = succeeded; red box = failed (non-zero exit).
- Click a box → driver logs → the task's `print()` summary (document counts,
  cost, quarantine rate, or the health-check failure reason).
- Re-run one task: **right-click the box → Run task** — e.g. re-run
  `index_to_search` after fixing a search endpoint, without re-extracting.

## Options you can flip

Task parameters are set in `databricks.yml` under each task's `parameters:`.

- **Crop-and-zoom for chart figures**: add `"--crop-zoom"` to `ingest_documents`.
- **Claim↔quant linking**: add `"--link-claims"` to `ingest_documents`.
- **Health-check strictness**: add `"--max-quarantine-rate", "0.15"` (etc.) to
  `run_health_check`.

## Honest caveats

- This is verified for **structure and Python validity** here (entry points
  import and run; the bundle YAML parses; the health check's logic is tested).
  It has **not** run against a real Databricks Job — the first deploy is where
  the wheel build, cluster config, and endpoint assumptions get confirmed.
- The `index_to_search` task currently re-discovers and re-processes to emit
  conflict/claim chunks; because writes are content-hash idempotent this does
  not duplicate Gold, but on a pure-Delta deployment you'd likely switch it to
  read *newly-approved Gold rows* and push their chunks (the
  `native/vector_sync.py` path). Decide which fits before production.
- Filenames must follow `entityID_entityName_documentTitle_reportDate.ext` for
  firm attribution (it's parsed from the name now, not the folder path). If
  yours differ, adjust `run_ingestion_job.entity_ref_from_filename`.
