# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Store & Index  *(async pipeline)*
# MAGIC
# MAGIC Persist results and index for retrieval. Direct storage/index writes are
# MAGIC synchronous, but the **full orchestrator `process_batch` is a coroutine**
# MAGIC (it drives the async extraction internally), so it's run with
# MAGIC `run_async(...)`. This is the single call a scheduled job runs.

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.storage import DeltaLakeStorage
from src.search.ai_search_store import AzureAISearchStore
from src.custom.production import process_batch

storage = DeltaLakeStorage()
search = AzureAISearchStore()
search.ensure_index()

# Direct writes for the single doc (sync helpers)
sub = submissions[0]  # noqa: F821
storage.write_silver(extraction.claims, team=sub["team"], report_type=sub["report_type"])  # noqa: F821
n_gold = storage.write_gold_metrics(gated["clean"], team=sub["team"], report_type=sub["report_type"])  # noqa: F821
print(f"Wrote {n_gold} Gold metrics for {sub['team']}/{sub['report_type']}")

# Full async orchestration for the whole batch (what a scheduled job runs):
# ASYNC CALL — wrapped in run_async (defined in stage 00)
result = run_async(process_batch(  # noqa: F821
    submissions, storage=storage, search_store=search,  # noqa: F821
    page_image_provider=None, link_claims=True, eval_every_n=10))
print(f"Batch: {result.documents} docs | {result.gold_metrics_total} Gold | "
      f"{result.needs_review} to review | eval'd {result.evaluated_documents}")
