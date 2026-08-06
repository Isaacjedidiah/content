# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Store & Index
# MAGIC
# MAGIC Persist the results and make them findable:
# MAGIC - **Silver/Gold** written to per-team_reporttype tables, entity as a
# MAGIC   column, idempotent (re-runs don't duplicate).
# MAGIC - **Vector index** receives conflicts, claim-links, and the raw
# MAGIC   narrative + footnotes (with page/bbox and shared footnote-markers) so
# MAGIC   the chat can retrieve context like a footnote qualifying a figure.

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.storage import DeltaLakeStorage
from src.search.ai_search_store import AzureAISearchStore

storage = DeltaLakeStorage()
search = AzureAISearchStore()
search.ensure_index()

sub = submissions[0]  # noqa: F821
storage.write_silver(extraction.claims, team=sub["team"], report_type=sub["report_type"])  # noqa: F821
n_gold = storage.write_gold_metrics(gated["clean"], team=sub["team"], report_type=sub["report_type"])  # noqa: F821
print(f"Wrote {n_gold} Gold metrics for {sub['team']}/{sub['report_type']}")

# Full orchestration (all stages 01-05 wired) for the whole batch is a single
# call — this is what a scheduled job runs:
from src.custom.production import process_batch
result = process_batch(submissions, storage=storage, search_store=search,  # noqa: F821
                       page_image_provider=None, link_claims=True, eval_every_n=10)
print(f"Batch: {result.documents} docs | {result.gold_metrics_total} Gold | "
      f"{result.needs_review} to review | eval'd {result.evaluated_documents}")
