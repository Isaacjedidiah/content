# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluate  *(async pipeline)*
# MAGIC
# MAGIC Register ground truth and read accuracy. **Synchronous** — registering
# MAGIC expected values and reading the eval summary are plain table operations.
# MAGIC (The per-document scoring itself runs *inside* `process_batch` in stage
# MAGIC 04 when `eval_every_n > 0`.)

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.shared.eval_registry import ExpectedValue, register_expected, list_missing
from src.custom.evaluator import eval_summary

register_expected(ExpectedValue(
    team="finance", report_type="primary_report", entity_ref="E1",
    metric="primary_ratio", expected_value="15.6", tolerance=0.05),
    updated_by="analyst")

print("Unregistered (no ground truth yet):", list_missing([("finance", "primary_report")]))
print("Eval summary:", eval_summary())
