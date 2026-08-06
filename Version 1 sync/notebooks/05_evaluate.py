# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluate
# MAGIC
# MAGIC Measure whether the extraction is **correct** — against ground truth
# MAGIC that is independent of the pipeline's own output:
# MAGIC - Teams register known-correct answers in the **eval registry**.
# MAGIC - Human-reviewed values are the other truth source.
# MAGIC - Scores a deterministic 1-in-N sample; reports accuracy on graded rows
# MAGIC   and the human-override rate. Ungraded values are never counted correct.
# MAGIC
# MAGIC This never grades against the Gold table the pipeline produced (that
# MAGIC would be circular).

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.shared.eval_registry import ExpectedValue, register_expected, list_missing
from src.custom.evaluator import eval_summary

# A team registers what the correct answer IS (done once, or as truth is known).
register_expected(ExpectedValue(
    team="finance", report_type="primary_report", entity_ref="E1",
    metric="primary_ratio", expected_value="15.6", tolerance=0.05),
    updated_by="analyst")

# Which (team, report_type) pairs still have NO registered truth?
print("Unregistered (can't be ground-truth evaluated yet):",
      list_missing([("finance", "primary_report")]))

# Aggregate eval across runs (accuracy + override rate), for the dashboard.
# The per-document scoring happens inside process_batch when eval_every_n>0.
print("Eval summary:", eval_summary())
