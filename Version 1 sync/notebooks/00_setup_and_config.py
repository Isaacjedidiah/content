# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup & Config
# MAGIC
# MAGIC First notebook in the DocExtract pipeline. Installs dependencies, puts
# MAGIC `src` on the path, loads config, and confirms the environment. Every
# MAGIC other stage notebook starts with the same bootstrap cell.
# MAGIC
# MAGIC **Pipeline stages:** 00 setup · 01 ingest & parse · 02 extract ·
# MAGIC 03 reconcile & gate · 04 store & index · 05 evaluate · 06 query.
# MAGIC
# MAGIC Logic lives in `src/` (imported, not inlined) so there is one source of
# MAGIC truth — edit the modules, not the notebooks.

# COMMAND ----------

# MAGIC %pip install pdfplumber Pillow openpyxl python-docx python-pptx pydantic openai azure-search-documents azure-identity sqlparse --quiet
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# ---- bootstrap: make `src` importable (repeated at the top of each stage) ----
import sys, os
REPO_ROOT = os.path.abspath("..")           # adjust to your repo layout
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.shared.config import CONFIG, Team

print("Teams / domains configured:", Team.values())
print("Gold catalog:", CONFIG.storage.eval_registry_table.split(".")[0])
print("Search endpoint set:", bool(getattr(CONFIG.search, "endpoint", None)))
print("Model tiers:", CONFIG.models.tier1_model, "->", CONFIG.models.tier2_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm the serving endpoints and search index are reachable before
# MAGIC running the pipeline stages. (Fill in real smoke-test calls for your
# MAGIC workspace — this is the place to fail fast if an endpoint is missing.)
