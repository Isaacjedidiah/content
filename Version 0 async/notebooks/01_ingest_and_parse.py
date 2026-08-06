# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest & Parse  *(async pipeline)*
# MAGIC
# MAGIC Discover submissions and parse each file into elements. **Parsing is
# MAGIC synchronous** (it's local work — no model calls), so nothing here needs
# MAGIC the async runner. Async only matters once we reach model calls (stage 02).

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.run_ingestion_job import discover_submissions
from src.custom.preprocessor import preprocess

submissions = discover_submissions(
    getattr(CONFIG.storage, "landing_root", "/Volumes/docextract/landing/submissions"))  # noqa: F821
print(f"Discovered {len(submissions)} submission(s)")
for s in submissions[:5]:
    print(" ", s["team"], "/", s["report_type"], "/",
          os.path.basename(s["path"]), "->", s.get("entity_ref"))

# COMMAND ----------

if submissions:
    sub = submissions[0]
    elements = preprocess(sub["path"], sub["team"], sub["report_type"], sub.get("entity_ref"))
    by_mod = {}
    for e in elements:
        by_mod[e.modality.value] = by_mod.get(e.modality.value, 0) + 1
    print(f"{os.path.basename(sub['path'])} -> {len(elements)} elements; by modality: {by_mod}")
