# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest & Parse
# MAGIC
# MAGIC Discover submissions in the landing volume and turn each file into a
# MAGIC list of **elements** (text, tables, figures) — for every supported
# MAGIC format: PDF, Word, Excel, PowerPoint, and email.
# MAGIC
# MAGIC - Text-layer content is read locally (cheap, no model call).
# MAGIC - Text paragraphs and footnotes carry page + bbox so they can be
# MAGIC   located and linked later.
# MAGIC - Footnote regions are kept whole (one chunk), not fragmented.
# MAGIC - Email attachments are ignored by design (teams drop them in the
# MAGIC   folder as their own submissions).

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.run_ingestion_job import discover_submissions
from src.custom.preprocessor import preprocess

# Discover what's waiting in the landing volume: team / report-type / file.
submissions = discover_submissions(CONFIG.storage.__dict__.get("landing_root", "/Volumes/docextract/landing/submissions"))  # noqa: F821
print(f"Discovered {len(submissions)} submission(s)")
for s in submissions[:5]:
    print(" ", s["team"], "/", s["report_type"], "/", os.path.basename(s["path"]), "->", s.get("entity_ref"))

# COMMAND ----------

# Parse one submission to elements (demonstration; the batch loop in stage 02
# does this for all of them). Shows that every format lands as the same
# Element shape, with page/bbox on text.
if submissions:
    sub = submissions[0]
    elements = preprocess(sub["path"], sub["team"], sub["report_type"], sub.get("entity_ref"))
    print(f"{os.path.basename(sub['path'])} -> {len(elements)} elements")
    by_modality = {}
    for e in elements:
        by_modality[e.modality.value] = by_modality.get(e.modality.value, 0) + 1
    print("  by modality:", by_modality)
    print("  sample element carries page/bbox:",
          elements[0].page, elements[0].bbox is not None)
