# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Extract
# MAGIC
# MAGIC Turn elements into **claims** (structured values). This is the heart of
# MAGIC the pipeline:
# MAGIC - **Content-aware routing** picks a cheap or strong model per element
# MAGIC   *before* any API call.
# MAGIC - The **cascade** escalates only low-confidence results.
# MAGIC - **Charts are cropped and zoomed** before the model reads them.
# MAGIC - Each claim carries provenance (how obtained) and confidence.

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.extractor import Extractor
from src.custom.preprocessor import preprocess
from src.search.figure_preprocessor import PdfPlumberPageImageProvider

extractor = Extractor()

# Extract from one parsed submission (elements from stage 01).
sub = submissions[0]  # noqa: F821  (from stage 01 / a shared setup)
elements = preprocess(sub["path"], sub["team"], sub["report_type"], sub.get("entity_ref"))

with open(sub["path"], "rb") as fh:
    doc_bytes = fh.read()

extraction = extractor.extract(
    elements, document_bytes=doc_bytes, filename=sub["path"],
    page_image_provider=PdfPlumberPageImageProvider())

print(f"Extracted {len(extraction.claims)} claims "
      f"| cost ${extraction.total_cost_usd:.4f} "
      f"| cropped figures: {extraction.cropped_figures} "
      f"| quarantined: {len(extraction.quarantine)}")
for c in extraction.claims[:5]:
    print(f"  {c.canonical_metric or c.field_name} = {c.value} "
          f"(conf {c.confidence:.2f}, {c.citation_tier.value})")
