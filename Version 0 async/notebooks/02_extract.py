# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Extract  *(async pipeline)*
# MAGIC
# MAGIC Turn elements into claims. **This is where async pays off:** every
# MAGIC element is extracted concurrently (bounded by the endpoint rate limit),
# MAGIC so 1000 elements run in overlapping waves instead of a 1000-long queue.
# MAGIC
# MAGIC `extract(...)` is a coroutine — it MUST be run with `run_async(...)`
# MAGIC (from stage 00). Calling it bare returns a coroutine that never executes.

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
sub = submissions[0]  # noqa: F821  (from stage 01)
elements = preprocess(sub["path"], sub["team"], sub["report_type"], sub.get("entity_ref"))

with open(sub["path"], "rb") as fh:
    doc_bytes = fh.read()

# ASYNC CALL — wrapped in run_async (defined in stage 00)
extraction = run_async(extractor.extract(  # noqa: F821
    elements, document_bytes=doc_bytes, filename=sub["path"],
    page_image_provider=PdfPlumberPageImageProvider()))

print(f"Extracted {len(extraction.claims)} claims | cost ${extraction.total_cost_usd:.4f} "
      f"| cropped figures: {extraction.cropped_figures} | quarantined: {len(extraction.quarantine)}")
for c in extraction.claims[:5]:
    print(f"  {c.canonical_metric or c.field_name} = {c.value} "
          f"(conf {c.confidence:.2f}, {c.citation_tier.value})")
