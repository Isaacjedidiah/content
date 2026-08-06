# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup & Config  *(async pipeline)*
# MAGIC
# MAGIC First notebook in the **async** DocExtract pipeline. Installs deps, puts
# MAGIC `src` on the path, loads config, and defines the async runner every
# MAGIC other stage uses.
# MAGIC
# MAGIC **Why an async runner?** The pipeline's model-call path is asynchronous —
# MAGIC `extract`, `process_batch`, `answer`, `match_all` are coroutines. Calling
# MAGIC them directly returns a coroutine that never runs. Each stage wraps them
# MAGIC in `run_async(...)` (defined below), which executes the coroutine.
# MAGIC
# MAGIC **Event-loop caveat (read this):** `asyncio.run()` fails if a loop is
# MAGIC already running, which some Databricks runtimes do inside notebooks. The
# MAGIC `run_async` helper below handles both cases — plain `asyncio.run` when no
# MAGIC loop is running, and `nest_asyncio` / existing-loop reuse when one is.
# MAGIC Confirm it works in your runtime on the first run; if you hit a loop
# MAGIC error, the helper's fallback branch is where to adjust.

# COMMAND ----------

# MAGIC %pip install pdfplumber Pillow openpyxl python-docx python-pptx pydantic openai azure-search-documents azure-identity sqlparse nest_asyncio --quiet
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# ---- bootstrap: make `src` importable + define the async runner ----
import sys, os, asyncio
REPO_ROOT = os.path.abspath("..")           # adjust to your repo layout
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_async(coro):
    """Run a coroutine from a notebook cell, whether or not an event loop is
    already running (Databricks runtimes differ). Plain asyncio.run when no
    loop is active; otherwise patch the running loop with nest_asyncio."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # a loop is already running in this runtime — reuse it
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)

# COMMAND ----------

from src.shared.config import CONFIG, Team

print("Teams / domains configured:", Team.values())
print("Gold catalog:", CONFIG.storage.eval_registry_table.split(".")[0])
print("Model tiers:", CONFIG.models.tier1_model, "->", CONFIG.models.tier2_model)
print("Async concurrency cap:", getattr(CONFIG, "max_model_concurrency", 20))

# COMMAND ----------

# MAGIC %md
# MAGIC Smoke-test the serving endpoints and search index here before running
# MAGIC the stages. The async path's behaviour under real rate limits only shows
# MAGIC on live endpoints — watch for throttling and tune the concurrency cap.
