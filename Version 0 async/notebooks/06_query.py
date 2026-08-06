# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Query (Supervisor Chat)  *(async pipeline)*
# MAGIC
# MAGIC Ask questions of the extracted data. `router.answer(...)` is a coroutine
# MAGIC (it awaits classification, SQL generation, and synthesis), so it's run
# MAGIC with `run_async(...)`. Conversational memory works the same as sync —
# MAGIC the follow-up resolves against the previous turn.

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.shared.llm_router import QueryRouter
from src.shared.supervisor_session import SupervisorSession
from src.search.ai_search_store import AzureAISearchStore

router = QueryRouter(search_store=AzureAISearchStore())
session = SupervisorSession()

# ASYNC CALLS — wrapped in run_async (defined in stage 00)
a1 = run_async(router.answer(  # noqa: F821
    "What is the primary ratio for entity E1?",
    team_filter="finance", report_type="primary_report", session=session))
print("Q1:", a1.get("answer") or a1.get("sql_result"))

a2 = run_async(router.answer(  # noqa: F821
    "Is that above the regulatory minimum?",
    team_filter="finance", report_type="primary_report", session=session))
print("Interpreted as:", a2["resolved_question"])
print("Q2:", a2.get("answer") or a2.get("sql_result"))
