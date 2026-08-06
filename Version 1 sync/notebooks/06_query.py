# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Query (Supervisor Chat)
# MAGIC
# MAGIC The consumption side — ask questions of what's been extracted:
# MAGIC - Routes a question to SQL (precise figure from Gold), narrative
# MAGIC   retrieval (context/footnotes from the vector store), or both.
# MAGIC - Grounded and cited; never invents figures.
# MAGIC - **Memory**: a follow-up ("...from that figure?") resolves against the
# MAGIC   prior turn. Every Q&A is logged; feedback can be attached.

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

a1 = router.answer("What is the primary ratio for entity E1?",
                   team_filter="finance", report_type="primary_report",
                   session=session)
print("Q1:", a1.get("answer") or a1.get("sql_result"))

# Follow-up — memory resolves "that figure" against the previous answer.
a2 = router.answer("Is that above the regulatory minimum?",
                   team_filter="finance", report_type="primary_report",
                   session=session)
print("Interpreted as:", a2["resolved_question"])
print("Q2:", a2.get("answer") or a2.get("sql_result"))
