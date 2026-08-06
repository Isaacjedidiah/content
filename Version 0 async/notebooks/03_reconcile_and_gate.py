# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Reconcile & Gate  *(async pipeline)*
# MAGIC
# MAGIC Quality control on the extracted claims. **Synchronous** — reconciliation
# MAGIC and gating are pure logic over already-extracted claims, no model calls,
# MAGIC so no async runner needed here.

# COMMAND ----------

import sys, os
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

from src.custom.validator import reconcile, gate_for_review

conflicts = reconcile(extraction.claims)  # noqa: F821 (from stage 02)
gated = gate_for_review(extraction.claims, conflicts)

print(f"Conflicts flagged: {len(conflicts)}")
print(f"Clean (-> Gold):   {len(gated['clean'])}")
print(f"Needs review:      {len(gated['needs_review'])}")
for c in gated["needs_review"][:5]:
    print(f"  REVIEW: {c.canonical_metric or c.field_name} = {c.value} (conf {c.confidence:.2f})")
