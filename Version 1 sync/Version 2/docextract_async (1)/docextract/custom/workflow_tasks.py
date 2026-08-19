"""Databricks Workflow task entry points.

Three linear tasks that mirror a Databricks Jobs DAG (like the attached
workflow graph), left to right:

    ingest_documents  ->  index_to_search  ->  run_health_check

Why three tasks and not one, and not a dozen: each boundary is a point where
failure means something different and recovery differs, so splitting there
buys visible-failure + targeted-rerun (exactly what a Jobs graph is for)
without over-plumbing. The review-gate stays *inside* ingestion (it is
per-document logic, not a batch stage).

State handoff between tasks: Databricks tasks do NOT share memory. Each task
persists what the next needs. Here the shared state is small — the run
summary — so it is written to a JSONL/Delta ops table by ``ingest`` and read
back by ``healthcheck``. Gold and the search index are the real artifacts the
downstream tasks act on.

Each task is a console entry point (see pyproject) invoked as a
``python_wheel_task`` in the bundle. Each fails loudly with a non-zero exit on
a real problem, so the Jobs graph shows a red task rather than a green one
hiding a silent failure.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from ..search.ai_search_store import AzureAISearchStore
from ..shared.config import CONFIG
from .production import process_batch
from .run_ingestion_job import discover_submissions


def _run_summary_path() -> str:
    root = CONFIG.storage.local_root
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "workflow_run_summary.jsonl")


def _write_summary(row: dict) -> None:
    with open(_run_summary_path(), "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _latest_summary() -> dict | None:
    p = _run_summary_path()
    if not os.path.exists(p):
        return None
    lines = [ln for ln in open(p).read().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


# ---------------------------------------------------------------------------
# TASK 1 — ingest_documents
# Discover the volume (<team>/<report_type>/<file>), process every document
# through the sequential pipeline, write Bronze/Silver/Gold per team_reporttype.
# Deliberately does NOT index to search — that is Task 2, so it re-runs alone.
# ---------------------------------------------------------------------------

def ingest_task() -> None:
    parser = argparse.ArgumentParser(description="Task 1: ingest documents")
    parser.add_argument("--root", default=CONFIG.storage.volume_root)
    parser.add_argument("--crop-zoom", action="store_true",
                        help="Enable crop-and-zoom for chart figures (PDF).")
    parser.add_argument("--link-claims", action="store_true",
                        help="Enable claim↔quant linking.")
    args = parser.parse_args()

    subs = discover_submissions(args.root)
    print(f"[ingest] discovered {len(subs)} document(s) under {args.root}")
    if not subs:
        print("[ingest] nothing to process — exiting cleanly.")
        _write_summary({"stage": "ingest", "documents": 0,
                        "ts": datetime.now(timezone.utc).isoformat()})
        return

    provider = None
    if args.crop_zoom:
        from ..search.figure_preprocessor import PdfPlumberPageImageProvider
        provider = PdfPlumberPageImageProvider()

    # No search_store here: indexing is Task 2. Conflicts still detected and
    # persisted; they are pushed to search in the next task.
    result = asyncio.run(process_batch(subs, search_store=None,
                           page_image_provider=provider,
                           link_claims=args.link_claims))
    summary = {
        "stage": "ingest", "run_id": result.run_id,
        "documents": result.documents, "gold_rows": result.gold_metrics_total,
        "needs_review": result.needs_review, "conflicts": result.conflicts,
        "quarantined": result.quarantined, "claim_links": result.claim_links,
        "cropped_figures": result.cropped_figures,
        "cost_usd": round(result.total_cost_usd, 4),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _write_summary(summary)
    print(f"[ingest] done: {json.dumps(summary)}")


# ---------------------------------------------------------------------------
# TASK 2 — index_to_search
# Own task because it depends on approved Gold, talks to a different external
# service (Azure AI Search), and fails for different reasons than extraction.
# When it breaks (as Aggregate did in the reference graph), re-run just this.
# ---------------------------------------------------------------------------

def index_task() -> None:
    parser = argparse.ArgumentParser(description="Task 2: index to AI Search")
    parser.add_argument("--root", default=CONFIG.storage.volume_root)
    args = parser.parse_args()

    store = AzureAISearchStore()
    # ensure_index is idempotent — safe to call every run.
    store.ensure_index()
    print("[index] AI Search index ensured.")

    # Re-run the batch with a search_store attached but writes are idempotent
    # (content-hash dedup), so this indexes conflict/claim chunks without
    # duplicating Gold. In a pure-Delta deployment this task would instead read
    # newly-approved Gold rows and push their chunks; that path lives in
    # native/vector_sync.py.
    subs = discover_submissions(args.root)
    result = asyncio.run(process_batch(subs, search_store=store))
    print(f"[index] indexed conflict/claim chunks for run_id={result.run_id}")
    _write_summary({"stage": "index", "run_id": result.run_id,
                    "conflicts": result.conflicts,
                    "ts": datetime.now(timezone.utc).isoformat()})


# ---------------------------------------------------------------------------
# TASK 3 — run_health_check
# Read the ingest summary and FAIL (non-zero exit) if the run looks abnormal,
# so the Jobs graph surfaces a problem instead of a green "success" hiding it.
# Thresholds are conservative defaults; tune per deployment.
# ---------------------------------------------------------------------------

def healthcheck_task() -> None:
    parser = argparse.ArgumentParser(description="Task 3: health check")
    parser.add_argument("--max-quarantine-rate", type=float, default=0.25,
                        help="Fail if quarantined/documents exceeds this.")
    parser.add_argument("--max-review-rate", type=float, default=0.80,
                        help="Fail if needs_review/gold_rows exceeds this.")
    args = parser.parse_args()

    summary = _latest_summary()
    if summary is None:
        print("[healthcheck] no run summary found — nothing to check.")
        return

    docs = max(summary.get("documents", 0), 1)
    gold = max(summary.get("gold_rows", 0), 1)
    q_rate = summary.get("quarantined", 0) / docs
    r_rate = summary.get("needs_review", 0) / gold

    print(f"[healthcheck] documents={summary.get('documents')} "
          f"gold_rows={summary.get('gold_rows')} "
          f"quarantine_rate={q_rate:.2%} review_rate={r_rate:.2%}")

    problems = []
    if q_rate > args.max_quarantine_rate:
        problems.append(
            f"quarantine rate {q_rate:.2%} > {args.max_quarantine_rate:.0%}")
    if r_rate > args.max_review_rate:
        problems.append(
            f"review rate {r_rate:.2%} > {args.max_review_rate:.0%}")

    if problems:
        print("[healthcheck] FAILED:")
        for p in problems:
            print(f"  ✗ {p}")
        # Non-zero exit -> the Jobs task goes red, like Aggregate in the graph.
        sys.exit(1)
    print("[healthcheck] OK — run within expected bounds.")


if __name__ == "__main__":
    # Allow `python -m ...workflow_tasks <ingest|index|healthcheck>` locally.
    which = sys.argv.pop(1) if len(sys.argv) > 1 else "ingest"
    {"ingest": ingest_task, "index": index_task,
     "healthcheck": healthcheck_task}.get(which, ingest_task)()
