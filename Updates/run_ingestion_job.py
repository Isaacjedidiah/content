"""Batch ingestion entry point (custom track).

Discovers submissions under the volume and hands them to the orchestrator.
Invoked by a Databricks Job / Workflow.

Layout: <root>/<team>/<report_type>/<file>. The orchestration scans EVERY
team folder, then EVERY report_type folder under it, and processes every
document found. entity_ref is NOT in the path — it is resolved from the
filename convention (entityID_entityName_documentTitle_reportDate.ext) and stays
a column in the data. A filename that doesn't parse still processes; its
entity_ref is left None and handled by the existing quarantine/UNKNOWN path
rather than crashing the run.

Idempotency comes from content-hash dedup in storage, so re-scanning already
processed files is a no-op.
"""
from __future__ import annotations

import asyncio
import argparse
import os
import re

from ..search.ai_search_store import AzureAISearchStore
from ..shared.config import CONFIG
from .production import process_batch

# entityID_entityName_documentTitle_reportDate.ext  (reportDate = YYYY-MM-DD)
_FILENAME_RE = re.compile(
    r"^(?P<entity_ref>[A-Za-z0-9]+)_[A-Za-z0-9\-]+_[A-Za-z0-9\-]+_"
    r"\d{4}-\d{2}-\d{2}\.[A-Za-z0-9]+$")


def entity_ref_from_filename(filename: str) -> str | None:
    """Extract entity_ref from the filename convention, or None if it doesn't
    match (the document still processes; attribution routes to review)."""
    m = _FILENAME_RE.match(filename.strip())
    return m.group("entity_ref") if m else None


def _normalise_entity(folder: str) -> str:
    """Normalise an entity folder name so casing/spacing differences don't
    fragment one entity into several (ACME_CORP vs 'Acme Corp'). Trim, collapse
    whitespace, and keep the given form otherwise — teams file under a canonical
    entity folder; this only guards against trivial variance."""
    return re.sub(r"\s+", " ", folder.strip())


def discover_submissions(root: str) -> list[dict]:
    """Scan <root>/<team>/<report_type>/<entity>/<file>.

    Attribution comes from the PATH, not the filename: a file is only collected
    when it sits inside a complete team/report_type/entity/ path, so every
    discovered file is fully attributed by construction. A file dropped loosely
    at the report_type level (no entity folder) is a *directory-less* path and
    is simply not collected — the walk descends into entity folders to find
    work, so there is no way to ingest a file without a resolved entity.

    entity_ref is taken from the entity folder name (normalised). The filename
    is free-form and plays no part in attribution.
    """
    subs: list[dict] = []
    if not os.path.isdir(root):
        return subs
    for team in sorted(os.listdir(root)):
        team_dir = os.path.join(root, team)
        if not os.path.isdir(team_dir) or team.startswith("."):
            continue
        for report_type in sorted(os.listdir(team_dir)):
            rt_dir = os.path.join(team_dir, report_type)
            if not os.path.isdir(rt_dir) or report_type.startswith("."):
                continue
            for entity in sorted(os.listdir(rt_dir)):
                entity_dir = os.path.join(rt_dir, entity)
                # Only descend into ENTITY folders. A loose file sitting
                # directly under report_type is not a directory, so it is
                # skipped here — it never becomes a submission without an
                # entity. This is the "no stray" property, by construction.
                if not os.path.isdir(entity_dir) or entity.startswith("."):
                    continue
                for fname in sorted(os.listdir(entity_dir)):
                    fpath = os.path.join(entity_dir, fname)
                    if fname.startswith(".") or not os.path.isfile(fpath):
                        continue
                    subs.append({
                        "path": fpath,
                        "entity_ref": _normalise_entity(entity),
                        "team": team,
                        "report_type": report_type,
                    })
    return subs


def main() -> None:
    parser = argparse.ArgumentParser(description="the pipeline regdata ingestion")
    parser.add_argument("--root", default=CONFIG.storage.volume_root)
    parser.add_argument("--no-index", action="store_true",
                        help="Skip Azure AI Search conflict indexing.")
    parser.add_argument("--crop-zoom", action="store_true",
                        help="Enable crop-and-zoom multimodal reading of "
                             "chart-like figures (PDF sources).")
    args = parser.parse_args()

    subs = discover_submissions(args.root)
    print(f"Discovered {len(subs)} submissions under {args.root}")

    search_store = None if args.no_index else AzureAISearchStore()
    if search_store is not None:
        search_store.ensure_index()

    # Crop-and-zoom is opt-in. The pdfplumber-backed provider renders figure
    # pages for legible re-reading; left off, extraction is text-only.
    page_image_provider = None
    if args.crop_zoom:
        from ..search.figure_preprocessor import PdfPlumberPageImageProvider
        page_image_provider = PdfPlumberPageImageProvider()

    result = asyncio.run(process_batch(subs, search_store=search_store,
                           page_image_provider=page_image_provider))
    print(
        f"run_id={result.run_id} documents={result.documents} "
        f"gold_rows={result.gold_metrics_total} "
        f"cost=${result.total_cost_usd:.4f} review={result.needs_review} "
        f"conflicts={result.conflicts} quarantined={result.quarantined} "
        f"cropped_figures={result.cropped_figures}"
    )


if __name__ == "__main__":
    main()
