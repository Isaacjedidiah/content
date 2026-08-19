"""Dictionary registry — synonym → canonical mappings as DATA, not code.

Mirrors prompt_registry and eval_registry: one global append-table
(``docextract.ops.dictionary_registry``); new mappings are appended (latest per
key wins on read); JSONL locally / Delta on Databricks; every write records who
set it. A team onboards a new synonym by appending a row — no code change, no
redeploy.

Two kinds of mapping share the table, distinguished by ``kind``:
  * ``metric``   — a metric synonym → canonical metric key
  * ``doctype``  — a document-type synonym → canonical doctype key

``dictionary.py`` consults this registry first and falls back to its in-code
seed maps, so an empty registry behaves exactly as before. Env override
DOCEXTRACT_DICTIONARY_REGISTRY_DIR (or PRA_DICTIONARY_REGISTRY_DIR) for
local/testing.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import CONFIG


def _norm(raw: str) -> str:
    """Lowercase, strip, collapse whitespace — same normalisation the
    dictionary uses, so registry keys match lookup keys."""
    return re.sub(r"\s+", " ", raw.strip().lower())


@dataclass
class TermMapping:
    """One synonym → canonical mapping registered by a team."""
    kind: str            # "metric" | "doctype"
    synonym: str         # the raw term as written (will be normalised on write)
    canonical: str       # the canonical key it maps to
    updated_by: str = "system"
    updated_at: str = ""


def _local_path() -> str:
    root = (os.environ.get("DOCEXTRACT_DICTIONARY_REGISTRY_DIR")
            or os.environ.get("PRA_DICTIONARY_REGISTRY_DIR")
            or CONFIG.storage.local_root)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "dictionary_registry.jsonl")


def _read_all(spark=None) -> list[dict]:
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.dictionary_registry_table).collect()]
        except Exception:
            return []
    p = _local_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def register_term(mapping: TermMapping, updated_by: str, spark=None) -> None:
    """Self-serve write: append a synonym → canonical mapping (latest wins)."""
    if mapping.kind not in ("metric", "doctype"):
        raise ValueError("kind must be 'metric' or 'doctype'")
    payload = {
        "kind": mapping.kind,
        "synonym": _norm(mapping.synonym),
        "canonical": mapping.canonical,
        "updated_by": updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if spark is not None:
        spark.createDataFrame([payload]).write.format("delta").mode(
            "append").saveAsTable(CONFIG.storage.dictionary_registry_table)
        return
    with open(_local_path(), "a") as fh:
        fh.write(json.dumps(payload) + "\n")


def get_mappings(kind: str, spark=None) -> dict[str, str]:
    """Return the resolved {normalised_synonym: canonical} map for a kind.
    Latest registered entry per synonym wins (later rows override earlier)."""
    resolved: dict[str, str] = {}
    for d in _read_all(spark):
        if d.get("kind") == kind:
            resolved[d["synonym"]] = d["canonical"]   # append order = latest wins
    return resolved


def list_registered(kind: str, spark=None) -> list[str]:
    """The synonyms currently registered for a kind (for visibility)."""
    return sorted(get_mappings(kind, spark).keys())
