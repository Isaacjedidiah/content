"""Tag registry — the governed vocabulary and mappings for CONTENT-DOMAIN tags.

Documents mix content domains (risk, financial, tax, ...). Each element gets an
ADVISORY domain tag so teams can find the components relevant to them — nothing
is gated on it; it's a findability aid, not an access control.

This registry is GOVERNED (written centrally, like the threshold registry), and
holds three things:

  * the valid domain vocabulary (kind="domain") — the allowed tags, so tagging
    can't produce free-form values. ``unknown`` is always valid as the reserved
    "not classified" fallback.
  * metric -> domain mappings (kind="metric_domain") — powers the cascade's
    "infer the domain from the canonical metrics present" rung, deterministically.
  * heading keyword -> domain mappings (kind="heading_domain") — powers the
    "the section heading announces its domain" rung.

Storage mirrors the other registries (JSONL local / Delta on Databricks,
append, latest-per-key wins, ``updated_by`` audit). Env override
DOCEXTRACT_TAG_REGISTRY_DIR (or PRA_TAG_REGISTRY_DIR).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG

UNKNOWN_TAG = "unknown"


def _norm(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


@dataclass
class TagEntry:
    """A governed tag-registry row.

    kind="domain":        ``value`` is a valid domain name (key unused).
    kind="metric_domain": ``key`` is a canonical metric, ``value`` its domain.
    kind="heading_domain":``key`` is a heading keyword, ``value`` its domain.
    """
    kind: str
    value: str
    key: Optional[str] = None
    updated_by: str = "governance"
    updated_at: str = ""


_KINDS = ("domain", "metric_domain", "heading_domain")


def _local_path() -> str:
    root = (os.environ.get("DOCEXTRACT_TAG_REGISTRY_DIR")
            or os.environ.get("PRA_TAG_REGISTRY_DIR")
            or CONFIG.storage.local_root)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "tag_registry.jsonl")


def _read_all(spark=None) -> list[dict]:
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.tag_registry_table).collect()]
        except Exception:
            return []
    p = _local_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def register_tag(entry: TagEntry, updated_by: str, spark=None) -> None:
    """Governed write: append a tag-registry row (latest per key wins)."""
    if entry.kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    payload = {
        "kind": entry.kind,
        "key": _norm(entry.key) if entry.key else None,
        "value": _norm(entry.value),
        "updated_by": updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if spark is not None:
        spark.createDataFrame([payload]).write.format("delta").mode(
            "append").saveAsTable(CONFIG.storage.tag_registry_table)
    else:
        with open(_local_path(), "a") as fh:
            fh.write(json.dumps(payload) + "\n")


def valid_domains(spark=None) -> set[str]:
    """The allowed domain vocabulary, always including ``unknown``."""
    out = {UNKNOWN_TAG}
    for d in _read_all(spark):
        if d.get("kind") == "domain":
            out.add(d["value"])
    return out


def metric_domain_map(spark=None) -> dict[str, str]:
    """{canonical_metric: domain}, latest per metric wins."""
    out: dict[str, str] = {}
    for d in _read_all(spark):
        if d.get("kind") == "metric_domain" and d.get("key"):
            out[d["key"]] = d["value"]
    return out


def heading_domain_map(spark=None) -> dict[str, str]:
    """{heading_keyword: domain}, latest per keyword wins."""
    out: dict[str, str] = {}
    for d in _read_all(spark):
        if d.get("kind") == "heading_domain" and d.get("key"):
            out[d["key"]] = d["value"]
    return out


def list_registered(spark=None) -> list[tuple]:
    """All rows as (kind, key, value) — for governance visibility."""
    return sorted((d["kind"], d.get("key") or "", d["value"])
                  for d in _read_all(spark))
