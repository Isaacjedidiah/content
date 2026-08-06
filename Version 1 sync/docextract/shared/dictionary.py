"""Canonical name resolution for metrics and document types.

Mappings are DATA: teams register synonym → canonical pairs in the dictionary
registry (docextract.ops.dictionary_registry). The in-code maps below are the
default SEED — used when the registry has no entry for a term, so an empty
registry behaves exactly as before. Registered entries override the seed.

Unmapped terms never block extraction: they flow through under a slugified raw
name, flagged as unmapped so downstream can surface them.
"""
from __future__ import annotations

import re

from .dictionary_registry import get_mappings, _norm as _reg_norm  # noqa: F401

# SEED mappings — the default vocabulary, used when the registry has no entry.
# Teams extend/override these by registering rows, not by editing this file.
METRIC_CANONICAL_SEED: dict[str, str] = {
    "primary ratio": "primary_ratio",
    "primary metric": "primary_ratio",
    "key ratio": "primary_ratio",
    "headline ratio": "primary_ratio",
    "secondary ratio": "secondary_ratio",
    "coverage ratio": "coverage_ratio",
    "coverage": "coverage_ratio",
    "critical metric": "critical_metric",
    "key indicator": "key_indicator",
    "kpi": "key_indicator",
}

DOCTYPE_CANONICAL_SEED: dict[str, str] = {
    "primary report": "primary_report",
    "coverage report": "coverage_report",
    "correspondence": "correspondence",
    "meeting minutes": "meeting_minutes",
    "assessment": "assessment",
}


def _norm(raw: str) -> str:
    """Lowercase, strip, and collapse internal whitespace runs to one space."""
    return re.sub(r"\s+", " ", raw.strip().lower())


def _resolved_metric_map(spark=None) -> dict[str, str]:
    """Seed overlaid with registered metric mappings (registry wins)."""
    m = dict(METRIC_CANONICAL_SEED)
    m.update(get_mappings("metric", spark))
    return m


def _resolved_doctype_map(spark=None) -> dict[str, str]:
    m = dict(DOCTYPE_CANONICAL_SEED)
    m.update(get_mappings("doctype", spark))
    return m


def canonical_metric(raw: str, spark=None) -> tuple[str, bool]:
    """Return ``(canonical_name, was_mapped)``.

    Consults registered mappings first (overlaid on the seed). Unmapped terms
    return a slug of the raw name and ``False`` — never blocked, just flagged.
    """
    key = _norm(raw)
    mapping = _resolved_metric_map(spark)
    if key in mapping:
        return mapping[key], True
    return key.replace(" ", "_"), False


def canonical_doctype(raw: str, spark=None) -> str:
    key = _norm(raw)
    return _resolved_doctype_map(spark).get(key, key)
