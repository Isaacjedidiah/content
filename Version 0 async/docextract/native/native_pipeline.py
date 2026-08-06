"""Lakeflow Declarative Pipeline (native track).

Declares Bronze -> Silver -> Gold as declarative tables. Bronze ingests via
Auto Loader; Silver applies ai_parse_document + ai_extract; a curation step
explodes the per-claim array, canonicalises metric names and threads entity_ref
from the file path; Gold filters on confidence.

Previously Silver produced a single ``claims`` column while Gold selected
per-claim columns that did not exist — the two did not line up. The
``silver_claims_exploded`` step is the missing bridge.

The ``dlt`` import is only available inside a pipeline run, so it is imported
lazily and guarded so the file can still be imported for unit tests of the
SQL builders.
"""
from __future__ import annotations

from ..shared.config import CONFIG
from .ai_functions_config import AI_CFG, extract_fields_array

try:  # pragma: no cover - only present inside a Lakeflow run
    import dlt  # type: ignore
    _HAS_DLT = True
except Exception:  # noqa: BLE001
    _HAS_DLT = False


def bronze_sql() -> str:
    # binaryFile load; entity_ref/team/report_type recovered from the path.
    return (
        "SELECT *, "
        "regexp_extract(path, '/([^/]+)/[^/]+/[^/]+/[^/]+$', 1) AS entity_ref, "
        "regexp_extract(path, '/([^/]+)/[^/]+/[^/]+$', 1) AS team, "
        "regexp_extract(path, '/([^/]+)/[^/]+$', 1) AS report_type "
        f"FROM cloud_files('{CONFIG.storage.volume_root}', 'binaryFile')"
    )


def silver_sql() -> str:
    # ai_extract returns an array of per-claim structs (one row's worth).
    return (
        "SELECT entity_ref, team, report_type, path, "
        "ai_extract(ai_parse_document(content), "
        f"array({extract_fields_array()})) AS claims "
        "FROM LIVE.bronze_documents"
    )


def silver_exploded_sql() -> str:
    """Bridge: explode claims to one row per claim and canonicalise metric.

    ``canonical_metric`` uses a CASE over the known synonym set; unmapped
    names pass through slugified (lower + underscores), never dropped."""
    cases = " ".join(
        f"WHEN lower(c.field_name) IN ({_syn(m)}) THEN '{m}'"
        for m, _syn_list in _METRIC_SYNONYMS.items()
        for _ in [0]
    )
    return (
        "SELECT entity_ref, "
        "c.field_name AS field_name, "
        f"CASE {cases} ELSE regexp_replace(lower(c.field_name), '\\\\s+', '_') "
        "END AS canonical_metric, "
        "c.value AS value, c.unit AS unit, "
        "c.reporting_basis AS reporting_basis, c.scale AS scale, "
        "c.as_at_date AS as_at_date, c.netting AS netting, "
        "CAST(c.confidence AS DOUBLE) AS confidence "
        "FROM LIVE.silver_claims LATERAL VIEW explode(claims) t AS c"
    )


def gold_sql() -> str:
    return (
        "SELECT entity_ref, canonical_metric, value, unit, reporting_basis, "
        "scale, as_at_date, netting FROM LIVE.silver_claims_exploded "
        f"WHERE confidence >= {AI_CFG.confidence_threshold}"
    )


# Metric synonyms for the SQL CASE (kept in sync with shared.dictionary).
_METRIC_SYNONYMS: dict[str, list[str]] = {
    "primary_ratio": ["primary ratio", "primary metric", "key ratio",
                      "headline ratio"],
    "secondary_ratio": ["secondary ratio"],
    "coverage_ratio": ["coverage ratio", "coverage"],
    "critical_metric": ["critical metric"],
    "key_indicator": ["key indicator", "kpi"],
}


def _syn(metric: str) -> str:
    return ", ".join(f"'{s}'" for s in _METRIC_SYNONYMS[metric])


if _HAS_DLT:  # pragma: no cover

    @dlt.table(comment="Raw files landed via Auto Loader")
    def bronze_documents():
        return spark.sql(bronze_sql())  # noqa: F821

    @dlt.table(comment="Parsed + extracted via native AI functions")
    def silver_claims():
        return spark.sql(silver_sql())  # noqa: F821

    @dlt.table(comment="One row per claim, canonicalised")
    def silver_claims_exploded():
        return spark.sql(silver_exploded_sql())  # noqa: F821

    @dlt.table(comment="Curated, SQL-queryable metrics")
    def gold_metrics():
        return spark.sql(gold_sql())  # noqa: F821
