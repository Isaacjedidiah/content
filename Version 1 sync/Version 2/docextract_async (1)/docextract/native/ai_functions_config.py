"""Configuration for the Databricks-native AI functions track.

Mirrors the custom track's config so the two are swappable. Declares the
serving endpoints backing ai_parse_document / ai_extract / ai_classify, the
extraction schema, and the doc-type labels.

The extraction schema is expressed as a response format so ai_extract returns
per-claim structured objects (value + unit + basis + scale + date +
confidence), matching the custom track's Claim shape. This is what makes the
two tracks genuinely swappable and lets the native Silver->Gold projection
select real columns.
"""
from __future__ import annotations

from dataclasses import dataclass


# Per-claim response schema for ai_extract, aligned to Claim. Databricks
# ai_extract accepts a JSON schema / labels array depending on runtime; this
# structure is rendered into the SQL builder below.
CLAIM_RESPONSE_FIELDS: tuple = (
    "field_name", "value", "unit", "reporting_basis", "netting",
    "scale", "as_at_date", "confidence",
)


@dataclass(frozen=True)
class AIFunctionsConfig:
    parse_endpoint: str = "databricks-ai-parse"
    extract_endpoint: str = "databricks-meta-llama"
    classify_endpoint: str = "databricks-meta-llama"
    claim_fields: tuple = CLAIM_RESPONSE_FIELDS
    metric_labels: tuple = (
        "primary_ratio", "secondary_ratio", "critical_metric",
        "coverage_ratio", "funding_ratio", "key_indicator",
    )
    doc_labels: tuple = (
        "primary_report", "coverage_report", "correspondence",
        "board_minutes", "risk_assessment",
    )
    confidence_threshold: float = 0.60


AI_CFG = AIFunctionsConfig()


def extract_fields_array() -> str:
    return ", ".join(f"'{f}'" for f in AI_CFG.claim_fields)


def classify_sql_template(source_table: str = "bronze.parsed_documents") -> str:
    labels = ", ".join(f"'{lbl}'" for lbl in AI_CFG.doc_labels)
    return (
        f"SELECT content, ai_classify(content, array({labels})) AS document_type "
        f"FROM {source_table}"
    )
