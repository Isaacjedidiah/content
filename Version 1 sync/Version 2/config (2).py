"""Central configuration for the DocExtract document data product.

Single source of truth for models, thresholds, storage locations, teams and
environment. Every other module imports ``CONFIG`` from here.

Design principle: **no silent fallbacks**. In a Databricks environment,
missing credentials raise immediately rather than degrading to stubs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Environment(str, Enum):
    LOCAL = "local"
    DATABRICKS = "databricks"


def detect_environment() -> Environment:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return Environment.DATABRICKS
    return Environment.LOCAL


class Team(str, Enum):
    """Canonical domains/teams. Single source of truth.

    The UIs, prompt registry and query scoping all resolve against this
    rather than hand-maintained string lists (which had drifted apart).

    These are neutral EXAMPLE domains — replace them with the teams or
    document categories of your own domain when reusing this pipeline.
    """
    FINANCE = "finance"
    OPERATIONS = "operations"
    COMPLIANCE = "compliance"
    RESEARCH = "research"

    @classmethod
    def values(cls) -> list[str]:
        return [t.value for t in cls]


@dataclass(frozen=True)
class ModelSpec:
    """One row in the model registry, addressed by role not vendor."""
    key: str
    endpoint: str
    input_cost_per_m: float
    output_cost_per_m: float
    tier: int


# Two-tier cascade: GPT-5 mini (Tier 1) -> Sonnet (Tier 2) -> human.
# Endpoints resolve to Databricks serving endpoints (Foundation Model APIs
# for Claude/Llama; External Model endpoint for GPT-5 mini).
MODELS: dict[str, ModelSpec] = {
    "tier1": ModelSpec("tier1", "gpt-5-mini", 0.15, 0.60, tier=1),
    "tier2": ModelSpec("tier2", "databricks-claude-sonnet", 3.00, 15.00, tier=2),
}
CLASSIFIER_MODEL = "tier1"
NL2SQL_MODEL = "tier1"


@dataclass(frozen=True)
class Thresholds:
    tier1_escalate: float = 0.75
    human_review: float = 0.60
    magnitude_conflict_ratio: float = 2.0

    def review_threshold(self, canonical_metric: Optional[str],
                         team: Optional[str] = None,
                         report_type: Optional[str] = None) -> float:
        """Threshold below which a claim for this metric needs human review.

        Per-metric bars are GOVERNED DATA in the threshold registry (written
        centrally, not by teams). This looks the registry up; if a metric has no
        registered bar it uses the global ``human_review``.

        The global is a FLOOR: a registered per-metric value can push the bar
        higher for a high-stakes metric, but can never drop it below the global.
        Resolved as ``max(global, registered)`` so scrutiny only ever increases
        from the baseline, never weakens — a governed guarantee, even against a
        mistaken registry entry below the floor.
        """
        if not canonical_metric:
            return self.human_review
        # lazy import avoids a config <-> registry import cycle
        from .threshold_registry import get_threshold
        registered = get_threshold(canonical_metric, team, report_type)
        if registered is None:
            return self.human_review
        return max(self.human_review, registered)


@dataclass(frozen=True)
class ClaimQuantPolicy:
    """Claim↔metric linking (adopted from the uploaded claim_quant_matcher)."""
    top_n_candidates: int = 5
    confidence_threshold: float = 0.70


@dataclass(frozen=True)
class RoutingPolicy:
    """Content-aware starting-tier policy.

    Decides which tier an element STARTS on, from cheap content signals,
    before any model call. Known-hard content starts at Tier 2 directly,
    skipping the wasted Tier 1 pass; everything else starts at Tier 1 and
    escalates reactively on low confidence as before.

    All rules are data so they can be tuned without code changes, and every
    decision carries a human-readable reason for the audit trail.
    """
    enabled: bool = True

    # Modalities that start on Tier 2. Tables are where extraction is hardest
    # and misreads are most expensive, so they skip Tier 1 by default.
    hard_modalities: tuple[str, ...] = ("table",)

    # Report types known to be dense/ambiguous enough to warrant Tier 2 first.
    # Empty by default: opt in per deployment once you have evidence.
    hard_report_types: tuple[str, ...] = ()

    # Character-length ceiling for Tier 1. Elements longer than this start on
    # Tier 2 (long context degrades the cheap model's extraction). 0 disables
    # the length rule.
    max_tier1_chars: int = 6000

    # Empty elements (e.g. flagged scanned pages) never warrant a paid Tier 2
    # pass; they start — and stay — on Tier 1, which will return nothing and
    # route to review/quarantine cheaply.
    skip_hard_routing_when_empty: bool = True


@dataclass(frozen=True)
class StoragePaths:
    # ONE catalog, ONE schema — every table lives in {catalog}.{schema}.*, with
    # the medallion layer expressed as a table-name PREFIX (bronze_/silver_/
    # gold_/ops_) rather than a separate schema. This matches a governed catalog
    # where a team is assigned a single schema. To re-home the whole pipeline,
    # change these two values only — every table name below derives from them.
    catalog: str = "docextract"
    schema: str = "reporting"
    volume_root: str = "/Volumes/docextract/landing/submissions"
    local_root: str = "./_local_store"

    def fqtn(self, table_name: str) -> str:
        """Fully-qualified table name: {catalog}.{schema}.{table_name}.
        The single place table names are assembled — change catalog/schema
        once and everything follows."""
        return f"{self.catalog}.{self.schema}.{table_name}"

    # Medallion + ops tables — layer is a name prefix, all in the one schema.
    @property
    def bronze_table(self) -> str: return self.fqtn("bronze_raw_elements")
    @property
    def silver_table(self) -> str: return self.fqtn("silver_extracted_claims")
    @property
    def gold_metrics(self) -> str: return self.fqtn("gold_metrics")
    @property
    def quarantine_table(self) -> str: return self.fqtn("bronze_quarantine")
    @property
    def run_metrics_table(self) -> str: return self.fqtn("ops_run_metrics")
    @property
    def review_table(self) -> str: return self.fqtn("ops_review_decisions")
    @property
    def audit_table(self) -> str: return self.fqtn("ops_audit_events")

    # Prompt registry: few-shots keyed by (team, report_type, kind), used for
    # extraction and the supervisor chat.
    @property
    def prompt_registry_table(self) -> str: return self.fqtn("ops_prompt_registry")

    # Supervisor chat query + feedback log.
    @property
    def supervisor_log_table(self) -> str: return self.fqtn("ops_supervisor_query_log")

    # Eval: registered expected values (answer key) + scoring results.
    @property
    def eval_registry_table(self) -> str: return self.fqtn("ops_eval_registry")
    @property
    def eval_results_table(self) -> str: return self.fqtn("ops_eval_results")

    # Dictionary: synonym -> canonical mappings.
    @property
    def dictionary_registry_table(self) -> str: return self.fqtn("ops_dictionary_registry")

    # Per-metric review thresholds (GOVERNED — written centrally).
    @property
    def threshold_registry_table(self) -> str: return self.fqtn("ops_threshold_registry")

    # Content-domain tag registry (GOVERNED — vocabulary + metric/heading maps).
    @property
    def tag_registry_table(self) -> str: return self.fqtn("ops_tag_registry")

    # AI Search: source chunks Delta table (pipeline appends here) + the Delta
    # Sync Index that mirrors it. Both governed Delta objects in the one schema.
    @property
    def search_chunks_table(self) -> str: return self.fqtn("ops_search_chunks")
    @property
    def search_index(self) -> str: return self.fqtn("ops_search_index")

    @staticmethod
    def slug(team: str, report_type: str) -> str:
        """Combined key used to name per-(team, report_type) storage, e.g.
        team='fx', report_type='exchange_report' -> 'fx_exchange_report'.
        Non-alphanumeric characters collapse to single underscores so the
        result is a valid table / directory name."""
        import re
        raw = f"{team}_{report_type}".strip().lower()
        return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    def layer_table(self, layer: str, team: str, report_type: str) -> str:
        """Table name for a medallion layer, partitioned by team+report_type,
        in the single assigned schema with the layer as a name prefix: e.g.
        layer='silver', fx/exchange_report ->
        'docextract.reporting.silver_fx_exchange_report'. entity_ref is a COLUMN
        inside each table, not part of the name."""
        return self.fqtn(f"{layer}_{self.slug(team, report_type)}")

    def layer_local(self, layer: str, team: str, report_type: str) -> str:
        """Local JSONL filename stem for the same partition, e.g.
        'silver__fx_exchange_report'."""
        return f"{layer}__{self.slug(team, report_type)}"


@dataclass(frozen=True)
class SearchConfig:
    """Databricks AI Search (Delta Sync) settings.

    Design: the caller performs chunking and APPENDS chunk rows to a governed
    Delta table; Databricks AI Search performs embedding and indexing. A Delta
    Sync Index mirrors the source table — after a batch appends chunks, a
    triggered sync embeds the new rows (via the managed embedding model) and
    makes them searchable. Our code never calls an embedding API directly.

    Everything lives in the one governed schema, consistent with the rest of
    the pipeline: the source chunks table and the index are both Delta objects
    in {catalog}.{schema}.
    """
    # The Vector Search / AI Search endpoint that serves the index (created in
    # the Databricks UI: Compute -> AI Search -> Create endpoint, Standard).
    endpoint_name: str = "docextract-vs"
    # Databricks-managed embedding model. Per Databricks' production guidance
    # for standard endpoints. No separate embedding deployment to provision —
    # the index uses this managed foundation model.
    embedding_model: str = "databricks-qwen3-embedding-0-6b"
    # Text column that gets embedded, and the primary key, in the source table.
    embedding_source_column: str = "content"
    primary_key: str = "chunk_id"
    # Triggered sync: sync() is called after each batch. Lower cost than
    # continuous for a batch pipeline (no always-on streaming cluster).
    sync_mode: str = "TRIGGERED"


@dataclass(frozen=True)
class AzureEndpoints:
    """Azure resource endpoints, read from environment / secrets."""

    @property
    def search_endpoint(self) -> Optional[str]:
        return os.environ.get("AZURE_SEARCH_ENDPOINT")

    @property
    def aoai_endpoint(self) -> Optional[str]:
        return os.environ.get("AZURE_OPENAI_ENDPOINT")


def _secret(scope: str, key: str, env_fallback: str) -> Optional[str]:
    """Read a secret from Databricks Secrets when on a cluster, else env.

    On Databricks this uses ``dbutils.secrets.get``; locally it reads an
    environment variable. Returns ``None`` if unset (callers decide whether
    that is fatal via ``Config.validate``).
    """
    try:
        # dbutils is injected into the notebook/cluster runtime.
        from pyspark.dbutils import DBUtils  # type: ignore
        from pyspark.sql import SparkSession

        dbutils = DBUtils(SparkSession.builder.getOrCreate())
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return os.environ.get(env_fallback)


@dataclass
class Config:
    environment: Environment = field(default_factory=detect_environment)
    models: dict = field(default_factory=lambda: dict(MODELS))
    thresholds: Thresholds = field(default_factory=Thresholds)
    routing: RoutingPolicy = field(default_factory=RoutingPolicy)
    claim_quant: ClaimQuantPolicy = field(default_factory=ClaimQuantPolicy)
    storage: StoragePaths = field(default_factory=StoragePaths)
    search: SearchConfig = field(default_factory=SearchConfig)
    azure: AzureEndpoints = field(default_factory=AzureEndpoints)
    secret_scope: str = "docextract"

    @property
    def is_databricks(self) -> bool:
        return self.environment == Environment.DATABRICKS

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(
                f"Unknown model key {key!r}. Known: {list(self.models)}"
            )
        return self.models[key]

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return _secret(self.secret_scope, "anthropic-api-key", "ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> Optional[str]:
        return _secret(self.secret_scope, "openai-api-key", "OPENAI_API_KEY")

    @property
    def search_api_key(self) -> Optional[str]:
        return _secret(self.secret_scope, "search-api-key", "AZURE_SEARCH_API_KEY")

    @property
    def aoai_api_key(self) -> Optional[str]:
        return _secret(self.secret_scope, "aoai-api-key", "AZURE_OPENAI_API_KEY")

    def validate(self) -> None:
        """Fail loudly if production is misconfigured."""
        if self.is_databricks:
            missing = [
                k for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")
                if not os.environ.get(k)
            ]
            # On a cluster these are implicit, so only warn via exception if
            # explicitly running off-cluster against Databricks.
            if missing and not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
                raise RuntimeError(
                    f"Databricks target requires {missing}; refusing to "
                    "fall back to stubs."
                )


CONFIG = Config()
CONFIG.validate()
