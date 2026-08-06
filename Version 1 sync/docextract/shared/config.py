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
    # Per-canonical-metric review thresholds: a misread low-risk field and a
    # misread critical metric don't carry the same real-world risk, so they
    # don't share one global bar. A metric absent here uses ``human_review``
    # above as the default. These are EXAMPLE metrics — replace the keys with
    # the high-stakes fields of your own domain.
    per_metric_review: dict = field(default_factory=lambda: {
        "primary_ratio": 0.95,
        "secondary_ratio": 0.95,
        "coverage_ratio": 0.95,
        "critical_metric": 0.95,
        "key_indicator": 0.93,
    })

    def review_threshold(self, canonical_metric: Optional[str]) -> float:
        """Threshold below which a claim for this metric needs human review."""
        if canonical_metric and canonical_metric in self.per_metric_review:
            return self.per_metric_review[canonical_metric]
        return self.human_review


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
    catalog: str = "docextract"
    volume_root: str = "/Volumes/docextract/landing/submissions"

    bronze_table: str = "docextract.bronze.raw_elements"
    silver_table: str = "docextract.silver.extracted_claims"
    gold_metrics: str = "docextract.gold.metrics"
    quarantine_table: str = "docextract.bronze.quarantine"
    run_metrics_table: str = "docextract.ops.run_metrics"
    review_table: str = "docextract.ops.review_decisions"
    audit_table: str = "docextract.ops.audit_events"

    # Prompt registry: stored few-shots keyed by (team, report_type, kind),
    # used both for extraction and for the supervisor chat.
    prompt_registry_table: str = "docextract.ops.prompt_registry"

    # Supervisor chat query + feedback log (audit of every Q&A, plus ratings).
    supervisor_log_table: str = "docextract.ops.supervisor_query_log"

    # Eval: team-registered expected values (the answer key), and the results
    # of scoring pipeline output against them + human-reviewed values.
    eval_registry_table: str = "docextract.ops.eval_registry"
    eval_results_table: str = "docextract.ops.eval_results"

    # Dictionary: team-registered synonym -> canonical mappings (metrics and
    # document types), so vocabulary is data, not code.
    dictionary_registry_table: str = "docextract.ops.dictionary_registry"

    local_root: str = "./_local_store"

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
        """Unity Catalog table name for a medallion layer, partitioned by
        team+report_type: e.g. layer='silver', fx/exchange_report ->
        'docextract.silver.fx_exchange_report'. entity_ref is a COLUMN inside
        each table, not part of the name."""
        return f"{self.catalog}.{layer}.{self.slug(team, report_type)}"

    def layer_local(self, layer: str, team: str, report_type: str) -> str:
        """Local JSONL filename stem for the same partition, e.g.
        'silver__fx_exchange_report'."""
        return f"{layer}__{self.slug(team, report_type)}"


@dataclass(frozen=True)
class SearchConfig:
    """Azure AI Search settings.

    Design: the caller performs chunking; Azure AI Search performs embedding
    (via an indexer/skillset AzureOpenAIEmbedding skill on push) and query-time
    vectorization (via a vectorizer bound to the vector field). Our code never
    calls an embedding API — it pushes chunk text and issues vector queries.
    """
    index_name: str = "regulatory-chunks"
    # text-embedding-3-large @ 3072 dims: strongest retrieval quality for
    # dense regulatory prose and acronym-heavy content. Configurable down to
    # 1536 later if latency/cost demands, by re-deploying the index.
    embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    vector_profile: str = "reg-hnsw-profile"
    vectorizer_name: str = "reg-aoai-vectorizer"
    semantic_config: str = "reg-semantic"
    api_version: str = "2024-07-01"


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
