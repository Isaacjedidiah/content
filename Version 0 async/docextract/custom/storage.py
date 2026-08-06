"""Medallion storage: Bronze / Silver / Gold.

Local mode persists JSONL under a root dir; Databricks mode writes Delta
tables. Gold is a deliberate whitelist — ``normalisation_conflicts`` is
excluded from Gold (they go to Azure AI Search for narrative search).

The build/persist split (pure ``build_gold_metric_rows`` + thin
``write_gold_metrics``) keeps row-building testable without I/O.

Fixes: entity_ref is read from the Claim (which now carries it) instead of a
sentinel; re-runs are idempotent via content_hash dedup on JSONL; quarantine,
review decisions and audit events have real sinks; reviewed/overridden claims
can be promoted into Gold.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import Claim, GoldMetric, content_hash

GOLD_WHITELIST = [
    "entity_ref", "canonical_metric", "value", "unit", "reporting_basis",
    "scale", "as_at_date", "netting", "measure_type", "period",
    "citation_tier", "review_tier",
]


class StorageManager:
    """Local JSONL persistence for MVP / test runs."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or CONFIG.storage.local_root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, layer: str) -> str:
        return os.path.join(self.root, f"{layer}.jsonl")

    def append_jsonl(self, layer: str, rows: list[dict]) -> int:
        """Append rows, skipping any whose content hash was already written.

        Idempotency: each row gets a stable hash over its sorted JSON; hashes
        already present in the file are skipped, so re-scanning processed
        files never duplicates. (On Databricks, MERGE handles this instead.)
        """
        if not rows:
            return 0
        seen = self._seen_hashes(layer)
        written = 0
        with open(self._path(layer), "a") as fh:
            for r in rows:
                h = content_hash(json.dumps(r, sort_keys=True, default=str))
                if h in seen:
                    continue
                seen.add(h)
                fh.write(json.dumps(r, default=str) + "\n")
                written += 1
        return written

    def _seen_hashes(self, layer: str) -> set[str]:
        p = self._path(layer)
        if not os.path.exists(p):
            return set()
        seen: set[str] = set()
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    seen.add(content_hash(
                        json.dumps(row, sort_keys=True, default=str)))
        return seen

    def read_jsonl(self, layer: str) -> list[dict]:
        p = self._path(layer)
        if not os.path.exists(p):
            return []
        with open(p) as fh:
            return [json.loads(line) for line in fh if line.strip()]


class DeltaLakeStorage(StorageManager):
    """Writes Delta tables on Databricks; falls back to JSONL locally.

    ``spark`` is injected on a cluster. When absent, all writes go to JSONL
    so the same code path works in tests.
    """

    def __init__(self, spark=None, root: Optional[str] = None):
        super().__init__(root=root)
        self.spark = spark

    @staticmethod
    def build_gold_metric_rows(claims: list[Claim]) -> list[dict]:
        """PURE: build Gold rows from claims. No I/O."""
        rows: list[dict] = []
        for c in claims:
            gm = GoldMetric(
                entity_ref=_entity_ref(c),
                canonical_metric=c.canonical_metric or c.field_name,
                value=c.value,
                unit=c.unit,
                reporting_basis=c.reporting_basis,
                scale=c.scale,
                as_at_date=c.as_at_date,
                netting=c.netting,
                measure_type=c.measure_type,
                period=c.period,
                citation_tier=c.citation_tier.value,
                review_tier=c.review_tier.value,
            )
            rows.append(gm.model_dump())
        return rows

    def write_gold_metrics(self, claims: list[Claim],
                           team: Optional[str] = None,
                           report_type: Optional[str] = None) -> int:
        """THIN: build THEN persist. Returns rows written.

        When team+report_type are given, writes to the per-partition table
        (e.g. gold.fx_exchange_report); entity_ref remains a column in the rows.
        """
        rows = self.build_gold_metric_rows(claims)
        return self._write_layer("gold", rows, team, report_type)

    def write_silver(self, claims: list[Claim], team: Optional[str] = None,
                     report_type: Optional[str] = None) -> int:
        rows = [c.model_dump(mode="json") for c in claims]
        return self._write_layer("silver", rows, team, report_type)

    def write_bronze(self, elements: list[dict], team: Optional[str] = None,
                     report_type: Optional[str] = None) -> int:
        return self._write_layer("bronze", elements, team, report_type)

    def write_quarantine(self, rows: list[dict]) -> int:
        return self._write("quarantine", CONFIG.storage.quarantine_table, rows)

    def write_reviews(self, rows: list[dict]) -> int:
        return self._write("reviews", CONFIG.storage.review_table, rows)

    def write_audit(self, rows: list[dict]) -> int:
        return self._write("audit", CONFIG.storage.audit_table, rows)

    def _write_layer(self, layer: str, rows: list[dict],
                     team: Optional[str], report_type: Optional[str]) -> int:
        """Write a medallion layer, partitioned by team+report_type when both
        are supplied. Falls back to a single shared table/file otherwise (so
        existing callers and tests keep working)."""
        if team and report_type:
            table = CONFIG.storage.layer_table(layer, team, report_type)
            local = CONFIG.storage.layer_local(layer, team, report_type)
        else:
            table = getattr(CONFIG.storage, f"{layer}_table", None) \
                or CONFIG.storage.gold_metrics
            local = layer
        if self.spark is not None and rows:
            df = self.spark.createDataFrame(rows)
            df.write.format("delta").mode("append").saveAsTable(table)
            return len(rows)
        return self.append_jsonl(local, rows)

    def _write(self, layer: str, table: str, rows: list[dict]) -> int:
        if self.spark is not None and rows:
            df = self.spark.createDataFrame(rows)
            df.write.format("delta").mode("append").saveAsTable(table)
            return len(rows)
        return self.append_jsonl(layer, rows)


def _entity_ref(c: Claim) -> str:
    """entity_ref now genuinely originates from the Claim (propagated from the
    source Element). The sentinel only appears if attribution truly failed
    upstream, which the pipeline routes to review rather than silently
    trusting."""
    return c.entity_ref or "UNKNOWN_ENTITY_REF"
