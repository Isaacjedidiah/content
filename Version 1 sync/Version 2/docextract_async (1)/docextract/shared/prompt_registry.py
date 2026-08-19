"""Stored prompt registry — few-shots as DATA, keyed by (team, report_type).

Two prompt kinds are stored and dispatched:
  * ``extraction`` — used by the extractor to build per-(team, report_type)
    extraction prompts.
  * ``query`` — used by the supervisor chat: at query time the team and report
    type are identified and that specific few-shot set is pulled, so the
    supervisor's answer is grounded in the right domain examples.

Design (mirrors the uploaded prompt_registry): entries are appended to a
JSONL file locally (or a Delta table on Databricks); the most recently
registered entry for a key wins. If a team hasn't registered its own, a
built-in DEFAULT seed is used so nothing is ever blocked — an unregistered
(team, report_type) still gets a working prompt. Every write records who
changed it, for the same audit posture as review decisions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import CONFIG

# --- default seeds (used when a team hasn't registered its own) -------------

_DEFAULT_EXTRACTION_PREAMBLE = (
    "You are extracting metrics from a document submission. "
    "Return only structured claims via the provided tool. Preserve exact field "
    "names and values; capture unit, reporting basis, netting, scale and as-at "
    "date when stated; leave a field empty rather than guessing."
)

_DEFAULT_QUERY_PREAMBLE = (
    "You are an analyst assistant helping a supervisor query extracted data. "
    "Use only the retrieved context; never invent figures, dates or entity "
    "names. State exact value, unit, entity and reporting date. If a figure "
    "was read from a chart, say it hasn't been human-verified."
)

# Default few-shots — used when a team hasn't registered its own, so the chat
# has worked examples out of the box, not just an instruction. Teams override
# these by registering their own (any number); their entry replaces the seed.
_DEFAULT_QUERY_FEWSHOTS = [
    # a data question -> grounded, cited answer with unit + entity + date
    {"input_text": "What is the primary ratio for ACME_CORP?",
     "output_text": ("The primary ratio for ACME_CORP is 15.6% as at "
                     "2025-03-31 (source: primary_report).")},
    # a value read from a chart -> flag it as not human-verified
    {"input_text": "What was the peak exposure shown in the chart?",
     "output_text": ("The chart shows a peak of about 750 (units as labelled). "
                     "This was read from a chart and has not been "
                     "human-verified.")},
    # missing data -> say so, don't invent
    {"input_text": "What is the coverage ratio for an entity with no data?",
     "output_text": ("There is no coverage ratio on record for that entity in "
                     "the retrieved context, so I can't provide one.")},
]

_DEFAULT_EXTRACTION_FEWSHOTS = [
    {"input_text": "Primary ratio was 15.6% as at 31 March 2025.",
     "output_text": ('{"field_name": "primary ratio", "value": "15.6", '
                     '"unit": "%", "as_at": "2025-03-31"}')},
]


@dataclass
class FewShot:
    input_text: str
    output_text: str


@dataclass
class PromptEntry:
    team: str
    report_type: str
    kind: str                       # "extraction" | "query"
    preamble: str
    fewshots: list[FewShot] = field(default_factory=list)
    version: int = 1
    updated_by: str = "system"
    updated_at: str = ""

    def render(self, body: str) -> str:
        blocks = "".join(
            f"\nExample input:\n{f.input_text}\nExample output:\n{f.output_text}\n"
            for f in self.fewshots)
        return f"{self.preamble}\n{blocks}\n{body}"


def _local_path() -> str:
    root = os.environ.get("PRA_PROMPT_REGISTRY_DIR") or CONFIG.storage.local_root
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "prompt_registry.jsonl")


def _read_all(spark=None) -> list[dict]:
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.prompt_registry_table).collect()]
        except Exception:
            return []
    p = _local_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def register_prompt(entry: PromptEntry, updated_by: str, spark=None) -> None:
    """Self-serve write: append an entry (most recent wins on read)."""
    payload = {
        "team": entry.team, "report_type": entry.report_type, "kind": entry.kind,
        "preamble": entry.preamble,
        "fewshots": [{"input_text": f.input_text, "output_text": f.output_text}
                     for f in entry.fewshots],
        "version": entry.version, "updated_by": updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if spark is not None:
        spark.createDataFrame([payload]).write.format("delta").mode(
            "append").saveAsTable(CONFIG.storage.prompt_registry_table)
        return
    with open(_local_path(), "a") as fh:
        fh.write(json.dumps(payload) + "\n")


def get_prompt(team: str, report_type: str, kind: str,
               spark=None) -> PromptEntry:
    """Return the registered entry for (team, report_type, kind), or a default
    seed if none is registered. Never returns None — prompts never block."""
    matches = [e for e in _read_all(spark)
               if e["team"] == team and e["report_type"] == report_type
               and e["kind"] == kind]
    if matches:
        latest = matches[-1]
        return PromptEntry(
            team=latest["team"], report_type=latest["report_type"],
            kind=latest["kind"], preamble=latest["preamble"],
            fewshots=[FewShot(**f) for f in latest.get("fewshots", [])],
            version=latest.get("version", 1),
            updated_by=latest.get("updated_by", "system"),
            updated_at=latest.get("updated_at", ""))
    preamble = (_DEFAULT_EXTRACTION_PREAMBLE if kind == "extraction"
                else _DEFAULT_QUERY_PREAMBLE)
    seed_fewshots = (_DEFAULT_EXTRACTION_FEWSHOTS if kind == "extraction"
                     else _DEFAULT_QUERY_FEWSHOTS)
    return PromptEntry(team="default", report_type=report_type, kind=kind,
                       preamble=preamble,
                       fewshots=[FewShot(**f) for f in seed_fewshots])


def list_missing(pairs: list[tuple[str, str]], spark=None) -> list[tuple]:
    """(team, report_type, kind) combinations with no registered entry — for
    monitoring backlog visibility."""
    have = {(e["team"], e["report_type"], e["kind"]) for e in _read_all(spark)}
    missing = []
    for team, report_type in pairs:
        for kind in ("extraction", "query"):
            if (team, report_type, kind) not in have:
                missing.append((team, report_type, kind))
    return missing
