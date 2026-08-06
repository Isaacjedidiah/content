"""Supervisor chat session — memory, audit logging, and feedback in one store.

A single per-session record serves three jobs that share the same data:

  1. MEMORY — recent turns (question + the key figures returned) are kept so a
     follow-up like "will firm A survive from that figure?" can be resolved
     against what was just said (see llm_router reference-resolution).
  2. AUDIT — every question, the resolved question, the answer, and the sources
     cited are persisted to a log table, for compliance ("who asked what, and
     what were they told").
  3. FEEDBACK — a supervisor's rating (up/down + optional note) attaches to the
     turn it concerns, giving a quality signal that can drive prompt-registry
     improvements and monitoring.

Scope is PER SESSION: memory resets when a supervisor's session ends; the
audit log persists regardless. A session is identified by ``session_id`` (the
Streamlit app supplies a stable id per user session).

Persistence mirrors the rest of the package: JSONL locally, a Delta table on
Databricks (via the same DeltaLakeStorage sink pattern). Writing is
best-effort and never blocks answering — a logging failure must not deny a
supervisor their answer.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG


@dataclass
class Turn:
    """One question/answer exchange within a session."""
    turn_id: str
    session_id: str
    team: Optional[str]
    report_type: Optional[str]
    raw_question: str                 # what the supervisor typed
    resolved_question: str            # after reference-resolution rewrite
    route: Optional[str] = None       # sql / narrative / both
    sql: Optional[str] = None
    answer_text: Optional[str] = None
    # compact figures pulled from the sql_result, kept for follow-up memory
    key_figures: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    prompt_source: Optional[str] = None
    ts: str = ""
    # feedback (filled in later, when the supervisor rates the turn)
    feedback_rating: Optional[str] = None   # "up" | "down" | None
    feedback_note: Optional[str] = None
    feedback_ts: Optional[str] = None


class SupervisorSession:
    """In-memory conversation state for ONE supervisor session, with a
    write-through to the audit/feedback log."""

    def __init__(self, session_id: Optional[str] = None, spark=None,
                 max_memory_turns: int = 6):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.spark = spark
        self.max_memory_turns = max_memory_turns
        self._turns: list[Turn] = []

    # -- memory ------------------------------------------------------------

    def recent_turns(self, n: Optional[int] = None) -> list[Turn]:
        n = n or self.max_memory_turns
        return self._turns[-n:]

    def memory_context(self) -> str:
        """A compact text digest of recent turns for reference-resolution.
        Includes the prior question and any key figures returned, since a
        follow-up like 'from that figure' points at those."""
        lines = []
        for t in self.recent_turns():
            lines.append(f"Q: {t.resolved_question}")
            if t.key_figures:
                figs = "; ".join(str(f) for f in t.key_figures[:6])
                lines.append(f"   (figures returned: {figs})")
            elif t.answer_text:
                lines.append(f"   (answer: {t.answer_text[:160]})")
        return "\n".join(lines)

    def has_history(self) -> bool:
        return bool(self._turns)

    # -- recording a turn --------------------------------------------------

    def record_turn(self, team, report_type, raw_question, resolved_question,
                    result: dict) -> Turn:
        """Append a completed turn to memory and persist it to the audit log."""
        turn = Turn(
            turn_id=uuid.uuid4().hex[:12],
            session_id=self.session_id,
            team=team, report_type=report_type,
            raw_question=raw_question,
            resolved_question=resolved_question,
            route=result.get("route"),
            sql=result.get("sql"),
            answer_text=result.get("answer"),
            key_figures=_extract_key_figures(result.get("sql_result")),
            sources=_extract_sources(result.get("narrative")),
            prompt_source=result.get("prompt_source"),
            ts=datetime.now(timezone.utc).isoformat(),
        )
        self._turns.append(turn)
        _persist(turn, self.spark)        # best-effort audit write
        return turn

    # -- feedback ----------------------------------------------------------

    def add_feedback(self, turn_id: str, rating: str,
                     note: Optional[str] = None) -> Optional[Turn]:
        """Attach feedback to a recorded turn and persist the update.

        rating must be 'up' or 'down'. Returns the updated Turn, or None if the
        turn isn't in this session's memory (feedback on an expired turn still
        writes a standalone feedback row for the audit log)."""
        rating = rating.lower().strip()
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        turn = next((t for t in self._turns if t.turn_id == turn_id), None)
        ts = datetime.now(timezone.utc).isoformat()
        if turn is not None:
            turn.feedback_rating = rating
            turn.feedback_note = note
            turn.feedback_ts = ts
            _persist(turn, self.spark, kind="feedback")
            return turn
        # turn expired from memory: still log the feedback event on its own
        _persist_feedback_only(self.session_id, turn_id, rating, note, ts,
                               self.spark)
        return None


# --- helpers ----------------------------------------------------------------

def _extract_key_figures(sql_result) -> list:
    """Pull a compact list of (label=value) figures from a SQL result so a
    follow-up can reference 'the figure'. Best-effort; never raises."""
    if not sql_result:
        return []
    figs = []
    try:
        for row in sql_result[:6]:
            if not isinstance(row, dict):
                continue
            metric = row.get("canonical_metric") or row.get("field_name") or "value"
            val = row.get("value")
            firm = row.get("entity_ref", "")
            unit = row.get("unit", "") or ""
            figs.append(f"{firm} {metric}={val}{unit}".strip())
    except Exception:
        return []
    return figs


def _extract_sources(narrative) -> list:
    if not narrative:
        return []
    out = []
    try:
        for h in narrative[:6]:
            md = h.get("metadata", {}) if isinstance(h, dict) else {}
            src = md.get("source_document_id") or md.get("entity_ref") or ""
            if src:
                out.append(src)
    except Exception:
        return []
    return out


def _log_path() -> str:
    root = os.environ.get("PRA_SUPERVISOR_LOG_DIR") or CONFIG.storage.local_root
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "supervisor_query_log.jsonl")


def _persist(turn: Turn, spark=None, kind: str = "answer") -> None:
    """Best-effort write of a turn (as answer or feedback update). Never
    raises — a logging failure must not deny the supervisor their answer."""
    try:
        row = asdict(turn)
        row["log_kind"] = kind
        if spark is not None:
            spark.createDataFrame([row]).write.format("delta").mode(
                "append").saveAsTable(CONFIG.storage.supervisor_log_table)
        else:
            with open(_log_path(), "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


def _persist_feedback_only(session_id, turn_id, rating, note, ts,
                           spark=None) -> None:
    try:
        row = {"log_kind": "feedback_only", "session_id": session_id,
               "turn_id": turn_id, "feedback_rating": rating,
               "feedback_note": note, "feedback_ts": ts}
        if spark is not None:
            spark.createDataFrame([row]).write.format("delta").mode(
                "append").saveAsTable(CONFIG.storage.supervisor_log_table)
        else:
            with open(_log_path(), "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


def load_query_log(spark=None) -> list[dict]:
    """Read back the supervisor query log (for monitoring / audit review)."""
    if spark is not None:
        try:
            return [r.asDict() for r in
                    spark.table(CONFIG.storage.supervisor_log_table).collect()]
        except Exception:
            return []
    p = _log_path()
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]
