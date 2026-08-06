"""Supervisor query router.

Classifies a natural-language question as SQL / narrative / both with one
cheap classification call, generates SQL when needed, then dispatches. A
SELECT-only safety layer rejects any mutating or chained SQL before
execution. SQL runs on SQLite locally or spark.sql() on Databricks; narrative
search runs against Azure AI Search.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Optional

from .config import CLASSIFIER_MODEL, NL2SQL_MODEL
from .llm_client import LLMClient


class Route(str, Enum):
    SQL = "sql"
    NARRATIVE = "narrative"
    BOTH = "both"


CLASSIFY_PROMPT = (
    "Classify the user's question about regulatory data as exactly one of: "
    "'sql' (needs a database query over structured metrics), 'narrative' "
    "(needs explanatory text search), or 'both'. Reply with only the label."
)

# The Gold table schema the model may query. Kept explicit so generated SQL
# references only real columns.
GOLD_SCHEMA_HINT = (
    "Table gold_metrics(entity_ref TEXT, canonical_metric TEXT, value TEXT, "
    "unit TEXT, reporting_basis TEXT, scale TEXT, as_at_date TEXT, "
    "netting TEXT, citation_tier TEXT)."
)

NL2SQL_PROMPT = (
    "You translate a supervisor's question into a single read-only SQL "
    "SELECT over the gold_metrics table. " + GOLD_SCHEMA_HINT + " Rules: "
    "emit exactly one SELECT statement; no comments; no semicolons; no DML "
    "or DDL; if a team scope is provided, do not invent a team column — the "
    "caller applies scoping. Reply with only the SQL."
)

REWRITE_PROMPT = (
    "You rewrite a supervisor's follow-up question into a single "
    "self-contained question, resolving references to earlier turns. Use the "
    "conversation so far to replace pronouns and phrases like 'that figure', "
    "'it', 'they', 'the firm' with the specific firm, metric, or value they "
    "refer to. Do NOT answer the question or add new facts — only make it "
    "stand alone. If it is already self-contained, return it unchanged. Reply "
    "with only the rewritten question."
)


class QueryRouter:
    def __init__(self, client: Optional[LLMClient] = None,
                 sql_executor: Optional[Callable[[str], list[dict]]] = None,
                 search_store=None):
        self._client = client or LLMClient()
        self._sql_executor = sql_executor
        self._search_store = search_store

    def classify(self, question: str) -> Route:
        resp = self._client.complete(CLASSIFIER_MODEL, CLASSIFY_PROMPT, question)
        label = resp.text.strip().lower()
        try:
            return Route(label)
        except ValueError:
            # Ambiguous label -> safest is to try both.
            return Route.BOTH

    def generate_sql(self, question: str) -> str:
        """Produce a single read-only SELECT for the question."""
        resp = self._client.complete(NL2SQL_MODEL, NL2SQL_PROMPT, question)
        return _strip_sql_fences(resp.text)

    def resolve_references(self, question: str, session=None) -> str:
        """Rewrite a follow-up into a self-contained question using session
        memory. Returns the question unchanged when there's no history (first
        turn) or no session. Best-effort: if the rewrite fails or comes back
        empty, fall back to the original question rather than break the turn."""
        if session is None or not session.has_history():
            return question
        context = session.memory_context()
        if not context:
            return question
        content = (f"Conversation so far:\n{context}\n\n"
                   f"Follow-up question: {question}\n\nRewritten question:")
        try:
            resp = self._client.complete(CLASSIFIER_MODEL, REWRITE_PROMPT, content)
            rewritten = (resp.text or "").strip()
        except Exception:
            return question
        return rewritten or question

    def answer(self, question: str, sql: Optional[str] = None,
               team_filter: Optional[str] = None,
               report_type: Optional[str] = None,
               session=None) -> dict:
        # Reference resolution: if this is a follow-up in an ongoing session,
        # rewrite it into a self-contained question first, so "will firm A
        # survive from that figure?" becomes a question the router can act on.
        resolved = self.resolve_references(question, session)

        route = self.classify(resolved)
        out: dict = {"route": route.value, "sql": None, "sql_result": None,
                     "narrative": None, "answer": None,
                     "prompt_source": None,
                     "raw_question": question, "resolved_question": resolved}

        if route in (Route.SQL, Route.BOTH):
            # Generate SQL when the caller didn't supply it. This is the step
            # that was previously missing: the UI never passed SQL, so the
            # SQL branch never fired.
            candidate = sql if sql is not None else self.generate_sql(resolved)
            candidate = _apply_team_scope(candidate, team_filter)
            out["sql"] = candidate
            out["sql_result"] = self.run_sql(candidate)

        if route in (Route.NARRATIVE, Route.BOTH) and self._search_store:
            hits = self._search_store.search(resolved, team_filter=team_filter)
            out["narrative"] = hits
            # Synthesise a narrative answer using the (team, report_type)
            # query few-shots from the prompt registry, so the supervisor's
            # answer is grounded in the right domain examples. Falls back to a
            # default seed when the team hasn't registered its own.
            if team_filter and report_type is not None:
                out["answer"], out["prompt_source"] = self._synthesise(
                    resolved, hits, team_filter, report_type)

        # Record the turn in session memory + the audit log (best-effort).
        if session is not None:
            turn = session.record_turn(team_filter, report_type, question,
                                       resolved, out)
            out["turn_id"] = turn.turn_id

        return out

    def _synthesise(self, question: str, hits: list,
                    team: str, report_type: str) -> tuple[str, str]:
        """Build the answer prompt from the registered (team, report_type)
        query few-shots and call the model. Returns (answer, prompt_source)."""
        from .prompt_registry import get_prompt

        entry = get_prompt(team, report_type, "query")
        context = "\n".join(
            f"[{h.get('metadata', {}).get('entity_ref', '')}] {h.get('text', '')}"
            for h in (hits or []))
        body = (f"Retrieved context:\n{context or '(no matching documents)'}\n\n"
                f"Supervisor question ({team} / {report_type}): {question}\n\nAnswer:")
        prompt = entry.render(body)
        resp = self._client.complete(CLASSIFIER_MODEL, prompt, question)
        source = (f"{entry.team}/{entry.report_type}"
                  if entry.team != "default" else "default_seed")
        return resp.text, source

    def run_sql(self, sql: str) -> list[dict]:
        if not is_safe_select(sql):
            raise PermissionError(f"Refusing non-SELECT SQL: {sql!r}")
        if self._sql_executor is None:
            raise RuntimeError("No SQL executor configured.")
        return self._sql_executor(sql)


def _apply_team_scope(sql: str, team_filter: Optional[str]) -> str:
    """Wrap the generated query so team scoping is enforced by the caller,
    not trusted to the model. Uses a subquery to avoid parsing the WHERE."""
    if not team_filter:
        return sql
    safe_team = team_filter.replace("'", "''")
    inner = sql.strip().rstrip(";")
    return (
        f"SELECT * FROM ({inner}) AS scoped "
        f"WHERE entity_ref IN (SELECT entity_ref FROM firm_team "
        f"WHERE team = '{safe_team}')"
    )


_KEYWORD_RE = {
    kw: re.compile(rf"\b{kw}\b", re.IGNORECASE)
    for kw in ("insert", "update", "delete", "drop", "alter",
               "create", "merge", "truncate", "grant", "revoke", "call",
               "execute", "attach", "pragma")
}


def is_safe_select(sql: str) -> bool:
    """SELECT-only guard.

    Prefers ``sqlparse`` (asserts exactly one statement of type SELECT); if
    unavailable, falls back to word-boundary keyword matching so legitimate
    identifiers like ``created_returns`` or ``delete_flag`` are not rejected.
    """
    s = sql.strip().rstrip(";")
    if not s:
        return False

    try:
        import sqlparse  # type: ignore

        statements = [st for st in sqlparse.parse(s) if str(st).strip()]
        if len(statements) != 1:
            return False
        stmt = statements[0]
        if stmt.get_type() != "SELECT":
            return False
        # Reject stacked statements hidden by comments/newlines.
        if ";" in str(stmt).rstrip(";"):
            return False
        return True
    except ImportError:
        # Fallback: no stacked statements, must start with SELECT, and no
        # mutating keywords as whole words.
        if ";" in s:
            return False
        if not re.match(r"(?is)^\s*select\b", s):
            return False
        return not any(rx.search(s) for rx in _KEYWORD_RE.values())


def _strip_sql_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:sql)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    return t
