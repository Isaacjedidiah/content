"""Structured, traceable audit logging.

Records structured events (not free text) so any claim can be traced
end-to-end: document -> model(s) -> confidence -> review decision. Events
can be exported to rows for persistence to the audit Delta table / JSONL,
so the trail survives process exit (it was previously in-memory only).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    run_id: str
    stage: str
    element_id: str
    detail: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class AuditLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def log(self, run_id: str, stage: str, element_id: str, **detail) -> None:
        self._events.append(Event(run_id, stage, element_id, detail))

    def trace(self, element_id: str) -> list[dict]:
        return [asdict(e) for e in self._events if e.element_id == element_id]

    def to_rows(self) -> list[dict]:
        """All events as flat rows for persistence."""
        rows = []
        for e in self._events:
            row = {"run_id": e.run_id, "stage": e.stage,
                   "element_id": e.element_id, "ts": e.ts}
            # Flatten detail into JSON string so the Delta schema stays stable
            # regardless of which keys a given stage logged.
            import json
            row["detail"] = json.dumps(e.detail, default=str)
            rows.append(row)
        return rows

    def clear(self) -> None:
        self._events.clear()
