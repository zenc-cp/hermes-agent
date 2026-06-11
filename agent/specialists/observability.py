"""ADR-032 observability sink — consumer side.

Mirrors the schema/contract of design-e/observability.py. Writes to the
same SQLite file (OBS_DB_PATH, default /var/lib/design-e/observability.sqlite)
so a single ``SELECT * WHERE task_id = ?`` returns the full timeline
across the design-e and consumer hops.

emit() is best-effort: any SQLite failure is logged but never propagated
to the caller. Observability MUST NOT break dispatch processing.
"""
from __future__ import annotations

import json as _json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OBS_DB_PATH = Path(os.getenv("OBS_DB_PATH", "/var/lib/design-e/observability.sqlite"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id     TEXT    NOT NULL,
  hop         TEXT    NOT NULL,
  event       TEXT    NOT NULL,
  timestamp   TEXT    NOT NULL,
  payload     TEXT,
  CHECK (length(task_id) BETWEEN 1 AND 128)
);
CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events(task_id);
CREATE INDEX IF NOT EXISTS idx_task_events_ts ON task_events(timestamp);
"""


def _resolve_path() -> Path:
    # Re-read module attribute each call so tests can monkeypatch OBS_DB_PATH.
    from agent.specialists import observability as _self  # type: ignore
    return _self.OBS_DB_PATH


def _connect() -> sqlite3.Connection:
    p = _resolve_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def emit(
    task_id: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one event to the sink (hop='consumer'). Best-effort: never raises."""
    if not task_id:
        return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        payload_str = _json.dumps(payload) if payload is not None else None
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO task_events (task_id, hop, event, timestamp, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, "consumer", event, ts, payload_str),
            )
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover - logged, not raised
        logger.error("observability.emit failed (task_id=%s event=%s): %s", task_id, event, e)
