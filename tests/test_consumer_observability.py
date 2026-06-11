"""ADR-032 observability sink — consumer side tests.

Verifies the consumer hop emits the right events to the shared SQLite sink:
- leased (after lease claim)
- completed (happy path)
- failed (malformed JSON, persona load error, invoke_agent error)
- expired (TTL exceeded)
- reaped (stale .processing/ file)

All emit() calls are best-effort; the events MUST land in the sink for the
ADR-032 acceptance criterion (single-query timeline).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.specialists import observability as obs
from agent.specialists.consumer import run_once, reap_stale_processing


@pytest.fixture
def obs_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "obs.sqlite"
    monkeypatch.setattr(obs, "OBS_DB_PATH", p, raising=False)
    return p


def _events_for(db: Path, task_id: str) -> list[tuple[str, str]]:
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as c:
        rows = c.execute(
            "SELECT hop, event FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    return rows


def _write_dispatch(
    inbox: Path,
    task_id: str,
    specialist: str = "Scout",
    *,
    created_at: str | None = None,
    ttl_sec: int = 3600,
) -> Path:
    f = inbox / f"{task_id}.json"
    f.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "specialist": specialist,
                "task": {"id": task_id, "role": "research", "payload": {}, "ttl_sec": ttl_sec},
                "context": {"session_id": "s", "user_id": "u"},
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return f


def _make_persona_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "Scout.yaml").write_text(
        "name: Scout\n"
        "system_prompt: stub\n"
        "allowed_tools: []\n"
        "default_model: gpt-5-chat\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    return d


def test_happy_path_emits_leased_then_completed(tmp_path: Path, obs_db: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = _make_persona_dir(tmp_path / "personas")
    _write_dispatch(inbox, "obs-ok")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert _events_for(obs_db, "obs-ok") == [
        ("consumer", "leased"),
        ("consumer", "completed"),
    ]


def test_malformed_json_emits_leased_then_failed(tmp_path: Path, obs_db: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = _make_persona_dir(tmp_path / "personas")
    (inbox / "obs-malformed.json").write_text("{not json", encoding="utf-8")

    with patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert _events_for(obs_db, "obs-malformed") == [
        ("consumer", "leased"),
        ("consumer", "failed"),
    ]


def test_ttl_expired_emits_leased_then_expired(tmp_path: Path, obs_db: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = _make_persona_dir(tmp_path / "personas")
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _write_dispatch(inbox, "obs-expired", created_at=old, ttl_sec=60)

    with patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert _events_for(obs_db, "obs-expired") == [
        ("consumer", "leased"),
        ("consumer", "expired"),
    ]


def test_persona_load_error_emits_leased_then_failed(tmp_path: Path, obs_db: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = tmp_path / "personas"
    personas.mkdir()  # no Scout.yaml -> load_persona raises
    _write_dispatch(inbox, "obs-no-persona")

    with patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert _events_for(obs_db, "obs-no-persona") == [
        ("consumer", "leased"),
        ("consumer", "failed"),
    ]


def test_invoke_error_emits_leased_then_failed(tmp_path: Path, obs_db: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = _make_persona_dir(tmp_path / "personas")
    _write_dispatch(inbox, "obs-boom")

    def boom(*_a, **_kw):
        raise RuntimeError("model unavailable")

    with patch("agent.specialists.consumer.invoke_agent", side_effect=boom), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert _events_for(obs_db, "obs-boom") == [
        ("consumer", "leased"),
        ("consumer", "failed"),
    ]


def test_reaper_emits_reaped(tmp_path: Path, obs_db: Path) -> None:
    processing = tmp_path / ".processing"
    processing.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    f = processing / "obs-stale.json"
    f.write_text(
        json.dumps({"task_id": "obs-stale", "specialist": "Scout"}),
        encoding="utf-8",
    )
    # Force file mtime way into the past
    import os
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
    os.utime(f, (old_ts, old_ts))

    with patch("agent.specialists.consumer.record_event"):
        n = reap_stale_processing(processing, results, max_age_s=60)

    assert n == 1
    assert _events_for(obs_db, "obs-stale") == [("consumer", "reaped")]


def test_emit_failure_does_not_break_run_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort contract: if observability.emit raises, run_once still completes."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    personas = _make_persona_dir(tmp_path / "personas")
    _write_dispatch(inbox, "obs-safe")

    def explode(*_a, **_kw):
        raise RuntimeError("sink down")

    monkeypatch.setattr(obs, "emit", explode)

    # If emit() were not safe-guarded by being best-effort internally,
    # the leased call would crash run_once. Our emit() catches its own
    # errors; the monkeypatch replaces the whole function, so we still
    # need run_once to either succeed (preferred) or raise cleanly.
    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        # Either passes through OK (because the import path is _obs.emit
        # captured at module load) OR raises — but if it raises, that's
        # a regression on the best-effort contract.
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    # Terminal result still written.
    assert (results / "obs-safe.json").exists()
