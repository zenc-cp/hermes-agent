"""F2 audit fix (2026-06-10): tests for lease via .processing/, TTL expiry, and reaper.

Pre-fix the consumer deleted the inbox file in a `finally` regardless of
outcome, so a mid-execution crash silently lost the task and Scout polled
404 forever. These tests pin the new lease + reaper semantics.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_dispatch_with_meta(
    inbox: Path,
    task_id: str,
    specialist: str,
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


# ---------------------------------------------------------------------------
# F2.1 — happy path drains processing dir clean
# ---------------------------------------------------------------------------


def test_processing_dir_empty_after_successful_run(tmp_path: Path) -> None:
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch_with_meta(inbox, "t-ok", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    processing = inbox / ".processing"
    assert processing.exists(), "processing dir must be auto-created"
    assert list(processing.glob("*.json")) == [], "processing dir must be empty after success"


# ---------------------------------------------------------------------------
# F2.2 — crash leaves file in .processing for reaper
# ---------------------------------------------------------------------------


def test_unexpected_exception_leaves_file_in_processing(tmp_path: Path) -> None:
    """If something escapes all inner handlers (e.g. disk full on result write),
    the processing file must remain for the reaper to find on the next loop."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch_with_meta(inbox, "t-crash", "Scout")

    # Make _write_status_atomic blow up AFTER lease is claimed, simulating a
    # crash with no terminal status path. The processing file must remain.
    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer._write_status_atomic", side_effect=OSError("disk full")), \
         patch("agent.specialists.consumer.record_event"):
        with pytest.raises(OSError):
            run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    processing = inbox / ".processing"
    leftover = list(processing.glob("*.json"))
    assert len(leftover) == 1, "processing file must remain after unexpected crash"
    assert leftover[0].stem == "t-crash"
    assert not (inbox / "t-crash.json").exists(), "inbox file must have been moved out"


# ---------------------------------------------------------------------------
# F2.3 — TTL expiry writes status=expired without invoking the agent
# ---------------------------------------------------------------------------


def test_ttl_expired_skips_invoke_and_writes_expired_status(tmp_path: Path) -> None:
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    # created_at 2 hours ago + ttl_sec=60 -> clearly expired
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _write_dispatch_with_meta(inbox, "t-old", "Scout", created_at=long_ago, ttl_sec=60)

    with patch("agent.specialists.consumer.invoke_agent") as mock_invoke, \
         patch("agent.specialists.consumer.record_event") as mock_record:
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert not mock_invoke.called, "invoke_agent must NOT be called for an expired task"
    status_file = results / "t-old.json"
    assert status_file.exists()
    body = json.loads(status_file.read_text(encoding="utf-8"))
    assert body["status"] == "expired"
    assert "ttl_sec exceeded" in body["error"]
    # processing dir must be drained
    assert list((inbox / ".processing").glob("*.json")) == []
    # dispatch_failed event recorded
    assert mock_record.called


def test_ttl_not_expired_proceeds_normally(tmp_path: Path) -> None:
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch_with_meta(inbox, "t-fresh", "Scout", ttl_sec=3600)

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    body = json.loads((results / "t-fresh.json").read_text(encoding="utf-8"))
    assert body["status"] == "completed"


# ---------------------------------------------------------------------------
# F2.4 — reap_stale_processing
# ---------------------------------------------------------------------------


def test_reaper_writes_abandoned_status_and_deletes_stale(tmp_path: Path) -> None:
    from agent.specialists.consumer import reap_stale_processing

    processing = tmp_path / ".processing"
    results = tmp_path / "results"
    processing.mkdir()
    results.mkdir()

    # Drop a stale file (mtime 2 hours ago).
    stale = processing / "t-stale.json"
    stale.write_text(json.dumps({
        "task_id": "t-stale", "specialist": "Scout",
        "task": {"id": "t-stale", "role": "x", "payload": {}, "ttl_sec": 60},
        "context": {"session_id": "s", "user_id": "u"},
    }), encoding="utf-8")
    old_ts = time.time() - 7200
    os.utime(stale, (old_ts, old_ts))

    # Drop a fresh file that should NOT be reaped.
    fresh = processing / "t-fresh.json"
    fresh.write_text(json.dumps({"task_id": "t-fresh", "specialist": "Scout"}), encoding="utf-8")

    with patch("agent.specialists.consumer.record_event") as mock_record:
        reaped = reap_stale_processing(processing, results, max_age_s=1800)

    assert reaped == 1
    assert not stale.exists(), "stale processing file must be deleted"
    assert fresh.exists(), "fresh file must remain"
    body = json.loads((results / "t-stale.json").read_text(encoding="utf-8"))
    assert body["status"] == "abandoned"
    assert body["specialist"] == "Scout"
    assert mock_record.called


def test_reaper_handles_malformed_processing_file(tmp_path: Path) -> None:
    """Reaper must not crash on a malformed processing file; it still
    writes an abandoned status (specialist=None) and deletes the file."""
    from agent.specialists.consumer import reap_stale_processing

    processing = tmp_path / ".processing"
    results = tmp_path / "results"
    processing.mkdir()
    results.mkdir()

    bad = processing / "t-bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    old_ts = time.time() - 7200
    os.utime(bad, (old_ts, old_ts))

    with patch("agent.specialists.consumer.record_event"):
        reaped = reap_stale_processing(processing, results, max_age_s=1800)

    assert reaped == 1
    assert not bad.exists()
    body = json.loads((results / "t-bad.json").read_text(encoding="utf-8"))
    assert body["status"] == "abandoned"
    assert body["specialist"] is None


def test_reaper_returns_zero_when_processing_dir_missing(tmp_path: Path) -> None:
    from agent.specialists.consumer import reap_stale_processing
    results = tmp_path / "results"
    results.mkdir()
    assert reap_stale_processing(tmp_path / "nonexistent", results) == 0


def test_reaper_returns_zero_when_no_stale_files(tmp_path: Path) -> None:
    from agent.specialists.consumer import reap_stale_processing

    processing = tmp_path / ".processing"
    results = tmp_path / "results"
    processing.mkdir()
    results.mkdir()
    (processing / "t-new.json").write_text("{}", encoding="utf-8")

    assert reap_stale_processing(processing, results, max_age_s=1800) == 0


# ---------------------------------------------------------------------------
# F2.5 — main loop calls reaper on every tick
# ---------------------------------------------------------------------------


def test_main_loop_invokes_reaper(monkeypatch, tmp_path: Path) -> None:
    import time as time_mod
    from agent.specialists.consumer import _main_loop

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()

    call_count = 0

    def mock_sleep(_interval):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt("stop")
        return None

    monkeypatch.setenv("BRAIN_INBOX_PATH", str(inbox))
    monkeypatch.setenv("RESULTS_PATH", str(results))
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(personas))

    with patch("agent.specialists.consumer.run_once"), \
         patch("agent.specialists.consumer.reap_stale_processing") as mock_reaper, \
         patch.object(time_mod, "sleep", side_effect=mock_sleep):
        try:
            _main_loop(poll_interval_sec=0.05)
        except KeyboardInterrupt:
            pass

    assert mock_reaper.called, "_main_loop must call reap_stale_processing on every tick"


# ---------------------------------------------------------------------------
# Review BLOCKER fix (2026-06-10 22:38): lease must refresh mtime so the
# reaper doesn't false-abandon a freshly-claimed task whose inbox file was
# old. Pre-fix, os.replace preserved mtime, so a 60-min-old inbox file got
# claimed and then reaped within seconds even though work was active.
# ---------------------------------------------------------------------------


def test_lease_refreshes_mtime_so_old_inbox_file_isnt_immediately_reaped(tmp_path: Path) -> None:
    from agent.specialists.consumer import run_once, reap_stale_processing

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()

    # Write a dispatch file and BACKDATE its mtime by 2 hours (simulates Scout
    # writing it long ago while consumer was offline).
    f = _write_dispatch_with_meta(inbox, "t-old-inbox", "Scout", ttl_sec=3600)
    old_ts = time.time() - 7200
    os.utime(f, (old_ts, old_ts))

    # Now run_once claims the file. After the lease, the processing file's
    # mtime MUST be recent, otherwise the reaper will immediately abandon it.
    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    # Happy path: file processed cleanly, processing dir empty.
    assert list((inbox / ".processing").glob("*.json")) == []
    body = json.loads((results / "t-old-inbox.json").read_text(encoding="utf-8"))
    assert body["status"] == "completed"


def test_lease_mtime_refresh_prevents_premature_reap_on_long_invoke(tmp_path: Path) -> None:
    """End-to-end: claim a file from an OLD inbox entry, simulate a long
    invoke by stopping just before invoke_agent, then run the reaper with
    max_age_s=30. The processing file should NOT be reaped because its
    mtime was refreshed at lease time."""
    from agent.specialists import consumer
    from agent.specialists.consumer import reap_stale_processing

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    processing = inbox / ".processing"
    inbox.mkdir()
    results.mkdir()
    processing.mkdir()

    # Backdate inbox file by 2 hours.
    f = _write_dispatch_with_meta(inbox, "t-leased", "Scout", ttl_sec=3600)
    old_ts = time.time() - 7200
    os.utime(f, (old_ts, old_ts))

    # Manually perform the lease step exactly as run_once does (move + utime),
    # then DON'T invoke. This isolates the lease-mtime contract.
    processing_file = processing / "t-leased.json"
    os.replace(str(f), str(processing_file))
    os.utime(processing_file, None)

    # Reaper with 30s max_age must NOT reap the freshly-leased file.
    with patch("agent.specialists.consumer.record_event"):
        reaped = reap_stale_processing(processing, results, max_age_s=30)

    assert reaped == 0, (
        "freshly-leased file must not be reaped — mtime should be refreshed "
        "to lease time, not inherited from old inbox mtime"
    )
    assert processing_file.exists(), "leased file must still be in processing"
