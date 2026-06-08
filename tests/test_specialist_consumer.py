"""
ADR-025 implementation tests 3–6: brain-inbox consumer behaviours.

RED-FIRST: these fail until agent/specialists/consumer.py exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_dispatch(inbox: Path, task_id: str, specialist: str) -> Path:
    f = inbox / f"{task_id}.json"
    f.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "specialist": specialist,
                "task": {"query": "test"},
                "context": {},
                "enqueued_at": "2026-06-07T15:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return f


def _make_persona_dir(d: Path) -> Path:
    """Drop a minimal Scout.yaml so the loader has something to find."""
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


def test_consumer_drains_one_file_then_exits_with_once_flag(tmp_path: Path) -> None:
    """Test 3 — --once drains exactly one dispatch then exits."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    f = _write_dispatch(inbox, "task-1", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert not f.exists(), "dispatch file should be drained from inbox"


def test_consumer_writes_status_file_atomically(tmp_path: Path) -> None:
    """Test 4 — status file write uses os.replace (atomic rename)."""
    from agent.specialists import consumer
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch(inbox, "task-2", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"), \
         patch.object(consumer.os, "replace", wraps=consumer.os.replace) as mock_replace:
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert mock_replace.called, "consumer must use os.replace for atomic status file write"


def test_consumer_records_dispatch_completed_event(tmp_path: Path) -> None:
    """Test 5 — successful dispatch produces exactly one dispatch_completed event."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch(inbox, "task-3", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event") as mock_record:
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert mock_record.call_count == 1
    call_kwargs = mock_record.call_args.kwargs or {}
    call_args = mock_record.call_args.args
    payload = call_args[0] if call_args else call_kwargs
    serialized = json.dumps(payload, default=str)
    assert "dispatch_completed" in serialized
    assert "task-3" in serialized
    assert "Scout" in serialized


def test_consumer_handles_persona_load_failure_gracefully(tmp_path: Path) -> None:
    """Test 6 — unknown specialist writes a failed status file, no crash."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")  # only Scout exists
    inbox.mkdir()
    results.mkdir()
    _write_dispatch(inbox, "task-4", "Ghost")  # Ghost has no YAML

    with patch("agent.specialists.consumer.invoke_agent"), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    status_file = results / "task-4.json"
    assert status_file.exists(), "failed dispatch must still write a status file"
    body = json.loads(status_file.read_text(encoding="utf-8"))
    assert body["status"] == "failed"
    assert "Ghost" in body.get("error", "") or "unknown" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# New reliability tests (fixes 1-5)
# ---------------------------------------------------------------------------

def test_consumer_handles_malformed_json(tmp_path: Path) -> None:
    """Fix 1 — malformed JSON must write a failed status file and delete inbox."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()

    bad_file = inbox / "bad-task.json"
    bad_file.write_text("{not json", encoding="utf-8")

    with patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert not bad_file.exists(), "inbox file must be deleted even on malformed JSON"
    status_file = results / "bad-task.json"
    assert status_file.exists(), "failed status file must be written for malformed JSON"
    body = json.loads(status_file.read_text(encoding="utf-8"))
    assert body["status"] == "failed"
    assert "malformed" in body["error"]


def test_consumer_continues_when_invoke_agent_raises(tmp_path: Path) -> None:
    """Fix 2 — invoke_agent exception must write failed status and delete inbox."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    f = _write_dispatch(inbox, "task-inv", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", side_effect=RuntimeError("boom")), \
         patch("agent.specialists.consumer.record_event") as mock_record:
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert not f.exists(), "inbox file must be deleted when invoke_agent raises"
    status_file = results / "task-inv.json"
    assert status_file.exists()
    body = json.loads(status_file.read_text(encoding="utf-8"))
    assert body["status"] == "failed"
    assert "boom" in body["error"]
    assert mock_record.called, "record_event must still be called on invoke_agent failure"


def test_consumer_continues_when_record_event_raises(tmp_path: Path) -> None:
    """Fix 3 — record_event failure must not propagate or block inbox deletion."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    f = _write_dispatch(inbox, "task-rec", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event", side_effect=RuntimeError("db down")):
        # Must not raise
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert not f.exists(), "inbox file must be deleted even when record_event raises"
    assert (results / "task-rec.json").exists(), "status file must be written before record_event"


def test_consumer_creates_results_dir_when_missing(tmp_path: Path) -> None:
    """Fix 4 — results_dir is created automatically if it doesn't exist."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results" / "nested" / "dir"  # does not exist yet
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    _write_dispatch(inbox, "task-mkdir", "Scout")

    with patch("agent.specialists.consumer.invoke_agent", return_value={"findings": []}), \
         patch("agent.specialists.consumer.record_event"):
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)

    assert results.is_dir(), "results_dir must be created automatically"
    assert (results / "task-mkdir.json").exists()


def test_consumer_returns_quietly_when_file_vanishes_between_glob_and_read(tmp_path: Path) -> None:
    """Fix 5 — TOCTOU: file disappears between glob and read must not raise."""
    from agent.specialists.consumer import run_once

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()
    _write_dispatch(inbox, "task-toctou", "Scout")

    original_read_text = Path.read_text

    def vanish_on_inbox(self: Path, *args, **kwargs):
        if self.parent.resolve() == inbox.resolve():
            raise FileNotFoundError(f"simulated race: {self}")
        return original_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", vanish_on_inbox):
        # Must return without raising
        run_once(inbox_dir=inbox, results_dir=results, persona_dir=personas)


# ---------------------------------------------------------------------------
# New __main__ entrypoint tests (G1)
# ---------------------------------------------------------------------------


def test_main_block_refuses_to_start_when_flag_disabled(monkeypatch) -> None:
    """G1.1 — __main__ block refuses to start when ZENOPS_CONSUMER_ENABLED != 'true'."""
    from agent.specialists.consumer import ConsumerDisabledError, start

    monkeypatch.setenv("ZENOPS_CONSUMER_ENABLED", "false")
    with pytest.raises(ConsumerDisabledError):
        start()


def test_main_loop_processes_one_then_continues(monkeypatch, tmp_path: Path) -> None:
    """G1.2 — _main_loop processes one dispatch then continues polling."""
    import time
    from agent.specialists.consumer import _main_loop

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()

    # Pre-populate inbox with one dispatch
    _write_dispatch(inbox, "task-main", "Scout")

    call_count = 0

    def mock_sleep(interval):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt("stop after 2 calls")
        return None

    monkeypatch.setenv("BRAIN_INBOX_PATH", str(inbox))
    monkeypatch.setenv("RESULTS_PATH", str(results))
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(personas))

    with patch("agent.specialists.consumer.run_once") as mock_run_once, \
         patch.object(time, "sleep", side_effect=mock_sleep):
        try:
            _main_loop(poll_interval_sec=0.05)
        except KeyboardInterrupt:
            pass

    # run_once should have been called at least once
    assert mock_run_once.call_count >= 1, "_main_loop should call run_once"
    # Check that run_once was passed the correct paths
    call_args = mock_run_once.call_args
    assert call_args[0][0] == inbox or call_args[1].get("inbox_dir") == inbox


def test_main_loop_survives_run_once_exception(monkeypatch, tmp_path: Path, capsys) -> None:
    """G1.3 — _main_loop catches and logs run_once exceptions, continues looping."""
    import time
    from agent.specialists.consumer import _main_loop

    inbox = tmp_path / "brain-inbox"
    results = tmp_path / "results"
    personas = _make_persona_dir(tmp_path / "personas")
    inbox.mkdir()
    results.mkdir()

    call_count = 0

    def mock_sleep(interval):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt("stop after 2 calls")
        return None

    monkeypatch.setenv("BRAIN_INBOX_PATH", str(inbox))
    monkeypatch.setenv("RESULTS_PATH", str(results))
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(personas))

    with patch("agent.specialists.consumer.run_once", side_effect=RuntimeError("boom")), \
         patch.object(time, "sleep", side_effect=mock_sleep):
        # Must not raise; exception should be caught and logged
        try:
            _main_loop(poll_interval_sec=0.05)
        except KeyboardInterrupt:
            pass

    # Verify exception was logged to stderr
    captured = capsys.readouterr()
    assert "boom" in captured.err, "_main_loop should log run_once exceptions to stderr"

