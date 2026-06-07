"""ADR-025 brain-inbox consumer — processes one dispatch file per call (run_once)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.specialists.persona import load_persona


class ConsumerDisabledError(RuntimeError):
    """Raised when ZENOPS_CONSUMER_ENABLED != 'true'."""


def invoke_agent(persona, task, context) -> dict:
    raise NotImplementedError("real Hermes integration lands in a follow-up plan")


def record_event(payload) -> None:
    pass


def start() -> None:
    if os.environ.get("ZENOPS_CONSUMER_ENABLED") != "true":
        raise ConsumerDisabledError("Set ZENOPS_CONSUMER_ENABLED=true to enable the consumer.")


def _safe_record_event(payload: dict) -> None:
    """Call record_event, swallowing any exception so it never blocks inbox deletion."""
    try:
        record_event(payload)
    except Exception as exc:  # Fix 3: recorder failure must not propagate
        print(f"[consumer] record_event failed (ignored): {exc}", file=sys.stderr)


def run_once(inbox_dir: Path, results_dir: Path, persona_dir: Path) -> None:
    """Process exactly one dispatch file from inbox_dir, then return."""
    dispatch_files = sorted(inbox_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not dispatch_files:
        return

    inbox_file = dispatch_files[0]

    # Fix 5: TOCTOU guard — another worker may delete the file between glob and read.
    try:
        raw = inbox_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[consumer] inbox file vanished before read: {inbox_file}", file=sys.stderr)
        return

    # Outer try/finally guarantees inbox deletion under every code path — including
    # unexpected exceptions that escape all inner handlers.  Broad-catch in the
    # finally is intentional: we own the queue and a crashed consumer must never
    # leave stale files that block every subsequent run.
    try:
        # Fix 1: Malformed JSON must not jam the inbox.
        try:
            dispatch = json.loads(raw)
        except json.JSONDecodeError as exc:
            task_id = inbox_file.stem
            now = datetime.now(timezone.utc).isoformat()
            status_payload = {
                "task_id": task_id,
                "specialist": None,
                "status": "failed",
                "error": f"malformed dispatch JSON: {repr(exc)}",
                "started_at": now,
                "finished_at": now,
                "latency_ms": 0,
                "model": None,
                "tool_calls": 0,
            }
            _write_status_atomic(results_dir, task_id, status_payload)
            _safe_record_event({
                "event_type": "dispatch_failed",
                "task_id": task_id,
                "specialist": None,
                "error": status_payload["error"],
                "started_at": now,
                "finished_at": now,
            })
            return

        task_id = dispatch["task_id"]
        specialist = dispatch["specialist"]
        task = dispatch["task"]
        context = dispatch["context"]
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            persona = load_persona(persona_dir / f"{specialist}.yaml")
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            status_payload = {
                "task_id": task_id,
                "specialist": specialist,
                "status": "failed",
                "error": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
                "latency_ms": 0,
                "model": None,
                "tool_calls": 0,
            }
            _write_status_atomic(results_dir, task_id, status_payload)
            _safe_record_event({
                "event_type": "dispatch_failed",
                "task_id": task_id,
                "specialist": specialist,
                "error": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
            })
            return

        # Fix 2: invoke_agent exceptions must not jam the inbox.
        try:
            output = invoke_agent(persona, task, context)
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            latency_ms = int(
                (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at))
                .total_seconds() * 1000
            )
            status_payload = {
                "task_id": task_id,
                "specialist": specialist,
                "status": "failed",
                "error": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
                "latency_ms": latency_ms,
                "model": persona.default_model,
                "tool_calls": 0,
            }
            _write_status_atomic(results_dir, task_id, status_payload)
            _safe_record_event({
                "event_type": "dispatch_failed",
                "task_id": task_id,
                "specialist": specialist,
                "error": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
            })
            return

        finished_at = datetime.now(timezone.utc).isoformat()
        latency_ms = int(
            (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at))
            .total_seconds() * 1000
        )
        status_payload = {
            "task_id": task_id,
            "specialist": specialist,
            "status": "completed",
            "output": output,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
            "model": persona.default_model,
            "tool_calls": 0,
        }
        _write_status_atomic(results_dir, task_id, status_payload)
        _safe_record_event({
            "event_type": "dispatch_completed",
            "task_id": task_id,
            "specialist": specialist,
            "model": persona.default_model,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
        })

    finally:
        if inbox_file.exists():
            inbox_file.unlink()


def _write_status_atomic(results_dir: Path, task_id: str, payload: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)  # Fix 4: auto-create if missing
    tmp_path = results_dir / f"{task_id}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, results_dir / f"{task_id}.json")
