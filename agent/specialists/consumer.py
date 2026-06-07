"""ADR-025 brain-inbox consumer — processes one dispatch file per call (run_once)."""
from __future__ import annotations

import json
import os
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


def run_once(inbox_dir: Path, results_dir: Path, persona_dir: Path) -> None:
    """Process exactly one dispatch file from inbox_dir, then return."""
    dispatch_files = sorted(inbox_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not dispatch_files:
        return

    dispatch_file = dispatch_files[0]
    dispatch = json.loads(dispatch_file.read_text(encoding="utf-8"))
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
        record_event({
            "event_type": "dispatch_failed",
            "task_id": task_id,
            "specialist": specialist,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": finished_at,
        })
        dispatch_file.unlink()
        return

    output = invoke_agent(persona, task, context)
    finished_at = datetime.now(timezone.utc).isoformat()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    latency_ms = int((finished_dt - started_dt).total_seconds() * 1000)

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
    record_event({
        "event_type": "dispatch_completed",
        "task_id": task_id,
        "specialist": specialist,
        "model": persona.default_model,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_ms": latency_ms,
    })
    dispatch_file.unlink()


def _write_status_atomic(results_dir: Path, task_id: str, payload: dict) -> None:
    tmp_path = results_dir / f"{task_id}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, results_dir / f"{task_id}.json")
