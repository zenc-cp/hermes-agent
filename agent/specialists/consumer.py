"""ADR-025 brain-inbox consumer â€” processes one dispatch file per call (run_once).

Audit F2 fix (2026-06-10): added lease-via-processing-dir and TTL expiry +
stale-file reaper. Pre-fix the consumer deleted the inbox file in a `finally`
regardless of outcome, so a mid-execution crash silently lost the task. Now:

  1. Read inbox file (handles TOCTOU)
  2. Atomic move inbox -> `.processing/{task_id}.json` (lease claim)
  3. TTL check: if now > created_at + ttl_sec, write status=expired
  4. Invoke agent, write result, delete from `.processing/` on terminal outcome
  5. Reaper (`reap_stale_processing`) sweeps `.processing/` for files older
     than `max_age_s`, writing status=abandoned and deleting. Called from the
     main loop on every tick so a crashed-then-restarted consumer cleans up.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.specialists.persona import load_persona
from agent.specialists import observability as _obs


def _safe_emit(task_id: str, event: str, payload: dict | None = None) -> None:
    """Defense-in-depth: observability MUST NOT break dispatch processing."""
    try:
        _obs.emit(task_id, event, payload)
    except Exception:
        pass


class ConsumerDisabledError(RuntimeError):
    """Raised when ZENOPS_CONSUMER_ENABLED != 'true'."""


# Map persona tool names â†’ toolset names that AIAgent.enabled_toolsets accepts.
# Verified against toolsets.py:TOOLSETS (2026-06-08).
_TOOL_TO_TOOLSET = {
    "web_search":   "web",
    "web_extract":  "web",
    "terminal":     "terminal",
    "read_file":    "file",
    "write_file":   "file",
}

# VM loopback inference shim â€” no auth, OpenAI-compatible.
_HERMES_BASE_URL = "http://127.0.0.1:8403/v1"

# Audit F2 (2026-06-10): reaper defaults.
# Tasks in `.processing/` older than this are considered abandoned (the worker
# that leased them crashed, was killed, or stalled past any reasonable bound).
# Default 30 min is well past p99 invoke_agent latency.
_DEFAULT_STALE_PROCESSING_MAX_AGE_S = 1800
_PROCESSING_SUBDIR = ".processing"


def _toolsets_for(persona) -> list[str]:
    out = set()
    for tool in persona.allowed_tools:
        if tool not in _TOOL_TO_TOOLSET:
            raise ValueError(
                f"persona {persona.name!r} requested tool {tool!r} not in "
                f"_TOOL_TO_TOOLSET; update the table"
            )
        out.add(_TOOL_TO_TOOLSET[tool])
    return sorted(out)


def _render_prompt(task: str, context: dict, schema: dict) -> str:
    import json
    return (
        f"TASK:\n{task}\n\n"
        f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        f"OUTPUT FORMAT â€” return ONLY a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"No prose. No markdown fences. Just the JSON object."
    )


def _extract_json(raw: str) -> dict:
    """Find the LAST balanced {...} block â€” models often write prose then JSON."""
    import json
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if s.startswith("json"):
                s = s[4:]
    last_close = s.rfind("}")
    if last_close == -1:
        raise ValueError(f"no JSON object in LLM output: {raw[:200]!r}")
    depth = 0
    for i in range(last_close, -1, -1):
        if s[i] == "}":
            depth += 1
        elif s[i] == "{":
            depth -= 1
            if depth == 0:
                return json.loads(s[i:last_close + 1])
    raise ValueError(f"unbalanced braces: {raw[:200]!r}")


def _validate(payload: dict, schema: dict) -> None:
    from jsonschema import Draft202012Validator, ValidationError
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as e:
        raise ValueError(
            f"output_schema violation: {e.message} at {list(e.absolute_path)}"
        )


def invoke_agent(persona, task, context) -> dict:
    # Lazy import keeps `__main__` startup light (AC5).
    from run_agent import AIAgent
    agent = AIAgent(
        model=persona.default_model,
        provider="openai",
        api_key="not-needed",          # loopback shim doesn't auth
        base_url=_HERMES_BASE_URL,
        api_mode="chat_completions",
        enabled_toolsets=_toolsets_for(persona),
        ephemeral_system_prompt=persona.system_prompt,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )
    raw = agent.chat(_render_prompt(task, context, persona.output_schema))
    payload = _extract_json(raw)
    _validate(payload, persona.output_schema)
    return payload


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


def _parse_created_at(value: str | None) -> datetime | None:
    """Parse ISO timestamp, accepting trailing 'Z'. Returns None on any failure."""
    if not value:
        return None
    try:
        # Normalize trailing Z to +00:00 for fromisoformat (py<3.11 quirk).
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _ttl_expired(envelope: dict) -> bool:
    """True if envelope.created_at + task.ttl_sec is in the past."""
    created = _parse_created_at(envelope.get("created_at") or envelope.get("enqueued_at"))
    if created is None:
        return False  # No created_at â†’ cannot judge, allow run
    task = envelope.get("task") or {}
    ttl_sec = task.get("ttl_sec")
    if not isinstance(ttl_sec, (int, float)) or ttl_sec <= 0:
        return False
    deadline = created.timestamp() + float(ttl_sec)
    return datetime.now(timezone.utc).timestamp() > deadline


def _safe_unlink(path: Path) -> None:
    """Delete file; swallow FileNotFoundError (idempotent cleanup)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_once(
    inbox_dir: Path,
    results_dir: Path,
    persona_dir: Path,
    processing_dir: Path | None = None,
) -> None:
    """Process exactly one dispatch file from inbox_dir, then return.

    Lease semantics (audit F2):
    - Atomically move inbox_file -> processing_dir before any work, so a
      concurrent consumer cannot pick up the same task.
    - On terminal outcome (success, handled failure, TTL expired): delete
      from processing_dir.
    - On uncaught exception escape: leave the file in processing_dir for the
      reaper (`reap_stale_processing`) to handle.
    """
    if processing_dir is None:
        processing_dir = inbox_dir / _PROCESSING_SUBDIR
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Skip the processing subdir when globbing inbox.
    dispatch_files = sorted(
        (p for p in inbox_dir.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not dispatch_files:
        return

    inbox_file = dispatch_files[0]

    # Fix 5 (preserved): TOCTOU guard â€” another worker may delete the file
    # between glob and read. Read FIRST so existing TOCTOU semantics hold
    # (mocks patching Path.read_text on inbox files still fire here).
    try:
        raw = inbox_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[consumer] inbox file vanished before read: {inbox_file}", file=sys.stderr)
        return

    # Audit F2: atomic lease claim. If another worker won the race, return.
    task_id = inbox_file.stem
    processing_file = processing_dir / f"{task_id}.json"
    try:
        os.replace(str(inbox_file), str(processing_file))
    except FileNotFoundError:
        print(f"[consumer] inbox file vanished before lease: {inbox_file}", file=sys.stderr)
        return

    # Review BLOCKER fix (2026-06-10 22:38): refresh mtime AFTER lease so the
    # reaper's "stale processing" check measures lease-age, not original
    # inbox-age. os.replace preserves mtime.
    # Review HIGH hardening (2026-06-10 22:50): if utime fails OR fails
    # silently (some container mounts return success without refreshing the
    # mtime), the reaper would false-abandon the freshly-claimed task â€” the
    # exact bug the F2 mtime fix was meant to prevent. Verify the refresh
    # took effect; on failure, roll back the lease so another worker can
    # retry rather than continuing with corrupt lease semantics.
    import time as _time
    try:
        os.utime(processing_file, None)
        refreshed_age_s = _time.time() - processing_file.stat().st_mtime
        if refreshed_age_s > 60:
            raise OSError(
                f"os.utime returned success but mtime still {refreshed_age_s:.0f}s old"
            )
    except OSError as exc:
        print(
            f"[consumer] lease mtime refresh failed for {task_id}: {exc}; rolling back lease",
            file=sys.stderr,
            flush=True,
        )
        try:
            os.replace(str(processing_file), str(inbox_file))
        except OSError as rollback_exc:
            print(
                f"[consumer] lease rollback also failed for {task_id}: {rollback_exc}",
                file=sys.stderr,
                flush=True,
            )
        return

    # From here on, we own processing_file. Delete it on every terminal branch.
    # If an unexpected exception escapes, the file stays for the reaper.

    # ADR-032: emit leased event (best-effort)
    _safe_emit(task_id, "leased")

    # Fix 1 (preserved): Malformed JSON must not jam the inbox.
    try:
        dispatch = json.loads(raw)
    except json.JSONDecodeError as exc:
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
        _safe_emit(task_id, "failed", {"reason": "malformed_json"})
        _safe_record_event({
            "event_type": "dispatch_failed",
            "task_id": task_id,
            "specialist": None,
            "error": status_payload["error"],
            "started_at": now,
            "finished_at": now,
        })
        _safe_unlink(processing_file)
        return

    # Audit F2: TTL check before invoking the agent.
    if _ttl_expired(dispatch):
        now = datetime.now(timezone.utc).isoformat()
        status_payload = {
            "task_id": dispatch.get("task_id", task_id),
            "specialist": dispatch.get("specialist"),
            "status": "expired",
            "error": f"task ttl_sec exceeded before invoke",
            "started_at": now,
            "finished_at": now,
            "latency_ms": 0,
            "model": None,
            "tool_calls": 0,
        }
        _write_status_atomic(results_dir, status_payload["task_id"], status_payload)
        _safe_emit(status_payload["task_id"], "expired")
        _safe_record_event({
            "event_type": "dispatch_failed",
            "task_id": status_payload["task_id"],
            "specialist": status_payload["specialist"],
            "error": status_payload["error"],
            "started_at": now,
            "finished_at": now,
        })
        _safe_unlink(processing_file)
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
        _safe_emit(task_id, "failed", {"reason": "persona_load", "specialist": specialist})
        _safe_record_event({
            "event_type": "dispatch_failed",
            "task_id": task_id,
            "specialist": specialist,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": finished_at,
        })
        _safe_unlink(processing_file)
        return

    # Fix 2 (preserved): invoke_agent exceptions must not jam the inbox.
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
        _safe_emit(task_id, "failed", {"reason": "invoke_error", "specialist": specialist, "latency_ms": latency_ms})
        _safe_record_event({
            "event_type": "dispatch_failed",
            "task_id": task_id,
            "specialist": specialist,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": finished_at,
        })
        _safe_unlink(processing_file)
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
    _safe_emit(task_id, "completed", {"specialist": specialist, "latency_ms": latency_ms, "model": persona.default_model})
    _safe_record_event({
        "event_type": "dispatch_completed",
        "task_id": task_id,
        "specialist": specialist,
        "model": persona.default_model,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_ms": latency_ms,
    })
    _safe_unlink(processing_file)


def _write_status_atomic(results_dir: Path, task_id: str, payload: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)  # Fix 4: auto-create if missing
    tmp_path = results_dir / f"{task_id}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, results_dir / f"{task_id}.json")


def reap_stale_processing(
    processing_dir: Path,
    results_dir: Path,
    max_age_s: int = _DEFAULT_STALE_PROCESSING_MAX_AGE_S,
) -> int:
    """Audit F2: sweep `.processing/` for files older than max_age_s.

    For each stale file, write a status=abandoned result and delete the
    processing file. A consumer that crashed mid-execution leaves a file
    here; the reaper guarantees Scout eventually sees a terminal status
    instead of polling 404 forever.

    Returns the count of files reaped (for tests and metrics).
    """
    if not processing_dir.exists():
        return 0
    now = datetime.now(timezone.utc).timestamp()
    reaped = 0
    for f in processing_dir.glob("*.json"):
        if not f.is_file():
            continue
        try:
            age_s = now - f.stat().st_mtime
        except FileNotFoundError:
            continue
        if age_s < max_age_s:
            continue
        task_id = f.stem
        # Try to recover specialist from the dispatch envelope; tolerate
        # malformed/truncated files.
        specialist: str | None = None
        try:
            envelope = json.loads(f.read_text(encoding="utf-8"))
            specialist = envelope.get("specialist")
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            pass
        now_iso = datetime.now(timezone.utc).isoformat()
        status_payload = {
            "task_id": task_id,
            "specialist": specialist,
            "status": "abandoned",
            "error": f"processing file older than {max_age_s}s without terminal result; consumer presumed crashed",
            "started_at": now_iso,
            "finished_at": now_iso,
            "latency_ms": 0,
            "model": None,
            "tool_calls": 0,
        }
        try:
            _write_status_atomic(results_dir, task_id, status_payload)
        except OSError as exc:
            print(f"[reaper] failed to write abandoned status for {task_id}: {exc}", file=sys.stderr)
            continue
        _safe_emit(task_id, "reaped", {"specialist": specialist, "age_s": int(age_s)})
        _safe_record_event({
            "event_type": "dispatch_failed",
            "task_id": task_id,
            "specialist": specialist,
            "error": status_payload["error"],
            "started_at": now_iso,
            "finished_at": now_iso,
        })
        _safe_unlink(f)
        reaped += 1
        print(f"[reaper] abandoned task={task_id} age={int(age_s)}s", file=sys.stderr, flush=True)
    return reaped


def _main_loop(poll_interval_sec: float = 2.0) -> None:
    """Production main loop. Polls brain-inbox forever, drains one file per tick.

    Reads paths from env: BRAIN_INBOX_PATH, RESULTS_PATH, HERMES_PERSONA_DIR.
    Defaults are the deployed VM paths.
    """
    import time
    inbox = Path(os.getenv("BRAIN_INBOX_PATH", "/var/lib/design-e/brain-inbox"))
    results = Path(os.getenv("RESULTS_PATH", "/var/lib/design-e/results"))
    personas = Path(
        os.getenv("HERMES_PERSONA_DIR", str(Path.home() / ".hermes" / "specialists"))
    )
    processing = inbox / _PROCESSING_SUBDIR
    reap_max_age = int(os.getenv("CONSUMER_REAP_MAX_AGE_S", str(_DEFAULT_STALE_PROCESSING_MAX_AGE_S)))
    print(
        f"zenops-consumer starting: inbox={inbox} results={results} "
        f"personas={personas} processing={processing} reap_max_age={reap_max_age}s",
        file=sys.stderr,
        flush=True,
    )
    while True:
        try:
            run_once(inbox, results, personas, processing)
        except Exception as exc:  # noqa: BLE001  defensive: loop must never die
            print(f"zenops-consumer run_once unexpected error: {exc!r}", file=sys.stderr, flush=True)
        try:
            reap_stale_processing(processing, results, max_age_s=reap_max_age)
        except Exception as exc:  # noqa: BLE001
            print(f"zenops-consumer reaper unexpected error: {exc!r}", file=sys.stderr, flush=True)
        time.sleep(poll_interval_sec)


if __name__ == "__main__":
    start()  # raises ConsumerDisabledError if ZENOPS_CONSUMER_ENABLED != "true"
    _main_loop()
