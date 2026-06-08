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


# Map persona tool names → toolset names that AIAgent.enabled_toolsets accepts.
# Verified against toolsets.py:TOOLSETS (2026-06-08).
_TOOL_TO_TOOLSET = {
    "web_search":   "web",
    "web_extract":  "web",
    "terminal":     "terminal",
    "read_file":    "file",
    "write_file":   "file",
}

# VM loopback inference shim — no auth, OpenAI-compatible.
_HERMES_BASE_URL = "http://127.0.0.1:8403/v1"


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
        f"OUTPUT FORMAT — return ONLY a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"No prose. No markdown fences. Just the JSON object."
    )


def _extract_json(raw: str) -> dict:
    """Find the LAST balanced {...} block — models often write prose then JSON."""
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
    print(
        f"zenops-consumer starting: inbox={inbox} results={results} personas={personas}",
        file=sys.stderr,
        flush=True,
    )
    while True:
        try:
            run_once(inbox, results, personas)
        except Exception as exc:  # noqa: BLE001  defensive: loop must never die
            print(f"zenops-consumer run_once unexpected error: {exc!r}", file=sys.stderr, flush=True)
        time.sleep(poll_interval_sec)


if __name__ == "__main__":
    start()  # raises ConsumerDisabledError if ZENOPS_CONSUMER_ENABLED != "true"
    _main_loop()
