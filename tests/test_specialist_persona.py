"""
ADR-025 implementation tests 1–2: specialist persona YAML loader.

RED-FIRST: these fail until agent/specialists/persona.py exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_persona_yaml_load_minimal(tmp_path: Path) -> None:
    """Test 1 — a well-formed Scout YAML loads with all required fields."""
    from agent.specialists.persona import load_persona

    yaml_text = (
        "name: Scout\n"
        "system_prompt: |\n"
        "  You are Scout, the ZenOps research persona.\n"
        "  Investigate the target and return structured findings.\n"
        "allowed_tools: [web_search, web_fetch]\n"
        "default_model: gpt-5-chat\n"
        "output_schema:\n"
        "  type: object\n"
        "  required: [findings]\n"
        "  properties:\n"
        "    findings: {type: array, items: {type: string}}\n"
    )
    f = tmp_path / "Scout.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    persona = load_persona(f)

    assert persona.name == "Scout"
    assert "Scout" in persona.system_prompt
    assert persona.allowed_tools == ["web_search", "web_fetch"]
    assert persona.default_model == "gpt-5-chat"
    assert persona.output_schema["type"] == "object"


def test_persona_yaml_rejects_unknown_specialist(tmp_path: Path) -> None:
    """Test 2 — a YAML with a name outside VALID_SPECIALISTS raises ValueError."""
    from agent.specialists.persona import load_persona

    yaml_text = (
        "name: Mallory\n"
        "system_prompt: I am not on the canonical list\n"
        "allowed_tools: []\n"
        "default_model: gpt-5-chat\n"
        "output_schema: {type: object}\n"
    )
    f = tmp_path / "Mallory.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Mallory"):
        load_persona(f)


def test_persona_yaml_rejects_non_dict_yaml(tmp_path: Path) -> None:
    """Test 3 — non-dict YAML (e.g., list) raises ValueError matching 'mapping'."""
    from agent.specialists.persona import load_persona

    yaml_text = "- a\n- b\n"
    f = tmp_path / "list.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_persona(f)


def test_persona_yaml_rejects_missing_required_field(tmp_path: Path) -> None:
    """Test 4 — missing default_model raises ValueError matching 'default_model'."""
    from agent.specialists.persona import load_persona

    yaml_text = (
        "name: Scout\n"
        "system_prompt: Test prompt\n"
        "allowed_tools: [web_search]\n"
        "output_schema: {type: object}\n"
    )
    f = tmp_path / "missing_field.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="default_model"):
        load_persona(f)


def test_persona_yaml_rejects_wrong_field_type(tmp_path: Path) -> None:
    """Test 5 — wrong field type (allowed_tools not list) raises ValueError."""
    from agent.specialists.persona import load_persona

    yaml_text = (
        "name: Scout\n"
        "system_prompt: Test prompt\n"
        "allowed_tools: not-a-list\n"
        "default_model: gpt-5-chat\n"
        "output_schema: {type: object}\n"
    )
    f = tmp_path / "wrong_type.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_tools"):
        load_persona(f)


def test_persona_yaml_rejects_empty_file(tmp_path: Path) -> None:
    """Test 6 — empty file (YAML parses to None) raises ValueError matching 'mapping'."""
    from agent.specialists.persona import load_persona

    yaml_text = ""
    f = tmp_path / "empty.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_persona(f)

