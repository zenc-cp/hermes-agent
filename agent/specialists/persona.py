"""ADR-025 specialist persona YAML loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


VALID_SPECIALISTS = {"Scout", "Hunter", "Sentinel", "Trader", "Scribe", "Ops"}


@dataclass
class Persona:
    """Specialist persona configuration."""
    name: str
    system_prompt: str
    allowed_tools: list[str]
    default_model: str
    output_schema: dict


def load_persona(path: Path) -> Persona:
    """Load and validate a Persona from a YAML file.
    
    Args:
        path: Path to the YAML file.
        
    Returns:
        Persona object with validated name.
        
    Raises:
        ValueError: If YAML is malformed, missing required fields, has wrong types,
                    or name is not in the canonical specialists set.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Issue 1: Non-dict YAML crashes with TypeError
    if not isinstance(data, dict):
        raise ValueError("Persona YAML must be a mapping at top level")
    
    # Issue 2: Missing required fields raise bare KeyError
    required_fields = ["name", "system_prompt", "allowed_tools", "default_model", "output_schema"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Persona YAML missing required field: {field}")
    
    # Issue 3: No type validation on fields
    name = data["name"]
    if not isinstance(name, str):
        raise ValueError(f"Persona YAML field name: expected str, got {type(name).__name__}")
    
    system_prompt = data["system_prompt"]
    if not isinstance(system_prompt, str):
        raise ValueError(f"Persona YAML field system_prompt: expected str, got {type(system_prompt).__name__}")
    
    allowed_tools = data["allowed_tools"]
    if not isinstance(allowed_tools, list):
        raise ValueError(f"Persona YAML field allowed_tools: expected list, got {type(allowed_tools).__name__}")
    for idx, tool in enumerate(allowed_tools):
        if not isinstance(tool, str):
            raise ValueError(f"Persona YAML field allowed_tools: expected list of str, got {type(tool).__name__} at index {idx}")
    
    default_model = data["default_model"]
    if not isinstance(default_model, str):
        raise ValueError(f"Persona YAML field default_model: expected str, got {type(default_model).__name__}")
    
    output_schema = data["output_schema"]
    if not isinstance(output_schema, dict):
        raise ValueError(f"Persona YAML field output_schema: expected dict, got {type(output_schema).__name__}")
    
    if name not in VALID_SPECIALISTS:
        raise ValueError(f"Unknown specialist: {name}")
    
    return Persona(
        name=name,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        default_model=default_model,
        output_schema=output_schema,
    )
