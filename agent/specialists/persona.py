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
        ValueError: If name is not in the canonical specialists set.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    name = data["name"]
    if name not in VALID_SPECIALISTS:
        raise ValueError(f"Unknown specialist: {name}")
    
    return Persona(
        name=name,
        system_prompt=data["system_prompt"],
        allowed_tools=data["allowed_tools"],
        default_model=data["default_model"],
        output_schema=data["output_schema"],
    )
