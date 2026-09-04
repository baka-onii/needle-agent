"""Canonical tool definition (spec §5).

One ``Tool`` object is the single source of truth. The framework derives both
the reasoning-model description and the Needle JSON schema from it — never
maintain both by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ToolError(Exception):
    """Recoverable tool failure. Becomes a natural-language observation (§34)."""


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., str] | None = None

    def reasoning_description(self) -> str:
        """Human-readable blurb for the reasoning model's system prompt."""
        lines = [f"- {self.name}: {self.description}"]
        props = (self.parameters or {}).get("properties", {})
        required = set((self.parameters or {}).get("required", []))
        for arg, schema in props.items():
            desc = schema.get("description", "") if isinstance(schema, dict) else ""
            opt = "" if arg in required else " (optional)"
            lines.append(f"    - {arg}{opt}: {desc}".rstrip())
        return "\n".join(lines)

    def needle_schema(self) -> dict[str, Any]:
        """Raw JSON-schema tool definition consumed by Needle."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }
