"""One canonical Tool generates both prose descriptions and Needle schemas."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolError(Exception):
    """Recoverable tool failure. Becomes a natural-language observation."""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    success: bool
    output: str = ""
    error: str | None = None


def truncate_text(text: str, limit: int) -> str:
    """The marker is included in the limit, even for very small budgets."""
    if len(text) <= limit:
        return text
    marker = "\n… [truncated]"
    if limit < len(marker):
        return "truncated"[:limit]
    return text[: limit - len(marker)] + marker


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., str] | None = None

    def reasoning_description(self) -> str:
        lines = [f"- {self.name}: {self.description}"]
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        for arg, schema in props.items():
            opt = "" if arg in required else " (optional)"
            lines.append(f"    - {arg}{opt}: {schema.get('description', '')}".rstrip())
        return "\n".join(lines)

    def needle_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": deepcopy(self.parameters or {"type": "object", "properties": {}}),
        }
