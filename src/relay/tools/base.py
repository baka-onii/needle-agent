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


def approval_summary(call: ToolCall) -> str:
    """One human-readable line per call for approval prompts (CLI and web)."""
    arguments = call.arguments
    target = (
        arguments.get("path")
        or arguments.get("source")
        or arguments.get("destination")
        or arguments.get("branch")
        or arguments.get("message")
        or arguments.get("question")
        or arguments.get("command")
        or arguments.get("code")
        or arguments.get("query")
        or ""
    )
    target = truncate_text(str(target), 120).replace("\n", " ")
    return f"{call.name} {target}".rstrip()


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., str] | None = None
    # Payload arguments filled from <content> / <text-N> blocks, in order, by
    # the runtime after translation. The translator never invents them.
    payload_args: tuple[str, ...] = ()

    def reasoning_description(self) -> str:
        """One compact line per tool: names and types stay exact, prose goes.

        The system prompt re-sends this block every turn, so verbosity here
        is a per-turn token tax. Argument descriptions live in the canonical
        schema (translators, validators, UI) rather than in this line.
        """
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))

        def arg_type(schema: dict[str, Any]) -> str:
            declared = schema.get("type", "string")
            types = declared if isinstance(declared, list) else [declared]
            return "/".join(str(t) for t in types)

        args = ", ".join(
            f"{name}{'' if name in required else '?'}: {arg_type(schema)}"
            for name, schema in props.items()
        )
        line = f"- {self.name}({args}): {self.description}"
        if len(self.payload_args) == 1:
            line += f" Payload: <content> fills {self.payload_args[0]!r}."
        elif self.payload_args:
            filled = ", ".join(
                f"<text-{i}> fills {arg!r}" for i, arg in enumerate(self.payload_args, 1)
            )
            line += f" Payloads: {filled}."
        return line

    def needle_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": deepcopy(self.parameters or {"type": "object", "properties": {}}),
        }
