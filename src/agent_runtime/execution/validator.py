"""Validation (spec §22). A high-confidence invalid call is still invalid."""

from __future__ import annotations

from typing import Any

from agent_runtime.tools.base import ToolCall, ToolError
from agent_runtime.tools.registry import ToolRegistry

_TYPE_NAMES = {
    "string": ("string",),
    "number": ("number",),
    "integer": ("integer",),
    "boolean": ("boolean",),
    "array": ("array",),
    "object": ("object",),
}


def _check_type(value: Any, expected: str, arg: str) -> None:
    if expected == "string" and isinstance(value, str):
        return
    if expected == "boolean" and isinstance(value, bool):
        return
    if expected == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return
    if expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return
    if expected == "array" and isinstance(value, list):
        return
    if expected == "object" and isinstance(value, dict):
        return
    raise ToolError(f"Argument {arg!r} must be {expected}, got {type(value).__name__}.")


def _check_constraints(value: Any, schema: dict[str, Any], arg: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"Argument {arg!r} must be one of {schema['enum']}.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, op, label in (
            ("minimum", lambda v, b: v >= b, ">="),
            ("maximum", lambda v, b: v <= b, "<="),
        ):
            if key in schema and not op(value, schema[key]):
                raise ToolError(f"Argument {arg!r} must be {label} {schema[key]}.")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolError(f"Argument {arg!r} is too short.")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolError(f"Argument {arg!r} is too long.")


def validate(call: ToolCall, registry: ToolRegistry) -> ToolCall:
    """Validate ``call`` against the registry. Returns it unchanged if valid."""
    tool = registry.get(call.name)  # raises ToolError for unknown tools
    params = tool.parameters or {}
    properties: dict[str, Any] = params.get("properties", {})
    required: list[str] = params.get("required", [])
    for arg in required:
        if arg not in call.arguments:
            raise ToolError(f"Tool {call.name!r} is missing required argument {arg!r}.")
    for arg, value in call.arguments.items():
        if arg not in properties:
            raise ToolError(f"Tool {call.name!r} has no argument {arg!r}.")
        schema = properties[arg] or {}
        expected = schema.get("type")
        if expected in _TYPE_NAMES:
            _check_type(value, expected, arg)
        _check_constraints(value, schema, arg)
    return call
