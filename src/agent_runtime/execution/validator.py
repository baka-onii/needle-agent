"""Strict validation of the supported JSON Schema subset, without coercion.

Unsupported schema keywords fail at registration, rather than silently skipping
constraints on a model-generated argument. Unknown object keys are always rejected.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from agent_runtime.tools.base import ToolCall, ToolError

if TYPE_CHECKING:
    from agent_runtime.tools.registry import ToolRegistry

_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}
_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
    "title",
    "default",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
}


def check_schema(schema: dict[str, Any], location: str = "parameters") -> None:
    """Validate tool definitions once, before they are exposed to either model."""
    if not isinstance(schema, dict):
        raise ValueError(f"{location}: schema must be an object.")
    unsupported = set(schema) - _KEYWORDS
    if unsupported:
        raise ValueError(f"{location}: unsupported schema keywords: {sorted(unsupported)}")
    types = schema.get("type", "object" if location == "parameters" else None)
    types = types if isinstance(types, list) else [types]
    if not types or any(not isinstance(t, str) or t not in _TYPES for t in types):
        raise ValueError(f"{location}: an explicit supported type is required.")
    if schema.get("additionalProperties", False) is not False:
        raise ValueError(f"{location}: additionalProperties must be false in V0.")
    if "object" in types:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or not isinstance(required, list):
            raise ValueError(f"{location}: invalid properties or required list.")
        if any(not isinstance(k, str) or k not in props for k in required):
            raise ValueError(f"{location}: required names must have property schemas.")
        for key, value in props.items():
            check_schema(value, f"{location}.{key}")
    if "array" in types:
        if "items" not in schema:
            raise ValueError(f"{location}: array schemas require items.")
        check_schema(schema["items"], f"{location}[]")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError) as exc:
            raise ValueError(f"{location}: invalid pattern.") from exc
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{location}: {key} must be a nonnegative integer.")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema:
            value = schema[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{location}: {key} must be a finite number.")
            if key == "multipleOf" and value <= 0:
                raise ValueError(f"{location}: multipleOf must be positive.")


def _is_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _check(value: Any, schema: dict[str, Any], arg: str) -> None:
    expected = schema.get("type", "object")
    types = expected if isinstance(expected, list) else [expected]
    if not any(_is_type(value, t) for t in types):
        raise ToolError(f"Argument {arg!r} must be {expected}, got {type(value).__name__}.")
    if "enum" in schema and not any(
        type(value) is type(option) and value == option for option in schema["enum"]
    ):
        raise ToolError(f"Argument {arg!r} must be one of {schema['enum']}.")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise ToolError(f"Argument {arg!r} must equal {schema['const']!r}.")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ToolError(f"Argument {arg!r} must be finite.")
        for key, invalid in (
            ("minimum", lambda b: value < b),
            ("maximum", lambda b: value > b),
            ("exclusiveMinimum", lambda b: value <= b),
            ("exclusiveMaximum", lambda b: value >= b),
        ):
            if key in schema and invalid(schema[key]):
                raise ToolError(f"Argument {arg!r} violates {key}={schema[key]}.")
        if "multipleOf" in schema:
            quotient = value / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                raise ToolError(f"Argument {arg!r} must be a multiple of {schema['multipleOf']}.")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ToolError(f"Argument {arg!r} is too short.")
        if len(value) > schema.get("maxLength", math.inf):
            raise ToolError(f"Argument {arg!r} is too long.")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ToolError(f"Argument {arg!r} does not match the required pattern.")
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ToolError(f"Argument {arg!r} has an invalid number of items.")
        if schema.get("uniqueItems") and any(item in value[:i] for i, item in enumerate(value)):
            raise ToolError(f"Argument {arg!r} must have unique items.")
        for i, item in enumerate(value):
            _check(item, schema["items"], f"{arg}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ToolError(f"Tool {arg!r} is missing required argument {required!r}.")
        if (
            not schema.get("minProperties", 0)
            <= len(value)
            <= schema.get("maxProperties", math.inf)
        ):
            raise ToolError(f"Argument {arg!r} has an invalid number of properties.")
        for key, item in value.items():
            if key not in props:
                raise ToolError(f"Tool {arg!r} has no argument {key!r}.")
            _check(item, props[key], f"{arg}.{key}")


def validate(call: ToolCall, registry: ToolRegistry) -> ToolCall:
    tool = registry.get(call.name)
    _check(call.arguments, tool.parameters, call.name)
    return call
