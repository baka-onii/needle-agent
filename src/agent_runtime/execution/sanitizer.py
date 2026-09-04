"""Sanitization (spec §21). Needle output is untrusted.

Normalize name/args, reject malformed structures. Never "repair" model output
into a potentially dangerous command: valid → continue, invalid → ToolError so
the reasoning model can retry.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.models.action import NeedleResult
from agent_runtime.tools.base import ToolCall, ToolError


def sanitize(result: NeedleResult) -> ToolCall:
    """Convert a ``NeedleResult`` into an executable-intent ``ToolCall``."""
    name = (result.selected_tool or "").strip()
    if not name:
        raise ToolError("Action model returned no tool selection.")
    if not isinstance(result.arguments, dict):
        raise ToolError("Action model returned non-object arguments.")
    arguments: dict[str, Any] = {}
    for key, value in result.arguments.items():
        if not isinstance(key, str) or not key.strip():
            raise ToolError("Action model returned an invalid argument name.")
        arguments[key.strip()] = value
    return ToolCall(name=name, arguments=arguments)
