"""Normalize only harmless whitespace. Never guess or repair a malformed call."""

from __future__ import annotations

import re
from typing import Any

from agent_runtime.models.action import NeedleResult
from agent_runtime.tools.base import ToolCall, ToolError

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def sanitize(result: NeedleResult) -> ToolCall:
    name = result.selected_tool
    if name is None or not isinstance(name, str) or not name.strip():
        raise ToolError("Action model returned no tool selection.")
    name = name.strip()
    if not _NAME.fullmatch(name):
        raise ToolError("Action model returned an invalid tool name.")
    if not isinstance(result.arguments, dict):
        raise ToolError("Action model returned non-object arguments.")
    arguments: dict[str, Any] = {}
    for key, value in result.arguments.items():
        if not isinstance(key, str) or not _NAME.fullmatch(key.strip()):
            raise ToolError("Action model returned an invalid argument name.")
        normalized = key.strip()
        if normalized in arguments:
            raise ToolError(f"Duplicate argument after normalization: {normalized!r}.")
        arguments[normalized] = value
    return ToolCall(name=name, arguments=arguments)
