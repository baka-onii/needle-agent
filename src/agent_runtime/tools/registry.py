"""Canonical tool registry, injected into the runtime as a dependency."""

from __future__ import annotations

import re
from collections.abc import Callable

from agent_runtime.config import AgentConfig
from agent_runtime.execution.validator import check_schema
from agent_runtime.tools.base import Tool, ToolError
from agent_runtime.tools.filesystem import filesystem_tools
from agent_runtime.tools.interaction import make_ask_user_tool
from agent_runtime.tools.utility import utility_tools

FORBIDDEN_TOOLS = frozenset({"shell", "terminal", "execute_command", "run_python"})


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tool.name):
            raise ValueError("Tool names must be identifiers.")
        if tool.name in FORBIDDEN_TOOLS:
            raise ValueError(f"Tool {tool.name!r} is outside V0's safety boundary.")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if tool.parameters.get("type", "object") != "object":
            raise ValueError("Tool parameter schemas must describe objects.")
        check_schema(tool.parameters)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name!r}") from exc

    def list(self) -> list[Tool]:
        return list(self._tools.values())


def create_default_registry(
    config: AgentConfig, ask_fn: Callable[[str], str] | None = None
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [*filesystem_tools(config), *utility_tools(config), make_ask_user_tool(ask_fn)]:
        registry.register(tool)
    return registry
