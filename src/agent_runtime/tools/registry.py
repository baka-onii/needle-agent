"""Tool registry (spec §6). Passed to the graph as a dependency."""

from __future__ import annotations

from collections.abc import Callable

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import Tool, ToolError
from agent_runtime.tools.filesystem import filesystem_tools
from agent_runtime.tools.interaction import make_ask_user_tool
from agent_runtime.tools.utility import utility_tools


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
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
    for tool in [
        *filesystem_tools(config),
        *utility_tools(),
        make_ask_user_tool(ask_fn),
    ]:
        registry.register(tool)
    return registry
