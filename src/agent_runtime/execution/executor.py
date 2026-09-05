"""Permission checks and execution. Only expected tool errors are recoverable."""

from __future__ import annotations

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import ToolCall, ToolError, ToolResult, truncate_text
from agent_runtime.tools.filesystem import resolve_safe_path
from agent_runtime.tools.registry import ToolRegistry

FILESYSTEM_TOOLS = frozenset({"read_file", "read_directory", "search_files", "write_file"})


def check_safety(call: ToolCall, config: AgentConfig) -> None:
    """Run before execution. Handlers also recheck paths at the operation boundary."""
    if call.name == "write_file" and config.read_only:
        raise ToolError("Workspace is read-only; writing is disabled.")
    if call.name in FILESYSTEM_TOOLS:
        resolve_safe_path(call.arguments.get("path", "."), config.workspace_root)


def execute(call: ToolCall, registry: ToolRegistry, config: AgentConfig) -> ToolResult:
    tool = registry.get(call.name)
    if tool.handler is None:
        raise RuntimeError(f"Tool {call.name!r} has no handler.")
    try:
        check_safety(call, config)
        output = tool.handler(**call.arguments)
    except ToolError as exc:
        return ToolResult(
            success=False, error=truncate_text(str(exc), config.max_tool_output_chars)
        )
    if not isinstance(output, str):
        raise TypeError(f"Tool {call.name!r} returned non-text output.")
    return ToolResult(success=True, output=truncate_text(output, config.max_tool_output_chars))
