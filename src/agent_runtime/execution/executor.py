"""Tool execution (spec §30 ``execute``). Runs validated calls; tool failures
become ``ToolResult`` failures (observations), never exceptions."""

from __future__ import annotations

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import ToolCall, ToolError, ToolResult
from agent_runtime.tools.registry import ToolRegistry


def execute(call: ToolCall, registry: ToolRegistry, config: AgentConfig) -> ToolResult:
    tool = registry.get(call.name)
    if tool.handler is None:
        return ToolResult(success=False, error=f"Tool {call.name!r} has no handler.")
    try:
        output = tool.handler(**call.arguments)
    except ToolError as exc:
        return ToolResult(success=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 — tool crashes become observations
        return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
    if len(output) > config.max_tool_output_chars:
        output = output[: config.max_tool_output_chars] + (
            f"\n... [truncated to {config.max_tool_output_chars} chars]"
        )
    return ToolResult(success=True, output=output)
