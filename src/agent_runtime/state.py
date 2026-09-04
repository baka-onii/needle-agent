"""LangGraph agent state. Plain data only — never model instances, registries,
executors, or config (spec §4). Those travel as graph dependencies."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    current_action: str | None
    needle_result: dict[str, Any] | None
    tool_call: dict[str, Any] | None
    last_tool_result: dict[str, Any] | None
    step_count: int
    max_tool_steps: int
    stall_count: int
    final_answer: str | None
    status: str


def create_initial_state(user_request: str, max_tool_steps: int) -> AgentState:
    return {
        "messages": [{"role": "user", "content": user_request}],
        "current_action": None,
        "needle_result": None,
        "tool_call": None,
        "last_tool_result": None,
        "step_count": 0,
        "max_tool_steps": max_tool_steps,
        "stall_count": 0,
        "final_answer": None,
        "status": "RUNNING",
    }
