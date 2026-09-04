"""LangGraph state machine (spec §29-33).

REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE → SAFETY →
EXECUTE → OBSERVE → UPDATE_CONTEXT → REASON, with FINAL/CONFIRM/MAX_STEPS exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager
from agent_runtime.execution.confidence import (
    is_confident,
    low_confidence_message,
    threshold_for,
)
from agent_runtime.execution.executor import execute
from agent_runtime.execution.sanitizer import sanitize
from agent_runtime.execution.validator import validate
from agent_runtime.models.action import ActionModel, NeedleResult, ToolRanking
from agent_runtime.models.reasoning import ReasoningModel
from agent_runtime.protocol.parser import parse_response
from agent_runtime.state import AgentState
from agent_runtime.tools.base import ToolCall, ToolError, ToolResult
from agent_runtime.tools.registry import ToolRegistry


@dataclass
class RuntimeDeps:
    reasoning: ReasoningModel
    action: ActionModel
    registry: ToolRegistry
    contexts: ContextManager
    config: AgentConfig


def _transcript(state: AgentState) -> list[dict[str, Any]]:
    return state["messages"]


def build_workflow(deps: RuntimeDeps):
    reasoning, action, registry, contexts, config = (
        deps.reasoning,
        deps.action,
        deps.registry,
        deps.contexts,
        deps.config,
    )

    def reason(state: AgentState) -> dict[str, Any]:
        try:
            raw = reasoning.generate(contexts.build(_transcript(state)))
        except Exception as exc:  # internal error → terminate (§34)
            return {"status": "ERROR", "final_answer": f"Reasoning model failed: {exc}"}
        return {"messages": [*state["messages"], {"role": "assistant", "content": raw}]}

    def parse(state: AgentState) -> dict[str, Any]:
        if state["status"] != "RUNNING" or state["final_answer"] is not None:
            return {}  # terminal state set upstream (e.g. reason ERROR) — keep it
        raw = state["messages"][-1].get("content", "")
        parsed = parse_response(raw if isinstance(raw, str) else str(raw))
        if parsed.final_answer is not None:
            return {"final_answer": parsed.final_answer, "status": "COMPLETED"}
        if parsed.actions:
            # One tool action per reasoning turn (§20, §32).
            return {"current_action": parsed.actions[0]}
        return {"final_answer": parsed.reasoning or "(no response)", "status": "COMPLETED"}

    def translate(state: AgentState) -> dict[str, Any]:
        try:
            result = action.translate(state["current_action"] or "", registry.list())
        except Exception as exc:  # internal error → terminate (§34)
            return {"status": "ERROR", "final_answer": f"Action model failed: {exc}"}
        return {"needle_result": result.model_dump()}

    def sanitize_node(state: AgentState) -> dict[str, Any]:
        try:
            call = sanitize(NeedleResult(**(state["needle_result"] or {})))
        except (ToolError, ValueError, TypeError) as exc:
            return _stall_or_retry(state, f"Tool error: {exc} Please try another action.")
        return {"tool_call": call.model_dump()}

    def validate_node(state: AgentState) -> dict[str, Any]:
        try:
            call = validate(ToolCall(**(state["tool_call"] or {})), registry)
        except (ToolError, ValueError, TypeError) as exc:
            return _stall_or_retry(state, f"Tool error: {exc} Please try another action.")
        return {"tool_call": call.model_dump()}

    def _stall_or_retry(state: AgentState, message: str) -> dict[str, Any]:
        stalls = state["stall_count"] + 1
        if stalls >= config.max_stalls:
            return {
                "stall_count": stalls,
                "tool_call": None,
                "current_action": None,
                "status": "STALLED",
                "final_answer": (
                    f"Stopped after {stalls} consecutive failed actions. Last error: {message}"
                ),
            }
        return {
            "tool_call": None,
            "current_action": None,
            "stall_count": stalls,
            "messages": [*_transcript(state), {"role": "user", "content": message}],
        }

    def confirm(state: AgentState) -> dict[str, Any]:
        stalls = state["stall_count"] + 1
        if stalls >= config.max_stalls:
            return {
                "stall_count": stalls,
                "current_action": None,
                "needle_result": None,
                "status": "STALLED",
                "final_answer": (
                    f"Stopped after {stalls} consecutive actions the translator "
                    "could not confidently map to a tool. Last action: "
                    f"{state['current_action']!r}"
                ),
            }
        needle = NeedleResult(**(state["needle_result"] or {}))
        rankings = needle.rankings or [
            ToolRanking(tool_name=needle.selected_tool or "?", confidence=needle.confidence)
        ]
        return {
            "current_action": None,
            "needle_result": None,
            "stall_count": stalls,
            "messages": [
                *_transcript(state),
                {
                    "role": "user",
                    "content": low_confidence_message(state["current_action"] or "", rankings),
                },
            ],
        }

    def safety(state: AgentState) -> dict[str, Any]:
        if state["step_count"] >= state["max_tool_steps"]:
            return {"status": "MAX_STEPS_REACHED"}
        return {}

    def execute_node(state: AgentState) -> dict[str, Any]:
        result = execute(ToolCall(**(state["tool_call"] or {})), registry, config)
        return {
            "last_tool_result": result.model_dump(),
            "step_count": state["step_count"] + 1,
            "stall_count": 0,  # progress resets the stall counter
        }

    def observe(state: AgentState) -> dict[str, Any]:
        result = ToolResult(**(state["last_tool_result"] or {}))
        call = ToolCall(**(state["tool_call"] or {}))
        if result.success:
            content = f"Observation from tool '{call.name}':\n{result.output}"
        else:
            content = (
                f"Tool error from '{call.name}': {result.error} "
                "You may want to try a different action."
            )
        return {
            "current_action": None,
            "tool_call": None,
            "needle_result": None,
            "messages": [*_transcript(state), {"role": "user", "content": content}],
        }

    def update_context(state: AgentState) -> dict[str, Any]:
        trimmed = contexts.build(_transcript(state))
        return {"messages": trimmed[1:]}  # strip the prepended system prompt

    def route_after_parse(state: AgentState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["current_action"]:
            return "translate"
        return "end"

    def route_after_sanitize(state: AgentState) -> str:
        if state["tool_call"]:
            return "validate"
        return "end" if state["status"] == "STALLED" else "reason"

    def route_after_validate(state: AgentState) -> str:
        if state["tool_call"]:
            return "confidence"
        return "end" if state["status"] == "STALLED" else "reason"

    def route_after_confidence(state: AgentState) -> str:
        needle = NeedleResult(**(state["needle_result"] or {}))
        gate = threshold_for(needle.selected_tool or "", config)
        return "safety" if is_confident(needle.confidence, gate) else "confirm"

    def route_after_safety(state: AgentState) -> str:
        return "end" if state["status"] == "MAX_STEPS_REACHED" else "execute"

    def route_after_translate(state: AgentState) -> str:
        return "end" if state["status"] == "ERROR" else "sanitize"

    def route_after_confirm(state: AgentState) -> str:
        return "end" if state["status"] == "STALLED" else "reason"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("parse", parse)
    graph.add_node("translate", translate)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("validate", validate_node)
    graph.add_node("confirm", confirm)
    graph.add_node("safety", safety)
    graph.add_node("execute", execute_node)
    graph.add_node("observe", observe)
    graph.add_node("update_context", update_context)

    graph.add_edge(START, "reason")
    graph.add_edge("reason", "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"translate": "translate", "end": END})
    graph.add_conditional_edges(
        "translate", route_after_translate, {"sanitize": "sanitize", "end": END}
    )
    graph.add_conditional_edges(
        "sanitize", route_after_sanitize, {"validate": "validate", "reason": "reason", "end": END}
    )
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"confidence": "confidence", "reason": "reason", "end": END},
    )
    # Confidence is a pure branch point: passthrough node + conditional edge.
    graph.add_node("confidence", lambda state: {})
    graph.add_conditional_edges(
        "confidence", route_after_confidence, {"safety": "safety", "confirm": "confirm"}
    )
    graph.add_conditional_edges(
        "safety", route_after_safety, {"execute": "execute", "end": END}
    )
    graph.add_conditional_edges(
        "confirm", route_after_confirm, {"reason": "reason", "end": END}
    )
    graph.add_edge("execute", "observe")
    graph.add_edge("observe", "update_context")
    graph.add_edge("update_context", "reason")
    return graph.compile()
