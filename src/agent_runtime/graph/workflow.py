"""The explicit LangGraph V0 loop. Runtime dependencies never enter state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager
from agent_runtime.execution.confidence import is_confident, low_confidence_message, threshold_for
from agent_runtime.execution.executor import check_safety, execute
from agent_runtime.execution.sanitizer import sanitize
from agent_runtime.execution.validator import validate
from agent_runtime.models.action import ActionModel, ActionOutputError, NeedleResult, ToolRanking
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
    cancelled: Callable[[], bool] | None = None
    approve: Callable[[ToolCall], bool] | None = None


def build_workflow(deps: RuntimeDeps):
    reasoning, action, registry, contexts, config = (
        deps.reasoning,
        deps.action,
        deps.registry,
        deps.contexts,
        deps.config,
    )

    def emit(event_type: str, **data: Any) -> None:
        get_stream_writer()({"type": event_type, **data})

    def append(state: AgentState, content: str, kind: str) -> list[dict]:
        return contexts.build(
            [
                *state["messages"],
                {"role": "user", "content": content, "kind": kind},
            ]
        )[1:]

    def retry(state: AgentState, message: str, stage: str) -> dict:
        stalls = state["stall_count"] + 1
        emit("rejected", stage=stage, message=message, stalls=stalls)
        update = {
            "stall_count": stalls,
            "tool_call": None,
            "needle_result": None,
            "current_action": None,
            "messages": append(state, message, "confirmation"),
        }
        if stalls >= config.max_stalls:
            update.update(
                status="STALLED",
                final_answer=(
                    f"Stopped after {stalls} consecutive non-executing actions. {message}"
                ),
            )
        return update

    def reason(state: AgentState) -> dict:
        prompt = contexts.build(state["messages"])
        raw = reasoning.generate(prompt)
        if not isinstance(raw, str):
            raise TypeError("Reasoning model must return text.")
        return {"messages": [*prompt[1:], {"role": "assistant", "content": raw}]}

    def parse(state: AgentState) -> dict:
        parsed = parse_response(state["messages"][-1]["content"])
        if parsed.final_answer is not None:
            return {
                "final_answer": parsed.final_answer,
                "status": "COMPLETED",
                "current_action": None,
            }
        if parsed.actions:
            emit("action", action=parsed.actions[0], ignored_actions=len(parsed.actions) - 1)
            return {"current_action": parsed.actions[0], "tool_call": None, "needle_result": None}
        return {"final_answer": parsed.reasoning or "(no response)", "status": "COMPLETED"}

    def translate(state: AgentState) -> dict:
        try:
            result = action.translate(state["current_action"] or "", registry.list())
            if not isinstance(result, NeedleResult):
                raise ActionOutputError("Action model must return a NeedleResult.")
            # Also validate model_construct() output from custom adapters.
            result = NeedleResult.model_validate(result.model_dump())
        except (ActionOutputError, ValidationError) as exc:
            return retry(
                state, f"Invalid translator output: {exc}. Please rephrase the action.", "translate"
            )
        emit("translation", **result.model_dump())
        return {"needle_result": result.model_dump()}

    def sanitize_node(state: AgentState) -> dict:
        try:
            call = sanitize(NeedleResult.model_validate(state["needle_result"]))
        except (ToolError, ValueError, TypeError) as exc:
            return retry(state, f"Tool error: {exc} Please try another action.", "sanitize")
        return {"tool_call": call.model_dump()}

    def validate_node(state: AgentState) -> dict:
        try:
            call = validate(ToolCall.model_validate(state["tool_call"]), registry)
        except (ToolError, ValueError, TypeError) as exc:
            return retry(state, f"Tool error: {exc} Please try another action.", "validate")
        emit("validated", tool=call.name)
        return {"tool_call": call.model_dump()}

    def confidence(state: AgentState) -> dict:
        call = ToolCall.model_validate(state["tool_call"])
        needle = NeedleResult.model_validate(state["needle_result"])
        gate = threshold_for(call.name, config)
        emit(
            "confidence",
            tool=call.name,
            score=needle.confidence,
            threshold=gate,
            accepted=is_confident(needle.confidence, gate),
        )
        return {}

    def confirm(state: AgentState) -> dict:
        needle = NeedleResult.model_validate(state["needle_result"])
        rankings = needle.rankings or [
            ToolRanking(tool_name=needle.selected_tool or "?", confidence=needle.confidence)
        ]
        return retry(
            state, low_confidence_message(state["current_action"] or "", rankings), "confidence"
        )

    def safety(state: AgentState) -> dict:
        if state["step_count"] >= state["max_tool_steps"]:
            return {
                "status": "MAX_STEPS_REACHED",
                "final_answer": (
                    f"Stopped at the limit of {state['max_tool_steps']} tool steps. "
                    "No further actions were executed. You can continue with a new message."
                ),
            }
        call = ToolCall.model_validate(state["tool_call"])
        try:
            check_safety(call, config)
            if call.name == "write_file" and deps.approve is not None and not deps.approve(call):
                raise ToolError("The user declined this write. Do not try the write again.")
        except ToolError as exc:
            return retry(state, f"Safety check blocked the action: {exc}", "safety")
        emit("safety", tool=call.name, allowed=True)
        return {}

    def execute_node(state: AgentState) -> dict:
        # A second guard at the actual execution boundary, not only in routing.
        if state["step_count"] >= state["max_tool_steps"]:
            return {"status": "MAX_STEPS_REACHED", "final_answer": "Tool step limit reached."}
        call = ToolCall.model_validate(state["tool_call"])
        emit("tool_start", tool=call.name, arguments=call.arguments, step=state["step_count"] + 1)
        result = execute(call, registry, config)
        emit("tool_result", tool=call.name, step=state["step_count"] + 1, **result.model_dump())
        return {
            "last_tool_result": result.model_dump(),
            "step_count": state["step_count"] + 1,
            "stall_count": 0,
        }

    def observe(state: AgentState) -> dict:
        result = ToolResult.model_validate(state["last_tool_result"])
        call = ToolCall.model_validate(state["tool_call"])
        content = (
            f"Observation from tool '{call.name}':\n{result.output}"
            if result.success
            else (
                f"Tool error from '{call.name}': {result.error}\nTry a different action or explain."
            )
        )
        return {
            "messages": [
                *state["messages"],
                {
                    "role": "user",
                    "content": content,
                    "kind": "observation",
                },
            ],
            "current_action": None,
            "tool_call": None,
            "needle_result": None,
        }

    def update_context(state: AgentState) -> dict:
        return {"messages": contexts.build(state["messages"])[1:]}

    def guarded(name: str, handler: Callable) -> Callable:
        def node(state: AgentState) -> dict:
            if deps.cancelled is not None and deps.cancelled():
                return {
                    "status": "CANCELLED",
                    "final_answer": "Run stopped. No further tools will run.",
                }
            emit("phase", node=name, step=state["step_count"])
            try:
                return handler(state)
            except Exception as exc:
                # ToolError is handled at tool boundaries. Everything else is an internal error.
                label = {"reason": "Reasoning model", "translate": "Action model"}.get(name, name)
                return {"status": "ERROR", "final_answer": f"{label} failed: {exc}"}

        return node

    def route(next_node: str | Callable[[AgentState], str]) -> Callable:
        def router(state: AgentState) -> str:
            if state["status"] != "RUNNING":
                return END
            return next_node(state) if callable(next_node) else next_node

        return router

    def after_confidence(state: AgentState) -> str:
        needle = NeedleResult.model_validate(state["needle_result"])
        call = ToolCall.model_validate(state["tool_call"])
        return (
            "safety"
            if is_confident(needle.confidence, threshold_for(call.name, config))
            else "confirm"
        )

    handlers = {
        "reason": reason,
        "parse": parse,
        "translate": translate,
        "sanitize": sanitize_node,
        "validate": validate_node,
        "confidence": confidence,
        "confirm": confirm,
        "safety": safety,
        "execute": execute_node,
        "observe": observe,
        "update_context": update_context,
    }
    transitions = {
        "reason": ("parse", ["parse"]),
        "parse": (lambda s: "translate" if s["current_action"] else END, ["translate"]),
        "translate": (
            lambda s: "sanitize" if s["needle_result"] is not None else "reason",
            ["sanitize", "reason"],
        ),
        "sanitize": (lambda s: "validate" if s["tool_call"] else "reason", ["validate", "reason"]),
        "validate": (
            lambda s: "confidence" if s["tool_call"] else "reason",
            ["confidence", "reason"],
        ),
        "confidence": (after_confidence, ["safety", "confirm"]),
        "confirm": ("reason", ["reason"]),
        "safety": (lambda s: "execute" if s["tool_call"] else "reason", ["execute", "reason"]),
        "execute": ("observe", ["observe"]),
        "observe": ("update_context", ["update_context"]),
        "update_context": ("reason", ["reason"]),
    }
    graph = StateGraph(AgentState)
    for name, handler in handlers.items():
        graph.add_node(name, guarded(name, handler))
    graph.add_edge(START, "reason")
    for name, (destination, choices) in transitions.items():
        graph.add_conditional_edges(name, route(destination), [*choices, END])
    return graph.compile()
