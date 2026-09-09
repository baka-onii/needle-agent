"""The explicit LangGraph V0 loop. Runtime dependencies never enter state."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager
from agent_runtime.execution.confidence import (
    is_confident,
    ranked_candidates,
    selection_review_message,
    threshold_for,
)
from agent_runtime.execution.executor import check_safety, execute
from agent_runtime.execution.sanitizer import sanitize
from agent_runtime.execution.validator import validate
from agent_runtime.models.action import ActionModel, ActionOutputError, NeedleResult
from agent_runtime.models.reasoning import ReasoningModel, StreamingReasoningModel
from agent_runtime.models.streaming import (
    GenerationCancelled,
    IncompleteGeneration,
    ModelDelta,
    TextDeltaBuffer,
    check_cancelled,
    check_finish,
    usage_counts,
)
from agent_runtime.protocol.intent import (
    check_atomic_action,
    is_write_permission_question,
    validate_intent,
)
from agent_runtime.protocol.parser import parse_response
from agent_runtime.protocol.stream import ResponseStream
from agent_runtime.state import AgentState
from agent_runtime.tools.base import ToolCall, ToolError, ToolResult, truncate_text
from agent_runtime.tools.filesystem import resolve_safe_path
from agent_runtime.tools.registry import ToolRegistry


@dataclass
class RuntimeDeps:
    reasoning: ReasoningModel | StreamingReasoningModel
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

    def review(state: AgentState, message: str, stage: str) -> str:
        needle = (
            NeedleResult.model_validate(state["needle_result"]) if state["needle_result"] else None
        )
        call = ToolCall.model_validate(state["tool_call"]) if state["tool_call"] else None
        if call is None and needle and needle.selected_tool:
            call = ToolCall(name=needle.selected_tool, arguments=needle.arguments)
        candidates = ranked_candidates(needle, registry.list())
        emit(
            "confirmation",
            stage=stage,
            reason=message,
            selected_tool=call.name if call else None,
            arguments=call.arguments if call else {},
            candidates=[candidate.model_dump() for candidate in candidates],
            suggested_tool=candidates[0].tool_name if candidates else None,
        )
        return selection_review_message(
            state["current_action"] or "",
            candidates,
            reason=message,
            call=call,
            tools=registry.list(),
            instructions=config.confirmation_prompt,
        )

    def retry(state: AgentState, message: str, stage: str) -> dict:
        stalls = state["stall_count"] + 1
        emit("rejected", stage=stage, message=message, stalls=stalls)
        feedback = review(state, message, stage)
        update = {
            "stall_count": stalls,
            "tool_call": None,
            "needle_result": None,
            "current_action": None,
            "current_payloads": [],
            "messages": append(state, feedback, "confirmation"),
        }
        if stalls >= config.max_stalls:
            update.update(
                status="STALLED",
                final_answer=f"Stopped after {stalls} consecutive non-executing actions. {message}",
            )
        return update

    # Tools whose handlers resolve paths must resolve identically here: an
    # escaping path is rejected in safety before any approval or execution.
    # Git paths are repo-scoped by git itself, so they are recorded raw.
    _RESOLVED_PATH_TOOLS = frozenset(
        {
            "write_file",
            "read_file",
            "read_directory",
            "search_files",
            "file_info",
            "create_directory",
            "delete_file",
            "replace_text",
            "insert_text",
            "delete_text",
            "apply_patch",
        }
    )

    def record(call: ToolCall, outcome: str, output: str = "") -> dict:
        arguments = dict(call.arguments)
        path = ""
        if call.name in _RESOLVED_PATH_TOOLS:
            path = str(resolve_safe_path(arguments.get("path", "."), config.workspace_root))
            arguments["path"] = path
        elif call.name in {"move_file", "copy_file"}:
            path = str(
                resolve_safe_path(arguments.get("source", ""), config.workspace_root)
            )
        elif call.name in {"git_diff", "git_log", "git_show"}:
            raw = arguments.get("path", ".")
            path = raw if isinstance(raw, str) else ""
        question = ""
        if call.name == "ask_user":
            question = re.sub(r"[^\w]+", " ", arguments["question"].casefold()).strip()
            arguments["question"] = question
        signature = hashlib.sha256(
            json.dumps(
                {"name": call.name, "arguments": arguments},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        return {
            "tool": call.name,
            "signature": signature,
            "path": path,
            "question": question,
            "outcome": outcome,
            "output": truncate_text(output, 1500),
        }

    def reason(state: AgentState) -> dict:
        prompt = contexts.build(state["messages"])
        turn = state["model_turn"] + 1
        model_id = f"reason-{turn}"
        stream_method = getattr(reasoning, "stream", None)
        streaming = config.llm_stream and callable(stream_method)
        emit(
            "model_start",
            model_id=model_id,
            component="reasoning",
            turn=turn,
            model=getattr(reasoning, "model_name", type(reasoning).__name__),
            streamed=streaming,
            input_messages=[{"role": m["role"], "content": m["content"]} for m in prompt]
            if config.capture_model_inputs
            else [],
            inputs_captured=config.capture_model_inputs,
            buffer_ms=config.stream_buffer_ms,
            max_lag_ms=config.stream_max_lag_ms,
        )
        projection = ResponseStream()
        started = time.monotonic()
        total_chars = 0
        usage = {}
        iterator = None
        projected_done = False
        declared_finished = False

        model_writer = get_stream_writer()

        def deliver(kind, text):
            parts = (
                projection.feed(text)
                if kind == "content"
                else [{"index": "provider", "kind": "reasoning", "text": text}]
            )
            model_writer(
                {
                    "type": "model_delta",
                    "model_id": model_id,
                    "channel": kind,
                    "delta": text,
                    "parts": parts,
                }
            )

        buffer = TextDeltaBuffer(deliver, config.stream_flush_ms, timed=streaming)

        def finish_projection():
            nonlocal projected_done
            if not projected_done:
                parts = projection.finish()
                if parts:
                    emit("model_delta", model_id=model_id, channel="content", delta="", parts=parts)
                projected_done = True

        try:
            if streaming:
                parameters = inspect.signature(stream_method).parameters.values()
                accepts_cancel = any(
                    p.name == "cancelled" or p.kind == p.VAR_KEYWORD for p in parameters
                )
                iterator = iter(
                    stream_method(prompt, cancelled=deps.cancelled)
                    if accepts_cancel
                    else stream_method(prompt)
                )
            else:
                iterator = iter([ModelDelta(text=reasoning.generate(prompt))])
            for delta in iterator:
                check_cancelled(deps.cancelled)
                if isinstance(delta, str):
                    delta = ModelDelta(text=delta)
                if not isinstance(delta, ModelDelta) or delta.kind not in {
                    "content",
                    "reasoning",
                    "metadata",
                }:
                    raise TypeError("Reasoning stream must yield text or ModelDelta values.")
                if not isinstance(delta.text, str):
                    raise TypeError("Reasoning model must return text.")
                if delta.kind == "metadata":
                    buffer.flush()
                    if delta.streamed is not None:
                        if type(delta.streamed) is not bool:
                            raise TypeError("ModelDelta.streamed must be a boolean.")
                        streaming = delta.streamed
                        emit("model_status", model_id=model_id, streamed=streaming)
                    if delta.usage:
                        usage.update(usage_counts(delta.usage))
                    check_finish(delta.finish_reason)
                    declared_finished |= delta.finish_reason is not None
                    continue
                if declared_finished and delta.text:
                    raise IncompleteGeneration("Model emitted text after declaring completion.")
                delta.text.encode("utf-8")
                total_chars += len(delta.text)
                if total_chars > config.max_model_output_chars:
                    raise ValueError(
                        "Model output exceeded max_model_output_chars; "
                        "no partial action was executed."
                    )
                buffer.append(delta.kind, delta.text)
                check_finish(delta.finish_reason)
                declared_finished |= delta.finish_reason is not None
            check_cancelled(deps.cancelled)
            buffer.close()
            finish_projection()
            raw = projection.assistant_text
            if not raw.strip():
                raise ValueError("Reasoning model returned no assistant response text.")
            parsed = parse_response(raw)
            emit(
                "model_end",
                model_id=model_id,
                status="completed",
                streamed=streaming,
                duration_ms=round((time.monotonic() - started) * 1000),
                output_chars=total_chars,
                usage=usage,
                decision="final" if parsed.final_answer is not None else "tool",
                answer=parsed.final_answer,
                action_count=len(parsed.actions),
            )
            return {
                "messages": [*prompt[1:], {"role": "assistant", "content": raw}],
                "model_turn": turn,
            }
        except Exception as exc:
            with contextlib.suppress(Exception):
                buffer.close()
            finish_projection()
            emit(
                "model_end",
                model_id=model_id,
                status="cancelled" if isinstance(exc, GenerationCancelled) else "error",
                error=str(exc),
                streamed=streaming,
                usage=usage,
                duration_ms=round((time.monotonic() - started) * 1000),
                output_chars=total_chars,
            )
            raise
        finally:
            buffer.close(check_error=False)
            if iterator is not None and callable(getattr(iterator, "close", None)):
                with contextlib.suppress(Exception):
                    iterator.close()

    def parse(state: AgentState) -> dict:
        parsed = parse_response(state["messages"][-1]["content"])
        if parsed.final_answer is not None:
            return {
                "final_answer": parsed.final_answer,
                "status": "COMPLETED",
                "current_action": None,
                "current_payloads": [],
            }
        if parsed.actions:
            first = parsed.actions[0]
            current = {
                **state,
                "current_action": first.instruction,
                "current_payloads": first.payloads,
                "tool_call": None,
                "needle_result": None,
            }
            emit(
                "action",
                action=first.instruction,
                payloads=first.payloads,
                ignored_actions=len(parsed.actions) - 1,
                model_id=f"reason-{state['model_turn']}",
            )
            try:
                check_atomic_action(first.instruction)
            except ToolError as exc:
                return retry(current, str(exc), "parse")
            return {
                "current_action": first.instruction,
                "current_payloads": first.payloads,
                "tool_call": None,
                "needle_result": None,
            }
        return {"final_answer": parsed.reasoning or "(no response)", "status": "COMPLETED"}

    def translate(state: AgentState) -> dict:
        model_id = f"translate-{state['model_turn']}"
        instruction = state["current_action"] or ""
        payloads = state.get("current_payloads") or []
        describe = getattr(action, "request_messages", None)
        inputs = []
        if config.capture_model_inputs:
            inputs = (
                describe(instruction)
                if callable(describe)
                else [{"role": "user", "content": instruction}]
            )
        emit(
            "model_start",
            model_id=model_id,
            component="translator",
            turn=state["model_turn"],
            parent_id=f"reason-{state['model_turn']}",
            model=type(action).__name__,
            streamed=False,
            input_messages=inputs if config.capture_model_inputs else [],
            inputs_captured=config.capture_model_inputs,
            buffer_ms=0,
            max_lag_ms=config.stream_max_lag_ms,
        )
        started = time.monotonic()
        try:
            result = action.translate(instruction, registry.list())
            if not isinstance(result, NeedleResult):
                raise ActionOutputError("Action model must return a NeedleResult.")
            result = NeedleResult.model_validate(result.model_dump())
            if payloads:
                try:
                    tool = registry.get(result.selected_tool or "")
                except ToolError as exc:
                    raise ActionOutputError(f"Unknown tool: {exc}") from exc
                expected = len(tool.payload_args)
                if len(payloads) != expected:
                    raise ActionOutputError(
                        f"Tool {tool.name!r} needs {expected} payload block(s), "
                        f"but the action carries {len(payloads)}. "
                        + (
                            f"Put the {tool.payload_args[0]!r} text in one <content> block."
                            if expected == 1
                            else (
                                "Put one <text-N> block per payload argument, numbered "
                                "from 1 without gaps."
                                if expected
                                else "This tool takes no payload blocks."
                            )
                        )
                    )
                # The translator only selects the destination and small
                # arguments; payload bytes always come from the reasoning
                # model's blocks. Its own payload output is discarded.
                # (Block-less actions keep the legacy fenced-payload path,
                # still verified downstream by validate_intent.)
                result = result.model_copy(
                    update={
                        "arguments": {
                            **result.arguments,
                            **dict(zip(tool.payload_args, payloads, strict=True)),
                        }
                    }
                )
        except Exception as exc:
            emit(
                "model_end",
                model_id=model_id,
                status="error",
                streamed=False,
                error=str(exc),
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            if isinstance(exc, (ActionOutputError, ValidationError)):
                return retry(
                    state,
                    f"Invalid translator output: {exc}. Please rephrase the action.",
                    "translate",
                )
            raise
        text = truncate_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            config.max_model_output_chars,
        )
        emit(
            "model_delta",
            model_id=model_id,
            channel="content",
            delta=text,
            parts=[{"index": 0, "kind": "text", "text": text, "complete": True}],
        )
        emit(
            "model_end",
            model_id=model_id,
            status="completed",
            streamed=False,
            duration_ms=round((time.monotonic() - started) * 1000),
            output_chars=len(text),
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
            validate_intent(
                state["current_action"] or "",
                call,
                config,
                list(state.get("current_payloads") or []),
            )
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
        call = ToolCall.model_validate(state["tool_call"])
        return retry(
            state,
            f"The action translator is uncertain: {call.name} scored {needle.confidence:.2f}, "
            f"below its {threshold_for(call.name, config):.2f} gate. Nothing was executed.",
            "confidence",
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
            proposal = record(call, "proposed")
            records = state["action_records"]
            same = [item for item in records if item["signature"] == proposal["signature"]]
            if sum(item["outcome"] == "failed" for item in same) >= config.max_repeated_failures:
                raise ToolError(
                    "This exact call already failed repeatedly. Choose a different tool or "
                    "correct its arguments; do not retry it unchanged."
                )
            if call.name == "ask_user":
                if is_write_permission_question(call.arguments["question"]):
                    raise ToolError(
                        "Do not use ask_user for write permission. The runtime handles approval. "
                        "Propose one exact write, or verify/report it if it already succeeded."
                    )
                answered = next((item for item in same if item["outcome"] == "succeeded"), None)
                if answered:
                    raise ToolError(
                        f"This question was already answered: {answered['output']!r}. "
                        "Use that answer instead of asking again."
                    )
            if call.name in config.require_approval_for or call.name == "write_file":
                if any(item["outcome"] == "succeeded" for item in same):
                    raise ToolError(
                        f"This exact {call.name} call already succeeded. "
                        "Do not repeat it or request permission again. "
                        "Verify the result or finish with a final answer."
                    )
            if call.name in config.require_approval_for:
                if any(
                    item["tool"] == call.name
                    and item["path"] == proposal["path"]
                    and item["outcome"] == "denied"
                    for item in records
                ):
                    raise ToolError(
                        f"The user already declined this {call.name} action during this run. "
                        "Do not request permission again."
                    )
                if deps.approve is not None and not deps.approve(call):
                    return {
                        **retry(
                            state,
                            f"The user declined this {call.name} action. Do not try it again.",
                            "safety",
                        ),
                        "action_records": [*records, record(call, "denied")],
                    }
        except ToolError as exc:
            return retry(state, f"Safety check blocked the action: {exc}", "safety")
        emit("safety", tool=call.name, allowed=True)
        if call.name in config.require_approval_for and deps.approve is not None:
            return {
                "messages": append(
                    state,
                    f"Runtime approval granted for this exact {call.name} call. "
                    "Do not ask for permission via ask_user. "
                    "This is not approval for any other action.",
                    "permission",
                )
            }
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
            "action_records": [
                *state["action_records"],
                record(
                    call,
                    "succeeded" if result.success else "failed",
                    result.output if result.success else result.error or "",
                ),
            ],
        }

    def observe(state: AgentState) -> dict:
        result = ToolResult.model_validate(state["last_tool_result"])
        call = ToolCall.model_validate(state["tool_call"])
        if result.success:
            content = f"Observation from tool '{call.name}':\n{result.output}"
            if call.name == "write_file":
                content += (
                    "\nThe write is complete. Do not repeat this write. "
                    "Verify with read_file if needed, then answer."
                )
        else:
            content = f"Tool error from '{call.name}': {result.error}\n" + review(
                state,
                f"Execution of {call.name!r} failed: {result.error}. "
                "Was the selected tool correct? Correct the tool or arguments before trying again.",
                "execute",
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
            "current_payloads": [],
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
            except GenerationCancelled:
                return {
                    "status": "CANCELLED",
                    "final_answer": "Run stopped. No further tools will run.",
                }
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
        "parse": (
            lambda s: "translate" if s["current_action"] else "reason",
            ["translate", "reason"],
        ),
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
