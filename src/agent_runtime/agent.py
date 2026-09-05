"""Public synchronous and streaming APIs, with per-run conversation and control."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager
from agent_runtime.graph.workflow import RuntimeDeps, build_workflow
from agent_runtime.models.action import ActionModel
from agent_runtime.models.needle import NeedleActionModel
from agent_runtime.models.reasoning import (
    OpenAICompatibleReasoningModel,
    ReasoningModel,
    build_system_prompt,
)
from agent_runtime.state import AgentState, create_initial_state
from agent_runtime.tools.base import ToolCall
from agent_runtime.tools.registry import ToolRegistry, create_default_registry


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        reasoning: ReasoningModel | None = None,
        action: ActionModel | None = None,
        registry: ToolRegistry | None = None,
        ask_fn: Callable[[str], str] | None = None,
        approve_fn: Callable[[ToolCall], bool] | None = None,
    ) -> None:
        config = config or AgentConfig()
        root = Path(config.workspace_root or Path.cwd()).resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace does not exist or is not a directory: {root}")
        self.config = replace(config, workspace_root=str(root))
        self.registry = registry or create_default_registry(self.config, ask_fn)
        self._contexts = ContextManager(self.config, build_system_prompt(self.registry.list()))
        self._reasoning = reasoning or OpenAICompatibleReasoningModel(
            base_url=config.llm_base_url,
            model=config.llm_model,
            timeout_s=config.llm_timeout_s,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            api_key=config.llm_api_key,
        )
        self._owns_action = action is None
        self._action = action or NeedleActionModel(
            self.registry.list(), weights=config.needle_weights
        )
        self._approve = approve_fn

    def stream(
        self,
        request: str,
        *,
        history: list[dict[str, Any]] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield phase/action/gate/result events, ending with a complete event + state.

        History is explicit so independent callers cannot leak conversations. The
        graph is built per invocation; no shared mutable per-run dependencies.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("A nonempty request is required.")
        initial = create_initial_state(request.strip(), self.config.max_tool_steps)
        if history:
            if any(
                m.get("role") not in {"user", "assistant"} or not isinstance(m.get("content"), str)
                for m in history
            ):
                raise ValueError("History must contain user/assistant text messages only.")
            initial["messages"] = [*[dict(m) for m in history], *initial["messages"]]
        deps = RuntimeDeps(
            self._reasoning,
            self._action,
            self.registry,
            self._contexts,
            self.config,
            cancelled,
            self._approve,
        )
        graph = build_workflow(deps)
        started, sequence, final = time.monotonic(), 0, initial
        # LangGraph's default 25 is too small even for three valid tool actions.
        limit = 12 * (self.config.max_tool_steps + 1) * (self.config.max_stalls + 1)
        try:
            for mode, chunk in graph.stream(
                initial,
                config={"recursion_limit": limit},
                stream_mode=["custom", "values"],
            ):
                if mode == "values":
                    final = chunk
                else:
                    sequence += 1
                    yield {
                        **chunk,
                        "sequence": sequence,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    }
        except Exception as exc:
            final = {**final, "status": "ERROR", "final_answer": f"Runtime failed: {exc}"}
        # Bound returned history too (a tagless/final response bypasses observe).
        try:
            final["messages"] = self._contexts.build(final["messages"])[1:]
        except ValueError as exc:
            final = {**final, "status": "ERROR", "final_answer": str(exc), "messages": []}
        yield {
            "type": "complete",
            "state": final,
            "sequence": sequence + 1,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }

    def run(
        self,
        request: str,
        *,
        history: list[dict[str, Any]] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AgentState:
        """Run to completion. Pass the returned messages as history for a follow-up."""
        for event in self.stream(request, history=history, cancelled=cancelled):
            if on_event is not None:
                on_event(event)
            if event["type"] == "complete":
                return event["state"]
        raise RuntimeError("Workflow ended without a terminal event.")

    def close(self) -> None:
        if self._owns_action and isinstance(self._action, NeedleActionModel):
            self._action.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
