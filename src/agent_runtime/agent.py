"""Agent entrypoint: wire defaults, expose ``run()``."""

from __future__ import annotations

from collections.abc import Callable

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager
from agent_runtime.graph.workflow import RuntimeDeps, build_workflow
from agent_runtime.models.action import ActionModel
from agent_runtime.models.needle import NeedleActionModel
from agent_runtime.models.reasoning import (
    LlamaServerReasoningModel,
    ReasoningModel,
    build_system_prompt,
)
from agent_runtime.state import AgentState, create_initial_state
from agent_runtime.tools.registry import ToolRegistry, create_default_registry


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        reasoning: ReasoningModel | None = None,
        action: ActionModel | None = None,
        registry: ToolRegistry | None = None,
        ask_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._registry = registry or create_default_registry(self._config, ask_fn)
        tools = self._registry.list()
        contexts = ContextManager(self._config, build_system_prompt(tools))
        self._reasoning = reasoning or LlamaServerReasoningModel(
            base_url=self._config.llm_base_url,
            model=self._config.llm_model,
            timeout_s=self._config.llm_timeout_s,
            max_tokens=self._config.llm_max_tokens,
        )
        self._action = action or NeedleActionModel(tools)
        deps = RuntimeDeps(
            reasoning=self._reasoning,
            action=self._action,
            registry=self._registry,
            contexts=contexts,
            config=self._config,
        )
        self._workflow = build_workflow(deps)

    def run(self, request: str) -> AgentState:
        """Run the loop to completion. Returns the final state."""
        initial = create_initial_state(request, self._config.max_tool_steps)
        return self._workflow.invoke(initial)
