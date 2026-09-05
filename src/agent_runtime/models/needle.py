"""Needle action-model adapter (spec §16).

Uses the single-turn ``complete()`` API — never Needle's full ``run()`` loop,
which belongs to this framework. The rest of the framework talks only to the
``ActionModel`` protocol.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.models.action import ActionModel, NeedleResult, ToolRanking
from agent_runtime.tools.base import Tool


class NeedleActionModel(ActionModel):
    """Bind one toolset to a Needle session (Needle sessions share a toolset)."""

    def __init__(
        self,
        tools: list[Tool],
        system: str | None = None,
        weights: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._tool_names = [t.name for t in tools]
        if client is not None:
            self._client = client
        else:
            from needle import Needle  # lazy: avoids engine load at import time

            self._client = Needle(
                tools=[t.needle_schema() for t in tools],
                system=system,
                weights=weights,
            )

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        if [t.name for t in tools] != self._tool_names:
            raise ValueError("NeedleActionModel is bound to a fixed toolset.")
        # One reasoning turn = one clean translation. The framework feeds tool
        # results back to the reasoning model, not to Needle, so a continuing
        # Needle session would accumulate result-less queries and drift.
        self._client.reset()
        response = self._client.complete(action, max_new_tokens=256)
        calls = response.get("function_calls") or []
        if response.get("type") == "call" and calls:
            selected = calls[0].get("name")
            arguments = calls[0].get("arguments") or {}
        else:
            # "respond" or empty call []: refusal / off-topic / done.
            selected, arguments = None, {}
        try:
            confidence = float(response.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rankings = [ToolRanking(tool_name=selected, confidence=confidence)] if selected else []
        return NeedleResult(
            selected_tool=selected,
            arguments=arguments if isinstance(arguments, dict) else {},
            confidence=confidence,
            rankings=rankings,
        )
