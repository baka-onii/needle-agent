"""Bounded context with original/current requests pinned and old observations removed first."""

from __future__ import annotations

from typing import Any

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import truncate_text


def message_chars(message: dict[str, Any]) -> int:
    return len(str(message.get("content", "")))


def total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(message_chars(message) for message in messages)


def _observation(message: dict[str, Any]) -> bool:
    return message.get("kind") in {"observation", "confirmation"} or str(
        message.get("content", "")
    ).startswith(("Observation from tool", "Tool error", "The action translator is uncertain"))


class ContextManager:
    def __init__(self, config: AgentConfig, system_prompt: str) -> None:
        if len(system_prompt) >= config.max_context_chars:
            raise ValueError(
                "Context budget is too small for the system prompt and tool descriptions."
            )
        self._config = config
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system = {"role": "system", "content": self._system_prompt}
        indexed = [(i, dict(message)) for i, message in enumerate(transcript)]
        # In a multi-turn conversation the current request is as important as the original.
        latest_request = next(
            (i for i, m in reversed(indexed) if m.get("role") == "user" and not _observation(m)), 0
        )
        pinned = [(i, m) for i, m in indexed if i in {0, latest_request}]
        remaining = self._config.max_context_chars - total_chars([system, *(m for _, m in pinned)])
        if remaining < 0:
            raise ValueError("Original request or current request exceeds the context budget.")
        tail = [(i, m) for i, m in indexed if i not in {0, latest_request}]
        while total_chars([m for _, m in tail]) > remaining and len(tail) > 1:
            index = next(
                (i for i, (_, message) in enumerate(tail[:-1]) if _observation(message)), 0
            )
            tail.pop(index)
        if total_chars([m for _, m in tail]) > remaining:
            tail[-1][1]["content"] = truncate_text(str(tail[-1][1].get("content", "")), remaining)
        return [system, *(message for _, message in sorted([*pinned, *tail]))]
