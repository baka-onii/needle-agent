"""Context manager (spec §26-27). Deliberately simple: no summarization in V0.

Keeps system prompt + original user request + most recent messages; drops
older middle messages (old observations) first when over budget.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.config import AgentConfig


def message_chars(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    return len(content) if isinstance(content, str) else len(str(content))


def total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(message_chars(m) for m in messages)


class ContextManager:
    def __init__(self, config: AgentConfig, system_prompt: str) -> None:
        self._config = config
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assemble the LLM context: system + transcript, trimmed to budget.

        ``transcript[0]`` must be the original user request; it is always kept.
        """
        messages = [{"role": "system", "content": self._system_prompt}, *transcript]
        budget = self._config.max_context_chars
        if total_chars(messages) <= budget or len(messages) <= 2:
            return messages
        head = messages[:2]  # system + original request
        kept: list[dict[str, Any]] = []
        remaining = budget - total_chars(head)
        for message in reversed(messages[2:]):
            cost = message_chars(message)
            if cost > remaining:
                break
            kept.append(message)
            remaining -= cost
        kept.reverse()
        # Guarantee progress: always keep at least the most recent message.
        if not kept:
            kept = [messages[-1]]
        return [*head, *kept]
