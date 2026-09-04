"""Reasoning-model interface (spec §14). The graph only sees ``generate()``."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol

from agent_runtime.tools.base import Tool


class ReasoningModel(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Return the raw model response text (may contain <tool>/<final>)."""
        ...


def build_system_prompt(tools: list[Tool]) -> str:
    """System prompt: reasoning focus, protocol tags, available tools."""
    descriptions = "\n".join(t.reasoning_description() for t in tools)
    return (
        "You are a reasoning agent. Think through the user's request step by step, "
        "then either request ONE tool action or give the final answer.\n"
        "You never emit JSON and never call tools directly. A separate translator "
        "turns your action sentence into a tool call.\n"
        "To request a tool action, write exactly:\n"
        "<tool>\nA single plain-language action sentence, e.g. Read src/auth.py.\n</tool>\n"
        "To finish, write exactly:\n"
        "<final>\nThe answer for the user.\n</final>\n"
        "Request only one action per turn. After each tool result, reason again.\n"
        "Write concrete actions that name exact files, words, or values:\n"
        "good: Read the file src/auth.py.\n"
        "good: Search for the word authentication in the project.\n"
        "good: Calculate 2 * (15 + 3).\n"
        "bad: Look around a bit. / bad: Check the thing.\n"
        "Prefer searching for a distinctive word over listing directories.\n"
        "Available tools:\n" + descriptions
    )


class LlamaServerReasoningModel:
    """Reasoning model backed by a llama.cpp OpenAI-compatible server.

    Stdlib HTTP only — no extra client dependency (spec §40).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "ornith",
        timeout_s: float = 120.0,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, messages: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Reasoning backend unreachable: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Bad reasoning backend response: {exc}") from exc
