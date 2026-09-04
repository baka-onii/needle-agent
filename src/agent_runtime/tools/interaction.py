"""Interaction tools: ask_user. The UI mechanism stays behind the handler."""

from __future__ import annotations

from collections.abc import Callable

from agent_runtime.tools.base import Tool


def make_ask_user_tool(ask_fn: Callable[[str], str] | None = None) -> Tool:
    def ask_user(question: str) -> str:
        if ask_fn is not None:
            return ask_fn(question)
        return input(question + " ")

    return Tool(
        name="ask_user",
        description="Ask the human a clarifying question. Use when the request is ambiguous.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the human.",
                }
            },
            "required": ["question"],
        },
        handler=ask_user,
    )
