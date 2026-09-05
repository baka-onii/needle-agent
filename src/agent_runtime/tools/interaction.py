"""ask_user's transport is injected: terminal input, web question, or a test."""

from __future__ import annotations

from collections.abc import Callable

from agent_runtime.tools.base import Tool, ToolError


def make_ask_user_tool(ask_fn: Callable[[str], str] | None = None) -> Tool:
    def ask_user(question: str) -> str:
        try:
            return ask_fn(question) if ask_fn is not None else input(question + " ")
        except EOFError as exc:
            raise ToolError("No interactive input available. Supply an ask_user handler.") from exc

    return Tool(
        name="ask_user",
        description="Ask the human a clarifying question when the request is ambiguous.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4_000,
                    "description": "The clarifying question to ask the human.",
                }
            },
            "required": ["question"],
        },
        handler=ask_user,
    )
