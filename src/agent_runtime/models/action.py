"""Action-model interface (spec §15). Needle implements this; the rest of the
framework must not call Needle directly."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent_runtime.tools.base import Tool


class ToolRanking(BaseModel):
    tool_name: str
    confidence: float


class NeedleResult(BaseModel):
    selected_tool: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    rankings: list[ToolRanking] = Field(default_factory=list)


class ActionModel(Protocol):
    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        """Map a natural-language action to a tool selection + arguments."""
        ...
