"""Action-model contract. Only the adapter knows about Needle's native API."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.base import Tool, ToolError


class ActionOutputError(ToolError):
    """Malformed translator output; retry reasoning without executing anything."""


class ToolRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class NeedleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selected_tool: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    rankings: list[ToolRanking] = Field(default_factory=list)


class ActionModel(Protocol):
    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        """Map one natural-language action to a selection and untrusted arguments."""
        ...
