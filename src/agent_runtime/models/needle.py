"""Needle's single-turn adapter. Never uses Needle.run() or executes its calls.

The C engine is process-global, so binding/reset/completion are serialized even
across multiple adapter instances. Loading is lazy and telemetry is off by default.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from pydantic import ValidationError

from agent_runtime.models.action import ActionOutputError, NeedleResult, ToolRanking
from agent_runtime.tools.base import Tool

_ENGINE_LOCK = threading.RLock()


def parse_needle_response(response: Any) -> NeedleResult:
    if not isinstance(response, dict) or response.get("type") not in {"call", "respond"}:
        raise ActionOutputError("Needle returned an invalid response envelope.")
    calls = response.get("function_calls", [])
    if not isinstance(calls, list):
        raise ActionOutputError("Needle returned malformed function_calls.")
    selected, arguments = None, {}
    if response["type"] == "call" and calls:
        call = calls[0]  # V0 only translates/executes the first action.
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ActionOutputError("Needle returned a malformed tool selection.")
        if not isinstance(call.get("arguments"), dict):
            raise ActionOutputError(
                "Needle returned non-object arguments; refusing to repair them."
            )
        selected, arguments = call["name"], call["arguments"]
    # Custom weights have no calibrated confidence (None). Fail closed, not 1.0.
    confidence = response.get("confidence")
    if confidence is None:
        confidence = 0.0
    try:
        rankings = response.get("rankings")
        if rankings is None:
            rankings = [ToolRanking(tool_name=selected, confidence=confidence)] if selected else []
        else:
            rankings = [ToolRanking.model_validate(item) for item in rankings]
        return NeedleResult(
            selected_tool=selected, arguments=arguments, confidence=confidence, rankings=rankings
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise ActionOutputError(
            "Needle returned invalid arguments, confidence, or rankings."
        ) from exc


class NeedleActionModel:
    def __init__(
        self,
        tools: list[Tool],
        system: str | None = None,
        weights: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._schemas = [tool.needle_schema() for tool in tools]
        self._signature = json.dumps(self._schemas, sort_keys=True)
        self._system, self._weights, self._client = system, weights, client

    def prepare(self) -> None:
        with _ENGINE_LOCK:
            if self._client is None:
                os.environ.setdefault("NEEDLE_TELEMETRY", "0")
                from needle import Needle

                self._client = Needle(
                    tools=self._schemas, system=self._system, weights=self._weights
                )

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        if json.dumps([tool.needle_schema() for tool in tools], sort_keys=True) != self._signature:
            raise ValueError("NeedleActionModel is bound to a fixed toolset.")
        with _ENGINE_LOCK:
            self.prepare()
            self._client.reset()
            response = self._client.complete(action, max_new_tokens=256)
        return parse_needle_response(response)

    def close(self) -> None:
        with _ENGINE_LOCK:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
            self._client = None
