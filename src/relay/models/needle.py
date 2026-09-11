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

from relay.config import AgentConfig
from relay.models.action import ActionOutputError, NeedleResult, ToolRanking
from relay.protocol.intent import deterministic_write_result
from relay.tools.base import Tool

_ENGINE_LOCK = threading.RLock()


def parse_needle_response(response: Any) -> NeedleResult:
    if not isinstance(response, dict) or response.get("type") not in {"call", "respond"}:
        raise ActionOutputError("Needle returned an invalid response envelope.")
    calls = response.get("function_calls", [])
    if not isinstance(calls, list):
        raise ActionOutputError("Needle returned malformed function_calls.")
    if len(calls) > 1:
        raise ActionOutputError(
            "Needle returned multiple calls for one action. Choose one atomic tool operation."
        )
    selected, arguments = None, {}
    if response["type"] == "call" and calls:
        call = calls[0]
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
        max_new_tokens: int | None = None,
    ) -> None:
        self._schemas = [tool.needle_schema() for tool in tools]
        self._signature = json.dumps(self._schemas, sort_keys=True)
        defaults = AgentConfig()
        self._system = system if system is not None else defaults.translator_prompt
        self._weights, self._client = weights, client
        self._external_client = client is not None
        self._max_new_tokens = (
            defaults.needle_max_tokens if max_new_tokens is None else max_new_tokens
        )
        if type(self._max_new_tokens) is not int or self._max_new_tokens <= 0:
            raise ValueError("Needle max_new_tokens must be a positive integer.")

    def request_messages(self, action: str) -> list[dict[str, str]]:
        # A caller-supplied native client may have its own unknown system context.
        system = [] if self._external_client else [{"role": "system", "content": self._system}]
        return [*system, {"role": "user", "content": action}]

    def set_system(self, system: str) -> None:
        """Refresh runtime workspace context between runs; keep loading lazy."""
        with _ENGINE_LOCK:
            if system != self._system:
                self.close()
                self._system = system

    def prepare(self) -> None:
        with _ENGINE_LOCK:
            if self._client is None:
                os.environ.setdefault("NEEDLE_TELEMETRY", "0")
                from needle import Needle

                self._client = Needle(
                    tools=self._schemas,
                    system=self._system,
                    weights=self._weights,
                    buffer_size=max(65_536, self._max_new_tokens * 8),
                )

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        if json.dumps([tool.needle_schema() for tool in tools], sort_keys=True) != self._signature:
            raise ValueError("NeedleActionModel is bound to a fixed toolset.")
        # Well-formed writes never reach the engine: it substitutes a nearby
        # quoted string for the fenced payload at every budget. Engine use is
        # reserved for selections it performs reliably (read/search/calc/time).
        shortcut = deterministic_write_result(action)
        if shortcut is not None:
            return shortcut
        # Small budgets only: over-long generations degenerate into repeated
        # memorized calls and can crash the native library. Engine faults stay
        # retryable so one bad turn degrades to CONFIRM/STALLED, never a hang.
        with _ENGINE_LOCK:
            self.prepare()
            self._client.reset()
            try:
                response = self._client.complete(action, max_new_tokens=self._max_new_tokens)
            except (OSError, RuntimeError) as exc:
                raise ActionOutputError(f"Needle engine error: {exc}") from exc
        return parse_needle_response(response)

    def close(self) -> None:
        with _ENGINE_LOCK:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
            self._client = None
