"""Provider-independent reasoning contract plus a stdlib OpenAI-compatible client."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from relay.config import AgentConfig, normalize_model_url
from relay.models.streaming import (
    GenerationCancelled,
    IncompleteGeneration,
    ModelDelta,
    check_cancelled,
    check_finish,
    decode_packet,
    response_deltas,
    sse_payloads,
    streaming_transport,
    usage_counts,
)
from relay.tools.base import Tool
from relay.tools.filesystem import workspace_description

MAX_RESPONSE_BYTES = 2_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a provider's Authorization header to a redirected host.
        return None


class ReasoningModel(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Return text in the <tool>/<final> protocol, never a structured tool call."""
        ...


class StreamingReasoningModel(Protocol):
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[ModelDelta | str]:
        """Optional streaming contract. Only returned content can be executable intent."""
        ...


def build_system_prompt(tools: list[Tool], config: AgentConfig | None = None) -> str:
    config = config or AgentConfig()
    descriptions = "\n".join(tool.reasoning_description() for tool in tools)
    return (
        config.reasoning_prompt.rstrip()
        + "\n\nRuntime workspace context:\n"
        + workspace_description(config)
        + "\n\nAvailable tools (generated from the canonical registry):\n"
        + descriptions
    )


def build_translator_prompt(config: AgentConfig) -> str:
    return (
        config.translator_prompt.rstrip()
        + "\n\nRuntime workspace context:\n"
        + workspace_description(config)
    )


def api_base_url(base_url: str) -> str:
    base = normalize_model_url(base_url)
    return base if base.endswith("/v1") else base + "/v1"


class OpenAICompatibleReasoningModel:
    """llama.cpp, Ollama, or a hosted compatible server; no SDK dependency."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
    ) -> None:
        defaults = AgentConfig()
        self._base = api_base_url(defaults.llm_base_url if base_url is None else base_url)
        self._model = defaults.llm_model if model is None else model
        self._timeout_s = defaults.llm_timeout_s if timeout_s is None else timeout_s
        self._max_tokens = defaults.llm_max_tokens if max_tokens is None else max_tokens
        self._temperature = defaults.llm_temperature if temperature is None else temperature
        self._api_key = api_key

    def _request(
        self, path: str, payload: dict | None = None, timeout: float | None = None
    ) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._base + path,
            data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
            headers=headers,
        )
        try:
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=timeout or self._timeout_s) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("Reasoning backend response exceeded the size limit.")
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise RuntimeError("Reasoning backend returned a non-object response.")
            return body
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Reasoning backend returned HTTP {exc.code}. Check the model, URL, and "
                "server-side API key configuration."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Reasoning backend is unreachable or timed out. Start the model server and "
                "check its URL from the machine running Relay."
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Reasoning backend returned invalid JSON.") from exc

    def generate(self, messages: list[dict[str, Any]]) -> str:
        # Internal context annotations never go to the provider. No tools/tool_choice.
        clean = [{"role": message["role"], "content": message["content"]} for message in messages]
        body = self._request(
            "/chat/completions",
            {
                "model": self._model,
                "messages": clean,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": False,
            },
        )
        try:
            choice = body["choices"][0]
            check_finish(choice.get("finish_reason"))
            message = choice["message"]
            # Non-streaming requests retain the same native-call/encoding guards.
            list(response_deltas(message))
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Missing text content.")
            return content
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise RuntimeError("Bad reasoning backend response: expected assistant text.") from exc

    @property
    def model_name(self) -> str:
        return self._model

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[ModelDelta]:
        """Read actual provider SSE. An application/json response is an honest fallback.

        A provider that rejects streaming should be configured with llm_stream=false;
        there is no automatic duplicate request, demo fallback, or invented token stream.
        """
        clean = [{"role": message["role"], "content": message["content"]} for message in messages]
        payload = {
            "model": self._model,
            "messages": clean,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._base + "/chat/completions",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers=headers,
        )
        try:
            with streaming_transport(cancelled, _NoRedirect(), timeout_s=self._timeout_s) as opener:
                with opener.open(request, timeout=self._timeout_s) as response:
                    content_type = response.headers.get_content_type()
                    if content_type == "application/json":
                        raw = response.read(MAX_RESPONSE_BYTES + 1)
                        check_cancelled(cancelled)
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raise RuntimeError(
                                "Reasoning backend response exceeded the size limit."
                            )
                        body = decode_packet(raw.decode("utf-8"))
                        choice = body["choices"][0]
                        check_finish(choice.get("finish_reason"))
                        yield ModelDelta(kind="metadata", streamed=False)
                        yield from response_deltas(choice["message"])
                        yield ModelDelta(
                            kind="metadata",
                            usage=usage_counts(body.get("usage")),
                            finish_reason="stop",
                        )
                        return
                    if content_type != "text/event-stream":
                        raise RuntimeError(
                            "Reasoning backend must return text/event-stream or application/json."
                        )
                    yield ModelDelta(kind="metadata", streamed=True)
                    finished = False
                    for data in sse_payloads(response, cancelled):
                        if data.strip() == "[DONE]":
                            finished = True
                            break
                        body = decode_packet(data)
                        counts = usage_counts(body.get("usage"))
                        if counts:
                            yield ModelDelta(kind="metadata", usage=counts)
                        choices = body.get("choices", [])
                        if not isinstance(choices, list):
                            raise RuntimeError("Reasoning stream returned an invalid choices list.")
                        for choice in choices:
                            if not isinstance(choice, dict):
                                raise RuntimeError("Reasoning stream returned an invalid choice.")
                            if choice.get("index", 0) != 0:
                                continue
                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict):
                                raise RuntimeError("Reasoning stream returned an invalid delta.")
                            if finished and (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or delta.get("reasoning")
                            ):
                                raise IncompleteGeneration(
                                    "Provider sent text after its finish marker."
                                )
                            yield from response_deltas(delta)
                            finish = choice.get("finish_reason")
                            check_finish(finish)
                            if finish is not None:
                                finished = True
                    check_cancelled(cancelled)
                    if not finished:
                        raise IncompleteGeneration(
                            "Reasoning stream disconnected before completion; "
                            "no partial tool call was executed."
                        )
                    yield ModelDelta(kind="metadata", finish_reason="stop")
        except (GenerationCancelled, IncompleteGeneration):
            raise
        except urllib.error.HTTPError as exc:
            check_cancelled(cancelled)
            exc.close()
            raise RuntimeError(
                f"Reasoning backend returned HTTP {exc.code}. "
                "Check the model, URL, and server-side "
                "API key. If this provider cannot stream, disable model streaming in Settings."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            check_cancelled(cancelled)
            raise RuntimeError(
                "Reasoning backend is unreachable, disconnected, or timed out during streaming."
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            check_cancelled(cancelled)
            raise RuntimeError(
                "Bad reasoning backend stream: expected assistant text deltas."
            ) from exc

    def check_connection(self) -> list[str]:
        body = self._request("/models", timeout=min(self._timeout_s, 10.0))
        return [
            item["id"]
            for item in body.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]


# Preserve the original public name used by V0 integrations.
LlamaServerReasoningModel = OpenAICompatibleReasoningModel
