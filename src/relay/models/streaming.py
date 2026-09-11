"""Streaming adapter data and bounded stdlib SSE/HTTP transport helpers."""

from __future__ import annotations

import contextlib
import contextvars
import http.client
import json
import socket
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

MAX_STREAM_BYTES = 16_000_000
MAX_SSE_LINE_BYTES = 1_000_000


class GenerationCancelled(Exception):
    """Cancellation is terminal for this generation, never a partial executable response."""


class IncompleteGeneration(RuntimeError):
    """EOF, token limits, or provider errors cannot authorize a partial tool call."""


@dataclass(frozen=True)
class ModelDelta:
    text: str = ""
    kind: Literal["content", "reasoning", "metadata"] = "content"
    streamed: bool | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise GenerationCancelled("Generation stopped.")


@contextlib.contextmanager
def streaming_transport(cancelled, *handlers, timeout_s: float):
    """Close only this request's socket on cancellation, including a stalled SSE read.

    HTTPConnection subclasses capture the socket through its public API. There is
    no detached reader thread or unbounded queue. Initial DNS/connect/TLS work is
    still bounded by the configured connection timeout.
    """
    finished = threading.Event()
    expired = threading.Event()
    started = time.monotonic()
    sockets: list[socket.socket] = []

    def connected(connection):
        sockets.append(connection.sock)
        check_cancelled(cancelled)
        if expired.is_set():
            raise TimeoutError("Model request exceeded its deadline.")

    class HTTPConnection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            connected(self)

    class HTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            connected(self)

    class HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(HTTPConnection, req)

    class HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(HTTPSConnection, req, context=self._context)

    def interrupt():
        while not finished.wait(0.05):
            stopping = cancelled is not None and cancelled()
            if time.monotonic() - started >= timeout_s:
                expired.set()
            if stopping or expired.is_set():
                for connection in sockets:
                    with contextlib.suppress(OSError):
                        connection.shutdown(socket.SHUT_RDWR)
                return

    watcher = threading.Thread(target=interrupt, daemon=True, name="model-cancellation")
    watcher.start()
    try:
        check_cancelled(cancelled)
        yield urllib.request.build_opener(HTTPHandler(), HTTPSHandler(), *handlers)
    except Exception as exc:
        check_cancelled(cancelled)
        if expired.is_set():
            raise TimeoutError("Model request exceeded its deadline.") from exc
        raise
    finally:
        finished.set()
        watcher.join(timeout=0.2)
        for connection in sockets:
            with contextlib.suppress(OSError):
                connection.close()


def sse_payloads(response, cancelled=None) -> Iterator[str]:
    """UTF-8, CRLF, comments, multiline data, and bounded packet/body sizes."""
    data: list[str] = []
    total = 0
    first = True
    while True:
        check_cancelled(cancelled)
        line = response.readline(MAX_SSE_LINE_BYTES + 1)
        check_cancelled(cancelled)
        total += len(line)
        if len(line) > MAX_SSE_LINE_BYTES or total > MAX_STREAM_BYTES:
            raise RuntimeError("Reasoning stream exceeded the transport size limit.")
        if not line:
            if data:
                yield "\n".join(data)
            return
        try:
            text = line.decode("utf-8-sig" if first else "utf-8").rstrip("\r\n")
        except UnicodeError as exc:
            raise RuntimeError("Reasoning stream contained invalid UTF-8.") from exc
        first = False
        if not text:
            if data:
                yield "\n".join(data)
                data.clear()
        elif text.startswith("data:"):
            value = text[5:]
            data.append(value[1:] if value.startswith(" ") else value)


def usage_counts(value: Any) -> dict[str, int]:
    """Only documented numeric counters, not arbitrary provider response metadata."""
    if not isinstance(value, dict):
        return {}
    counts = {
        key: value[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
        if type(value.get(key)) is int and 0 <= value[key] <= 2**53 - 1
    }
    details = value.get("completion_tokens_details")
    if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
        if 0 <= details["reasoning_tokens"] <= 2**53 - 1:
            counts["reasoning_tokens"] = details["reasoning_tokens"]
    return counts


def response_deltas(message: dict) -> Iterator[ModelDelta]:
    """Expose only provider-returned text. Never invent or request private reasoning."""
    if message.get("tool_calls") or message.get("function_call"):
        raise RuntimeError(
            "Reasoning returned native tool calls; use the <tool>/<final> text protocol."
        )
    # Compatible servers use one of these names, not an undisclosed internal channel.
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    for kind, text in (("reasoning", reasoning), ("content", message.get("content"))):
        if text is None or text == "":
            continue
        if not isinstance(text, str):
            raise RuntimeError("Reasoning backend returned non-text content.")
        text.encode("utf-8")  # Reject unpaired surrogate escapes before emitting JSON/SSE.
        yield ModelDelta(text=text, kind=kind)


def decode_packet(payload: str) -> dict:
    try:
        packet = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Reasoning stream contained invalid JSON.") from exc
    if not isinstance(packet, dict) or packet.get("error"):
        raise RuntimeError("Reasoning backend reported a stream error or invalid packet.")
    return packet


def check_finish(reason: str | None) -> None:
    if reason is not None and reason not in {"stop", "eos", "end_turn"}:
        detail = "the output token limit" if reason == "length" else "a provider stop condition"
        raise IncompleteGeneration(
            f"Generation ended at {detail}; no partial tool call was executed."
        )


class TextDeltaBuffer:
    """Bounded time/size coalescing that flushes even while the provider is idle.

    The caller supplies a thread-safe event writer. A scoped timer (not a reader
    thread or unbounded queue) prevents the final small batch from waiting for
    another upstream token. close() joins before the model's terminal event.
    """

    def __init__(self, deliver, interval_ms: int, *, timed: bool = True):
        self._deliver = deliver
        self._interval = interval_ms / 1000
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._pending: list[tuple[str, str]] = []
        self._chars = 0
        self._emitted = False
        self._error: Exception | None = None
        self._thread = None
        if timed:
            context = contextvars.copy_context()
            self._thread = threading.Thread(
                target=context.run, args=(self._timer,), daemon=True, name="model-delta-buffer"
            )
            self._thread.start()

    def _timer(self):
        while not self._stop.wait(self._interval):
            try:
                self.flush()
            except Exception as exc:
                self._error = exc
                return

    def append(self, kind: str, text: str):
        if not text:
            return
        with self._lock:
            if self._error:
                raise self._error
            if self._stop.is_set():
                raise RuntimeError("Cannot append to a closed model stream buffer.")
            if self._pending and self._pending[-1][0] == kind:
                self._pending[-1] = (kind, self._pending[-1][1] + text)
            else:
                self._pending.append((kind, text))
            self._chars += len(text)
            if not self._emitted or self._chars >= 1024:
                self.flush()

    def flush(self):
        with self._lock:
            if self._error:
                raise self._error
            for kind, text in self._pending:
                self._deliver(kind, text)
            if self._pending:
                self._emitted = True
            self._pending.clear()
            self._chars = 0

    def close(self, *, check_error: bool = True):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if check_error:
            self.flush()
