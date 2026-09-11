"""Unified token counting against each server's own tokenizer.

The runtime budgets context per model, not per character guess. Each backend
asks the model server to tokenize with the exact vocabulary it serves:
llama.cpp exposes POST /tokenize, Ollama exposes POST /api/tokenize. Hosted
providers without a tokenize endpoint fall back to a chars/4 heuristic.

Counting never breaks a run: backends fail open to None, and callers fall
back to character budgets. Only the standard library is used.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


def _post(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read(1_000_000 + 1))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise TokenizerError(f"Tokenize request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise TokenizerError("Tokenize backend returned a non-object response.")
    return body


class TokenizerError(Exception):
    """The backend tokenizer is unreachable or returned unusable data."""


def _server_root(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


class TokenCounter(Protocol):
    kind: str

    def count(self, text: str) -> int | None:
        """Tokens for text, or None when the backend is unavailable."""
        ...


class LlamaCppCounter:
    """llama-server POST /tokenize (reasoning servers and the FG translator)."""

    kind = "llamacpp"

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self._url = _server_root(base_url) + "/tokenize"
        self._timeout_s = timeout_s
        self._dead = False

    def count(self, text: str) -> int | None:
        if self._dead or not text:
            return 0 if not text else None
        try:
            body = _post(self._url, {"content": text}, self._timeout_s)
            tokens = body.get("tokens")
            if not isinstance(tokens, list):
                raise TokenizerError("Tokenize response has no token list.")
            return len(tokens)
        except TokenizerError:
            self._dead = True
            return None


class OllamaCounter:
    """Ollama POST /api/tokenize."""

    kind = "ollama"

    def __init__(
        self, base_url: str, model: str, timeout_s: float = 10.0
    ) -> None:
        self._url = _server_root(base_url) + "/api/tokenize"
        self._model = model
        self._timeout_s = timeout_s
        self._dead = False

    def count(self, text: str) -> int | None:
        if self._dead or not text:
            return 0 if not text else None
        try:
            body = _post(
                self._url, {"model": self._model, "prompt": text}, self._timeout_s
            )
            tokens = body.get("tokens")
            if not isinstance(tokens, list):
                raise TokenizerError("Tokenize response has no token list.")
            return len(tokens)
        except TokenizerError:
            self._dead = True
            return None


class HeuristicCounter:
    """chars/4 fallback for providers without a tokenize endpoint."""

    kind = "heuristic"

    def count(self, text: str) -> int | None:
        return max(1, len(text) // 4) if text else 0


def detect_counter(
    base_url: str, model: str = "", timeout_s: float = 10.0
) -> TokenCounter:
    """Pick a backend by URL hint. Counting still fails open per call."""
    lowered = base_url.lower()
    if "11434" in lowered or "ollama" in lowered:
        return OllamaCounter(base_url, model, timeout_s)
    if not lowered:
        return HeuristicCounter()
    return LlamaCppCounter(base_url, timeout_s)


def count_texts(counter: TokenCounter | None, texts: list[str]) -> int | None:
    """One backend call over joined text; None when unavailable."""
    if counter is None:
        return None
    return counter.count("\n".join(texts))
