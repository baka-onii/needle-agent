"""Unified token counting: server tokenizers first, heuristic fallback."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_runtime.models.tokens import (
    HeuristicCounter,
    LlamaCppCounter,
    OllamaCounter,
    count_texts,
    detect_counter,
)


class _TokenHandler(BaseHTTPRequestHandler):
    mode = "llamacpp"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize" and self.server.mode == "llamacpp":
            text = payload["content"]
            body = {"tokens": list(range(len(text.split())))}
        elif self.path == "/api/tokenize" and self.server.mode == "ollama":
            text = payload["prompt"]
            body = {"tokens": list(range(len(text.split())))}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def servers():
    instances = []
    for mode in ("llamacpp", "ollama"):
        server = HTTPServer(("127.0.0.1", 0), _TokenHandler)
        server.mode = mode
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        instances.append(server)
    yield instances
    for server in instances:
        server.shutdown()
        server.server_close()


def test_llamacpp_counts_words(servers):
    counter = LlamaCppCounter(f"http://127.0.0.1:{servers[0].server_port}/v1")
    assert counter.kind == "llamacpp"
    assert counter.count("one two three") == 3
    assert counter.count("") == 0


def test_ollama_counts_words(servers):
    counter = OllamaCounter(f"http://127.0.0.1:{servers[1].server_port}", "m")
    assert counter.kind == "ollama"
    assert counter.count("one two three four") == 4


def test_backends_fail_open_to_none():
    dead = LlamaCppCounter("http://127.0.0.1:1")
    assert dead.count("hello world") is None
    # Stays dead: no repeated slow timeouts.
    assert dead.count("hello world") is None
    assert OllamaCounter("http://127.0.0.1:1", "m").count("hi") is None
    assert count_texts(None, ["a"]) is None


def test_heuristic_counts_chars():
    counter = HeuristicCounter()
    assert counter.count("abcd") == 1
    assert counter.count("a" * 40) == 10
    assert counter.count("") == 0
    assert count_texts(counter, ["ab", "cdef"]) == 1


def test_detect_counter_by_url(servers):
    assert (
        detect_counter(f"http://127.0.0.1:{servers[0].server_port}/v1").kind
        == "llamacpp"
    )
    assert (
        detect_counter("http://127.0.0.1:11434", "m").kind == "ollama"
    )
    assert detect_counter("", "m").kind == "heuristic"


def test_count_texts_joins(servers):
    counter = LlamaCppCounter(f"http://127.0.0.1:{servers[0].server_port}")
    assert count_texts(counter, ["one two", "three"]) == 3


def test_context_manager_trims_to_token_budget(tmp_path):
    from agent_runtime import AgentConfig
    from agent_runtime.context.manager import ContextManager

    class StubCounter:
        kind = "stub"

        def count(self, text):
            return 100000 if "huge" in text else 1

    config = AgentConfig(workspace_root=str(tmp_path), max_context_tokens=10)
    manager = ContextManager(config, "sys", StubCounter())
    built = manager.build(
        [
            {"role": "user", "content": "original"},
            {"role": "user", "content": "huge observation", "kind": "observation"},
            {"role": "user", "content": "current request"},
        ]
    )
    assert not any("huge" in str(message.get("content", "")) for message in built)
    assert manager.last_tokens is not None and manager.last_tokens <= 10
    assert manager.last_chars > 0
