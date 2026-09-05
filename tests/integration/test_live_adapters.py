"""Live adapter contracts exercised against a local HTTP server and a fake C client.

No demo planner here: requests go through the real reasoning HTTP client and the
real NeedleActionModel envelope parser. Model inference itself requires weights.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_runtime import Agent, AgentConfig
from agent_runtime.models.needle import NeedleActionModel
from agent_runtime.models.reasoning import OpenAICompatibleReasoningModel
from agent_runtime.tools.registry import create_default_registry


@pytest.fixture()
def reasoning_server():
    requests = []
    redirect = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), payload))
            if redirect:
                self.send_response(302)
                self.send_header("Location", redirect[0])
                self.end_headers()
                return
            observed = any("Observation from tool" in m["content"] for m in payload["messages"])
            text = "<final>42</final>" if observed else "<tool>Calculate 6 * 7.</tool>"
            data = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            requests.append((self.path, dict(self.headers), None))
            data = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests, redirect
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_reasoning_and_needle_single_turn_contract(reasoning_server):
    base, requests, _ = reasoning_server

    class CClient:
        resets = 0
        calls = []

        def reset(self):
            self.resets += 1

        def complete(self, action, max_new_tokens):
            self.calls.append(action)
            assert max_new_tokens == 256
            return {
                "type": "call",
                "confidence": 0.92,
                "function_calls": [{"name": "calculator", "arguments": {"expression": "6*7"}}],
            }

    config = AgentConfig(
        llm_base_url=base + "/v1", llm_model="test-model", llm_api_key="test-only-key"
    )
    registry = create_default_registry(config)
    client = CClient()
    action = NeedleActionModel(registry.list(), client=client)
    agent = Agent(config, action=action, registry=registry)
    state = agent.run("Multiply six and seven")
    assert state["status"] == "COMPLETED"
    assert state["final_answer"] == "42"
    assert state["step_count"] == 1
    assert client.resets == 1 and client.calls == ["Calculate 6 * 7."]
    assert len(requests) == 2
    for path, headers, payload in requests:
        assert path == "/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-only-key"
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["model"] == "test-model"
        assert all(set(m) == {"role", "content"} for m in payload["messages"])


def test_connection_check_uses_model_list(reasoning_server):
    base, requests, _ = reasoning_server
    assert OpenAICompatibleReasoningModel(base).check_connection() == ["test-model"]
    assert requests[-1][0] == "/v1/models"


def test_auth_is_not_forwarded_on_redirect(reasoning_server):
    base, requests, redirect = reasoning_server
    redirect.append(base + "/should-not-be-requested")
    model = OpenAICompatibleReasoningModel(base, api_key="test-only-key")
    with pytest.raises(RuntimeError, match="HTTP 302") as error:
        model.generate([{"role": "user", "content": "hi"}])
    assert len(requests) == 1
    assert "test-only-key" not in str(error.value)
