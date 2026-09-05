"""P5: Needle adapter mapping, system prompt, llama-server client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_runtime.config import AgentConfig
from agent_runtime.models.needle import NeedleActionModel
from agent_runtime.models.reasoning import LlamaServerReasoningModel, build_system_prompt
from agent_runtime.tools.base import Tool
from agent_runtime.tools.registry import create_default_registry


class StubNeedleClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def complete(self, action: str, max_new_tokens: int = 256) -> dict:
        assert action and max_new_tokens
        return self.response


def _tools():
    return create_default_registry(AgentConfig()).list()


def test_translate_call() -> None:
    tools = _tools()
    client = StubNeedleClient(
        {
            "type": "call",
            "function_calls": [{"name": "search_files", "arguments": {"query": "auth"}}],
            "confidence": 0.94,
        }
    )
    result = NeedleActionModel(tools, client=client).translate("find auth", tools)
    assert result.selected_tool == "search_files"
    assert result.arguments == {"query": "auth"}
    assert result.confidence == 0.94
    assert result.rankings[0].tool_name == "search_files"


def test_translate_refusal_maps_to_no_selection() -> None:
    tools = _tools()
    for response in (
        {"type": "respond", "function_calls": [], "confidence": 0.9},
        {"type": "call", "function_calls": [], "confidence": 0.3},
    ):
        result = NeedleActionModel(tools, client=StubNeedleClient(response)).translate(
            "off topic", tools
        )
        assert result.selected_tool is None
        assert result.rankings == []


def test_translate_resets_session_each_turn() -> None:
    tools = _tools()
    client = StubNeedleClient({"type": "respond", "function_calls": [], "confidence": 0.0})
    adapter = NeedleActionModel(tools, client=client)
    adapter.translate("a", tools)
    adapter.translate("b", tools)
    assert client.resets == 2


def test_translate_rejects_toolset_mismatch() -> None:
    tools = _tools()
    adapter = NeedleActionModel(tools, client=StubNeedleClient({"type": "respond"}))
    with pytest.raises(ValueError):
        adapter.translate("x", tools[:3])


def test_system_prompt_lists_tools_and_tags() -> None:
    prompt = build_system_prompt(_tools())
    assert "<tool>" in prompt and "<final>" in prompt
    for name in ("read_file", "search_files", "calculator", "ask_user"):
        assert name in prompt


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.mode == "error":
            self.send_response(500)
            self.end_headers()
            return
        body = {"choices": [{"message": {"content": "<final>hi</final>"}}]}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # noqa: ANN002, ANN003
        pass


@pytest.fixture()
def server_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_llama_server_generate(server_url: str) -> None:
    _Handler.mode = "ok"
    model = LlamaServerReasoningModel(base_url=server_url, timeout_s=5.0)
    out = model.generate([{"role": "user", "content": "hi"}])
    assert out == "<final>hi</final>"


def test_llama_server_error_raises(server_url: str) -> None:
    _Handler.mode = "error"
    model = LlamaServerReasoningModel(base_url=server_url, timeout_s=5.0)
    with pytest.raises(RuntimeError):
        model.generate([{"role": "user", "content": "hi"}])
    _Handler.mode = "ok"


def test_tool_schema_is_needle_compatible() -> None:
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    schema = tool.needle_schema()
    assert set(schema) == {"name", "description", "parameters"}


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {"type": "call", "function_calls": "not a list"},
        {"type": "call", "function_calls": [{"name": "get_time", "arguments": "garbage"}]},
        {"type": "call", "function_calls": [{"name": "get_time", "arguments": None}]},
        {"type": "call", "function_calls": [None]},
        {
            "type": "call",
            "function_calls": [{"name": "get_time", "arguments": {}}],
            "confidence": float("nan"),
        },
    ],
)
def test_malformed_needle_output_is_not_repaired(response) -> None:
    from agent_runtime.models.action import ActionOutputError
    from agent_runtime.models.needle import parse_needle_response

    with pytest.raises(ActionOutputError):
        parse_needle_response(response)


def test_uncalibrated_needle_output_fails_closed() -> None:
    from agent_runtime.models.needle import parse_needle_response

    result = parse_needle_response(
        {
            "type": "call",
            "function_calls": [{"name": "get_time", "arguments": {}}],
            "confidence": None,
        }
    )
    assert result.confidence == 0.0


def test_v1_base_url_not_duplicated(server_url: str) -> None:
    model = LlamaServerReasoningModel(base_url=server_url + "/v1/")
    assert model._base == server_url + "/v1"
    assert "hi" in model.generate([{"role": "user", "content": "hello", "kind": "request"}])
