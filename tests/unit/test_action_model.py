"""P5: Needle adapter mapping, system prompt, llama-server client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from relay.config import AgentConfig
from relay.models.needle import NeedleActionModel
from relay.models.reasoning import LlamaServerReasoningModel, build_system_prompt
from relay.tools.base import Tool
from relay.tools.registry import create_default_registry


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
    from relay.models.action import ActionOutputError
    from relay.models.needle import parse_needle_response

    with pytest.raises(ActionOutputError):
        parse_needle_response(response)


def test_uncalibrated_needle_output_fails_closed() -> None:
    from relay.models.needle import parse_needle_response

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


def test_multiple_native_calls_are_rejected_as_non_atomic():
    from relay.models.action import ActionOutputError
    from relay.models.needle import parse_needle_response

    with pytest.raises(ActionOutputError, match="multiple calls"):
        parse_needle_response(
            {
                "type": "call",
                "confidence": 0.99,
                "function_calls": [
                    {"name": "read_directory", "arguments": {"path": "."}},
                    {"name": "read_file", "arguments": {"path": "a.txt"}},
                ],
            }
        )


def test_needle_receives_configured_generation_budget_and_default_instructions():
    class Client(StubNeedleClient):
        def complete(self, action, max_new_tokens):
            assert max_new_tokens == 4096
            return self.response

    tools = _tools()
    adapter = NeedleActionModel(
        tools,
        client=Client({"type": "respond", "confidence": 0.1}),
        max_new_tokens=4096,
    )
    adapter.translate("hello", tools)
    assert "not a planner" in adapter._system
    assert "full finished file content" in adapter._system


class _CannedFunctionGemma:
    """FunctionGemma adapter with the HTTP layer replaced by canned outputs."""

    def __init__(self, tools, outputs):
        from relay.models.functiongemma import FunctionGemmaActionModel

        self.adapter = FunctionGemmaActionModel.__new__(FunctionGemmaActionModel)
        FunctionGemmaActionModel.__init__(self.adapter, tools)
        self.outputs = list(outputs)
        self.prompts = []

    def translate(self, action, tools):
        text, finished = self.outputs.pop(0)
        self.adapter._complete = lambda prompt: self._record(prompt, text, finished)
        return self.adapter.translate(action, tools)

    def _record(self, prompt, text, finished):
        self.prompts.append(prompt)
        return text, finished


def _fg_tools():
    return _tools()


def test_functiongemma_translates_single_call() -> None:
    tools = _fg_tools()
    canned = _CannedFunctionGemma(
        tools,
        [("<start_function_call>call:calculator{expression:<escape>6*7<escape>}", True)],
    )
    result = canned.translate("Calculate 6 * 7.", tools)
    assert result.selected_tool == "calculator"
    assert result.arguments == {"expression": "6*7"}
    assert result.confidence == 1.0
    assert "<start_function_declaration>" in canned.prompts[0]
    assert "Calculate 6 * 7." in canned.prompts[0]


def test_functiongemma_accepts_server_consumed_stop_token() -> None:
    tools = _fg_tools()
    canned = _CannedFunctionGemma(
        tools,
        [("<start_function_call>call:get_time{timezone:<escape>UTC<escape>}", True)],
    )
    result = canned.translate("Get the time.", tools)
    assert result.selected_tool == "get_time"


def test_functiongemma_rejects_truncated_and_multiple_calls() -> None:
    from relay.models.action import ActionOutputError

    tools = _fg_tools()
    for text, finished in (
        ("<start_function_call>call:calculator{expression:<escape>6*7", False),
        ("no call here", True),
        (
            "<start_function_call>call:a{}<end_function_call>"
            "<start_function_call>call:b{}<end_function_call>",
            True,
        ),
        ("<start_function_call>call:nope{arg:<escape>x<escape>}<end_function_call>", True),
    ):
        canned = _CannedFunctionGemma(tools, [(text, finished)])
        with pytest.raises(ActionOutputError):
            canned.translate("Do something.", tools)


def test_functiongemma_parses_bare_none_and_commas_in_values() -> None:
    from relay.models.functiongemma import parse_function_call

    name, arguments = parse_function_call(
        "<start_function_call>call:search_files{query:<escape>a, b<escape>,"
        "path:<escape>.<escape>}<end_function_call>"
    )
    assert (name, arguments) == ("search_files", {"query": "a, b", "path": "."})


def test_functiongemma_render_parse_round_trip() -> None:
    from relay import AgentConfig
    from relay.models.functiongemma import (
        _coerce_arguments,
        parse_function_call,
        render_function_call,
    )
    from relay.tools.registry import create_default_registry

    tools = {t.name: t for t in create_default_registry(AgentConfig()).list()}
    cases = [
        ("get_time", {}),
        ("read_file", {"path": "src/auth.py"}),
        ("git_log", {"limit": 5, "path": "."}),
        ("search_files", {"query": "a, b {c}", "path": "."}),
        ("get_time", {"timezone": None}),
        ("run_python", {"code": "print('hi')", "timeout_s": 30}),
    ]
    for name, arguments in cases:
        parsed = parse_function_call(render_function_call(name, arguments))
        assert parsed[0] == name
        assert _coerce_arguments(tools[name], parsed[1]) == arguments


class ExplodingNeedleClient:
    """Fails if touched: well-formed writes must never reach the engine."""

    resets = 0

    def reset(self) -> None:
        self.resets += 1

    def complete(self, action: str, max_new_tokens: int) -> dict:
        raise AssertionError("engine must not be consulted for a clean write action")


def test_clean_write_action_copies_payload_without_engine() -> None:
    tools = _tools()
    client = ExplodingNeedleClient()
    result = NeedleActionModel(tools, client=client).translate(
        'Use write_file to write the file "test.txt" with this exact text:\n'
        "```text\ntest Hello World\n```",
        tools,
    )
    assert result.selected_tool == "write_file"
    assert result.arguments == {"path": "test.txt", "content": "test Hello World"}
    assert result.confidence == 1.0
    assert client.resets == 0


def test_write_without_literal_payload_falls_through_to_engine() -> None:
    tools = _tools()
    client = StubNeedleClient({"type": "respond", "function_calls": [], "confidence": 0.4})
    result = NeedleActionModel(tools, client=client).translate(
        'Use write_file to write the file "test.txt".', tools
    )
    assert result.selected_tool is None
    assert client.resets == 1


def test_non_write_action_still_uses_engine() -> None:
    tools = _tools()
    client = StubNeedleClient(
        {
            "type": "call",
            "function_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}],
            "confidence": 0.9,
        }
    )
    result = NeedleActionModel(tools, client=client).translate("Calculate 1 + 1.", tools)
    assert result.selected_tool == "calculator"
    assert client.resets == 1
