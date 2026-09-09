"""Real HTTP/SSE, live LangGraph events, termination, cancellation, and execution boundaries."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_runtime import Agent, AgentConfig
from agent_runtime.models.action import NeedleResult
from agent_runtime.models.streaming import ModelDelta


def packet(text="", *, channel="content", finish=None, **extras):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {channel: text}, "finish_reason": finish}],
                **extras,
            },
            ensure_ascii=False,
        )
        + "\r\n\r\n"
    ).encode()


@pytest.fixture()
def backend():
    class Backend:
        script = None
        content_type = "text/event-stream"
        requests = []
        connected = threading.Event()
        release = threading.Event()
        before_headers = False

    state = Backend()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((payload, dict(self.headers)))
            state.connected.set()
            if state.before_headers:
                state.release.wait(5)
            try:
                self.send_response(200)
                self.send_header("Content-Type", state.content_type)
                self.send_header("Connection", "close")
                self.end_headers()
                for data in state.script(payload):
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    state.release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


class Calculator:
    def __init__(self):
        self.calls = []

    def translate(self, action, tools):
        self.calls.append(action)
        return NeedleResult(
            selected_tool="calculator", arguments={"expression": "6*7"}, confidence=0.99
        )


def make_agent(backend, **overrides):
    action = Calculator()
    config = AgentConfig(
        llm_base_url=backend.url,
        llm_model="stream-test",
        llm_api_key="private-test-key",
        **overrides,
    )
    return Agent(config, action=action), action


def test_real_tokens_arrive_before_generation_finishes_and_tools_wait_for_eof(backend):
    def script(payload):
        if any("Observation from tool" in message["content"] for message in payload["messages"]):
            yield packet("<final>The result is ")
            yield packet("42.</final>", finish="stop", usage={"completion_tokens": 9})
        else:
            yield b": heartbeat\r\n\r\n"
            yield packet("An exposed planning note.", channel="reasoning_content")
            for text in ("<to", "ol>Calculate ", "6*7.</to", "ol>"):
                yield packet(text)
            assert backend.release.wait(3), (
                "No streaming event was delivered before inference ended."
            )
            yield packet(finish="stop")
        yield b"data: [DONE]\r\n\r\n"

    backend.script = script
    agent, translator = make_agent(backend, stream_flush_ms=10)
    events = []
    for event in agent.stream("Multiply six and seven"):
        events.append(event)
        if event["type"] == "model_delta" and not backend.release.is_set():
            assert translator.calls == []
            backend.release.set()
    assert events[-1]["state"]["final_answer"] == "The result is 42."
    assert events[-1]["state"]["step_count"] == 1
    assert any(e["type"] == "model_delta" and e["channel"] == "reasoning" for e in events)
    first_end = next(i for i, e in enumerate(events) if e["type"] == "model_end")
    assert first_end < next(i for i, e in enumerate(events) if e["type"] == "action")
    assert (
        next(e for e in events if e["type"] == "model_start")["input_messages"]
        == backend.requests[0][0]["messages"]
    )
    assert any(e.get("usage", {}).get("completion_tokens") == 9 for e in events)
    assert "private-test-key" not in json.dumps(events)
    assert all(
        request["stream"] is True and "tools" not in request for request, _ in backend.requests
    )


@pytest.mark.parametrize(
    "ending",
    [
        b"",
        packet(finish="length"),
        b'data: {"error":{"message":"private-test-key"}}\n\n',
        b"data: not-json\n\n",
    ],
)
def test_interrupted_or_invalid_stream_never_executes_even_a_closed_tool(backend, ending):
    backend.script = lambda payload: iter([packet("<tool>Calculate 6*7.</tool>"), ending])
    agent, translator = make_agent(backend)
    events = list(agent.stream("Calculate"))
    assert events[-1]["state"]["status"] == "ERROR"
    assert translator.calls == []
    assert any(event["type"] == "model_end" and event["status"] == "error" for event in events)
    assert "private-test-key" not in events[-1]["state"]["final_answer"]


def test_a_later_final_still_overrides_an_earlier_tool_preview(backend):
    backend.script = lambda payload: iter(
        [
            packet("<tool>Calculate 6*7.</tool>"),
            packet("<final>I can answer without executing.</final>", finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    agent, translator = make_agent(backend)
    events = list(agent.stream("Answer"))
    assert events[-1]["state"]["final_answer"] == "I can answer without executing."
    assert translator.calls == []
    assert any(
        part["kind"] == "tool"
        for event in events
        if event["type"] == "model_delta"
        for part in event["parts"]
    )


def test_reasoning_channel_and_inline_thoughts_cannot_smuggle_tools(backend):
    backend.script = lambda payload: iter(
        [
            packet("<tool>Calculate 6*7.</tool>", channel="reasoning"),
            packet("<think><tool>Another thought, not an action.</tool></think>"),
            packet("<final>Safe answer.</final>", finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    agent, translator = make_agent(backend)
    events = list(agent.stream("Think, then answer"))
    assert events[-1]["state"]["final_answer"] == "Safe answer."
    assert translator.calls == []
    assert not any(
        "Another thought" in message["content"] for message in events[-1]["state"]["messages"]
    )


def test_json_fallback_is_labelled_buffered_without_a_second_request(backend):
    backend.content_type = "application/json"
    backend.script = lambda payload: iter(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "<final>Buffered.</final>"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ]
    )
    agent, _ = make_agent(backend)
    events = list(agent.stream("Test"))
    assert events[-1]["state"]["final_answer"] == "Buffered."
    assert len(backend.requests) == 1
    assert any(event["type"] == "model_status" and event["streamed"] is False for event in events)


def test_cancel_interrupts_a_stalled_socket_after_a_partial_response(backend):
    cancelled = threading.Event()

    def script(payload):
        yield packet("<final>Partial response.")
        backend.release.wait(5)
        yield packet(" Do not deliver this.</final>", finish="stop")

    backend.script = script
    agent, translator = make_agent(backend)
    stopped_at = None
    events = []
    for event in agent.stream("Test cancellation", cancelled=cancelled.is_set):
        events.append(event)
        if event["type"] == "model_delta" and stopped_at is None:
            stopped_at = time.monotonic()
            cancelled.set()
    assert time.monotonic() - stopped_at < 1.0
    assert events[-1]["state"]["status"] == "CANCELLED" and translator.calls == []
    assert "Do not deliver" not in json.dumps(events)


def test_cancel_can_interrupt_waiting_for_response_headers(backend):
    backend.before_headers = True
    backend.script = lambda payload: iter([packet("<final>Too late.</final>", finish="stop")])
    agent, translator = make_agent(backend)
    cancelled = threading.Event()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(agent.run("Wait", cancelled=cancelled.is_set))
    )
    worker.start()
    assert backend.connected.wait(2)
    cancelled.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result[0]["status"] == "CANCELLED" and not translator.calls


def test_streaming_can_be_disabled_and_input_capture_is_optional(backend):
    backend.content_type = "application/json"
    backend.script = lambda payload: iter(
        [json.dumps({"choices": [{"message": {"content": "Done."}}]}).encode()]
    )
    agent, _ = make_agent(backend, llm_stream=False, capture_model_inputs=False)
    events = list(agent.stream("Don't capture model input"))
    assert events[-1]["state"]["status"] == "COMPLETED"
    assert backend.requests[0][0]["stream"] is False
    start = next(event for event in events if event["type"] == "model_start")
    assert start["input_messages"] == [] and start["streamed"] is False


def test_output_limit_includes_provider_reasoning_and_stops_without_translating():
    class Model:
        def stream(self, messages):
            yield ModelDelta("x" * 100, kind="reasoning")
            yield ModelDelta("<tool>Calculate 6*7.</tool>")

    action = Calculator()
    events = list(
        Agent(AgentConfig(max_model_output_chars=110), reasoning=Model(), action=action).stream(
            "Test"
        )
    )
    assert events[-1]["state"]["status"] == "ERROR" and not action.calls
    assert "max_model_output_chars" in events[-1]["state"]["final_answer"]


def test_generate_only_adapters_still_work_and_are_marked_buffered():
    class Model:
        def generate(self, messages):
            return "<final>Legacy adapter.</final>"

    events = list(Agent(reasoning=Model(), action=Calculator()).stream("Test"))
    assert events[-1]["state"]["final_answer"] == "Legacy adapter."
    assert next(event for event in events if event["type"] == "model_start")["streamed"] is False


def test_small_final_batch_flushes_on_time_while_generator_waits():
    release = threading.Event()

    class Model:
        def stream(self, messages):
            yield ModelDelta("Planning note", kind="reasoning")
            yield ModelDelta("<tool>Calculate 6*7.")
            assert release.wait(2), "Small token batch waited for the provider instead of flushing."
            yield ModelDelta("</tool><final>Final wins.</final>")

    events = []
    for event in Agent(
        AgentConfig(stream_flush_ms=25), reasoning=Model(), action=Calculator()
    ).stream("Test"):
        events.append(event)
        if event["type"] == "model_delta" and "Calculate" in event["delta"]:
            release.set()
    assert events[-1]["state"]["final_answer"] == "Final wins."
    assert not any(thread.name == "model-delta-buffer" for thread in threading.enumerate())


def test_custom_stream_finish_reason_cannot_bypass_incomplete_output_guard():
    class Model:
        def stream(self, messages):
            yield ModelDelta("<tool>Calculate 6*7.</tool>")
            yield ModelDelta(kind="metadata", finish_reason="length")

    action = Calculator()
    state = Agent(reasoning=Model(), action=action).run("Test")
    assert state["status"] == "ERROR" and action.calls == []


def test_utf8_bytes_and_multiline_sse_survive_network_fragmentation(backend):
    message = "<final>नमस्ते 雪 🐍</final>"
    encoded = packet(message, finish="stop")
    # Whitespace/newlines outside the JSON string are legal multiline SSE data.
    encoded = encoded.replace(b'"choices":', b'\r\ndata: "choices":')
    backend.script = lambda payload: (bytes([byte]) for byte in encoded + b"data: [DONE]\n\n")
    agent, _ = make_agent(backend)
    assert agent.run("Read Unicode")["final_answer"] == "नमस्ते 雪 🐍"


def test_native_tool_call_deltas_are_not_executable(backend):
    body = {"choices": [{"delta": {"tool_calls": [{"name": "calculator"}]}}]}
    backend.script = lambda payload: iter(
        [b"data: " + json.dumps(body).encode() + b"\n\n", b"data: [DONE]\n\n"]
    )
    agent, action = make_agent(backend)
    state = agent.run("Test")
    assert state["status"] == "ERROR" and not action.calls
    assert "native tool calls" in state["final_answer"]


def test_reasoning_stream_has_a_wall_clock_deadline_even_with_heartbeats(backend):
    def script(payload):
        for _ in range(30):
            yield b": still here\n\n"
            if backend.release.wait(0.05):
                return

    backend.script = script
    agent, action = make_agent(backend, llm_timeout_s=0.3)
    started = time.monotonic()
    state = agent.run("Wait for output")
    assert state["status"] == "ERROR" and not action.calls
    assert time.monotonic() - started < 1.0
    assert "timed out" in state["final_answer"]


def test_oversized_stream_is_bounded_before_execution(backend, monkeypatch):
    monkeypatch.setattr("agent_runtime.models.streaming.MAX_STREAM_BYTES", 20)
    backend.script = lambda payload: iter([packet("<tool>Calculate 6*7.</tool>", finish="stop")])
    agent, action = make_agent(backend)
    state = agent.run("Test")
    assert state["status"] == "ERROR" and not action.calls
    assert "size limit" in state["final_answer"]
