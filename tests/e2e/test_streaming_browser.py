"""Controlled live streams through the real HTTP service and runtime (no inference weights)."""

import os
import threading
from pathlib import Path

import pytest

from relay import Agent, AgentConfig
from relay.models.action import NeedleResult
from relay.models.streaming import ModelDelta, check_cancelled
from relay.server import WorkspaceService, make_server

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect


@pytest.fixture()
def streaming_page(tmp_path, request):
    class Control:
        reasoning_ready = threading.Event()
        more_reasoning = threading.Event()
        output_ready = threading.Event()
        finish = threading.Event()
        closed_tool = threading.Event()
        allow_completion = threading.Event()
        calls = []

    control = Control()

    def wait_for(event, cancelled):
        while not event.wait(0.025):
            check_cancelled(cancelled)
        check_cancelled(cancelled)

    class Model:
        model_name = "Scripted stream"

        def stream(self, messages, *, cancelled=None):
            if any("Observation from tool" in message["content"] for message in messages):
                yield ModelDelta("<final>The calculator returned 42.</final>")
                return
            query = next(
                message["content"] for message in reversed(messages) if message["role"] == "user"
            )
            yield ModelDelta(kind="metadata", streamed=True)
            reasoning = "\n\n".join(
                f"Visible planning note {i}: inspect the request carefully." for i in range(60)
            )
            reasoning += '\n\n<img src=x onerror="window.injected=true">'
            yield ModelDelta(reasoning, kind="reasoning")
            control.reasoning_ready.set()
            wait_for(control.more_reasoning, cancelled)
            yield ModelDelta("\n\nThe next planning note has arrived.", kind="reasoning")
            if query == "tool":
                for text in ("<to", "ol>Calculate ", "6*7."):
                    yield ModelDelta(text)
                control.output_ready.set()
                wait_for(control.finish, cancelled)
                yield ModelDelta("</tool>")
                # A metadata packet flushes buffered text without ending generation.
                yield ModelDelta(kind="metadata", streamed=True)
                control.closed_tool.set()
                wait_for(control.allow_completion, cancelled)
            else:
                for text in ("<fi", "nal># Live result\n", "```python\n", "print('雪 🐍')\n"):
                    yield ModelDelta(text)
                yield ModelDelta(kind="metadata", streamed=True)
                control.output_ready.set()
                wait_for(control.finish, cancelled)
                if query == "failure":
                    raise RuntimeError("The scripted provider disconnected.")
                yield ModelDelta("```\n\nThe response is complete.</fi")
                yield ModelDelta("nal>")
            yield ModelDelta(kind="metadata", usage={"completion_tokens": 42}, finish_reason="stop")

    class Action:
        def translate(self, action, tools):
            control.calls.append(action)
            return NeedleResult(
                selected_tool="calculator", arguments={"expression": "6*7"}, confidence=0.99
            )

    config = AgentConfig(
        workspace_root=str(tmp_path),
        stream_buffer_ms=getattr(request, "param", 0),
        llm_api_key="server-only-key-do-not-expose",
    )
    service = WorkspaceService(config)
    service.agent_factory = lambda settings, run: Agent(
        service.run_config(settings), reasoning=Model(), action=Action()
    )
    server = make_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            executable = os.getenv("RELAY_BROWSER_EXECUTABLE") or p.chromium.executable_path
            if not Path(executable).is_file():
                pytest.skip("Install Chromium or set RELAY_BROWSER_EXECUTABLE")
            browser = p.chromium.launch(
                executable_path=executable,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--no-zygote"],
            )
            page = browser.new_page(
                viewport={"width": 1440, "height": 950}, reduced_motion="reduce"
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            expect(page.locator("#runtime-status")).to_contain_text("Runtime ready")
            yield page, control
            assert not errors, errors
            browser.close()
    finally:
        service.close()
        control.more_reasoning.set()
        control.finish.set()
        control.allow_completion.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def send(page, text):
    page.locator("#message-input").fill(text)
    page.locator("#send-message").click()


def test_reasoning_disclosure_resize_scroll_and_request_inspection_survive_updates(streaming_page):
    page, control = streaming_page
    send(page, "answer")
    expect(page.locator(".reasoning-card")).to_have_count(1)
    page.locator(".reasoning-card > summary").click()
    expect(page.locator(".model-output")).to_contain_text("Visible planning note 0")
    scroller = page.locator(".reasoning-card .model-scroll")
    initial_height = scroller.bounding_box()["height"]
    page.get_by_role("button", name="Expand", exact=True).click()
    assert scroller.bounding_box()["height"] > initial_height + 100
    page.get_by_role("button", name="Shrink", exact=True).click()
    scroller.evaluate("node => { node.scrollTop = 0; window.savedScroller = node; }")
    page.get_by_role("button", name="Model conversation", exact=True).click()
    expect(page.locator(".model-message")).to_have_count(
        4
    )  # system, user, raw reply, provider reasoning
    page.locator(".model-message").first.locator("summary").click()
    expect(page.locator(".model-message[open] pre")).to_contain_text("Available tools")
    scroller.evaluate("""node => {
      const text = node.querySelector('.model-output p').firstChild;
      const range = document.createRange(); range.setStart(text, 0); range.setEnd(text, 20);
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      window.selectedBeforeStream = selection.toString();
    }""")
    control.more_reasoning.set()
    expect(page.locator(".model-output")).to_contain_text("The next planning note")
    assert page.evaluate("getSelection().toString() === window.selectedBeforeStream")
    assert scroller.evaluate("node => node === window.savedScroller")
    assert scroller.evaluate("node => node.scrollTop") == 0
    assert page.locator(".reasoning-card").evaluate("node => node.open")
    expect(page.locator(".model-message").first).to_have_attribute("open", "")
    assert page.evaluate("window.injected") is None
    expect(page.locator(".model-output img")).to_have_count(0)
    page.locator(".reasoning-card > summary").click()
    control.finish.set()
    expect(page.locator(".message-meta")).to_contain_text("Completed")
    assert not page.locator(".reasoning-card").evaluate("node => node.open")


def test_partial_final_and_open_code_fence_render_before_completion_and_survive_reload(
    streaming_page,
):
    page, control = streaming_page
    send(page, "answer")
    control.more_reasoning.set()
    expect(page.locator(".streaming-answer")).to_contain_text("print('雪 🐍')")
    expect(page.locator(".streaming-answer .code-block")).to_have_count(1)
    expect(page.locator(".streaming-answer")).not_to_contain_text("<final>")
    expect(page.locator(".streaming-answer")).not_to_contain_text("```python")
    expect(page.locator(".message-meta")).to_have_count(0)
    page.reload()
    expect(page.locator(".streaming-answer")).to_contain_text("print('雪 🐍')")
    expect(page.locator(".reasoning-card")).to_have_count(1)
    control.finish.set()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("The response is complete.")
    expect(page.locator(".streaming-answer")).to_have_count(0)
    expect(page.locator(".assistant-body > .markdown .code-block")).to_have_count(1)
    expect(page.locator(".message-meta")).to_contain_text("Completed")


def test_draft_tool_waits_for_whole_response_and_merges_into_validated_card(streaming_page):
    page, control = streaming_page
    send(page, "tool")
    control.more_reasoning.set()
    expect(page.locator(".draft-tool")).to_have_count(1)
    page.locator(".draft-tool summary").click()
    expect(page.locator(".draft-tool pre")).to_contain_text("Calculate 6*7.")
    page.locator(".draft-tool").evaluate("node => window.draftCard = node")
    assert control.calls == []
    control.finish.set()
    assert control.closed_tool.wait(2)
    expect(page.locator(".draft-tool .tool-card-status")).to_contain_text("drafting")
    assert control.calls == []
    control.allow_completion.set()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("42")
    expect(page.locator(".draft-tool")).to_have_count(0)
    expect(page.locator(".tool-card")).to_have_count(1)
    assert page.locator(".tool-card").evaluate("node => node === window.draftCard && node.open")
    assert control.calls == ["Calculate 6*7."]
    expect(page.locator(".translator-card")).to_contain_text("buffered response")
    expect(page.locator(".message-meta")).to_contain_text("1 tool step")


@pytest.mark.parametrize("query", ["answer", "failure"])
def test_stop_and_stream_errors_retain_partial_output_without_claiming_success(
    streaming_page, query
):
    page, control = streaming_page
    send(page, query)
    control.more_reasoning.set()
    expect(page.locator(".streaming-answer")).to_contain_text("print(")
    if query == "answer":
        page.locator("#stop-run").click()
        expect(page.locator(".message-meta")).to_contain_text("Stopped")
    else:
        control.finish.set()
        expect(page.locator(".message-meta")).to_contain_text("Error")
    expect(page.locator(".partial-response")).to_contain_text("generation did not complete")
    expect(page.locator(".partial-response")).to_contain_text("print('雪 🐍')")
    expect(page.locator(".streaming-answer")).to_have_count(0)
    assert control.calls == []
    page.reload()
    expect(page.locator(".partial-response")).to_contain_text("print('雪 🐍')")


@pytest.mark.parametrize("streaming_page", [1000], indirect=True)
def test_paced_display_and_mobile_streaming_controls(streaming_page):
    page, control = streaming_page
    page.emulate_media(reduced_motion="no-preference")
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Switch to dark mode").click()
    send(page, "answer")
    page.locator(".reasoning-card > summary").click()
    expect(page.locator(".model-output")).to_contain_text("Visible planning note 0")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.get_by_role("button", name="Expand", exact=True).click()
    control.more_reasoning.set()
    expect(page.locator(".streaming-answer")).to_contain_text("print('雪 🐍')")
    control.finish.set()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("The response is complete.")
    expect(page.locator(".streaming-answer")).to_have_count(0)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    with page.expect_download() as download:
        page.locator(".inspect-run").click()
        page.get_by_role("button", name="Export trace", exact=True).click()
    trace = Path(download.value.path()).read_text()
    assert "server-only-key-do-not-expose" not in trace
    assert '"type": "model_delta"' in trace
