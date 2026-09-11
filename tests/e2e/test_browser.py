"""Optional real-browser tests: uv sync --group browser; playwright install chromium.

Set RELAY_BROWSER_EXECUTABLE to use an existing Chromium instead. No live LLM
or API key is needed; the demo runs the same graph and real sandboxed tools.
"""

import os
import shutil
import threading
from pathlib import Path

import pytest

from relay import Agent, AgentConfig
from relay.server import WorkspaceService, make_server

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect


@pytest.fixture()
def browser_page(tmp_path, request):
    root = tmp_path / "workspace"
    shutil.copytree(Path(__file__).parents[2] / "examples" / "workspace", root)
    (root / "untrusted.md").write_text(
        '<img src=x onerror="window.injected=true"><script>window.injected=true</script>'
    )
    service = WorkspaceService(AgentConfig(workspace_root=str(root)), demo=True)
    if getattr(request, "param", None) == "selection-review":
        from relay.models.action import NeedleResult

        class ReviewReasoning:
            turn = 0

            def generate(self, messages):
                if self.turn == 1:
                    assert (
                        "Is the highest-ranked available tool 'calculator' correct?"
                        in messages[-1]["content"]
                    )
                response = [
                    "<tool>Work out six times seven.</tool>",
                    "<tool>Use calculator to calculate 6*7.</tool>",
                    "<final>The result is 42.</final>",
                ][self.turn]
                self.turn += 1
                return response

        class ReviewTranslator:
            turn = 0

            def translate(self, action, tools):
                self.turn += 1
                return NeedleResult(
                    selected_tool="calculator",
                    arguments={"expression": "6*7"},
                    confidence=0.2 if self.turn == 1 else 0.99,
                )

        service.agent_factory = lambda settings, run: Agent(
            service.run_config(settings),
            reasoning=ReviewReasoning(),
            action=ReviewTranslator(),
        )
    server = make_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            executable = os.environ.get("RELAY_BROWSER_EXECUTABLE") or p.chromium.executable_path
            if not Path(executable).is_file():
                pytest.skip("Install Chromium with: playwright install chromium")
            browser = p.chromium.launch(
                executable_path=executable,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--no-zygote"],
            )
            page = browser.new_page(
                viewport={"width": 1440, "height": 950}, reduced_motion="reduce"
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            expect(page.locator("#runtime-status")).to_contain_text("Runtime ready")
            yield page, root
            assert not errors, errors
            browser.close()
    finally:
        service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def send(page, text):
    page.locator("#message-input").fill(text)
    page.locator("#send-message").click()


def test_search_inspection_reload_and_followup(browser_page):
    page, _ = browser_page
    expect(page.locator("#mode-badge")).to_contain_text("Offline demo")
    page.locator('[data-prompt="Find the authentication implementation"]').click()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("src/auth.py")
    expect(page.locator(".message-meta")).to_contain_text("2 tool steps")
    expect(page.locator(".phase-list .done")).to_have_count(10)
    page.locator(".tool-card summary").first.click()
    expect(page.locator(".tool-card[open]")).to_contain_text("authentication")
    expect(page.locator(".tool-card[open]")).to_contain_text("Synthetic demo score")
    page.reload()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("src/auth.py")
    send(page, "Calculate 24 * 18 + 120")
    expect(page.locator(".assistant-body > .markdown").last).to_contain_text("552")
    expect(page.locator(".message.user")).to_have_count(2)
    page.locator('[data-view="history"]').click()
    expect(page.locator(".history-row:not(.table-header)")).to_have_count(2)


def test_question_and_write_survive_reload(browser_page):
    page, root = browser_page
    page.locator('[data-prompt="Create a note"]').click()
    expect(page.locator(".pending-card")).to_contain_text("What would you like me to write")
    page.reload()
    expect(page.locator(".pending-card")).to_contain_text("What would you like me to write")
    send(page, "A useful note from the browser.")
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("Wrote and verified")
    assert (root / "note.txt").read_text() == "A useful note from the browser."
    expect(page.locator(".message-meta")).to_contain_text("3 tool steps")
    with page.expect_download() as download:
        page.get_by_role("button", name="Export trace").click()
    assert download.value.suggested_filename.startswith("relay-run-")


def test_write_completes_without_approval_and_cancellation(browser_page):
    page, root = browser_page
    send(page, 'Write "goes straight through" to direct.txt')
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("Wrote and verified")
    expect(page.locator(".message-meta")).to_contain_text("tool steps")
    assert (root / "direct.txt").exists()
    page.locator("#new-session").click()
    send(page, "Create a note")
    expect(page.locator(".pending-card")).to_be_visible()
    page.locator("#stop-run").click()
    expect(page.locator(".message-meta")).to_contain_text("Stopped")
    assert not (root / "note.txt").exists()


def test_files_tools_and_settings(browser_page):
    page, _ = browser_page
    page.locator('[data-view="workspace"]').click()
    expect(page.locator(".file-row")).to_have_count(5)
    page.locator('[data-file-path="src"]').click()
    page.locator('[data-file-path="src/auth.py"]').click()
    expect(page.locator(".file-content")).to_contain_text("def authenticate_user")
    page.locator('[data-view="tools"]').click()
    expect(page.locator(".tool-catalog-card")).to_have_count(33)
    page.locator('.tool-catalog-card[data-tool="write_file"]').click()
    expect(page.locator("#detail-dialog")).to_be_visible()
    expect(page.locator(".parameter-table")).to_contain_text("content")
    page.locator("#detail-dialog .dialog-close").click()
    page.locator("#open-settings").click()
    page.locator("#test-connection").click()
    expect(page.locator("#connection-result")).to_contain_text("real tools")
    page.locator(".advanced-settings summary").click()
    page.locator("#max-steps").fill("5")
    page.locator("#save-settings").click()
    expect(page.locator("#settings-dialog")).not_to_be_visible()
    page.locator('[data-view="playground"]').click()
    expect(page.locator("#step-limit")).to_contain_text("5 steps")


def test_live_backend_failure_is_not_silently_demo(browser_page):
    page, _ = browser_page
    page.locator("#open-settings").click()
    page.locator('input[name="mode"][value="live"]').check()
    page.locator("#base-url").fill("http://127.0.0.1:1/v1")
    page.locator("#model-name").fill("unavailable-model")
    page.locator("#save-settings").click()
    expect(page.locator("#mode-badge")).to_contain_text("Live models")
    send(page, "Calculate 2+2")
    expect(page.locator(".message-meta")).to_contain_text("Error")
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("unreachable")
    expect(page.locator(".tool-card")).to_have_count(0)


def test_untrusted_file_is_text_not_html(browser_page):
    page, _ = browser_page
    send(page, "Read the file untrusted.md")
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("onerror")
    assert page.evaluate("window.injected") is None
    expect(page.locator(".conversation img")).to_have_count(0)
    expect(page.locator(".conversation script")).to_have_count(0)


def test_mobile_navigation_and_chat(browser_page):
    page, _ = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    expect(page.locator("#menu-button")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator('[data-prompt="Calculate 24 * 18 + 120"]').click()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("552")
    page.locator("#menu-button").click()
    page.locator('[data-view="tools"]').click()
    expect(page.locator("#tools-view")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator("#mode-badge").click()
    expect(page.locator("#settings-dialog")).to_be_visible()
    assert page.locator("#settings-dialog").bounding_box()["width"] < 390


def test_theme_respects_system_then_persists_explicit_choice(browser_page):
    page, _ = browser_page
    page.emulate_media(color_scheme="dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(page.get_by_role("button", name="Switch to light mode")).to_be_visible()
    assert (
        page.locator(".composer").evaluate("e => getComputedStyle(e).backgroundColor")
        == "rgb(28, 42, 34)"
    )
    page.get_by_role("button", name="Switch to light mode").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.reload()
    expect(page.locator("#runtime-status")).to_contain_text("Runtime ready")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.get_by_role("button", name="Switch to dark mode").click()
    page.reload()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.locator("#open-settings").click()
    assert (
        page.locator("#settings-dialog").evaluate("e => getComputedStyle(e).backgroundColor")
        == "rgb(28, 42, 34)"
    )


def test_delete_conversations_cancels_cleanly_and_survives_reload(browser_page):
    page, _ = browser_page
    send(page, "Calculate 2+2")
    expect(page.locator(".message-meta")).to_contain_text("Completed")
    page.locator("#new-session").click()
    send(page, "Calculate 3+3")
    expect(page.locator(".message-meta")).to_contain_text("Completed")
    expect(page.locator(".recent-row")).to_have_count(2)
    page.get_by_role("button", name="Delete conversation: Calculate 2+2", exact=True).click()
    page.locator("#delete-dialog").get_by_role("button", name="Cancel", exact=True).click()
    expect(page.locator(".recent-row")).to_have_count(2)
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("6")
    page.get_by_role("button", name="Delete conversation: Calculate 2+2", exact=True).click()
    page.locator("#confirm-delete").click()
    expect(page.locator("#delete-dialog")).not_to_be_visible()
    expect(page.locator(".recent-row")).to_have_count(1)
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("6")
    page.locator('[data-view="history"]').click()
    expect(page.locator(".history-row:not(.table-header)")).to_have_count(1)
    page.reload()
    expect(page.locator(".recent-row")).to_have_count(1)
    page.locator(".recent-delete").click()
    page.locator("#confirm-delete").click()
    expect(page.locator("#welcome")).to_be_visible()
    expect(page.locator(".recent-row")).to_have_count(0)
    expect(page.locator(".message")).to_have_count(0)
    page.reload()
    expect(page.locator("#welcome")).to_be_visible()
    expect(page.locator(".recent-row")).to_have_count(0)


def test_delete_waits_for_active_run_and_never_deletes_written_files(browser_page):
    page, root = browser_page
    send(page, 'Write "keep the contents" to keep.txt')
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("Wrote and verified")
    expect(page.locator(".message-meta")).to_contain_text("Completed")
    expect(page.locator(".recent-delete")).to_be_enabled()
    page.locator(".recent-delete").click()
    expect(page.locator("#delete-dialog")).to_contain_text("Workspace files are not deleted")
    page.locator("#confirm-delete").click()
    expect(page.locator("#welcome")).to_be_visible()
    assert (root / "keep.txt").read_text() == "keep the contents"


def test_prompts_and_limits_save_reload_export_and_import(browser_page):
    page, _ = browser_page
    page.locator("#open-settings").click()
    page.locator(".prompt-settings > summary").click()
    page.locator("#reasoning-prompt").fill("Custom reasoning instructions: use one tool at a time.")
    page.locator("#translator-prompt").fill(
        "Custom translation instructions: preserve the payload."
    )
    page.locator("#confirmation-prompt").fill("Custom review: is the highest-ranked tool correct?")
    page.locator(".model-settings > summary").click()
    page.locator("#needle-max-tokens").fill("3072")
    page.locator("#llm-max-tokens").fill("5000")
    page.locator("#save-settings").click()
    expect(page.locator("#settings-dialog")).not_to_be_visible()
    page.reload()
    expect(page.locator("#runtime-status")).to_contain_text("Runtime ready")
    page.locator("#open-settings").click()
    page.locator(".prompt-settings > summary").click()
    expect(page.locator("#reasoning-prompt")).to_have_value(
        "Custom reasoning instructions: use one tool at a time."
    )
    expect(page.locator("#confirmation-prompt")).to_have_value(
        "Custom review: is the highest-ranked tool correct?"
    )
    with page.expect_download() as download:
        page.locator("#export-settings").click()
    content = Path(download.value.path()).read_text()
    assert "Custom translation instructions" in content and "needle_max_tokens = 3072" in content
    assert "llm_api_key" not in content and "workspace_root" not in content
    page.locator("#config-file").set_input_files(
        {
            "name": "import.toml",
            "mimeType": "text/plain",
            "buffer": b"[runtime]\nmax_tool_steps = 9\n[prompts]\n"
            b'reasoning_prompt = "Imported instructions"\n',
        }
    )
    expect(page.locator("#connection-result")).to_contain_text("imported and applied")
    expect(page.locator("#reasoning-prompt")).to_have_value("Imported instructions")
    page.locator("#settings-dialog .dialog-close").click()
    expect(page.locator("#step-limit")).to_contain_text("9 steps")
    send(page, "Calculate 6*7")
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("42")


def test_mobile_dark_theme_configuration_and_delete(browser_page):
    page, _ = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Switch to dark mode").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    send(page, "Calculate 9*9")
    expect(page.locator(".message-meta")).to_contain_text("Completed")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator("#mode-badge").click()
    page.locator(".prompt-settings > summary").click()
    expect(page.locator("#reasoning-prompt")).to_be_visible()
    assert page.locator("#settings-dialog").bounding_box()["width"] < 390
    page.locator("#settings-dialog .dialog-close").click()
    page.locator("#menu-button").click()
    page.locator(".recent-delete").click()
    page.locator("#confirm-delete").click()
    expect(page.locator(".recent-row")).to_have_count(0)
    expect(page.locator("#delete-dialog")).not_to_be_visible()


@pytest.mark.parametrize("browser_page", ["selection-review"], indirect=True)
def test_candidate_confirmation_is_visible_and_returns_through_reasoning(browser_page):
    page, _ = browser_page
    send(page, "Multiply six and seven")
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("42")
    expect(page.locator(".message-meta")).to_contain_text("1 tool step")
    expect(page.locator(".tool-card")).to_have_count(2)
    page.locator(".tool-card summary").first.click()
    expect(page.locator(".selection-review")).to_contain_text("Is calculator the correct tool?")
    expect(page.locator(".selection-review")).to_contain_text("0.20")
    expect(page.locator(".tool-card").first).to_contain_text("not executed")
    expect(page.locator(".phase-list")).to_contain_text("Request selection review")
