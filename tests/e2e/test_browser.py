"""Optional real-browser tests: uv sync --group browser; playwright install chromium.

Set NEEDLE_BROWSER_EXECUTABLE to use an existing Chromium instead. No live LLM
or API key is needed; the demo runs the same graph and real sandboxed tools.
"""

import os
import shutil
import threading
from pathlib import Path

import pytest

from agent_runtime import AgentConfig
from agent_runtime.server import WorkspaceService, make_server

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect


@pytest.fixture()
def browser_page(tmp_path):
    root = tmp_path / "workspace"
    shutil.copytree(Path(__file__).parents[2] / "examples" / "workspace", root)
    (root / "untrusted.md").write_text(
        '<img src=x onerror="window.injected=true"><script>window.injected=true</script>'
    )
    service = WorkspaceService(AgentConfig(workspace_root=str(root)), demo=True)
    server = make_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            executable = os.environ.get("NEEDLE_BROWSER_EXECUTABLE") or p.chromium.executable_path
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


def test_question_and_approval_survive_reload(browser_page):
    page, root = browser_page
    page.locator('[data-prompt="Create a note"]').click()
    expect(page.locator(".pending-card")).to_contain_text("What would you like me to write")
    page.reload()
    expect(page.locator(".pending-card")).to_contain_text("What would you like me to write")
    send(page, "A useful note from the browser.")
    expect(page.locator(".pending-card")).to_contain_text("Your permission is needed")
    assert not (root / "note.txt").exists()
    page.get_by_role("button", name="Allow write", exact=True).click()
    expect(page.locator(".assistant-body > .markdown")).to_contain_text("Wrote and verified")
    assert (root / "note.txt").read_text() == "A useful note from the browser."
    expect(page.locator(".message-meta")).to_contain_text("3 tool steps")
    with page.expect_download() as download:
        page.get_by_role("button", name="Export trace").click()
    assert download.value.suggested_filename.startswith("needle-run-")


def test_denial_and_cancellation(browser_page):
    page, root = browser_page
    send(page, 'Write "do not write this" to denied.txt')
    expect(page.locator(".pending-card")).to_contain_text("Allow writing to denied.txt")
    page.get_by_role("button", name="Deny", exact=True).click()
    expect(page.locator(".message-meta")).to_contain_text("0 tool steps")
    assert not (root / "denied.txt").exists()
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
    expect(page.locator(".tool-catalog-card")).to_have_count(7)
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
