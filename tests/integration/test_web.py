"""HTTP/SSE tests, no model downloads. Real graph and tools behind the demo adapters."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent_runtime import AgentConfig
from agent_runtime.server import WorkspaceService, make_server


@pytest.fixture()
def web(tmp_path: Path):
    (tmp_path / "README.md").write_text("# A small project\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("# authentication\ndef authenticate_user(): pass\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("protected")
    service = WorkspaceService(
        AgentConfig(workspace_root=str(tmp_path), llm_api_key="never-leak"), demo=True
    )
    server = make_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield service, f"http://127.0.0.1:{server.server_port}", tmp_path
    service.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(base, path, *, method="GET", data=None, token=None, headers=None, raw=False):
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    if token:
        hdrs["X-Needle-Session"] = token
    req = urllib.request.Request(
        base + path,
        method=method,
        headers=hdrs,
        data=json.dumps(data).encode() if data is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read()
            return response.status, body.decode() if raw else json.loads(body)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def setup_conversation(base):
    _, bootstrap = request(base, "/api/session")
    token = bootstrap["session_token"]
    _, conversation = request(base, "/api/conversations", method="POST", data={}, token=token)
    return token, conversation["id"]


def start(base, token, conversation, message):
    status, run = request(
        base,
        f"/api/conversations/{conversation}/runs",
        method="POST",
        data={"message": message},
        token=token,
    )
    assert status == 202
    return run


def wait_pending(service, token, run_id):
    run = service.sessions[token].runs[run_id]
    with run.condition:
        assert run.condition.wait_for(lambda: run.pending is not None or run.done, timeout=5)
        assert run.pending
        return dict(run.pending)


def finish(base, token, run_id):
    status, text = request(base, f"/api/runs/{run_id}/events", token=token, raw=True)
    assert status == 200
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "complete"
    return events


def test_real_tools_stream_and_history(web):
    _, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Find the authentication implementation")
    events = finish(base, token, run["id"])
    assert events[-1]["status"] == "COMPLETED"
    assert events[-1]["steps"] == 2
    assert "src/auth.py" in events[-1]["final_answer"]
    assert [e["tool"] for e in events if e["type"] == "tool_result"] == [
        "search_files",
        "read_file",
    ]
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    _, saved = request(base, f"/api/conversations/{conversation}", token=token)
    assert saved["messages"][1]["content"] == events[-1]["final_answer"]
    _, bootstrap = request(base, "/api/session", token=token)
    assert "never-leak" not in json.dumps(bootstrap)
    assert bootstrap["api_key_configured"] is True


def test_questions_then_write_approval_pause_and_resume(web):
    service, base, root = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Create a note")
    question = wait_pending(service, token, run["id"])
    assert question["kind"] == "question"
    assert not (root / "note.txt").exists()
    status, _ = request(
        base,
        f"/api/runs/{run['id']}/answer",
        method="POST",
        token=token,
        data={"question_id": question["question_id"], "answer": "Hello from the browser"},
    )
    assert status == 200
    real_run = service.sessions[token].runs[run["id"]]
    with real_run.condition:
        assert real_run.condition.wait_for(
            lambda: real_run.pending is not None and real_run.pending["kind"] == "approval",
            timeout=5,
        )
        approval = dict(real_run.pending)
    assert not (root / "note.txt").exists()
    assert approval["call"]["arguments"]["content"] == "Hello from the browser"
    status, _ = request(
        base,
        f"/api/runs/{run['id']}/answer",
        method="POST",
        token=token,
        data={"question_id": approval["question_id"], "approved": True},
    )
    assert status == 200
    events = finish(base, token, run["id"])
    assert events[-1]["status"] == "COMPLETED"
    assert events[-1]["steps"] == 3
    assert (root / "note.txt").read_text() == "Hello from the browser"


def test_deny_write_never_mutates_file(web):
    service, base, root = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, 'Write "hello" to note.txt')
    approval = wait_pending(service, token, run["id"])
    status, _ = request(
        base,
        f"/api/runs/{run['id']}/answer",
        method="POST",
        token=token,
        data={"question_id": approval["question_id"], "approved": False},
    )
    assert status == 200
    events = finish(base, token, run["id"])
    assert events[-1]["steps"] == 0
    assert not (root / "note.txt").exists()


def test_cancel_wakes_question_and_prevents_new_tools(web):
    service, base, root = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Create a note")
    wait_pending(service, token, run["id"])
    status, _ = request(base, f"/api/runs/{run['id']}/cancel", method="POST", data={}, token=token)
    assert status == 200
    events = finish(base, token, run["id"])
    assert events[-1]["status"] == "CANCELLED"
    assert not (root / "note.txt").exists()
    assert not any(e.get("tool") == "write_file" for e in events)


def test_session_isolation_and_unauthorized_requests(web):
    _, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Calculate 3+4")
    finish(base, token, run["id"])
    _, other = request(base, "/api/session")
    assert (
        request(base, f"/api/conversations/{conversation}", token=other["session_token"])[0] == 404
    )
    assert request(base, f"/api/runs/{run['id']}", token=other["session_token"])[0] == 404
    assert request(base, "/api/files")[0] == 401
    assert (
        request(
            base,
            "/api/conversations",
            method="POST",
            data={},
            token=token,
            headers={"Sec-Fetch-Site": "cross-site"},
        )[0]
        == 403
    )


def test_bad_input_and_unsafe_file_paths(web):
    _, base, _ = web
    token, conversation = setup_conversation(base)
    for message in (" ", 42, "x" * 8001):
        assert (
            request(
                base,
                f"/api/conversations/{conversation}/runs",
                method="POST",
                token=token,
                data={"message": message},
            )[0]
            == 400
        )
    for path in ("../outside", ".git/config", "/etc/passwd"):
        assert request(base, f"/api/file?path={path}", token=token)[0] == 400
    assert request(base, "/api/conversations", method="POST", data=[], token=token)[0] == 400
    assert (
        request(
            base,
            "/api/settings",
            method="POST",
            token=token,
            data={"mode": "demo", "workspace_root": "/etc"},
        )[0]
        == 400
    )


def test_settings_busy_run_and_stale_answers(web):
    service, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Create a note")
    question = wait_pending(service, token, run["id"])
    assert question
    assert (
        request(
            base,
            f"/api/conversations/{conversation}/runs",
            method="POST",
            token=token,
            data={"message": "Calculate 4+4"},
        )[0]
        == 409
    )
    assert (
        request(base, "/api/settings", method="POST", token=token, data={"mode": "demo"})[0] == 409
    )
    assert (
        request(
            base,
            f"/api/runs/{run['id']}/answer",
            method="POST",
            token=token,
            data={"question_id": "old", "answer": "no"},
        )[0]
        == 409
    )


def test_static_assets_preview_headers_and_no_cors(web):
    _, base, _ = web
    for path in ("/", "/app.js", "/style.css", "/favicon.svg"):
        req = urllib.request.Request(base + path, headers={"Host": "3000-preview.e2b.app"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
            assert response.headers.get("X-Frame-Options") is None
            assert "frame-ancestors *" in response.headers["Content-Security-Policy"]
            assert not response.headers.get("Access-Control-Allow-Origin")
    assert request(base, "/health")[1] == {"status": "ok"}
    assert request(base, "/../pyproject.toml")[0] == 404


def test_read_only_server_cannot_be_escalated_from_browser(web):
    service, base, root = web
    from dataclasses import replace

    service.config = replace(service.config, read_only=True)
    token, _ = setup_conversation(base)
    status, body = request(
        base, "/api/settings", method="POST", token=token, data={"mode": "demo", "read_only": False}
    )
    assert status == 403
    assert "read-only" in body["error"]
    assert service.run_config(service.sessions[token].settings).read_only


def test_browser_cannot_redirect_server_owned_api_key(web):
    _, base, _ = web
    token, _ = setup_conversation(base)
    status, body = request(
        base,
        "/api/settings",
        method="POST",
        token=token,
        data={"mode": "live", "base_url": "https://different-provider.example"},
    )
    assert status == 403
    assert "API key is bound" in body["error"]
    assert "never-leak" not in str(body)
