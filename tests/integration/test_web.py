"""HTTP/SSE tests, no model downloads. Real graph and tools behind the demo adapters."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from relay import AgentConfig
from relay.server import WorkspaceService, make_server


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
        hdrs["X-Relay-Session"] = token
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


def test_questions_then_write_completes_without_approval(web):
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
    events = finish(base, token, run["id"])
    assert events[-1]["status"] == "COMPLETED"
    assert events[-1]["steps"] == 3
    assert (root / "note.txt").read_text() == "Hello from the browser"


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
    for path in ("/", "/app.js", "/style.css", "/favicon.svg", "/theme.js"):
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


def test_editable_prompts_model_limits_and_portable_config(web):
    service, base, root = web
    token, _ = setup_conversation(base)
    _, original = request(base, "/api/session", token=token)
    settings = {
        **original["settings"],
        "reasoning_prompt": "REASON CUSTOM",
        "translator_prompt": "TRANSLATE CUSTOM",
        "confirmation_prompt": "REVIEW CUSTOM",
        "llm_max_tokens": 5000,
        "needle_max_tokens": 3000,
        "max_stalls": 5,
        "max_search_results": 8,
    }
    status, result = request(base, "/api/settings", method="POST", token=token, data=settings)
    assert status == 200
    config = service.run_config(service.sessions[token].settings)
    assert (
        config.reasoning_prompt == "REASON CUSTOM"
        and config.translator_prompt == "TRANSLATE CUSTOM"
    )
    assert config.confirmation_prompt == "REVIEW CUSTOM" and config.needle_max_tokens == 3000
    assert config.max_stalls == 5 and config.max_search_results == 8
    _, exported = request(base, "/api/settings/export", token=token)
    assert "never-leak" not in exported["content"]
    assert (
        "workspace_root" not in exported["content"] and "needle_weights" not in exported["content"]
    )
    assert "REASON CUSTOM" in exported["content"] and "REVIEW CUSTOM" in exported["content"]
    _, other = request(base, "/api/session")
    assert (
        request(
            base,
            "/api/settings",
            method="POST",
            token=other["session_token"],
            data={**other["settings"], "default_timezone": "UTC"},
        )[0]
        == 200
    )
    status, imported = request(
        base,
        "/api/settings/import",
        method="POST",
        token=other["session_token"],
        data={"content": exported["content"], "format": "toml"},
    )
    assert status == 200 and imported["settings"] == result["settings"]
    assert service.config.workspace_root == str(root)
    assert other["settings"]["reasoning_prompt"] != "REASON CUSTOM"


def test_import_cannot_load_server_prompt_files_or_change_workspace(web):
    service, base, root = web
    token, _ = setup_conversation(base)
    content = '[prompts]\nreasoning_prompt_file = "/etc/passwd"\n'
    status, error = request(
        base,
        "/api/settings/import",
        method="POST",
        token=token,
        data={"content": content},
    )
    assert status == 400 and "embed prompts" in error["error"]
    content = '[workspace]\nworkspace_root = "/etc"\n[models]\nneedle_weights = "/secret"\n'
    status, result = request(
        base,
        "/api/settings/import",
        method="POST",
        token=token,
        data={"content": content},
    )
    assert status == 200 and result["ignored"] == ["needle_weights", "workspace_root"]
    assert service.config.workspace_root == str(root) and service.config.needle_weights is None


def test_invalid_config_import_is_atomic_and_prompts_are_bounded(web):
    service, base, _ = web
    token, _ = setup_conversation(base)
    before = service.sessions[token].settings.model_dump()
    for content in (
        "[runtime]\nmax_stalls = 0",
        '[prompts]\nreasoning_prompt = ""',
        "[models]\nllm_temperature = 6",
        "[runtime]\nmax_stallz = 5",
        "[runtime]\nmax_stalls = 9\n[models]\nllm_temperature = 6",
    ):
        status, _ = request(
            base,
            "/api/settings/import",
            method="POST",
            token=token,
            data={"content": content},
        )
        assert status == 400
        assert service.sessions[token].settings.model_dump() == before
    settings = {**before, "reasoning_prompt": "x" * 20001}
    assert request(base, "/api/settings", method="POST", token=token, data=settings)[0] == 400
    settings = {**before, "max_context_chars": 8000, "reasoning_prompt": "x" * 10000}
    assert request(base, "/api/settings", method="POST", token=token, data=settings)[0] == 400


def test_config_import_respects_read_only_floor_and_api_key_origin(web):
    from dataclasses import replace

    service, base, _ = web
    token, _ = setup_conversation(base)
    service.config = replace(service.config, read_only=True)
    for content in (
        "[workspace]\nread_only = false",
        '[workspace]\nread_only = true\n[models]\nmode = "live"\nllm_base_url = "https://other.invalid"',
    ):
        status, _ = request(
            base,
            "/api/settings/import",
            method="POST",
            token=token,
            data={"content": content},
        )
        assert status == 403


def test_delete_chat_removes_history_and_runs_not_workspace_files(web):
    service, base, root = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, 'Write "keep this file" to kept.txt')
    finish(base, token, run["id"])
    status, deleted = request(
        base, f"/api/conversations/{conversation}", method="DELETE", token=token
    )
    assert status == 200 and deleted["deleted"] == conversation
    assert (root / "kept.txt").read_text() == "keep this file"
    assert request(base, f"/api/conversations/{conversation}", token=token)[0] == 404
    assert request(base, f"/api/runs/{run['id']}", token=token)[0] == 404
    assert request(base, f"/api/runs/{run['id']}/events", token=token)[0] == 404
    _, snapshot = request(base, "/api/session", token=token)
    assert snapshot["conversations"] == [] and snapshot["runs"] == []
    assert (
        request(base, f"/api/conversations/{conversation}", method="DELETE", token=token)[0] == 404
    )


def test_delete_active_chat_is_rejected_until_stopped(web):
    service, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Create a note")
    wait_pending(service, token, run["id"])
    path = f"/api/conversations/{conversation}"
    assert request(base, path, method="DELETE", token=token)[0] == 409
    assert request(base, path, token=token)[0] == 200
    request(base, f"/api/runs/{run['id']}/cancel", method="POST", token=token, data={})
    finish(base, token, run["id"])
    assert request(base, path, method="DELETE", token=token)[0] == 200


def test_delete_is_session_scoped_and_cross_site_protected(web):
    _, base, _ = web
    token, conversation = setup_conversation(base)
    _, other = request(base, "/api/session")
    path = f"/api/conversations/{conversation}"
    assert request(base, path, method="DELETE")[0] == 401
    assert request(base, path, method="DELETE", token=other["session_token"])[0] == 404
    assert (
        request(base, path, method="DELETE", token=token, headers={"Sec-Fetch-Site": "cross-site"})[
            0
        ]
        == 403
    )
    assert request(base, path, token=token)[0] == 200


def test_eviction_does_not_leave_orphaned_run_history(web, monkeypatch):
    _, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Calculate 2+2")
    finish(base, token, run["id"])
    monkeypatch.setattr("relay.server.MAX_CONVERSATIONS", 1)
    status, new = request(base, "/api/conversations", method="POST", token=token, data={})
    assert status == 201 and new["id"] != conversation
    assert request(base, f"/api/runs/{run['id']}", token=token)[0] == 404


def test_streaming_settings_roundtrip_and_can_disable_input_capture(web):
    service, base, _ = web
    token, conversation = setup_conversation(base)
    _, bootstrap = request(base, "/api/session", token=token)
    settings = {
        **bootstrap["settings"],
        "stream_buffer_ms": 0,
        "stream_flush_ms": 20,
        "llm_stream": False,
        "capture_model_inputs": False,
    }
    assert request(base, "/api/settings", method="POST", token=token, data=settings)[0] == 200
    run = start(base, token, conversation, "Calculate 6*7")
    events = finish(base, token, run["id"])
    starts = [event for event in events if event["type"] == "model_start"]
    assert starts and all(event["input_messages"] == [] for event in starts)
    assert all(event["streamed"] is False for event in starts)
    assert service.sessions[token].settings.stream_buffer_ms == 0
    _, exported = request(base, "/api/settings/export", token=token)
    assert "stream_buffer_ms = 0" in exported["content"]
    assert "llm_stream = false" in exported["content"]


def test_model_trace_budget_does_not_break_run_or_final_answer(web, monkeypatch):
    service, base, _ = web
    monkeypatch.setattr("relay.server.MAX_MODEL_TRACE_BYTES", 80)
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Calculate 6*7")
    events = finish(base, token, run["id"])
    assert events[-1]["status"] == "COMPLETED" and "42" in events[-1]["final_answer"]
    assert any(event["type"] == "model_trace_limited" for event in events)
    assert not any(event["type"] == "model_delta" for event in events)
    assert [event["id"] for event in events] == list(range(1, len(events) + 1))
    _, snapshot = request(base, f"/api/runs/{run['id']}", token=token)
    assert snapshot["trace_limited"] and service.sessions[token].runs[run["id"]].done
    # Replay resumes from the retained event cursor, even when details were capped.
    _, replay = request(
        base, f"/api/runs/{run['id']}/events?after={events[-2]['id']}", token=token, raw=True
    )
    assert replay.count('"type": "complete"') == 1


def test_model_streams_are_session_scoped(web):
    service, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Create a note")
    wait_pending(service, token, run["id"])
    _, other = request(base, "/api/session")
    assert request(base, f"/api/runs/{run['id']}/events", token=other["session_token"])[0] == 404
    assert request(base, f"/api/runs/{run['id']}", token=other["session_token"])[0] == 404


def test_streaming_module_is_packaged_and_served_with_preview_headers(web):
    _, base, _ = web
    status, text = request(base, "/stream.js", raw=True)
    assert status == 200 and "PacedText" in text


def test_run_tracks_context_chars_in_snapshot(web):
    service, base, _ = web
    token, conversation = setup_conversation(base)
    run = start(base, token, conversation, "Calculate 6*7")
    events = finish(base, token, run["id"])
    starts = [
        event
        for event in events
        if event["type"] == "model_start" and event.get("component") == "reasoning"
    ]
    assert starts and all(event["context_chars"] > 0 for event in starts)
    _, snapshot = request(base, f"/api/runs/{run['id']}", token=token)
    assert snapshot["context_chars"] == starts[-1]["context_chars"] > 0


def test_approval_answer_event_carries_tool_name(web):
    import threading

    from relay.server import Answer, Run
    from relay.tools.base import ToolCall

    service, _, _ = web
    run = Run(id="r", conversation_id="c", mode="demo")
    call = ToolCall(name="run_python", arguments={"code": "print(1)"})
    outcome = {}
    worker = threading.Thread(
        daemon=True,
        target=lambda: outcome.setdefault(
            "reply", run.wait_for_user("approval", "Allow Python run?", call)
        )
    )
    worker.start()
    with run.condition:
        assert run.condition.wait_for(lambda: run.pending is not None, timeout=5)
    service.answer(run, Answer(question_id=run.pending["question_id"], approved=True))
    worker.join(timeout=5)
    assert outcome["reply"] == {"approved": True, "call": None}
    answers = [event for event in run.events if event["type"] == "user_answer"]
    assert len(answers) == 1
    assert answers[0]["kind"] == "approval" and answers[0]["tool"] == "run_python"
    assert answers[0]["answer"] is True and answers[0]["edited"] is False


def test_approval_pending_carries_diff_and_accepts_edits(web):
    import threading

    from relay.server import Answer, Run
    from relay.tools.base import ToolCall

    service, _, root = web
    (root / "note.txt").write_text("line one\nline two\n")
    session = service.session(None, create=True)
    run = Run(id="r", conversation_id="c", mode="demo")
    call = ToolCall(
        name="replace_text",
        arguments={"path": "note.txt", "old_text": "line two", "new_text": "LINE TWO"},
    )
    agent = service._make_agent(session.settings, run)
    outcome = {}
    worker = threading.Thread(
        daemon=True,
        target=lambda: outcome.setdefault("reply", agent._approve(call))
    )
    worker.start()
    with run.condition:
        assert run.condition.wait_for(lambda: run.pending is not None, timeout=5)
    diff = run.pending["diff"]
    assert diff["label"] == "Replace text" and "-line two" in diff["text"]
    assert "+LINE TWO" in diff["text"] and "near line 2" in diff["text"]
    service.answer(
        run,
        Answer(
            question_id=run.pending["question_id"],
            approved=True,
            arguments={
                "path": "note.txt",
                "old_text": "line two",
                "new_text": "LINE 2!",
            },
        ),
    )
    worker.join(timeout=5)
    assert isinstance(outcome["reply"], ToolCall)
    assert outcome["reply"].arguments["new_text"] == "LINE 2!"
    answers = [event for event in run.events if event["type"] == "user_answer"]
    assert answers[0]["edited"] is True


def test_approval_edits_are_validated(web):
    from relay.server import Answer, Run, WebError

    service, _, _ = web

    def pending_run(**pending):
        run = Run(id="r", conversation_id="c", mode="demo")
        run.pending = {
            "question_id": "q",
            "kind": "approval",
            "question": "Allow?",
            "call": {
                "name": "run_python",
                "arguments": {"code": "print(1)"},
            },
            **pending,
        }
        return run

    run = pending_run()
    with pytest.raises(WebError):
        service.answer(run, Answer(question_id="q", approved=True, arguments={"bogus": 1}))
    with pytest.raises(ValueError):
        Answer(question_id="q", approved=True, arguments=["nope"])
    with pytest.raises(WebError):
        service.answer(
            run, Answer(question_id="q", approved=False, arguments={"code": "print(2)"})
        )
    assert run.reply is None and run.pending is not None
    service.answer(
        run, Answer(question_id="q", approved=True, arguments={"code": "print(2)"})
    )
    assert run.reply == {
        "approved": True,
        "call": {"name": "run_python", "arguments": {"code": "print(2)"}},
    }


def test_auto_approve_is_session_only_and_validated(web):
    service, base, _ = web
    assert "run_python" in service.config.require_approval_for
    token, _ = setup_conversation(base)
    _, original = request(base, "/api/session", token=token)
    status, body = request(
        base,
        "/api/settings",
        method="POST",
        token=token,
        data={**original["settings"], "auto_approve": ["run_python", "bogus_tool"]},
    )
    assert status == 400 and "bogus_tool" in body["error"]
    status, result = request(
        base,
        "/api/settings",
        method="POST",
        token=token,
        data={**original["settings"], "auto_approve": ["run_python", "git_commit"]},
    )
    assert status == 200
    assert result["settings"]["auto_approve"] == ["run_python", "git_commit"]
    config = service.run_config(service.sessions[token].settings)
    assert "run_python" not in config.require_approval_for
    assert "git_commit" not in config.require_approval_for
    assert "delete_file" in config.require_approval_for
    _, exported = request(base, "/api/settings/export", token=token)
    assert "auto_approve" not in exported["content"]
    _, fresh = request(base, "/api/session")
    assert fresh["settings"]["auto_approve"] == []
