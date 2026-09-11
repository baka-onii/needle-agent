"""Extended toolset: registry completeness plus per-category behavior."""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_runtime import AgentConfig
from agent_runtime.execution.confidence import threshold_for
from agent_runtime.models.functiongemma import _coerce_arguments
from agent_runtime.tools.base import ToolError
from agent_runtime.tools.editing import apply_unified_diff
from agent_runtime.tools.registry import FORBIDDEN_TOOLS, create_default_registry
from agent_runtime.tools.web import html_to_text, parse_duckduckgo, search_blocked

REQUIRED = [
    "read_file", "read_directory", "search_files", "write_file", "file_info",
    "create_directory", "move_file", "copy_file", "delete_file", "replace_text",
    "insert_text", "delete_text", "apply_patch", "run_python", "run_powershell",
    "run_process", "git_status", "git_diff", "git_log", "git_show", "git_branch_list",
    "git_stage", "git_commit", "git_checkout", "web_search", "web_open", "web_extract",
    "get_working_directory", "find_executable", "process_info", "calculator",
    "get_time", "ask_user",
]


def _registry(tmp_path: Path):
    return create_default_registry(AgentConfig(workspace_root=str(tmp_path)))


def _run(registry, tool_name: str, **arguments):
    return registry.get(tool_name).handler(**arguments)


def test_registry_matches_required_tools_list(tmp_path: Path) -> None:
    names = sorted(tool.name for tool in _registry(tmp_path).list())
    assert names == sorted(REQUIRED)
    assert "run_python" not in FORBIDDEN_TOOLS
    assert {"shell", "terminal", "execute_command"} <= FORBIDDEN_TOOLS


def test_read_only_gates_cover_new_inspection_tools(tmp_path: Path) -> None:
    config = AgentConfig(workspace_root=str(tmp_path))
    for name in (
        "file_info", "git_status", "git_diff", "git_log", "git_show",
        "git_branch_list", "web_search", "web_open", "web_extract",
        "get_working_directory", "find_executable", "process_info",
    ):
        assert threshold_for(name, config) == config.read_only_threshold
    for name in (
        "write_file", "delete_file", "run_python", "git_commit", "replace_text",
    ):
        assert threshold_for(name, config) == config.confidence_threshold
    assert threshold_for("ask_user", config) == config.confidence_threshold


def test_filesystem_mutations_and_info(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert "Created" in _run(registry, "create_directory", path="sub")
    assert "already exists" in _run(registry, "create_directory", path="sub")
    assert "Created" in _run(registry, "create_directory", path="sub/nested")
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    info = _run(registry, "file_info", path="a.txt")
    assert "Type: file" in info and "Lines: 2" in info
    assert "Moved" in _run(registry, "move_file", source="a.txt", destination="sub/b.txt")
    assert "Copied" in _run(registry, "copy_file", source="sub/b.txt", destination="c.txt")
    assert "Deleted" in _run(registry, "delete_file", path="c.txt")
    with pytest.raises(ToolError):
        _run(registry, "delete_file", path="sub")
    with pytest.raises(ToolError):
        _run(registry, "read_file", path="../outside.txt")
    with pytest.raises(ToolError):
        _run(registry, "move_file", source="sub/b.txt", destination="sub/b.txt")


def test_editing_tools(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    (tmp_path / "note.txt").write_text("one\ntwo\nthree\n")
    assert "1 occurrence" in _run(
        registry, "replace_text", path="note.txt", old_text="two", new_text="TWO"
    )
    assert "line 1" in _run(registry, "insert_text", path="note.txt", line=1, text="zero")
    assert "1-2" in _run(registry, "delete_text", path="note.txt", start_line=1, end_line=2)
    assert (tmp_path / "note.txt").read_text() == "TWO\nthree\n"
    with pytest.raises(ToolError):
        _run(registry, "replace_text", path="note.txt", old_text="missing", new_text="x")
    with pytest.raises(ToolError):
        _run(registry, "delete_text", path="note.txt", start_line=9, end_line=10)


def test_apply_unified_diff(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    (tmp_path / "code.py").write_text("a = 1\nb = 2\nc = 3\n")
    out = _run(
        registry,
        "apply_patch",
        path="code.py",
        patch="--- a/code.py\n+++ b/code.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n+b = 20\n c = 3\n",
    )
    assert "Patched" in out
    assert (tmp_path / "code.py").read_text() == "a = 1\nb = 20\nc = 3\n"
    with pytest.raises(ToolError):
        apply_unified_diff("x\n", "@@ -1,1 +1,1 @@\n-y\n+z\n")


def test_run_python_and_process(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert _run(registry, "run_python", code="print(2 + 2)") == "4"
    with pytest.raises(ToolError):
        _run(registry, "run_python", code="raise SystemExit(3)")
    with pytest.raises(ToolError):
        _run(registry, "run_process", command="definitely-not-a-real-binary-xyz")
    with pytest.raises(ToolError):
        _run(registry, "run_python", code="x", timeout_s=0)


def test_git_tools_against_scratch_repo(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("hi\n")
    registry = _registry(tmp_path)
    assert "f.txt" in _run(registry, "git_status")
    _run(registry, "git_stage", paths="f.txt")
    assert "Add f" in _run(registry, "git_commit", message="Add f")
    assert "Add f" in _run(registry, "git_log", limit=1)
    assert "f.txt" in _run(registry, "git_show", revision="HEAD")
    assert "master" in _run(registry, "git_branch_list") or "main" in _run(
        registry, "git_branch_list"
    )
    with pytest.raises(ToolError):
        _run(registry, "git_checkout", branch="no-such-branch-xyz")


def test_web_helpers_without_network() -> None:
    page = (
        '<html><head><script>var x = 1;</script></head><body>'
        '<nav>menu</nav><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>'
    )
    text = html_to_text(page)
    assert "Title" in text and "Hello world" in text
    assert "var x" not in text and "menu" not in text
    ddg = (
        '<a class="result__a" href="https://example.com/a">Alpha</a>'
        '<div class="result__snippet">First snippet here.</div>'
    )
    assert parse_duckduckgo(ddg, 5) == [
        ("https://example.com/a", "Alpha", "First snippet here.")
    ]


def test_search_blocked_only_matches_bot_walls() -> None:
    challenge = (
        "<html><body>Unfortunately, bots use DuckDuckGo too. "
        "Please complete the following challenge to confirm "
        "this search was made by a human.</body></html>"
    )
    assert search_blocked(challenge)
    assert not search_blocked("<html><body>No results.</body></html>")
    assert not search_blocked("Anubis, god of funerary rites, guarded graves.")


class _WebHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        data = b"<html><body><p>" + b"x" * 100 + b" readable words here.</p></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_web_open_and_extract_against_local_server(tmp_path: Path) -> None:
    server = HTTPServer(("127.0.0.1", 0), _WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = _registry(tmp_path)
        url = f"http://127.0.0.1:{server.server_port}/"
        assert "readable words" in _run(registry, "web_open", url=url)
        assert "readable words" in _run(registry, "web_extract", url=url)
        with pytest.raises(ToolError):
            _run(registry, "web_open", url="ftp://example.com/x")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_environment_tools(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert _run(registry, "get_working_directory") == str(tmp_path.resolve())
    found = _run(registry, "find_executable", name="python")
    assert "not found" not in found or shutil.which("python") is None
    import os

    assert f"pid: {os.getpid()}" in _run(registry, "process_info")
    assert "alive: no" in _run(registry, "process_info", pid=2**30)
    with pytest.raises(ToolError):
        _run(registry, "find_executable", name="a/b")


def test_functiongemma_coerces_declared_scalars(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in _registry(tmp_path).list()}
    assert _coerce_arguments(tools["git_log"], {"limit": "5", "path": "."}) == {
        "limit": 5,
        "path": ".",
    }
    assert _coerce_arguments(tools["git_log"], {"limit": "many"}) == {"limit": "many"}
    assert _coerce_arguments(tools["read_file"], {"path": "a.txt"}) == {"path": "a.txt"}


def test_tool_schemas_validate(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    schemas = json.dumps([tool.needle_schema() for tool in registry.list()], sort_keys=True)
    assert len(schemas) > 1000
    for tool in registry.list():
        assert tool.name and tool.description and tool.parameters.get("type") == "object"


def test_payload_contracts_cover_exactly_the_bulk_text_tools(tmp_path: Path) -> None:
    by_name = {tool.name: tool for tool in _registry(tmp_path).list()}
    assert by_name["write_file"].payload_args == ("content",)
    assert by_name["run_python"].payload_args == ("code",)
    assert by_name["run_process"].payload_args == ("command",)
    assert by_name["run_powershell"].payload_args == ("command",)
    assert by_name["insert_text"].payload_args == ("text",)
    assert by_name["apply_patch"].payload_args == ("patch",)
    assert by_name["replace_text"].payload_args == ("old_text", "new_text")
    assert by_name["read_file"].payload_args == ()
    assert by_name["delete_file"].payload_args == ()


def test_approval_summary_names_tool_and_target(tmp_path: Path) -> None:
    from agent_runtime.tools.base import ToolCall, approval_summary

    assert (
        approval_summary(ToolCall(name="delete_file", arguments={"path": "a.txt"}))
        == "delete_file a.txt"
    )
    assert approval_summary(
        ToolCall(name="run_python", arguments={"code": "print(1)"})
    ).startswith("run_python print(1)")
    assert approval_summary(ToolCall(name="get_time", arguments={})) == "get_time"


def test_approval_diff_covers_severe_tools(tmp_path: Path) -> None:
    from agent_runtime.tools.base import ToolCall
    from agent_runtime.tools.preview import approval_diff

    config = AgentConfig(workspace_root=str(tmp_path))
    (tmp_path / "note.txt").write_text("line one\nline two\n")

    diff = approval_diff(
        ToolCall(name="write_file", arguments={"path": "new.txt", "content": "hi\n"}),
        config,
    )
    assert diff["label"].startswith("New file") and "hi" in diff["text"]

    diff = approval_diff(
        ToolCall(
            name="write_file",
            arguments={"path": "note.txt", "content": "line one\nLINE TWO\n"},
        ),
        config,
    )
    assert diff["label"] == "Edit note.txt"
    assert "-line two" in diff["text"] and "+LINE TWO" in diff["text"]

    diff = approval_diff(
        ToolCall(
            name="replace_text",
            arguments={"path": "note.txt", "old_text": "line two", "new_text": "2"},
        ),
        config,
    )
    assert "near line 2" in diff["text"] and "-line two" in diff["text"]

    diff = approval_diff(
        ToolCall(
            name="insert_text",
            arguments={"path": "note.txt", "line": 1, "text": "middle"},
        ),
        config,
    )
    assert "after line 1" in diff["text"] and "+middle" in diff["text"]

    diff = approval_diff(
        ToolCall(
            name="delete_text",
            arguments={"path": "note.txt", "start_line": 1, "end_line": 1},
        ),
        config,
    )
    assert "-line one" in diff["text"]

    diff = approval_diff(
        ToolCall(name="delete_file", arguments={"path": "note.txt"}), config
    )
    assert diff["label"] == "Delete note.txt" and "line one" in diff["text"]

    diff = approval_diff(
        ToolCall(name="apply_patch", arguments={"path": "n", "patch": "@@ x"}),
        config,
    )
    assert "@@ x" in diff["text"]

    assert (
        approval_diff(ToolCall(name="run_python", arguments={"code": "x"}), config)
        is None
    )
    assert (
        approval_diff(
            ToolCall(name="write_file", arguments={"path": "../evil", "content": "x"}),
            config,
        )
        is None
    )
