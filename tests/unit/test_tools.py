"""P2: tools, registry, path safety."""

from pathlib import Path

import pytest

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import ToolError
from agent_runtime.tools.filesystem import resolve_safe_path
from agent_runtime.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def authenticate_user(u): ...\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "notes.bin").write_bytes(b"\x00\x01binary\x00")
    return tmp_path


@pytest.fixture()
def registry(workspace: Path) -> ToolRegistry:
    return create_default_registry(AgentConfig(workspace_root=str(workspace)))


def test_default_registry_has_seven_tools(registry: ToolRegistry) -> None:
    names = sorted(t.name for t in registry.list())
    assert names == [
        "ask_user",
        "calculator",
        "get_time",
        "read_directory",
        "read_file",
        "search_files",
        "write_file",
    ]


def test_registry_duplicate_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register(registry.get("read_file"))


def test_read_file(registry: ToolRegistry) -> None:
    out = registry.get("read_file").handler(path="main.py")
    assert "print('hi')" in out


def test_read_file_missing(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError):
        registry.get("read_file").handler(path="nope.py")


def test_read_directory(registry: ToolRegistry) -> None:
    out = registry.get("read_directory").handler(path=".")
    assert "main.py" in out and "src" in out


def test_search_files_finds_with_line_numbers(registry: ToolRegistry) -> None:
    out = registry.get("search_files").handler(query="authenticate")
    assert "src/auth.py:1" in out


def test_search_files_skips_binaries_and_empty_query(registry: ToolRegistry) -> None:
    out = registry.get("search_files").handler(query="binary")
    assert "No matches" in out
    with pytest.raises(ToolError):
        registry.get("search_files").handler(query="  ")


def test_search_files_skips_venv(workspace: Path, registry: ToolRegistry) -> None:
    venv_file = workspace / ".venv" / "lib.py"
    venv_file.parent.mkdir()
    venv_file.write_text("authenticate secret\n", encoding="utf-8")
    out = registry.get("search_files").handler(query="authenticate")
    assert ".venv" not in out


def test_write_file_roundtrip(workspace: Path, registry: ToolRegistry) -> None:
    out = registry.get("write_file").handler(path="out.txt", content="hello")
    assert "Successfully wrote 5 characters" in out
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_no_parent_creation_by_default(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError):
        registry.get("write_file").handler(path="newdir/f.txt", content="x")


def test_write_file_parent_creation_when_allowed(workspace: Path) -> None:
    config = AgentConfig(workspace_root=str(workspace), allow_create_parent_dirs=True)
    reg = create_default_registry(config)
    reg.get("write_file").handler(path="newdir/f.txt", content="x")
    assert (workspace / "newdir" / "f.txt").exists()


def test_path_escape_blocked(workspace: Path) -> None:
    with pytest.raises(ToolError):
        resolve_safe_path("../outside.txt", str(workspace))
    outside = str(workspace.parent / "peer.txt")
    with pytest.raises(ToolError):
        resolve_safe_path(outside, str(workspace))


def test_calculator(registry: ToolRegistry) -> None:
    calc = registry.get("calculator").handler
    assert calc(expression="2 * (15 + 3)") == "36"
    assert calc(expression="10 % 3") == "1"
    with pytest.raises(ToolError):
        calc(expression="__import__('os').system('x')")
    with pytest.raises(ToolError):
        calc(expression="1/0")
    with pytest.raises(ToolError):
        calc(expression="True + 1")


def test_get_time(registry: ToolRegistry) -> None:
    assert "T" in registry.get("get_time").handler()
    assert "+00:00" in registry.get("get_time").handler(timezone="UTC")
    with pytest.raises(ToolError):
        registry.get("get_time").handler(timezone="Mars/Olympus")


def test_ask_user_injected() -> None:
    reg = create_default_registry(AgentConfig(), ask_fn=lambda q: f"answer:{q}")
    assert reg.get("ask_user").handler(question="q?") == "answer:q?"
