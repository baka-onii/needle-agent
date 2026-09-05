"""Regression tests for malformed output, resource limits, and workspace safety."""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_runtime.config import AgentConfig
from agent_runtime.execution.sanitizer import sanitize
from agent_runtime.execution.validator import validate
from agent_runtime.models.action import NeedleResult
from agent_runtime.protocol.parser import parse_response
from agent_runtime.tools.base import Tool, ToolCall, ToolError, truncate_text
from agent_runtime.tools.filesystem import resolve_safe_path
from agent_runtime.tools.registry import ToolRegistry, create_default_registry


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, True, "0.99"])
def test_invalid_confidence_is_not_accepted(score):
    with pytest.raises(ValidationError):
        NeedleResult(selected_tool="write_file", confidence=score)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tool_steps": 0},
        {"max_stalls": -1},
        {"max_context_chars": 0},
        {"confidence_threshold": math.nan},
        {"read_only_threshold": 1.1},
        {"search_context_lines": -1},
        {"max_search_results": True},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        AgentConfig(**kwargs)


def test_colliding_argument_names_are_rejected():
    with pytest.raises(ToolError, match="Duplicate"):
        sanitize(
            NeedleResult(
                selected_tool="write_file",
                confidence=0.99,
                arguments={"path": "a", " path ": "b", "content": "x"},
            )
        )


@pytest.mark.parametrize(
    "text",
    [
        "<tool><tool>Write a file</tool></tool>",
        "<tool>Read a file</final>",
        "<final>unfinished <tool>Write a file</tool>",
    ],
)
def test_malformed_nested_protocol_never_executes(text):
    assert parse_response(text).actions == []


def test_empty_final_still_wins_over_tool():
    assert parse_response("<tool>Write a file</tool><final></final>").final_answer == ""


def test_output_cap_includes_marker():
    for limit in (1, 10, 20, 100):
        assert len(truncate_text("x" * 200, limit)) <= limit


def test_filesystem_symlinks_and_metadata(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (root / "link.txt").symlink_to(outside)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("secret")
    registry = create_default_registry(AgentConfig(workspace_root=str(root)))
    for path in ("link.txt", "escape/secret.txt", "escape/new.txt", ".git/config", "x/../ok"):
        with pytest.raises(ToolError):
            resolve_safe_path(path, str(root))
    assert "link.txt" not in registry.get("read_directory").handler(path=".")
    assert "No matches" in registry.get("search_files").handler(query="secret")
    with pytest.raises(ToolError):
        registry.get("write_file").handler(path="escape/new.txt", content="no")
    assert not (tmp_path / "new.txt").exists()


def test_reads_and_searches_are_bounded(tmp_path: Path):
    (tmp_path / "large.txt").write_text("x" * 100_000)
    config = AgentConfig(
        workspace_root=str(tmp_path), max_tool_output_chars=80, max_search_file_bytes=100
    )
    registry = create_default_registry(config)
    assert len(registry.get("read_file").handler(path="large.txt")) <= 80
    assert "No matches" in registry.get("search_files").handler(query="x")
    (tmp_path / "late-binary").write_bytes(b"x" * 20 + b"\x00")
    assert "No matches" in registry.get("search_files").handler(query="x")


def test_read_only_blocks_direct_handler(tmp_path: Path):
    reg = create_default_registry(AgentConfig(workspace_root=str(tmp_path), read_only=True))
    with pytest.raises(ToolError, match="read-only"):
        reg.get("write_file").handler(path="x.txt", content="no")
    assert not (tmp_path / "x.txt").exists()


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** (9 ** 9)",
        "2 ** 999999999",
        "1e999",
        "(-1) ** 0.5",
        "[]",
        "sum([1, 2])",
        "1+" * 500 + "1",
        "1 << 10",
    ],
)
def test_calculator_resource_limits(expression):
    calc = create_default_registry(AgentConfig()).get("calculator").handler
    with pytest.raises(ToolError):
        calc(expression=expression)


def test_nullable_timezone():
    registry = create_default_registry(AgentConfig(default_timezone="UTC"))
    call = validate(ToolCall(name="get_time", arguments={"timezone": None}), registry)
    assert "+00:00" in registry.get("get_time").handler(**call.arguments)


def test_nested_validation_and_fail_closed_schemas():
    registry = ToolRegistry()
    registry.register(
        Tool(
            "nested",
            "example",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {"n": {"type": "integer", "minimum": 1}},
                            "required": ["n"],
                        },
                    },
                },
                "required": ["items"],
            },
        )
    )
    validate(ToolCall(name="nested", arguments={"items": [{"n": 1}]}), registry)
    for items in ([{"n": 0}], [{"n": True}], [{"n": 1, "extra": 2}], [{}], [1]):
        with pytest.raises(ToolError):
            validate(ToolCall(name="nested", arguments={"items": items}), registry)
    with pytest.raises(ValueError, match="unsupported"):
        registry.register(Tool("bad_schema", "d", {"type": "object", "anyOf": []}))
