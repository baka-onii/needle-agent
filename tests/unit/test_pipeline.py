"""P4: sanitizer, validator, confidence, executor."""

import pytest

from agent_runtime.config import AgentConfig
from agent_runtime.execution.confidence import (
    is_confident,
    low_confidence_message,
    threshold_for,
)
from agent_runtime.execution.executor import execute
from agent_runtime.execution.sanitizer import sanitize
from agent_runtime.execution.validator import validate
from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.tools.base import Tool, ToolCall, ToolError
from agent_runtime.tools.registry import create_default_registry


@pytest.fixture()
def registry():
    return create_default_registry(AgentConfig(workspace_root="."))


def test_sanitize_valid() -> None:
    call = sanitize(
        NeedleResult(selected_tool=" read_file ", arguments={"path": "x.py"}, confidence=0.9)
    )
    assert call == ToolCall(name="read_file", arguments={"path": "x.py"})


def test_sanitize_rejects_empty_selection() -> None:
    with pytest.raises(ToolError):
        sanitize(NeedleResult(selected_tool=None, arguments={}, confidence=0.1))
    with pytest.raises(ToolError):
        sanitize(NeedleResult(selected_tool="  ", arguments={}, confidence=0.1))


def test_validate_ok(registry) -> None:
    call = validate(ToolCall(name="calculator", arguments={"expression": "1+1"}), registry)
    assert call.name == "calculator"


def test_validate_unknown_tool_rejected_even_if_confident(registry) -> None:
    with pytest.raises(ToolError):
        validate(ToolCall(name="shell", arguments={}), registry)


def test_validate_missing_required(registry) -> None:
    with pytest.raises(ToolError):
        validate(ToolCall(name="read_file", arguments={}), registry)


def test_validate_wrong_type_and_unexpected_arg(registry) -> None:
    with pytest.raises(ToolError):
        validate(ToolCall(name="read_file", arguments={"path": 42}), registry)
    with pytest.raises(ToolError):
        validate(ToolCall(name="read_file", arguments={"path": "x", "extra": 1}), registry)


def test_validate_constraints() -> None:
    reg = create_default_registry(AgentConfig())
    tool = Tool(
        name="bounded",
        description="t",
        parameters={
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 5},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["level", "mode"],
        },
        handler=lambda level, mode: "ok",
    )
    reg.register(tool)
    validate(ToolCall(name="bounded", arguments={"level": 3, "mode": "a"}), reg)
    with pytest.raises(ToolError):
        validate(ToolCall(name="bounded", arguments={"level": 9, "mode": "a"}), reg)
    with pytest.raises(ToolError):
        validate(ToolCall(name="bounded", arguments={"level": 1, "mode": "z"}), reg)


def test_per_tool_thresholds() -> None:
    config = AgentConfig()
    assert threshold_for("read_file", config) == 0.5
    assert threshold_for("search_files", config) == 0.5
    assert threshold_for("write_file", config) == 0.85
    assert threshold_for("ask_user", config) == 0.85
    assert threshold_for("unknown_tool", config) == 0.85


def test_confidence_boundary_and_message() -> None:
    assert is_confident(0.85, 0.85) is True
    assert is_confident(0.849, 0.85) is False
    msg = low_confidence_message(
        "do stuff", [ToolRanking(tool_name="search_files", confidence=0.52)]
    )
    assert "uncertain" in msg and "search_files" in msg and "do stuff" in msg


def test_execute_success_and_tool_error(registry) -> None:
    ok = execute(
        ToolCall(name="calculator", arguments={"expression": "2+2"}), registry, AgentConfig()
    )
    assert ok.success and ok.output == "4"
    fail = execute(
        ToolCall(name="calculator", arguments={"expression": "1/0"}), registry, AgentConfig()
    )
    assert not fail.success and fail.error


def test_execute_truncates() -> None:
    config = AgentConfig(max_tool_output_chars=10)
    reg = create_default_registry(config)
    result = execute(
        ToolCall(name="calculator", arguments={"expression": "123456 + 0"}), reg, config
    )
    assert result.success  # short output untouched
    reg.get("calculator").handler = lambda expression: "x" * 50
    result = execute(ToolCall(name="calculator", arguments={"expression": "1"}), reg, config)
    assert result.success and "truncated" in result.output
