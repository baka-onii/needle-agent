"""P1: core models, Tool definition, config, state."""

from agent_runtime.config import AgentConfig
from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.state import create_initial_state
from agent_runtime.tools.base import Tool, ToolCall, ToolResult


def test_tool_call_defaults() -> None:
    call = ToolCall(name="read_file")
    assert call.name == "read_file"
    assert call.arguments == {}


def test_tool_result_defaults() -> None:
    result = ToolResult(success=True, output="hi")
    assert result.success is True
    assert result.error is None


def test_needle_result_rankings() -> None:
    result = NeedleResult(
        selected_tool="search_files",
        arguments={"query": "auth"},
        confidence=0.94,
        rankings=[ToolRanking(tool_name="search_files", confidence=0.94)],
    )
    assert result.selected_tool == "search_files"
    assert result.rankings[0].confidence == 0.94


def test_tool_derives_both_views() -> None:
    tool = Tool(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path."}},
            "required": ["path"],
        },
        handler=lambda path: path,
    )
    schema = tool.needle_schema()
    assert schema["name"] == "read_file"
    assert schema["parameters"]["required"] == ["path"]
    desc = tool.reasoning_description()
    assert "read_file" in desc and "path" in desc


def test_config_defaults_match_spec() -> None:
    config = AgentConfig()
    assert config.confidence_threshold == 0.85
    assert config.read_only_threshold == 0.5
    assert config.max_stalls == 3
    assert config.max_tool_steps == 20
    assert config.max_tool_output_chars == 20_000
    assert config.max_search_results == 50
    assert config.search_context_lines == 2
    assert config.max_search_file_bytes == 2_000_000


def test_initial_state() -> None:
    state = create_initial_state("hello", 20)
    assert state["messages"] == [{"role": "user", "content": "hello"}]
    assert state["step_count"] == 0
    assert state["stall_count"] == 0
    assert state["status"] == "RUNNING"
    assert state["final_answer"] is None
