"""P8: full loop with mocked reasoning + action models (no LLM, spec §37)."""

from pathlib import Path
from typing import Any

import pytest

from agent_runtime.agent import Agent
from agent_runtime.config import AgentConfig
from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.tools.base import Tool
from agent_runtime.tools.registry import create_default_registry


class ScriptedReasoning:
    """Return queued responses in order; repeat the last when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]]) -> str:
        assert messages[0]["role"] == "system"  # system prompt always present
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


class StubAction:
    def __init__(self, mapping: dict[str, NeedleResult]) -> None:
        self.mapping = mapping

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        for key, result in self.mapping.items():
            if key in action:
                return result
        return NeedleResult(selected_tool=None, arguments={}, confidence=0.0)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "config.py").write_text("AUTH_BACKEND = 'local'\n", encoding="utf-8")
    (tmp_path / "src" / "auth.py").write_text(
        "def authenticate_user(user):\n    return True\n", encoding="utf-8"
    )
    return tmp_path


def _agent(workspace: Path, reasoning: ScriptedReasoning, action: StubAction, **kw: Any) -> Agent:
    config = AgentConfig(workspace_root=str(workspace), **kw)
    registry = create_default_registry(config)
    return Agent(config=config, reasoning=reasoning, action=action, registry=registry)


def test_reasoning_backend_failure_ends_run_gracefully(workspace: Path) -> None:
    class DeadBackend:
        def generate(self, messages: list[dict[str, Any]]) -> str:
            raise RuntimeError("connection refused")

    action = StubAction({})
    state = _agent(workspace, DeadBackend(), action).run("hello?")
    assert state["status"] == "ERROR"
    assert state["step_count"] == 0
    assert "Reasoning model failed" in (state["final_answer"] or "")


def test_repeated_low_confidence_stalls_with_bounded_turns(workspace: Path) -> None:
    reasoning = ScriptedReasoning(["<tool>\nDo something vague.\n</tool>"])
    action = StubAction(
        {
            "vague": NeedleResult(
                selected_tool="search_files",
                arguments={"query": "x"},
                confidence=0.10,
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("vague request")
    assert state["status"] == "STALLED"
    assert state["step_count"] == 0
    assert reasoning.calls == 3  # initial + 2 retries, then stall terminates


def test_read_only_tool_clears_lower_gate(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            "<tool>\nAdd one and one.\n</tool>",
            "<final>\nTwo.\n</final>",
        ]
    )
    action = StubAction(
        {
            "Add": NeedleResult(
                selected_tool="calculator", arguments={"expression": "1+1"}, confidence=0.55
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("add")
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 1  # 0.55 clears the 0.5 read-only gate


def test_mutating_tool_held_to_strict_gate(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            "<tool>\nWrite hi.\n</tool>",
            "I will not write without confidence.",
        ]
    )
    action = StubAction(
        {
            "Write": NeedleResult(
                selected_tool="write_file",
                arguments={"path": "h.txt", "content": "hi"},
                confidence=0.60,
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("write")
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 0  # 0.60 < 0.85 strict gate: never executes
    assert not (workspace / "h.txt").exists()


def test_end_to_end_auth_search(workspace: Path) -> None:
    """Spec §38 scenario: search → read → final, with real filesystem tools."""
    reasoning = ScriptedReasoning(
        [
            "<tool>\nSearch the project for authentication-related code.\n</tool>",
            "<tool>\nRead src/auth.py.\n</tool>",
            "<final>\nThe authentication implementation is in src/auth.py.\n</final>",
        ]
    )
    action = StubAction(
        {
            "authentication": NeedleResult(
                selected_tool="search_files",
                arguments={"query": "authentication", "path": "."},
                confidence=0.94,
            ),
            "Read src/auth.py": NeedleResult(
                selected_tool="read_file",
                arguments={"path": "src/auth.py"},
                confidence=0.99,
            ),
        }
    )
    state = _agent(workspace, reasoning, action).run("Find the auth implementation.")
    assert state["status"] == "COMPLETED"
    assert "src/auth.py" in (state["final_answer"] or "")
    assert state["step_count"] == 2
    assert reasoning.calls == 3  # reason after every execution (§32)


def test_low_confidence_returns_to_reasoning_without_executing(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            "<tool>\nDo something vague.\n</tool>",
            "I don't have enough information to perform that action.",
        ]
    )
    action = StubAction(
        {
            "vague": NeedleResult(
                selected_tool="search_files",
                arguments={"query": "x"},
                confidence=0.40,
                rankings=[
                    ToolRanking(tool_name="search_files", confidence=0.40),
                    ToolRanking(tool_name="read_directory", confidence=0.37),
                ],
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("vague request")
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 0
    assert any("uncertain" in m.get("content", "") for m in state["messages"])


def test_max_steps_terminates(workspace: Path) -> None:
    reasoning = ScriptedReasoning(["<tool>\nAdd one and one.\n</tool>"])
    action = StubAction(
        {
            "Add": NeedleResult(
                selected_tool="calculator",
                arguments={"expression": "1+1"},
                confidence=0.99,
            )
        }
    )
    state = _agent(workspace, reasoning, action, max_tool_steps=2).run("loop forever")
    assert state["status"] == "MAX_STEPS_REACHED"
    assert state["step_count"] == 2


def test_invalid_tool_becomes_observation_and_recovers(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            "<tool>\nOpen a shell.\n</tool>",
            "<final>\nNo shell tool is available; done.\n</final>",
        ]
    )
    action = StubAction(
        {"shell": NeedleResult(selected_tool="shell", arguments={}, confidence=0.99)}
    )
    state = _agent(workspace, reasoning, action).run("open a shell")
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 0  # high-confidence invalid never executes
    assert any("Unknown tool" in m.get("content", "") for m in state["messages"])
