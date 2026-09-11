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
    approve_fn = kw.pop("approve_fn", None)
    config = AgentConfig(workspace_root=str(workspace), **kw)
    registry = create_default_registry(config)
    return Agent(
        config=config,
        reasoning=reasoning,
        action=action,
        registry=registry,
        approve_fn=approve_fn,
    )


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


def test_content_block_write_attaches_without_approval(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            '<tool>Use write_file to write the file "note.txt". '
            "<content>Hello block</content></tool>",
            "<final>Done.</final>",
        ]
    )
    action = StubAction(
        {
            "write_file": NeedleResult(
                selected_tool="write_file", arguments={"path": "note.txt"}, confidence=1.0
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("write a note")
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 1
    assert (workspace / "note.txt").read_text() == "Hello block"


def test_translator_payload_output_is_discarded(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            '<tool>Use write_file to write the file "note.txt". '
            "<content>real bytes</content></tool>",
            "<final>Done.</final>",
        ]
    )
    action = StubAction(
        {
            "write_file": NeedleResult(
                selected_tool="write_file",
                arguments={"path": "note.txt", "content": "invented garbage"},
                confidence=1.0,
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("write a note")
    assert state["step_count"] == 1
    assert (workspace / "note.txt").read_text() == "real bytes"


def test_two_block_replace_attaches_positionally_with_approval(workspace: Path) -> None:
    (workspace / "note.txt").write_text("old words here")
    approvals = []
    reasoning = ScriptedReasoning(
        [
            '<tool>Use replace_text to fix "note.txt". '
            "<text-1>old words</text-1> <text-2>new words</text-2></tool>",
            "<final>Done.</final>",
        ]
    )
    action = StubAction(
        {
            "replace_text": NeedleResult(
                selected_tool="replace_text", arguments={"path": "note.txt"}, confidence=1.0
            )
        }
    )
    agent = _agent(
        workspace,
        reasoning,
        action,
        approve_fn=lambda call: approvals.append(call) or True,
    )
    state = agent.run("fix the note")
    assert state["step_count"] == 1
    assert len(approvals) == 1
    assert (workspace / "note.txt").read_text() == "new words here"


def test_block_count_mismatch_never_executes(workspace: Path) -> None:
    (workspace / "note.txt").write_text("keep me")
    reasoning = ScriptedReasoning(
        [
            '<tool>Use replace_text to fix "note.txt". <content>only one</content></tool>',
            "<final>Gave up.</final>",
        ]
    )
    action = StubAction(
        {
            "replace_text": NeedleResult(
                selected_tool="replace_text", arguments={"path": "note.txt"}, confidence=1.0
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("fix the note")
    assert state["step_count"] == 0
    assert (workspace / "note.txt").read_text() == "keep me"


def test_blocks_on_blockless_tool_are_rejected(workspace: Path) -> None:
    reasoning = ScriptedReasoning(
        [
            '<tool>Use read_file to read the file "main.py". <content>stowaway</content></tool>',
            "<final>Gave up.</final>",
        ]
    )
    action = StubAction(
        {
            "read_file": NeedleResult(
                selected_tool="read_file", arguments={"path": "main.py"}, confidence=1.0
            )
        }
    )
    state = _agent(workspace, reasoning, action).run("read main")
    assert state["step_count"] == 0


def test_severe_tool_asks_and_honors_denial(workspace: Path) -> None:
    approvals = []
    reasoning = ScriptedReasoning(
        ["<tool>Use run_python to run code.</tool>", "<final>Denied.</final>"]
    )
    action = StubAction(
        {
            "run_python": NeedleResult(
                selected_tool="run_python", arguments={"code": "print(1)"}, confidence=1.0
            )
        }
    )
    agent = _agent(
        workspace,
        reasoning,
        action,
        approve_fn=lambda call: approvals.append(call) or False,
    )
    state = agent.run("run code")
    assert state["step_count"] == 0
    assert len(approvals) == 1


def test_severe_tool_executes_when_approved(workspace: Path) -> None:
    approvals = []
    reasoning = ScriptedReasoning(
        ["<tool>Use run_python to run code.</tool>", "<final>Ran it.</final>"]
    )
    action = StubAction(
        {
            "run_python": NeedleResult(
                selected_tool="run_python", arguments={"code": "print(41 + 1)"}, confidence=1.0
            )
        }
    )
    agent = _agent(
        workspace,
        reasoning,
        action,
        approve_fn=lambda call: approvals.append(call) or True,
    )
    state = agent.run("run code")
    assert state["step_count"] == 1
    assert len(approvals) == 1


def test_reasoning_model_start_reports_context_chars(workspace: Path) -> None:
    reasoning = ScriptedReasoning(["<final>Done.</final>"])
    events = list(_agent(workspace, reasoning, StubAction({})).stream("hello"))
    starts = [
        event
        for event in events
        if event.get("type") == "model_start" and event.get("component") == "reasoning"
    ]
    assert starts
    assert all(event["context_chars"] > 0 for event in starts)


def test_translator_placeholder_never_executes(workspace: Path) -> None:
    placeholder = NeedleResult(
        selected_tool="run_python",
        arguments={"code": "__PAYLOAD_CODE__"},
        confidence=1.0,
    )
    # With a real block the payload wins and the tool runs.
    reasoning = ScriptedReasoning(
        [
            "<tool>Use run_python to run this. <content>print('BLOCK_WINS')</content></tool>",
            "<final>Done.</final>",
        ]
    )
    events = list(
        _agent(workspace, reasoning, StubAction({"run_python": placeholder}),
               approve_fn=lambda call: True).stream("block wins")
    )
    starts = [e for e in events if e.get("type") == "tool_start"]
    assert [e["arguments"] for e in starts] == [{"code": "print('BLOCK_WINS')"}]
    # Without a block the placeholder fails closed: retry, zero executions.
    reasoning = ScriptedReasoning(["<tool>Use run_python to run this.</tool>"] * 4)
    events = list(
        _agent(workspace, reasoning, StubAction({"run_python": placeholder}),
               approve_fn=lambda call: True).stream("no block")
    )
    assert not [e for e in events if e.get("type") == "tool_start"]
    assert any(
        e.get("type") == "rejected" and "__PAYLOAD" not in e.get("message", "")
        and "content" in e.get("message", "")
        for e in events
    )


def test_approval_with_edited_arguments_executes_the_edit(workspace: Path) -> None:
    from agent_runtime.tools.base import ToolCall

    reasoning = ScriptedReasoning(
        ["<tool>Use run_python to run code.</tool>", "<final>Ran it.</final>"]
    )
    action = StubAction(
        {
            "run_python": NeedleResult(
                selected_tool="run_python",
                arguments={"code": "print(1)"},
                confidence=1.0,
            )
        }
    )
    edited = ToolCall(name="run_python", arguments={"code": "print(40 + 2)"})
    agent = _agent(workspace, reasoning, action, approve_fn=lambda call: edited)
    state = agent.run("run code")
    assert state["step_count"] == 1
    assert state["last_tool_result"]["output"] == "42"


VALID_SUMMARY = """task:
  objective: "Add OAuth authentication"

constraints:
  - "Keep existing JWT authentication"

completed:
  - "Refactored token validation"

current:
  subtask: "Implement Google provider"

files:
  primary:
    - src/auth/google.py
  related:
    - src/auth/auth.py

decisions:
  - "Use existing JWT implementation"

blockers: []
"""


class SummaryReasoning(ScriptedReasoning):
    """Answer snapshot requests with a fixed summary, else queued actions."""

    def __init__(self, responses: list[str], summary: str) -> None:
        super().__init__(responses)
        self.summary = summary

    def generate(self, messages: list[dict[str, Any]]) -> str:
        if "structured state snapshot" in messages[-1]["content"]:
            return self.summary
        return super().generate(messages)


def _big_history() -> list[dict[str, Any]]:
    filler = "context filler text. " * 60  # ~1260 chars of history
    return [
        {"role": "user", "content": "original big task"},
        {"role": "assistant", "content": filler},
        {"role": "user", "content": filler},
        {"role": "assistant", "content": filler},
    ]


def test_loop_compresses_past_threshold(workspace: Path) -> None:
    reasoning = SummaryReasoning(
        [
            "<tool>Use calculator to calculate 1 + 1.</tool>",
            "<final>Two.</final>",
        ],
        VALID_SUMMARY,
    )
    action = StubAction(
        {
            "calculate": NeedleResult(
                selected_tool="calculator",
                arguments={"expression": "1 + 1"},
                confidence=1.0,
            )
        }
    )
    agent = _agent(
        workspace, reasoning, action, max_context_chars=12000, max_context_tokens=100000
    )
    events = list(agent.stream("Calculate 1+1", history=_big_history()))
    compressed = [e for e in events if e.get("type") == "context_compressed"]
    assert len(compressed) == 1
    assert compressed[0]["after_chars"] < compressed[0]["before_chars"]
    final = [e for e in events if e.get("type") == "complete"]
    assert final and final[0]["state"]["status"] == "COMPLETED"


def test_loop_skips_compression_when_summary_unusable(workspace: Path) -> None:
    reasoning = SummaryReasoning(
        [
            "<tool>Use calculator to calculate 1 + 1.</tool>",
            "<final>Two.</final>",
        ],
        "Just some prose, no shape.",
    )
    action = StubAction(
        {
            "calculate": NeedleResult(
                selected_tool="calculator",
                arguments={"expression": "1 + 1"},
                confidence=1.0,
            )
        }
    )
    agent = _agent(
        workspace, reasoning, action, max_context_chars=12000, max_context_tokens=100000
    )
    events = list(agent.stream("Calculate 1+1", history=_big_history()))
    assert any(e.get("type") == "context_compression_skipped" for e in events)
    assert not [e for e in events if e.get("type") == "context_compressed"]
    final = [e for e in events if e.get("type") == "complete"]
    assert final and final[0]["state"]["status"] == "COMPLETED"
