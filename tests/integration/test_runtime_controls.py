"""Long runs, event ordering, cancellation, permissions, and conversation recovery."""

from pathlib import Path

from agent_runtime import Agent, AgentConfig, Tool, ToolRegistry
from agent_runtime.models.action import NeedleResult


class Reasoning:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.contexts = []

    def generate(self, messages):
        self.contexts.append(messages)
        return next(self.turns)


class CalculatorAction:
    def translate(self, action, tools):
        return NeedleResult(
            selected_tool="calculator", arguments={"expression": "1+1"}, confidence=0.9
        )


def test_default_twenty_step_limit_does_not_hit_langgraph_limit():
    model = Reasoning(["<tool>Calculate 1+1.</tool>"] * 21)
    state = Agent(reasoning=model, action=CalculatorAction()).run("Keep calculating")
    assert state["status"] == "MAX_STEPS_REACHED"
    assert state["step_count"] == 20
    assert len(model.contexts) == 21  # last observation gets a reasoning turn
    assert "20" in state["final_answer"]


def test_stream_pipeline_order_and_single_action():
    model = Reasoning(
        ["<tool>Calculate 1+1.</tool><tool>Calculate 2+2.</tool>", "<final>2</final>"]
    )
    events = list(Agent(reasoning=model, action=CalculatorAction()).stream("One calculation"))
    phases = [event["node"] for event in events if event["type"] == "phase"]
    assert phases == [
        "reason",
        "parse",
        "translate",
        "sanitize",
        "validate",
        "confidence",
        "safety",
        "execute",
        "observe",
        "update_context",
        "reason",
        "parse",
    ]
    assert events[-1]["type"] == "complete"
    assert events[-1]["state"]["step_count"] == 1
    assert any("Observation" in m["content"] for m in model.contexts[1])
    assert next(e for e in events if e["type"] == "action")["ignored_actions"] == 1


def test_invalid_call_never_reaches_confidence():
    class InvalidAction:
        def translate(self, action, tools):
            return NeedleResult(selected_tool="write_file", arguments={"path": "a"}, confidence=1.0)

    model = Reasoning(["<tool>Write a.</tool>", "No write was made."])
    events = list(Agent(reasoning=model, action=InvalidAction()).stream("Write"))
    assert not any(e["type"] in {"confidence", "tool_start"} for e in events)
    assert events[-1]["state"]["status"] == "COMPLETED"


def test_cancel_before_execution():
    stopped = False
    model = Reasoning(["<tool>Calculate 1+1.</tool>"])
    agent = Agent(reasoning=model, action=CalculatorAction())
    events = []

    def cancelled():
        return stopped

    for event in agent.stream("Calculate", cancelled=cancelled):
        events.append(event)
        if event["type"] == "confidence":
            stopped = True
    assert events[-1]["state"]["status"] == "CANCELLED"
    assert not any(e["type"] == "tool_start" for e in events)


def test_followup_receives_prior_context():
    model = Reasoning(["<final>Your favorite color is green.</final>", "<final>Green.</final>"])
    agent = Agent(reasoning=model, action=CalculatorAction())
    first = agent.run("My favorite color is green.")
    second = agent.run("What is my favorite color?", history=first["messages"])
    assert second["status"] == "COMPLETED"
    assert any("My favorite color is green" in m["content"] for m in model.contexts[-1])


def test_write_approval_and_path_safety(tmp_path: Path):
    class WriteAction:
        path = "note.txt"

        def translate(self, action, tools):
            return NeedleResult(
                selected_tool="write_file",
                arguments={"path": self.path, "content": "hello"},
                confidence=0.99,
            )

    action = WriteAction()
    approved = []
    config = AgentConfig(workspace_root=str(tmp_path))
    agent = Agent(
        config,
        Reasoning(["<tool>Write note.txt.</tool>", "Denied."]),
        action,
        approve_fn=lambda call: approved.append(call) or False,
    )
    assert agent.run("Write")["step_count"] == 0
    assert len(approved) == 1
    assert not (tmp_path / "note.txt").exists()
    action.path = "../outside.txt"
    approved.clear()
    agent = Agent(
        config,
        Reasoning(["<tool>Write outside.</tool>", "Blocked."]),
        action,
        approve_fn=lambda call: approved.append(call) or True,
    )
    assert agent.run("Write")["step_count"] == 0
    assert approved == []  # unsafe path never even asks permission


def test_internal_handler_error_terminates():
    registry = ToolRegistry()

    def broken():
        raise RuntimeError("Internal bug")

    registry.register(Tool("broken", "d", {"type": "object"}, broken))

    class Action:
        def translate(self, action, tools):
            return NeedleResult(selected_tool="broken", confidence=1.0)

    agent = Agent(
        reasoning=Reasoning(["<tool>Try broken.</tool>"]), action=Action(), registry=registry
    )
    state = agent.run("Test")
    assert state["status"] == "ERROR"
    assert "Internal bug" in state["final_answer"]


def test_ask_user_answer_returns_to_reasoning():
    class AskAction:
        def translate(self, action, tools):
            return NeedleResult(
                selected_tool="ask_user", arguments={"question": "Which file?"}, confidence=0.99
            )

    model = Reasoning(["<tool>Ask which file.</tool>", "<final>Thanks.</final>"])
    agent = Agent(reasoning=model, action=AskAction(), ask_fn=lambda question: "README.md")
    assert agent.run("Read a file")["step_count"] == 1
    assert any("README.md" in m["content"] for m in model.contexts[-1])
