"""Reported live-model failure patterns reproduced deterministically, with real tools."""

import pytest

from agent_runtime import Agent, AgentConfig
from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.protocol.intent import write_action


class Reasoning:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.contexts = []

    def generate(self, messages):
        self.contexts.append(messages)
        return next(self.responses)


class Translator:
    def __init__(self, results):
        self.results = iter(results)
        self.actions = []

    def translate(self, action, tools):
        self.actions.append(action)
        return next(self.results)


def call(name, arguments, confidence=0.99, rankings=None):
    return NeedleResult(
        selected_tool=name, arguments=arguments, confidence=confidence, rankings=rankings or []
    )


def write(content, path="test.txt", confidence=0.99):
    return call("write_file", {"path": path, "content": content}, confidence)


def tool(action):
    return f"<tool>{action}</tool>"


def feedback(model):
    return "\n".join(message["content"] for message in model.contexts[-1][1:])


def run(tmp_path, responses, calls, **kwargs):
    model, action = Reasoning(responses), Translator(calls)
    agent = Agent(AgentConfig(workspace_root=str(tmp_path)), model, action, **kwargs)
    events = list(agent.stream("Work on the requested files."))
    return events[-1]["state"], events, model, action


def test_vague_content_is_rejected_then_reasoner_composes_ten_functions(tmp_path):
    text = "\n".join(
        f"{i}. {name}(): {description}"
        for i, (name, description) in enumerate(
            [
                ("print", "Display values"),
                ("len", "Count items"),
                ("range", "Create a range"),
                ("type", "Inspect a type"),
                ("int", "Convert to integer"),
                ("str", "Convert to text"),
                ("list", "Build a list"),
                ("sum", "Sum items"),
                ("sorted", "Return sorted items"),
                ("enumerate", "Pair indices and values"),
            ],
            start=1,
        )
    )
    approvals = []
    state, events, model, action = run(
        tmp_path,
        [
            tool("Write ten commonly used Python functions to test.txt."),
            tool(write_action("test.txt", text)),
            "<final>Wrote the ten functions.</final>",
        ],
        [write("ten commonly used Python functions"), write(text)],
        approve_fn=lambda proposal: approvals.append(proposal) or True,
    )
    assert state["status"] == "COMPLETED" and state["step_count"] == 1
    assert approvals == [] and len(action.actions) == 2
    assert (tmp_path / "test.txt").read_text() == text
    assert len(text.splitlines()) == 10
    first_rejection = next(event for event in events if event["type"] == "rejected")
    assert first_rejection["stage"] == "validate"
    assert "finished literal content" in first_rejection["message"]
    assert "Compose the requested content yourself" in feedback(model)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  indented\n\n",
        "literal \\n, not a newline",
        "Windows line endings\r\nare preserved\r\n",
        "नमस्ते 🐍\nprint('hi')\n",
        "```python\nprint('x')\n```\n<final>data</final>\n<tool>not a call</tool>",
        "Read one file and then write another. These words are literal file content.",
    ],
)
def test_literal_payload_survives_fences_unicode_and_protocol_tags(tmp_path, text):
    state, _, _, _ = run(
        tmp_path,
        [tool(write_action("test.txt", text)), "<final>Done.</final>"],
        [write(text)],
        approve_fn=lambda proposal: True,
    )
    assert state["status"] == "COMPLETED" and state["step_count"] == 1
    assert (tmp_path / "test.txt").read_bytes() == text.encode("utf-8")


def test_translator_cannot_summarize_or_truncate_a_payload(tmp_path):
    approvals = []
    state, events, _, _ = run(
        tmp_path,
        [tool(write_action("test.txt", "First line.\nSecond line.")), "Stopped."],
        [write("First line.")],
        approve_fn=lambda proposal: approvals.append(proposal) or True,
    )
    assert state["step_count"] == 0 and approvals == []
    assert not (tmp_path / "test.txt").exists()
    assert not any(event["type"] == "confidence" for event in events)
    assert any("changed or truncated" in event.get("message", "") for event in events)


@pytest.mark.parametrize(
    "action",
    [
        'Read "a.txt" and then write "b.txt".',
        'Use read_directory to list "."; read the file "a.txt".',
        'Search for "authentication" and read the matching file.',
        'Read the files "a.txt" and "b.txt".',
    ],
)
def test_compound_operations_never_reach_translator(tmp_path, action):
    state, events, _, adapter = run(tmp_path, [tool(action), "One operation at a time."], [])
    assert state["step_count"] == 0 and adapter.actions == []
    assert any(event["type"] == "rejected" and event["stage"] == "parse" for event in events)


def test_permission_questions_before_and_after_write_are_not_asked(tmp_path):
    approvals, questions = [], []
    state, events, _, _ = run(
        tmp_path,
        [
            tool("Ask the user for write permission."),
            tool(write_action("test.txt", "done")),
            tool("Ask whether I can save the file."),
            "<final>The write is complete.</final>",
        ],
        [
            call("ask_user", {"question": "May I write test.txt?"}),
            write("done"),
            call("ask_user", {"question": "Do you approve saving test.txt?"}),
        ],
        ask_fn=lambda question: questions.append(question) or "yes",
        approve_fn=lambda proposal: approvals.append(proposal) or True,
    )
    assert state["status"] == "COMPLETED" and state["step_count"] == 1
    assert approvals == [] and questions == []
    assert (tmp_path / "test.txt").read_text() == "done"
    assert sum(event["type"] == "tool_start" for event in events) == 1


def test_completed_delete_and_denied_path_are_not_reapproved(tmp_path):
    for approved in (True, False):
        target = tmp_path / "target.txt"
        target.write_text("remove me")
        approvals = []
        responses = [tool('Use delete_file to delete the file "target.txt".')] * 2
        responses.append("<final>Done.</final>")
        calls = [
            call("delete_file", {"path": "target.txt"}),
            call("delete_file", {"path": "target.txt"}),
        ]
        state, _, _, _ = run(
            tmp_path,
            responses,
            calls,
            approve_fn=lambda proposal, approvals=approvals, approved=approved: (
                approvals.append(proposal) or approved
            ),
        )
        assert len(approvals) == 1
        assert state["step_count"] == int(approved)
        assert target.exists() != approved


def test_denied_delete_cannot_reask_for_same_path(tmp_path):
    approvals = []
    state, _, _, _ = run(
        tmp_path,
        [
            tool('Use delete_file to delete the file "target.txt".'),
            tool('Use delete_file to delete the file "target.txt".'),
            "Denied.",
        ],
        [
            call("delete_file", {"path": "target.txt"}),
            call("delete_file", {"path": "target.txt"}),
        ],
        approve_fn=lambda proposal: approvals.append(proposal) or False,
    )
    assert state["step_count"] == 0 and len(approvals) == 1
    assert not (tmp_path / "target.txt").exists()


def test_answered_clarification_is_not_repeated(tmp_path):
    questions = []
    state, _, model, _ = run(
        tmp_path,
        [tool("Ask which filename."), tool("Ask which filename."), "Thanks."],
        [
            call("ask_user", {"question": "Which filename?"}),
            call("ask_user", {"question": "WHICH filename?!"}),
        ],
        ask_fn=lambda question: questions.append(question) or "report.txt",
    )
    assert state["step_count"] == 1 and len(questions) == 1
    assert "already answered" in feedback(model) and "report.txt" in feedback(model)


@pytest.mark.parametrize("bad_path", ["root", "user/home", "/", "/home/user"])
def test_root_aliases_are_rejected_not_silently_rewritten(tmp_path, bad_path):
    (tmp_path / "actual.txt").write_text("hello")
    intent = 'Use read_directory to list the directory ".".'
    state, events, model, _ = run(
        tmp_path,
        [tool(intent), tool(intent), "<final>Listed the workspace.</final>"],
        [call("read_directory", {"path": bad_path}), call("read_directory", {"path": "."})],
    )
    assert state["step_count"] == 1
    assert next(event for event in events if event["type"] == "tool_start")["arguments"] == {
        "path": "."
    }
    assert str(tmp_path) in model.contexts[0][0]["content"]
    assert '"actual.txt"' in model.contexts[0][0]["content"]
    assert "Preserve the specified path exactly" in feedback(model)


def test_real_directory_named_root_is_not_an_alias(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "a.txt").write_text("a")
    state, events, _, _ = run(
        tmp_path,
        [tool('Use read_directory to list the directory "root".'), "Done."],
        [call("read_directory", {"path": "root"})],
    )
    assert state["step_count"] == 1
    assert "root/a.txt" in next(
        event["output"] for event in events if event["type"] == "tool_result"
    )


def test_low_confidence_review_names_highest_candidate_then_retranslates(tmp_path):
    rankings = [
        ToolRanking(tool_name="read_file", confidence=0.1),
        ToolRanking(tool_name="read_directory", confidence=0.4),
    ]
    state, events, model, translator = run(
        tmp_path,
        [
            tool("Inspect the workspace."),
            tool('Use read_directory to list the directory ".".'),
            "Done.",
        ],
        [
            call("read_directory", {"path": "."}, 0.4, rankings),
            call("read_directory", {"path": "."}),
        ],
    )
    review = next(event for event in events if event["type"] == "confirmation")
    assert review["stage"] == "confidence" and review["suggested_tool"] == "read_directory"
    assert review["candidates"][0] == {"tool_name": "read_directory", "confidence": 0.4}
    assert review["arguments"] == {"path": "."}
    next_context = "\n".join(item["content"] for item in model.contexts[1])
    assert "Is the highest-ranked available tool 'read_directory' correct?" in next_context
    assert "Proposed arguments" in next_context and "<tool>" in next_context
    assert translator.actions[1].startswith("Use read_directory")
    assert state["step_count"] == 1 and state["status"] == "COMPLETED"


def test_reasoner_can_choose_a_different_candidate(tmp_path):
    state, events, _, _ = run(
        tmp_path,
        [
            tool("Inspect the project."),
            tool('Use read_directory to list the directory ".".'),
            "Done.",
        ],
        [
            call(
                "search_files",
                {"query": "project"},
                0.4,
                [
                    ToolRanking(tool_name="search_files", confidence=0.4),
                    ToolRanking(tool_name="read_directory", confidence=0.35),
                ],
            ),
            call("read_directory", {"path": "."}),
        ],
    )
    assert state["step_count"] == 1
    assert [event["tool"] for event in events if event["type"] == "tool_start"] == [
        "read_directory"
    ]


def test_needles_selection_must_match_reasoners_explicit_choice(tmp_path):
    state, events, model, _ = run(
        tmp_path,
        [tool('Use read_directory to list the directory ".".'), "Wrong selection."],
        [call("search_files", {"query": "anything"})],
    )
    assert state["step_count"] == 0
    assert "reasoning model selected 'read_directory'" in feedback(model)
    assert not any(event["type"] == "confidence" for event in events)


def test_confirmation_never_overrides_low_confidence_or_approval(tmp_path):
    approvals = []
    intent = tool(write_action("test.txt", "hello"))
    state, events, _, _ = run(
        tmp_path,
        [intent] * 3,
        [write("hello", confidence=0.6)] * 3,
        approve_fn=lambda proposal: approvals.append(proposal) or True,
    )
    assert state["status"] == "STALLED" and state["step_count"] == 0
    assert not approvals and not (tmp_path / "test.txt").exists()
    assert sum(event["type"] == "confirmation" for event in events) == 3


def test_execution_failure_queries_reasoner_with_tool_arguments_and_candidates(tmp_path):
    state, events, model, _ = run(
        tmp_path,
        [
            tool('Use read_file to read the file "missing.py".'),
            tool('Use read_directory to list the directory ".".'),
            "Done.",
        ],
        [call("read_file", {"path": "missing.py"}), call("read_directory", {"path": "."})],
    )
    assert state["step_count"] == 2
    review = next(event for event in events if event["type"] == "confirmation")
    assert review["stage"] == "execute" and review["selected_tool"] == "read_file"
    assert review["arguments"]["path"] == "missing.py"
    text = "\n".join(item["content"] for item in model.contexts[1])
    assert "Was the selected tool correct?" in text and "missing.py" in text


def test_same_failed_call_has_a_bounded_retry_budget(tmp_path):
    intent = tool('Use read_file to read the file "missing.py".')
    state, events, _, _ = run(
        tmp_path, [intent] * 5, [call("read_file", {"path": "missing.py"})] * 5
    )
    assert state["status"] == "STALLED" and state["step_count"] == 2
    assert sum(event["type"] == "tool_start" for event in events) == 2
