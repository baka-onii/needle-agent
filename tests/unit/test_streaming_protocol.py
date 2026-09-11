"""Display parsing across arbitrary token boundaries must not change executable intent."""

import pytest

from relay.protocol.intent import write_action
from relay.protocol.parser import parse_response
from relay.protocol.stream import ResponseStream

CASES = [
    ("Hello, world.", [("text", "Hello, world.")]),
    ("I will look. <tool>Read a.txt.</tool>", [("text", "I will look. "), ("tool", "Read a.txt.")]),
    (
        "<final >Hello <tool>an example</tool>.</final >",
        [("final", "Hello <tool>an example</tool>.")],
    ),
    (
        "<think>Check <tool>inert</tool>.</think><final>Done.</final>",
        [("reasoning", "Check <tool>inert</tool>."), ("final", "Done.")],
    ),
    ("<tool>A</tool>\n<tool>B</tool>", [("tool", "A"), ("text", "\n"), ("tool", "B")]),
    ("```text\n<tool>literal</tool>\n```\n", [("text", "```text\n<tool>literal</tool>\n```\n")]),
    (
        "<final>नमस्ते 🐍\n```python\nprint('hi')\n```</final>",
        [("final", "नमस्ते 🐍\n```python\nprint('hi')\n```")],
    ),
]


@pytest.mark.parametrize("text, expected", CASES)
def test_projection_at_every_split(text, expected):
    for split in range(len(text) + 1):
        projection = ResponseStream()
        updates = [
            *projection.feed(text[:split]),
            *projection.feed(text[split:]),
            *projection.finish(),
        ]
        assert [(part.kind, part.text) for part in projection.parts] == expected
        reconstructed = {}
        for update in updates:
            reconstructed.setdefault(update["index"], "")
            reconstructed[update["index"]] += update["text"]
        assert [reconstructed[i] for i in range(len(expected))] == [body for _, body in expected]


def test_partial_tag_names_are_not_exposed_as_content():
    projection = ResponseStream()
    projection.feed("Look <fi")
    assert projection.parts[0].text == "Look "
    projection.feed("nal>Hel")
    assert [(part.kind, part.text) for part in projection.parts] == [
        ("text", "Look "),
        ("final", "Hel"),
    ]
    projection.feed("lo</fi")
    assert projection.parts[-1].text == "Hello" and not projection.parts[-1].complete
    projection.feed("nal>")
    assert projection.parts[-1].complete


def test_literal_payload_fences_do_not_open_fake_final_or_tool_sections():
    content = '```python\nprint("hello")\n```\n<final>literal</final>\n<tool>also literal</tool>'
    text = f"<tool>{write_action('report.txt', content)}</tool>"
    projection = ResponseStream()
    for char in text:
        projection.feed(char)
    projection.finish()
    assert len(projection.parts) == 1
    assert projection.parts[0].kind == "tool"
    assert projection.assistant_text == text
    assert parse_response(projection.assistant_text).actions
    assert parse_response(projection.assistant_text).final_answer is None


def test_thought_tags_never_authorize_execution_even_if_nested_or_unclosed():
    for text in (
        "<think><tool>Do something.</tool></think>",
        "<think><think>inner</think><tool>Still a thought.</tool></think>",
        "<think>Unclosed thought <tool>Do not execute.</tool>",
    ):
        projection = ResponseStream()
        for char in text:
            projection.feed(char)
        projection.finish()
        assert projection.assistant_text == ""
        assert not parse_response(text).actions


def test_fenced_think_tags_are_literal_file_content():
    text = f"<tool>{write_action('report.txt', '<think>Literal</think>')}</tool>"
    assert parse_response(text).actions[0].instruction.endswith("<think>Literal</think>\n```")


def test_late_final_keeps_the_completed_tool_preview_non_authoritative():
    projection = ResponseStream()
    projection.feed("<tool>Calculate 2+2.</tool>")
    assert projection.parts[0].complete  # Display completion is not execution authorization.
    projection.feed("<final>Use this answer instead.</final>")
    projection.finish()
    parsed = parse_response(projection.assistant_text)
    assert parsed.final_answer == "Use this answer instead." and parsed.actions == []


def test_unclosed_block_is_marked_incomplete_and_cannot_be_extended_after_finish():
    projection = ResponseStream()
    projection.feed("<tool>Read the")
    projection.finish()
    assert not projection.parts[0].complete
    assert not parse_response(projection.assistant_text).actions
    with pytest.raises(ValueError, match="completed"):
        projection.feed(" rest")
