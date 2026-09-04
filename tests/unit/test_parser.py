"""P3: protocol parser."""

from agent_runtime.protocol.parser import parse_response


def test_basic_tool_block() -> None:
    parsed = parse_response("I need to inspect first.\n\n<tool>\nRead the directory.\n</tool>")
    assert parsed.actions == ["Read the directory."]
    assert parsed.final_answer is None
    assert "inspect" in parsed.reasoning


def test_multiline_action() -> None:
    parsed = parse_response("<tool>\nSearch for auth\nin all files.\n</tool>")
    assert parsed.actions == ["Search for auth\nin all files."]


def test_multiple_tool_blocks_in_order() -> None:
    parsed = parse_response("<tool>A</tool>\ntext\n<tool>B</tool>")
    assert parsed.actions == ["A", "B"]


def test_final_block() -> None:
    parsed = parse_response("<final>\nDone: it is in src/auth.py.\n</final>")
    assert parsed.final_answer == "Done: it is in src/auth.py."
    assert parsed.actions == []


def test_tagless_response_is_final() -> None:
    parsed = parse_response("I don't have enough information.")
    assert parsed.final_answer == "I don't have enough information."
    assert parsed.actions == []


def test_unclosed_tool_tag_is_not_executable() -> None:
    parsed = parse_response("Please <tool>\nRead everything")
    assert parsed.actions == []


def test_empty_tool_block_dropped() -> None:
    parsed = parse_response("<tool>   </tool>")
    assert parsed.actions == []
    assert parsed.final_answer == "<tool>   </tool>"


def test_whitespace_tags() -> None:
    parsed = parse_response("<tool >  spaced  </tool >")
    assert parsed.actions == ["spaced"]
