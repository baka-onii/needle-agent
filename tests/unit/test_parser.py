"""P3: protocol parser, including <content> payload blocks."""

from relay.protocol.parser import parse_response


def _acts(parsed):
    return [(a.instruction, a.content) for a in parsed.actions]


def test_basic_tool_block() -> None:
    parsed = parse_response("I need to inspect first.\n\n<tool>\nRead the directory.\n</tool>")
    assert _acts(parsed) == [("Read the directory.", None)]
    assert parsed.final_answer is None
    assert "inspect" in parsed.reasoning


def test_multiline_action() -> None:
    parsed = parse_response("<tool>\nSearch for auth\nin all files.\n</tool>")
    assert _acts(parsed) == [("Search for auth\nin all files.", None)]


def test_multiple_tool_blocks_in_order() -> None:
    parsed = parse_response("<tool>A</tool>\ntext\n<tool>B</tool>")
    assert [a.instruction for a in parsed.actions] == ["A", "B"]


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
    assert _acts(parsed) == [("spaced", None)]


def test_tool_tags_inside_final_are_inert_answer_text() -> None:
    parsed = parse_response("<final>Example: <tool>Read a file.</tool></final>")
    assert parsed.final_answer == "Example: <tool>Read a file.</tool>"
    assert parsed.actions == []


def test_content_block_splits_instruction_and_payload() -> None:
    parsed = parse_response(
        '<tool>Use write_file to write the file "a.txt". <content>Hello World</content></tool>'
    )
    assert _acts(parsed) == [('Use write_file to write the file "a.txt".', "Hello World")]
    assert parsed.final_answer is None


def test_content_masks_inner_protocol_tags() -> None:
    parsed = parse_response(
        "<tool>Use write_file. <content>Hello <tool>World</tool> and </final>done</content></tool>"
    )
    assert _acts(parsed) == [
        ("Use write_file.", "Hello <tool>World</tool> and </final>done")
    ]


def test_content_masks_inner_final() -> None:
    parsed = parse_response(
        '<tool>Write docs. <content>Example: <final>hi</final></content></tool>'
    )
    assert parsed.final_answer is None
    assert _acts(parsed) == [("Write docs.", "Example: <final>hi</final>")]


def test_stray_content_block_is_not_executable() -> None:
    parsed = parse_response('<tool>Read "a".</tool> <content>stray</content>')
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_two_content_blocks_in_one_action_are_rejected() -> None:
    parsed = parse_response(
        "<tool>Write <content>one</content> plus <content>two</content>.</tool>"
    )
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_unclosed_content_block_is_not_executable() -> None:
    parsed = parse_response("<tool>Write <content>oops.</tool>")
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_final_answer_keeps_content_markup_as_text() -> None:
    parsed = parse_response("<final>done <content>x</content></final>")
    assert parsed.final_answer == "done <content>x</content>"
    assert parsed.actions == []


def _payloads(parsed):
    return [(a.instruction, a.content, a.payloads) for a in parsed.actions]


def test_numbered_blocks_split_positionally() -> None:
    parsed = parse_response(
        '<tool>Use replace_text. <text-1>old words</text-1> '
        "middle <text-2>new words</text-2></tool>"
    )
    assert len(parsed.actions) == 1
    action = parsed.actions[0]
    assert action.instruction == "Use replace_text.  middle"
    assert action.payloads == ["old words", "new words"]
    assert action.content is None


def test_numbered_blocks_reorder_by_number() -> None:
    parsed = parse_response(
        "<tool>Fix it. <text-2>second</text-2> <text-1>first</text-1></tool>"
    )
    assert parsed.actions[0].payloads == ["first", "second"]


def test_numbered_block_gap_is_not_executable() -> None:
    parsed = parse_response("<tool>Fix it. <text-1>a</text-1> <text-3>c</text-3></tool>")
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_duplicate_block_number_is_not_executable() -> None:
    parsed = parse_response("<tool>Fix it. <text-1>a</text-1> <text-1>b</text-1></tool>")
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_mixed_content_and_numbered_blocks_are_rejected() -> None:
    parsed = parse_response("<tool>Do it. <content>a</content> <text-1>b</text-1></tool>")
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_stray_numbered_block_is_not_executable() -> None:
    parsed = parse_response('<tool>Read "a".</tool> <text-1>stray</text-1>')
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_mismatched_numbered_tags_are_not_executable() -> None:
    parsed = parse_response("<tool>Fix it. <text-1>a</text-2></tool>")
    assert parsed.actions == []
    assert parsed.final_answer is not None


def test_inner_markup_inside_blocks_is_inert_data() -> None:
    parsed = parse_response(
        "<tool>Write docs. <content>See <text-1>not a block</text-1></content></tool>"
    )
    assert _payloads(parsed) == [
        ("Write docs.", "See <text-1>not a block</text-1>", ["See <text-1>not a block</text-1>"])
    ]
    assert parsed.final_answer is None


def test_tool_tags_inside_numbered_block_are_masked() -> None:
    parsed = parse_response(
        "<tool>Replace. <text-1>a <tool>b</tool></text-1> <text-2>c</text-2></tool>"
    )
    assert parsed.actions[0].payloads == ["a <tool>b</tool>", "c"]
    assert parsed.final_answer is None


def test_block_only_action_without_instruction_is_dropped() -> None:
    parsed = parse_response("<tool><content>orphan payload</content></tool>")
    assert parsed.actions == []
