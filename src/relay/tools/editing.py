"""Surgical UTF-8 text edits. Exact matching only, never fuzzy or semantic."""

from __future__ import annotations

import re

from relay.config import AgentConfig
from relay.tools.base import Tool, ToolError, truncate_text
from relay.tools.filesystem import _path_schema, resolve_safe_path

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _read_text(path: str, config: AgentConfig) -> tuple[str, object]:
    resolved = resolve_safe_path(path, config.workspace_root)
    try:
        if not resolved.is_file():
            raise ToolError(f"File does not exist or is not a regular file: {path!r}")
        with resolved.open("rb") as file:
            raw = file.read(config.max_write_chars + 1)
        if len(raw) > config.max_write_chars:
            raise ToolError(f"File exceeds the {config.max_write_chars}-character edit limit.")
        if b"\x00" in raw:
            raise ToolError(f"Cannot edit binary file as text: {path!r}")
        # Normalize CRLF so line numbers and patches behave identically on
        # every platform; edited files are written back with LF endings.
        return raw.decode("utf-8", errors="replace").replace("\r\n", "\n"), resolved
    except OSError as exc:
        raise ToolError(f"Cannot read file {path!r}: {exc.strerror}") from exc


def _write_text(resolved: object, path: str, text: str, config: AgentConfig) -> str:
    if config.read_only:
        raise ToolError("Workspace is read-only; editing is disabled.")
    if "\x00" in text:
        raise ToolError("Only UTF-8 text edits are supported; NUL bytes are not allowed.")
    if len(text) > config.max_write_chars:
        raise ToolError(f"Result exceeds the {config.max_write_chars}-character edit limit.")
    try:
        resolved.write_text(text, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"Cannot write file {path!r}: {exc.strerror}") from exc
    return text


def make_replace_text_tool(config: AgentConfig) -> Tool:
    def replace_text(path: str, old_text: str, new_text: str, count: int = 1) -> str:
        if not old_text:
            raise ToolError("The text to replace must not be empty.")
        if type(count) is not int or count < 0:
            raise ToolError("Count must be a nonnegative integer (0 replaces all).")
        text, resolved = _read_text(path, config)
        found = text.count(old_text)
        if not found:
            raise ToolError(f"Text not found in {path!r}; no changes made.")
        if count == 0:
            updated = text.replace(old_text, new_text)
            replaced = found
        else:
            if found < count:
                raise ToolError(
                    f"Found {found} occurrence(s) in {path!r}, fewer than {count}; no changes made."
                )
            updated = text.replace(old_text, new_text, count)
            replaced = count
        _write_text(resolved, path, updated, config)
        return f"Replaced {replaced} occurrence(s) in {path}."

    return Tool(
        name="replace_text",
        description="Replace exact literal text inside a file (first match, COUNT, or all). "
        "The file must already exist.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Workspace-relative text file to edit."),
                "old_text": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact literal text to find, e.g. 'def old_name('",
                },
                "new_text": {
                    "type": "string",
                    "description": "Exact literal replacement text.",
                    "maxLength": config.max_write_chars,
                },
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Matches to replace; 1 for the first, 0 for all.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=replace_text,
        payload_args=("old_text", "new_text"),
    )


def make_insert_text_tool(config: AgentConfig) -> Tool:
    def insert_text(path: str, line: int, text: str) -> str:
        if type(line) is not int or line < 1:
            raise ToolError("Line must be a 1-based line number.")
        content, resolved = _read_text(path, config)
        lines = content.split("\n")
        if line > len(lines) + 1:
            raise ToolError(f"File {path!r} has {len(lines)} line(s); cannot insert at {line}.")
        lines.insert(line - 1, text)
        _write_text(resolved, path, "\n".join(lines), config)
        return f"Inserted text at line {line} of {path}."

    return Tool(
        name="insert_text",
        description="Insert text as new lines before LINE (1-based; past-the-end appends).",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Workspace-relative text file to edit."),
                "line": {"type": "integer", "minimum": 1, "description": "1-based line number."},
                "text": {
                    "type": "string",
                    "description": "Text to insert; may span lines.",
                    "maxLength": config.max_write_chars,
                },
            },
            "required": ["path", "line", "text"],
        },
        handler=insert_text,
        payload_args=("text",),
    )


def make_delete_text_tool(config: AgentConfig) -> Tool:
    def delete_text(path: str, start_line: int, end_line: int) -> str:
        for value in (start_line, end_line):
            if type(value) is not int or value < 1:
                raise ToolError("Line numbers must be 1-based integers.")
        if end_line < start_line:
            raise ToolError("End line must not precede the start line.")
        content, resolved = _read_text(path, config)
        lines = content.split("\n")
        if end_line > len(lines):
            raise ToolError(f"File {path!r} has {len(lines)} line(s); range is out of bounds.")
        del lines[start_line - 1 : end_line]
        _write_text(resolved, path, "\n".join(lines), config)
        return f"Deleted lines {start_line}-{end_line} of {path}."

    return Tool(
        name="delete_text",
        description="Delete an inclusive 1-based line range from a text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Workspace-relative text file to edit."),
                "start_line": {"type": "integer", "minimum": 1, "description": "First line."},
                "end_line": {"type": "integer", "minimum": 1, "description": "Last line."},
            },
            "required": ["path", "start_line", "end_line"],
        },
        handler=delete_text,
    )


def apply_unified_diff(original: str, patch: str) -> str:
    """Apply single-file unified hunks with exact context matching."""
    lines = original.split("\n")
    hunks: list[tuple[int, list[str]]] = []
    for raw in patch.splitlines():
        if raw.startswith(("---", "+++")) or not raw.strip():
            continue
        match = _HUNK.match(raw)
        if match:
            start = int(match.group(1))
            hunks.append((start, []))
        elif raw[0] in (" ", "-", "+") and hunks:
            hunks[-1][1].append(raw)
        elif raw.startswith("\\"):
            continue
        else:
            raise ToolError(f"Unsupported patch line: {raw[:60]!r}")
    if not hunks:
        raise ToolError("Patch contains no @@ hunks.")
    offset = 0
    for start, entries in hunks:
        cursor = start - 1 + offset
        if cursor < 0 or cursor > len(lines):
            raise ToolError("Patch hunk starts outside the file.")
        replacement: list[str] = []
        consumed = 0
        for entry in entries:
            kind, body = entry[0], entry[1:]
            if kind == " ":
                if cursor + consumed >= len(lines) or lines[cursor + consumed] != body:
                    raise ToolError("Patch context does not match the file; no changes made.")
                replacement.append(body)
                consumed += 1
            elif kind == "-":
                if cursor + consumed >= len(lines) or lines[cursor + consumed] != body:
                    raise ToolError("Patch removal does not match the file; no changes made.")
                consumed += 1
            else:
                replacement.append(body)
        lines[cursor : cursor + consumed] = replacement
        offset += len(replacement) - consumed
    return "\n".join(lines)


def make_apply_patch_tool(config: AgentConfig) -> Tool:
    def apply_patch(path: str, patch: str) -> str:
        if not patch.strip():
            raise ToolError("Patch must not be empty.")
        content, resolved = _read_text(path, config)
        updated = apply_unified_diff(content, patch)
        _write_text(resolved, path, updated, config)
        return truncate_text(f"Patched {path}.", config.max_tool_output_chars)

    return Tool(
        name="apply_patch",
        description="Apply unified-diff @@ hunks to one file; context must match exactly.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Workspace-relative text file to patch."),
                "patch": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Unified diff hunks, e.g. '@@ -1,2 +1,2 @@'.",
                },
            },
            "required": ["path", "patch"],
        },
        handler=apply_patch,
        payload_args=("patch",),
    )


def editing_tools(config: AgentConfig) -> list[Tool]:
    return [
        make_replace_text_tool(config),
        make_insert_text_tool(config),
        make_delete_text_tool(config),
        make_apply_patch_tool(config),
    ]
