"""Read-only previews for approval prompts (server question events and CLI).

Every preview is computed without executing anything: current file content is
read (bounded), proposed content comes from the call arguments, and diffs use
difflib from the standard library. Anything unreadable falls back to the raw
arguments the pending card already shows, never to an error.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from relay.config import AgentConfig
from relay.tools.base import ToolCall, truncate_text

_PREVIEW_LINES = 40
_PREVIEW_CHARS = 4000


def _read_lines(path: Path, limit: int = _PREVIEW_LINES * 4) -> list[str] | None:
    try:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = []
            for line in handle:
                lines.append(line.rstrip("\n"))
                if len(lines) >= limit:
                    break
            return lines
    except OSError:
        return None


def _resolve(path_value: Any, config: AgentConfig) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    root = Path(config.workspace_root or Path.cwd()).resolve()
    try:
        candidate = (root / path_value).resolve()
    except OSError:
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _unified(old: list[str], new: list[str], path: str) -> str:
    return "".join(
        difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}")
    )


def approval_diff(call: ToolCall, config: AgentConfig) -> dict[str, str] | None:
    """Describe what an approval would change. Returns {label, text} or None."""
    args = call.arguments
    name = call.name
    if name == "write_file":
        path = _resolve(args.get("path"), config)
        new = str(args.get("content", "")).splitlines()
        if path is None:
            return None
        old = _read_lines(path)
        if old is None:
            return {
                "label": f"New file {args.get('path')} ({len(new)} lines)",
                "text": truncate_text("\n".join(new), _PREVIEW_CHARS),
            }
        return {
            "label": f"Edit {args.get('path')}",
            "text": truncate_text(
                _unified(old, new, str(args.get("path"))) or "(no visible changes)",
                _PREVIEW_CHARS,
            ),
        }
    if name == "replace_text":
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        lines = [f"--- {args.get('path')}"]
        path = _resolve(args.get("path"), config)
        if path is not None:
            current = _read_lines(path)
            if current is not None:
                joined = "\n".join(current)
                index = joined.find(old_text)
                if index >= 0:
                    lineno = joined.count("\n", 0, index) + 1
                    lines[0] += f" (near line {lineno})"
        lines.extend(f"-{line}" for line in old_text.splitlines() or [""])
        lines.extend(f"+{line}" for line in new_text.splitlines() or [""])
        return {"label": "Replace text", "text": truncate_text("\n".join(lines), _PREVIEW_CHARS)}
    if name == "insert_text":
        path = _resolve(args.get("path"), config)
        current = _read_lines(path) if path is not None else None
        try:
            lineno = int(args.get("line", 0))
        except (TypeError, ValueError):
            lineno = 0
        lines = [f"+++ {args.get('path')} after line {lineno}"]
        if current:
            start = max(0, lineno - 3)
            lines.extend(f" {line}" for line in current[start:lineno])
        lines.extend(f"+{line}" for line in str(args.get("text", "")).splitlines() or [""])
        if current:
            lines.extend(f" {line}" for line in current[lineno : lineno + 3])
        return {"label": "Insert text", "text": truncate_text("\n".join(lines), _PREVIEW_CHARS)}
    if name == "delete_text":
        path = _resolve(args.get("path"), config)
        current = _read_lines(path) if path is not None else None
        try:
            start = int(args.get("start_line", 0))
            end = int(args.get("end_line", start))
        except (TypeError, ValueError):
            start = end = 0
        lines = [f"--- {args.get('path')} lines {start}-{end}"]
        if current:
            lines.extend(f"-{line}" for line in current[max(0, start - 1) : max(0, end)])
        else:
            lines.append("(current content unavailable)")
        return {"label": "Delete text", "text": truncate_text("\n".join(lines), _PREVIEW_CHARS)}
    if name == "apply_patch":
        return {
            "label": f"Apply patch to {args.get('path')}",
            "text": truncate_text(str(args.get("patch", "")), _PREVIEW_CHARS),
        }
    if name == "delete_file":
        path = _resolve(args.get("path"), config)
        current = _read_lines(path, _PREVIEW_LINES) if path is not None else None
        return {
            "label": f"Delete {args.get('path')}",
            "text": truncate_text(
                "\n".join(current) if current else "(current content unavailable)",
                _PREVIEW_CHARS,
            ),
        }
    return None
