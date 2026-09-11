"""Structured context compression: replace history with a validated summary.

When a built context passes ~80% of budget, the transcript is unloaded and
the reasoning model is asked for a structured state snapshot in a fixed
shape. The next turn starts from summary + current request instead of the
full history. Anything malformed falls back to the normal trim path: a bad
summary is dropped, never installed.
"""

from __future__ import annotations

import re
from typing import Any

COMPRESSION_THRESHOLD = 0.8
MAX_SUMMARY_CHARS = 4000
MAX_SUMMARY_INPUT_CHARS = 24000

SUMMARY_PROMPT = """Compress this agent conversation into a structured state snapshot.
Reply with ONLY the snapshot in exactly this shape (no prose, no fences):

task:
  objective: "<one line: what the user ultimately wants>"

constraints:
  - "<each hard requirement or limit, one per line>"

completed:
  - "<each finished subtask with its concrete result>"

current:
  subtask: "<the single next step to take>"

files:
  primary:
    - <paths being changed>
  related:
    - <paths read or relevant>

decisions:
  - "<each decision already made, with brief why>"

blockers: []

Rules: paths stay workspace-relative; drop stale exploration; keep every
pending requirement; empty sections stay present but empty.
"""

_SECTION = re.compile(r"^([A-Za-z_]+):\s*$")
_ITEM = re.compile(r"^\s*-\s+(.*\S)\s*$")
_FIELD = re.compile(r"^\s+([A-Za-z_]+):\s*(.*\S)?\s*$")


def _empty() -> dict[str, Any]:
    return {
        "objective": "",
        "constraints": [],
        "completed": [],
        "subtask": "",
        "files_primary": [],
        "files_related": [],
        "decisions": [],
        "blockers": [],
    }


def parse_summary(text: str) -> dict[str, Any] | None:
    """Lenient line parser for the snapshot shape. None when unusable."""
    data = _empty()
    section: str | None = None
    subsection: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        top = _SECTION.match(line)
        if top and not line.startswith((" ", "\t")):
            section = top.group(1).lower()
            subsection = None
            continue
        if section is None:
            continue
        item = _ITEM.match(line)
        if item:
            value = item.group(1).strip("\"'")
            if section == "constraints":
                data["constraints"].append(value)
            elif section == "completed":
                data["completed"].append(value)
            elif section == "decisions":
                data["decisions"].append(value)
            elif section == "blockers":
                data["blockers"].append(value)
            elif section == "files" and subsection == "primary":
                data["files_primary"].append(value)
            elif section == "files" and subsection == "related":
                data["files_related"].append(value)
            continue
        field = _FIELD.match(line)
        if field:
            key, value = field.group(1).lower(), (field.group(2) or "").strip("\"'")
            if section == "task" and key == "objective":
                data["objective"] = value
            elif section == "current" and key == "subtask":
                data["subtask"] = value
            elif section == "files" and value == "":
                subsection = key
            continue
    if not data["objective"] or not data["subtask"]:
        return None
    return data


def render_summary(data: dict[str, Any]) -> str:
    """Canonical rendering with fixed section order for the next context."""

    def items(values: list[str]) -> str:
        return "\n".join(f"  - {value}" for value in values) or "  []"

    return (
        "Compressed conversation state (earlier turns were summarized):\n"
        f"task:\n  objective: {data['objective']!r}\n\n"
        f"constraints:\n{items(data['constraints'])}\n\n"
        f"completed:\n{items(data['completed'])}\n\n"
        f"current:\n  subtask: {data['subtask']!r}\n\n"
        "files:\n  primary:\n"
        + "".join(f"    - {path}\n" for path in data["files_primary"])
        + "  related:\n"
        + "".join(f"    - {path}\n" for path in data["files_related"])
        + f"\ndecisions:\n{items(data['decisions'])}\n\n"
        f"blockers: {data['blockers']!r}"
    )


def should_compress(
    chars: int, tokens: int | None, max_chars: int, max_tokens: int
) -> bool:
    if tokens is not None:
        return tokens > max_tokens * COMPRESSION_THRESHOLD
    return chars > max_chars * COMPRESSION_THRESHOLD


def generate_text(reasoning: Any, messages: list[dict[str, Any]]) -> str:
    """One non-executable completion from any reasoning adapter."""
    generate = getattr(reasoning, "generate", None)
    if callable(generate):
        return str(generate(messages))
    parts = []
    for delta in reasoning.stream(messages):
        parts.append(delta if isinstance(delta, str) else (delta.text or ""))
    return "".join(parts)


def compress_messages(
    reasoning: Any,
    messages: list[dict[str, Any]],
    config: Any,
    emit: Any,
) -> list[dict[str, Any]] | None:
    """Replace history with a validated structured summary, or None."""
    from agent_runtime.context.manager import _observation
    from agent_runtime.models.streaming import GenerationCancelled
    from agent_runtime.tools.base import truncate_text

    before_chars = sum(len(str(message.get("content", ""))) for message in messages)
    dump = "\n\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages
    )
    if len(dump) > MAX_SUMMARY_INPUT_CHARS:
        dump = dump[:MAX_SUMMARY_INPUT_CHARS] + "\n… [earlier turns truncated]"
    try:
        text = generate_text(
            reasoning,
            [{"role": "user", "content": SUMMARY_PROMPT + "\n\n" + dump}],
        )
        data = parse_summary(text)
    except GenerationCancelled:
        raise
    except Exception as exc:
        emit("context_compression_skipped", reason=f"summarizer failed: {exc}")
        return None
    if data is None:
        emit("context_compression_skipped", reason="summary shape was unusable")
        return None
    rendered = truncate_text(render_summary(data), MAX_SUMMARY_CHARS)
    latest = next(
        (
            message
            for message in reversed(messages)
            if message.get("role") == "user" and not _observation(message)
        ),
        None,
    )
    replacement = [{"role": "user", "content": rendered, "kind": "summary"}]
    if latest is not None:
        replacement.append(dict(latest))
    after_chars = sum(len(str(message.get("content", ""))) for message in replacement)
    emit(
        "context_compressed",
        before_chars=before_chars,
        after_chars=after_chars,
        turns=len(messages),
    )
    return replacement
