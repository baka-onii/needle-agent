"""Incremental, display-only projection of model text into protocol sections.

This scanner never authorizes execution. The full response still goes through
parse_response and every runtime gate after generation finishes successfully.
Pending tag/fence prefixes are retained across arbitrary transport boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG = re.compile(r"<(/?)(tool|final|think)\s*>")
_NAMES = ("tool", "final", "think")


@dataclass
class StreamPart:
    index: int
    kind: str
    chunks: list[str] = field(default_factory=list)
    complete: bool = False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


class ResponseStream:
    """Linear scanner; emitted updates append to stable section IDs, not snapshots."""

    def __init__(self) -> None:
        self.parts: list[StreamPart] = []
        self._part: StreamPart | None = None
        self._mode = "text"
        self._before_think = "text"
        self._thought_depth = 0
        self._pending = ""
        self._line_start = True
        self._fence: tuple[str, int] | None = None
        self._assistant: list[str] = []
        self._updates: list[dict] = []
        self._finished = False

    @property
    def assistant_text(self) -> str:
        """Content without out-of-band inline <think> sections; fences stay literal."""
        return "".join(self._assistant)

    def _ensure_part(self) -> StreamPart:
        if self._part is None:
            self._part = StreamPart(len(self.parts), self._mode)
            self.parts.append(self._part)
            self._updates.append({"index": self._part.index, "kind": self._mode, "text": ""})
        return self._part

    def _append(self, text: str) -> None:
        if not text:
            return
        part = self._ensure_part()
        part.chunks.append(text)
        if self._updates and self._updates[-1]["index"] == part.index:
            self._updates[-1]["text"] += text
        else:
            self._updates.append({"index": part.index, "kind": part.kind, "text": text})
        if self._mode != "reasoning":
            self._assistant.append(text)

    def _close_part(self, complete: bool = True) -> None:
        if self._part is not None:
            self._part.complete = complete
            self._updates.append(
                {
                    "index": self._part.index,
                    "kind": self._part.kind,
                    "text": "",
                    "complete": complete,
                }
            )
            self._part = None

    def _fence_prefix(self, value: str, final: bool) -> tuple[str, int]:
        """Return wait/open/close/text plus the consumed fence-prefix length."""
        ws = len(value) - len(value.lstrip(" \t"))
        tail = value[ws:]
        if not tail:
            return ("text" if final else "wait", 0)
        marker = tail[0]
        if marker not in "`~" or (self._fence and marker != self._fence[0]):
            return "text", 0
        count = len(tail) - len(tail.lstrip(marker))
        if count == len(tail) and not final:
            return "wait", 0
        if self._fence is None:
            return ("open", ws + count) if count >= 3 else ("text", 0)
        if count < self._fence[1]:
            return "text", 0
        suffix = tail[count:]
        spaces = len(suffix) - len(suffix.lstrip(" \t"))
        suffix = suffix[spaces:]
        if not suffix or suffix == "\r":
            return ("close", ws + count + spaces) if final else ("wait", 0)
        if suffix.startswith(("\n", "\r\n")) or re.match(r"</(?:tool|final)\s*>", suffix):
            return "close", ws + count + spaces
        if any(tag.startswith(suffix) for tag in ("</tool>", "</final>")) and not final:
            return "wait", 0
        return "text", 0

    @staticmethod
    def _possible_tag(value: str) -> bool:
        return any(
            f"<{slash}{name}>".startswith(value)
            or re.fullmatch(r"<" + slash + name + r"\s*", value)
            for name in _NAMES
            for slash in ("", "/")
        )

    def _tag(self, token: str, closing: str, name: str) -> None:
        if self._mode == "reasoning":
            if name == "think" and not closing:
                self._thought_depth += 1
                self._append(token)
            elif closing and name == "think":
                self._thought_depth -= 1
                if self._thought_depth == 0:
                    self._close_part()
                    self._mode = self._before_think
                else:
                    self._append(token)
            else:
                self._append(token)  # Tool-looking thought text is never executable.
            return
        if self._mode == "final" and not (closing and name == "final"):
            self._append(token)  # Literal tool tags in final answers remain answer text.
            return
        if name == "think" and not closing:
            self._close_part()
            self._before_think, self._mode = self._mode, "reasoning"
            self._thought_depth = 1
            self._ensure_part()
            return
        if not closing and name in {"tool", "final"}:
            if self._mode == "tool" and name == "tool":
                self._append(token)  # Final parser rejects nested actions.
                return
            self._close_part(complete=self._mode == "text")
            self._mode = name
            self._assistant.append(token)
            self._ensure_part()
        elif closing and name == self._mode:
            self._assistant.append(token)
            self._close_part()
            self._mode = "text"
        else:
            self._append(token)

    def feed(self, text: str, *, final: bool = False) -> list[dict]:
        if self._finished:
            raise ValueError("Cannot append to a completed response projection.")
        self._pending += text
        self._updates = []
        cursor = 0
        source = self._pending
        while cursor < len(source):
            if self._line_start:
                decision, length = self._fence_prefix(source[cursor:], final)
                if decision == "wait":
                    break
                if decision in {"open", "close"}:
                    prefix = source[cursor : cursor + length]
                    if decision == "open":
                        marker = prefix.lstrip(" \t")
                        self._fence = (marker[0], len(marker))
                    else:
                        self._fence = None
                    self._append(prefix)
                    cursor += length
                self._line_start = False
                if cursor == len(source):
                    break
            if source[cursor] == "\n":
                self._append("\n")
                cursor += 1
                self._line_start = True
                continue
            if self._fence is None and source[cursor] == "<":
                candidate = source[cursor:]
                tag = _TAG.match(candidate)
                if tag:
                    self._tag(tag[0], tag[1], tag[2])
                    cursor += len(tag[0])
                    continue
                if not final and self._possible_tag(candidate):
                    break
                self._append("<")
                cursor += 1
                continue
            boundary = source.find("\n", cursor)
            if boundary < 0:
                boundary = len(source)
            if self._fence is None:
                tag_start = source.find("<", cursor)
                if tag_start >= 0:
                    boundary = min(boundary, tag_start)
            self._append(source[cursor:boundary])
            cursor = boundary
        self._pending = source[cursor:]
        if final:
            self._close_part(complete=self._mode == "text")
            self._finished = True
        return self._updates

    def finish(self) -> list[dict]:
        return self.feed("", final=True)
