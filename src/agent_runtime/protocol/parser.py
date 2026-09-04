"""Action protocol parser (spec §17-20). Independent from LangGraph.

Only content inside ``<tool>`` is executable intent. ``<final>`` carries the
answer. No ``<tool>`` and no ``<final>`` → the whole response is the final
answer (tag-less model fallback, §19).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOOL_RE = re.compile(r"<tool\s*>(.*?)</tool\s*>", re.DOTALL)
_FINAL_RE = re.compile(r"<final\s*>(.*?)</final\s*>", re.DOTALL)


@dataclass
class ParsedResponse:
    reasoning: str = ""
    actions: list[str] = field(default_factory=list)
    final_answer: str | None = None


def parse_response(text: str) -> ParsedResponse:
    actions = [m.group(1).strip() for m in _TOOL_RE.finditer(text)]
    actions = [a for a in actions if a]  # drop empty blocks
    final_match = _FINAL_RE.search(text)
    final_answer = final_match.group(1).strip() if final_match else None
    if final_answer == "":
        final_answer = None
    reasoning = _TOOL_RE.sub("", _FINAL_RE.sub("", text)).strip()
    if not actions and final_answer is None:
        final_answer = text.strip() or None  # §19 tag-less fallback
    return ParsedResponse(reasoning=reasoning, actions=actions, final_answer=final_answer)
