"""Independent, fail-closed parser for the <tool>/<final> text protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG = re.compile(r"<(/?)(tool|final)\s*>")
_FINAL = re.compile(r"<final\s*>(.*?)</final\s*>", re.DOTALL)


@dataclass
class ParsedResponse:
    reasoning: str = ""
    actions: list[str] = field(default_factory=list)
    final_answer: str | None = None


def parse_response(text: str) -> ParsedResponse:
    # A complete final always wins; literal tool tags in an answer are inert data.
    final = _FINAL.search(text)
    if final is not None:
        return ParsedResponse(
            reasoning=(text[: final.start()] + text[final.end() :]).strip(),
            final_answer=final.group(1).strip(),
        )
    actions: list[str] = []
    finals: list[str] = []
    prose: list[str] = []
    opened: tuple[str, int] | None = None
    cursor = 0
    malformed = False
    for tag in _TAG.finditer(text):
        closing, name = tag.group(1, 2)
        if not closing:
            if opened is not None:
                malformed = True
                break
            prose.append(text[cursor : tag.start()])
            opened = (name, tag.end())
        else:
            if opened is None or opened[0] != name:
                malformed = True
                break
            content = text[opened[1] : tag.start()].strip()
            if name == "final":
                finals.append(content)  # even an empty final forbids execution
            elif content:
                actions.append(content)
            opened = None
            cursor = tag.end()
    if malformed or opened is not None:
        # A nested or unclosed block never smuggles an inner action through.
        return ParsedResponse(reasoning=text.strip(), final_answer=text.strip() or None)
    prose.append(text[cursor:])
    reasoning = "".join(prose).strip()
    if finals:
        return ParsedResponse(reasoning=reasoning, actions=actions, final_answer=finals[0])
    if actions:
        return ParsedResponse(reasoning=reasoning, actions=actions)
    return ParsedResponse(reasoning=reasoning, final_answer=text.strip() or None)
