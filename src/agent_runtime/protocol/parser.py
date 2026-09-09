"""Independent, fail-closed parser for the <tool>/<final> text protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG = re.compile(r"<(/?)(tool|final)\s*>")
_FINAL = re.compile(r"<final\s*>(.*?)</final\s*>", re.DOTALL)
_PAYLOAD = re.compile(
    r"<content\s*>(?P<content>.*?)</content\s*>"
    r"|<text-(?P<num>\d+)\s*>(?P<numbered>.*?)</text-(?P=num)\s*>",
    re.DOTALL,
)
_STRAY_CONTENT = re.compile(r"</?content\b|</?text\b")


@dataclass
class ToolAction:
    """One executable intent: instruction-only text plus opaque payload blocks.

    Payload blocks are data, never protocol: they are masked before tag
    scanning so tags inside a payload cannot smuggle actions through.
    Simple tools carry one <content> block; tools with several payload
    arguments carry ordered <text-1>, <text-2>, ... blocks.
    """

    instruction: str
    payloads: list[str] = field(default_factory=list)

    @property
    def content(self) -> str | None:
        """The single payload, or None for zero or several blocks."""
        return self.payloads[0] if len(self.payloads) == 1 else None


@dataclass
class ParsedResponse:
    reasoning: str = ""
    actions: list[ToolAction] = field(default_factory=list)
    final_answer: str | None = None


def _extract_payload_spans(masked: str) -> tuple[list[tuple[int, int, str, int]], bool]:
    """Cut payload blocks out of fence-masked text, preserving offsets.

    Returns ((start, end, payload, number)...) with number 0 for <content>
    and N for <text-N>, plus whether stray payload markup remains.
    One left-to-right pass: the outer block always wins, so markup nested
    inside a payload is inert data. Fence interiors are already blanked by
    _mask_fences, so literal markup inside fenced payloads is never a tag.
    """
    spans: list[tuple[int, int, str, int]] = []

    def blank(match: re.Match) -> str:
        if match.group("num") is None:
            spans.append((match.start(), match.end(), match.group("content"), 0))
        else:
            spans.append(
                (match.start(), match.end(), match.group("numbered"), int(match.group("num")))
            )
        return " " * (match.end() - match.start())

    remainder = _PAYLOAD.sub(blank, masked)
    return spans, _STRAY_CONTENT.search(remainder) is not None


def _mask_fences(text: str) -> str:
    """Protocol-looking tags in fenced payloads are data, including literal finals."""
    result: list[str] = []
    fence = ""
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not fence:
            opening = re.match(r"^(`{3,}|~{3,})", stripped)
            if opening:
                fence = opening[1]
            else:
                result.append(line)
                continue
        else:
            closing = re.match(
                r"^[ \t]*"
                + re.escape(fence[0])
                + "{"
                + str(len(fence))
                + r",}[ \t]*(?=</(?:tool|final)\s*>|$)",
                line.rstrip("\r\n"),
            )
            if closing:
                fence = ""
                result.append(" " * closing.end() + line[closing.end() :])
                continue
        result.append("".join("\n" if c == "\n" else " " for c in line))
    return "".join(result)


def parse_response(text: str) -> ParsedResponse:
    # Only provider-visible assistant content can be executable. Inline thought
    # sections are display data, even if they contain complete tool/final tags.
    if "<think" in text:
        from agent_runtime.protocol.stream import ResponseStream

        stream = ResponseStream()
        stream.feed(text, final=True)
        text = stream.assistant_text
    # A complete final always wins; literal tool tags in an answer are inert data.
    # Payload blocks are blanked first so markup inside them cannot forge
    # a final (or a tool block) either.
    masked = _mask_fences(text)
    payload_spans, stray_content = _extract_payload_spans(masked)
    if stray_content:
        return ParsedResponse(reasoning=text.strip(), final_answer=text.strip() or None)
    demasked = _PAYLOAD.sub(lambda m: " " * (m.end() - m.start()), masked)
    final = _FINAL.search(demasked)
    if final is not None:
        return ParsedResponse(
            reasoning=(text[: final.start()] + text[final.end() :]).strip(),
            final_answer=text[final.start(1) : final.end(1)].strip(),
        )
    actions: list[ToolAction] = []
    action_ranges: list[tuple[int, int]] = []
    finals: list[str] = []
    prose: list[str] = []
    opened: tuple[str, int] | None = None
    cursor = 0
    malformed = False
    for tag in _TAG.finditer(demasked):
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
            if name == "final":
                finals.append(text[opened[1] : tag.start()].strip())
            elif text[opened[1] : tag.start()].strip():
                action_ranges.append((opened[1], tag.start()))
            opened = None
            cursor = tag.end()
    if malformed or opened is not None:
        # A nested or unclosed block never smuggles an inner action through.
        return ParsedResponse(reasoning=text.strip(), final_answer=text.strip() or None)
    used = [False] * len(payload_spans)
    for start, end in action_ranges:
        inner = [
            (i, num, payload)
            for i, (s, e, payload, num) in enumerate(payload_spans)
            if s >= start and e <= end
        ]
        numbers = sorted(num for _, num, _ in inner)
        if 0 in numbers and len(numbers) > 1:
            malformed = True  # <content> never mixes with <text-N> blocks.
            break
        if any(num == 0 for _, num, _ in inner):
            ordered = [payload for _, _, payload in inner]
        else:
            if numbers != list(range(1, len(numbers) + 1)):
                malformed = True  # Gaps or duplicates in <text-N> numbering.
                break
            ordered = [payload for _, _, payload in sorted(inner, key=lambda item: item[1])]
        for i, _, _ in inner:
            used[i] = True
        # Cut payload spans out of the instruction, preserving other text.
        instruction = text[start:end]
        cuts = sorted(
            ((s, e) for s, e, _, _ in (payload_spans[i] for i, _, _ in inner)),
            reverse=True,
        )
        for s, e in cuts:
            instruction = instruction[: s - start] + instruction[e - start :]
        instruction = instruction.strip()
        if instruction:
            actions.append(ToolAction(instruction=instruction, payloads=ordered))
    if malformed or not all(used):
        # Payload outside any tool block, or mixed/duplicated block numbering.
        return ParsedResponse(reasoning=text.strip(), final_answer=text.strip() or None)
    prose.append(text[cursor:])
    reasoning = "".join(prose).strip()
    if finals:
        return ParsedResponse(reasoning=reasoning, actions=actions, final_answer=finals[0])
    if actions:
        return ParsedResponse(reasoning=reasoning, actions=actions)
    return ParsedResponse(reasoning=reasoning, final_answer=text.strip() or None)
