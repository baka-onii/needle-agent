"""Conservative checks on NL intent, not a second translator or argument repairer.

The small translator cannot invent missing file content. Explicit paths, tool
names, and literal write payloads are contracts checked against its proposal.
General natural-language atomicity cannot be proved; reject common compound
forms, then still require exactly one translated call and the normal gates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_runtime.config import AgentConfig
from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.tools.base import ToolCall, ToolError

_EXACT = re.compile(r"\bwith (?:this )?exact (?:text|content):[ \t]*\r?\n", re.I)
_FENCE = re.compile(r"(?P<fence>`{3,}|~{3,})[^\r\n]*\r?\n")
_QUOTED = re.compile(r""""(?:\\.|[^"\\])*"|'[^'\n]*'|`[^`\n]*`""")
_NAMED = re.compile(r"^\s*Use\s+`?([A-Za-z_][A-Za-z_0-9]*)`?\s+to\b", re.I)
_COMPOUND = re.compile(
    r"(?:\b(?:and(?:\s+then)?|then|also|afterwards)\b|;)\s+"
    r"(?:please\s+)?(?:use\s+)?"
    r"(?:read(?:_file|_directory)?|open|list|search(?:_files)?|find|write(?:_file)?|"
    r"create|save|calculate|calculator|compute|ask(?:_user)?|get_time|summarize|"
    r"run|delete|remove|move|copy|replace|insert|commit|stage|checkout|fetch)\b",
    re.I,
)


def literal_write_content(action: str) -> str:
    marker = _EXACT.search(action)
    if not marker:
        raise ToolError(
            "write_file requires the finished literal content, not an instruction to generate it. "
            "Compose the requested content yourself and reissue one action with "
            "'with this exact text:' followed by the full content in a fenced block."
        )
    payload = action[marker.end() :]
    opening = _FENCE.match(payload)
    if not opening:
        # Preserve the original explicit-text NL form for existing integrations.
        # New prompts use fences to preserve leading/trailing whitespace exactly.
        if payload.startswith(("```", "~~~")):
            raise ToolError("The literal write payload has a malformed fence.")
        return payload
    fence = opening["fence"]
    if payload[opening.end() :].rstrip(" \t") == fence:
        return ""
    closing = re.search(r"\r?\n" + re.escape(fence) + r"[ \t]*\Z", payload)
    if closing is None or closing.start() < opening.end():
        raise ToolError(
            "The literal write payload needs a matching closing fence, with no following action."
        )
    return payload[opening.end() : closing.start()]


def write_action(path: str, content: str) -> str:
    """Emit NL, not tool-call JSON. A longer fence keeps embedded Markdown inert."""
    fence = "`" * max(3, max((len(m[0]) + 1 for m in re.finditer(r"`+", content)), default=0))
    return (
        f"Use write_file to write the file {json.dumps(path, ensure_ascii=False)} "
        "with this exact text:\n"
        f"{fence}text\n{content}\n{fence}\n"
    )


def instruction_header(action: str) -> str:
    marker = _EXACT.search(action)
    return action[: marker.start()] if marker else action


def check_atomic_action(action: str) -> None:
    header = _QUOTED.sub("VALUE", instruction_header(action))
    if _COMPOUND.search(header):
        raise ToolError(
            "The action combines multiple operations. Choose one tool and one target now; "
            "wait for its observation before reading, writing, "
            "summarizing, or asking anything else."
        )
    if re.search(r"\b(?:read|write|open)\b", header, re.I) and re.search(
        r"\bVALUE\s+(?:and|&)\s+VALUE\b", header, re.I
    ):
        raise ToolError(
            "Use one file target per action. Inspect the first file, then reason again."
        )


def named_tool(action: str) -> str | None:
    match = _NAMED.match(action)
    return match[1] if match else None


def explicit_path(action: str) -> str | None:
    header = instruction_header(action)
    for match in re.finditer(r"\b(?:file|directory|folder|under)\s+", header, re.I):
        quoted = _QUOTED.match(header, match.end())
        if quoted:
            value = quoted[0]
            if value.startswith('"'):
                try:
                    return json.loads(value)
                except ValueError:
                    return value[1:-1]
            return value[1:-1]
    return None


def deterministic_write_result(action: str) -> NeedleResult | None:
    """Exact copy for well-formed write actions, without involving the engine.

    The translator model provably cannot copy a fenced payload: it substitutes
    a nearby quoted string (usually the filename) for the content at every
    generation budget, and large budgets degenerate further into memorized
    samples. Copying here is exact string surgery on the reasoning model's own
    text, so 1.0 is an honest confidence, matching the demo translator. The
    normal gates still apply: validate_intent rechecks tool, path, and content,
    and safety still requires runtime approval before any write executes.
    Returns None when the action is not a clean write, leaving engine
    translation and its error messages untouched.
    """
    if named_tool(action) != "write_file":
        return None
    try:
        content = literal_write_content(action)
        path = explicit_path(action)
    except ToolError:
        return None
    if path is None:
        return None
    return NeedleResult(
        selected_tool="write_file",
        arguments={"path": path, "content": content},
        confidence=1.0,
        rankings=[ToolRanking(tool_name="write_file", confidence=1.0)],
    )


def validate_intent(
    action: str, call: ToolCall, config: AgentConfig, payloads: list[str] | None = None
) -> None:
    selected = named_tool(action)
    if selected is not None and selected != call.name:
        raise ToolError(
            f"The reasoning model selected {selected!r}, "
            f"but the translator proposed {call.name!r}. "
            "Explicitly confirm or correct the intended tool; "
            "do not substitute another tool silently."
        )
    payloads = list(payloads or [])
    if call.name == "write_file":
        # Preferred source is the payload block the runtime attached in
        # translate; the legacy fenced payload inside the instruction remains.
        expected = payloads[0] if payloads else literal_write_content(action)
        if call.arguments["content"] != expected:
            raise ToolError(
                "The translator changed or truncated the literal file content. "
                "Nothing was written. "
                "Reissue the complete finished content exactly in one fenced block. "
                "Do not replace it with the task description."
            )
    if call.name in {"read_file", "read_directory", "search_files", "write_file"}:
        expected = explicit_path(action)
        if expected is not None:
            # Compare literal paths lexically; never turn aliases or outside paths into '.'.
            root = Path(config.workspace_root or Path.cwd()).resolve()
            provided = Path(call.arguments.get("path", "."))
            target = Path(expected)
            if root / provided != root / target:
                raise ToolError(
                    f"The action specified path {expected!r}, but the translator proposed "
                    f"{str(provided)!r}. Preserve the specified path exactly. "
                    "The workspace root is '.', not 'root', '/' or 'user/home'."
                )


def is_write_permission_question(question: str) -> bool:
    text = " ".join(question.casefold().split())
    # Requests for missing content/destinations are not requests for permission.
    if re.match(r"^(?:what|which|where|how|when|who)\b", text):
        return False
    if not re.search(
        r"\b(?:write|writing|create|creating|save|saving|overwrite|overwriting|"
        r"run|running|execute|executing|delete|deleting|commit|committing)\b",
        text,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:permission|approval|approve|allow|authorize|authorise|consent)\b|"
            r"\b(?:may|can|could|should|shall) (?:i|we)\b|"
            r"\b(?:is it (?:ok|okay)|do you (?:want|wish)|would you like|please confirm)\b",
            text,
        )
    )
