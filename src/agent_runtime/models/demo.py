"""Explicit offline demonstration adapters, NOT an LLM or Needle inference.

A small, deterministic planner supports workspace exploration, literal search,
reading/writing text, arithmetic, time, and clarification. It only emits NL tags;
the separate demo translator supplies synthetic confidence=1.0. All actual tool
operations still go through the production graph, validation, gates, and safety.
Use live mode for open-ended reasoning and calibrated Needle confidence.
"""

from __future__ import annotations

import re
from typing import Any

from agent_runtime.models.action import NeedleResult, ToolRanking
from agent_runtime.tools.base import Tool

_FILE = re.compile(r"[\w./-]+\.[A-Za-z0-9_-]+")
_MATCH = re.compile(r"^([\w./-]+):\d+\s*$", re.MULTILINE)


def _final(text: str) -> str:
    return f"<final>{text}</final>"


def _output(message: dict) -> str:
    return message["content"].partition(":\n")[2]


class DemoReasoningModel:
    """Ground answers in actual observations; never claim arbitrary AI capability."""

    def generate(self, messages: list[dict[str, Any]]) -> str:
        user_index = max(
            i for i, m in enumerate(messages) if m["role"] == "user" and not m.get("kind")
        )
        request = messages[user_index]["content"].strip()
        lower = request.lower()
        current = messages[user_index + 1 :]
        observations = [m for m in current if m.get("kind") == "observation"]
        blocked = [m for m in current if m.get("kind") == "confirmation"]
        if blocked:
            return _final("The runtime did not execute this action.\n\n" + blocked[-1]["content"])
        if observations and observations[-1]["content"].startswith("Tool error"):
            return _final(observations[-1]["content"])
        outputs = [_output(m) for m in observations]
        completed = len(observations)
        plans: list[str] = []
        task = ""
        file_match = _FILE.search(request)
        filename = file_match.group() if file_match else None
        quoted = re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', request)
        quoted = [a or b for a, b in quoted]

        if re.search(r"\b(write|create|save|make a note)\b", lower):
            task = "write"
            filename = filename or "note.txt"
            content = quoted[0] if quoted else None
            if content is None:
                plans.append(f"Ask the user: What would you like me to write in {filename}?")
                if outputs:
                    content = outputs[0]
            if content is not None:
                plans.extend(
                    [
                        f"Write the file {filename} with this exact text:\n{content}",
                        f"Read the file {filename}.",
                    ]
                )
        elif re.search(r"\b(time|date)\b", lower):
            task = "time"
            zone = re.search(
                r"\b(?:in|for)\s+([A-Za-z_]+(?:/[A-Za-z_+\-]+)+|UTC|GMT)\b", request, re.I
            )
            plans = [f"Get the current time in {zone.group(1) if zone else 'UTC'}."]
        elif re.search(r"\d", request) and (
            re.search(r"\b(calculate|compute|what|evaluate)\b", lower)
            or re.fullmatch(r"[\d\s()+*/%.\-]+", request)
        ):
            task = "calculator"
            expression = re.sub(
                r"(?i)^(?:calculate|compute|evaluate|what(?:'s| is))\s+", "", request
            )
            expression = expression.strip().rstrip("? .")
            for word, symbol in (
                ("multiplied by", "*"),
                ("times", "*"),
                ("plus", "+"),
                ("minus", "-"),
                ("divided by", "/"),
            ):
                expression = re.sub(rf"\b{word}\b", symbol, expression, flags=re.I)
            plans = [f"Calculate {expression}."]
        elif re.search(r"\b(search|find|where|locate)\b", lower):
            task = "search"
            if quoted:
                query = quoted[0]
            elif "auth" in lower:
                query = "authentication"
            else:
                match = re.search(r"(?:for|defines?|called)\s+([\w_-]+)", request, re.I)
                query = (
                    match.group(1)
                    if match
                    else re.sub(r"(?i)^(?:search|find|locate)\s+", "", request).strip(" .?")
                )
            plans = [f'Search for "{query}" under .']
            if outputs and not lower.startswith("search"):
                matches = list(dict.fromkeys(_MATCH.findall(outputs[0])))
                if matches:
                    preferred = next((p for p in matches if p.startswith("src/")), matches[0])
                    plans.append(f"Read the file {preferred}.")
        elif re.search(r"\b(read|open|show)\b", lower) and filename:
            task = "read"
            plans = [f"Read the file {filename}."]
        elif re.search(r"\b(list|explore|workspace|project|directory|files)\b", lower):
            task = "explore"
            plans = ["Read the directory ."]
            if outputs and "README.md" in outputs[0] and not lower.startswith("list"):
                plans.append("Read the file README.md.")
        elif re.search(r"\b(ask|clarify|question)\b", lower):
            task = "ask"
            plans = ["Ask the user: What would you like to explore in this workspace?"]

        if completed < len(plans):
            return f"<tool>{plans[completed]}</tool>"
        if not plans:
            return _final(
                "You're in **offline demo mode**. I can explore this workspace, read a file, "
                "search for a phrase, write a note with your approval, "
                "calculate, or check the time.\n\n"
                "Try **Find the authentication implementation** or **Calculate 24 * 18 + 120**. "
                "For open-ended conversation and reasoning, connect your model in **Settings**."
            )
        if task == "calculator":
            return _final(
                f"The result is **{outputs[-1]}**.\n\n"
                "Calculated with the restricted AST calculator."
            )
        if task == "time":
            return _final(
                f"The current time is **{outputs[-1]}**.\n\n"
                "Returned by the runtime's clock, not a model estimate."
            )
        if task == "write":
            return _final(
                f"Wrote and verified **{filename}**.\n\n```text\n{outputs[-1]}\n```\n\n"
                "The file was read back after writing."
            )
        if task == "read":
            language = {"py": "python", "md": "markdown", "json": "json"}.get(
                filename.rsplit(".", 1)[-1], "text"
            )
            return _final(f"Here’s **{filename}**:\n\n```{language}\n{outputs[-1]}\n```")
        if task == "search":
            if len(outputs) > 1:
                path = re.match(r"Read the file (.*)\.", plans[-1]).group(1)
                return _final(
                    f"Found a matching implementation in **{path}**.\n\n"
                    f"```python\n{outputs[-1]}\n```\n\n"
                    "Located by searching the workspace, then reading the matching file."
                )
            return _final(f"Search results:\n\n```text\n{outputs[0]}\n```")
        if task == "ask":
            return _final(
                f"Thanks — you said: **{outputs[0]}**.\n\n"
                "Your answer was returned to the agent as a normal tool observation."
            )
        result = "Here’s what’s in your workspace:\n\n```text\n" + outputs[0] + "\n```"
        if len(outputs) > 1:
            result += "\n\n### From the project README\n\n" + outputs[1]
        return _final(result)


class DemoActionModel:
    """Deterministic NL translator; its 1.0 scores are synthetic, not calibrated."""

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        patterns = [
            (r"Read the directory (.*)\.", "read_directory", ("path",)),
            (r"Read the file (.*)\.", "read_file", ("path",)),
            (r'Search for "(.*)" under (.*)', "search_files", ("query", "path")),
            (r"Calculate (.*)\.", "calculator", ("expression",)),
            (r"Get the current time in (.*)\.", "get_time", ("timezone",)),
            (r"Ask the user: (.*)", "ask_user", ("question",)),
            (r"Write the file (.*) with this exact text:\n(.*)", "write_file", ("path", "content")),
        ]
        for pattern, name, keys in patterns:
            match = re.fullmatch(pattern, action, re.DOTALL)
            if match and name in {tool.name for tool in tools}:
                return NeedleResult(
                    selected_tool=name,
                    arguments=dict(zip(keys, match.groups(), strict=True)),
                    confidence=1.0,
                    rankings=[ToolRanking(tool_name=name, confidence=1.0)],
                )
        return NeedleResult(selected_tool=None, confidence=0.0)
