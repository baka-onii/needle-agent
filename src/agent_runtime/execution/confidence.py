"""Confidence routing (spec §23-24)."""

from __future__ import annotations

from agent_runtime.config import AgentConfig
from agent_runtime.models.action import ToolRanking

# Tools with no world side effects: a mistaken call costs one observation,
# so they clear a lower gate. Everything else (write_file, ask_user) uses
# the strict default threshold.
READ_ONLY_TOOLS = frozenset(
    {"read_file", "read_directory", "search_files", "calculator", "get_time"}
)


def threshold_for(tool_name: str, config: AgentConfig) -> float:
    if tool_name in READ_ONLY_TOOLS:
        return config.read_only_threshold
    return config.confidence_threshold


def is_confident(confidence: float, threshold: float) -> bool:
    return confidence >= threshold


def low_confidence_message(action: str, rankings: list[ToolRanking]) -> str:
    lines = [
        "The action translator is uncertain.",
        "",
        "Requested action:",
        action,
        "",
        "Candidate tools:",
    ]
    for i, ranking in enumerate(rankings, start=1):
        lines.append(f"{i}. {ranking.tool_name} — {ranking.confidence:.2f}")
    lines.append(
        "Decide whether to clarify the action, choose another action, "
        "or continue without using a tool."
    )
    return "\n".join(lines)
