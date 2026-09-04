"""Runtime configuration. All tunable limits live here, not as magic numbers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """V0 defaults follow docs/spec-v0.md."""

    # Confidence gating (§23) and loop guards (§33).
    # Default gate (mutating / interactive tools). Read-only tools use the
    # lower gate below: Needle is accurate but under-confident on them.
    confidence_threshold: float = 0.85
    read_only_threshold: float = 0.5
    max_tool_steps: int = 20
    # Max consecutive low-confidence/invalid turns without an execution before
    # the run terminates as STALLED (liveness guard; confirm cycles don't
    # consume tool steps).
    max_stalls: int = 3

    # Tool output + context budgets (§27-28).
    max_tool_output_chars: int = 20_000
    max_context_chars: int = 32_000

    # Filesystem sandbox (§7, §35).
    workspace_root: str | None = None
    allow_create_parent_dirs: bool = False

    # search_files limits (§9).
    max_search_results: int = 50
    max_matches_per_file: int = 5
    search_context_lines: int = 2
    max_search_file_bytes: int = 2_000_000

    # Local reasoning backend (llama.cpp OpenAI-compatible server).
    llm_base_url: str = "http://127.0.0.1:8080"
    llm_model: str = "ornith"
    llm_timeout_s: float = 120.0
    llm_max_tokens: int = 512
