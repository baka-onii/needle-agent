"""Runtime configuration and resource limits. No model objects or secrets in graph state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_EXPRESSION_CHARS = 1_000
MAX_AST_NODES = 128
MAX_INTEGER_BITS = 4_096
MAX_EXPONENT = 1_000


@dataclass(frozen=True)
class AgentConfig:
    """V0 defaults follow docs/spec-v0.md. Invalid limits fail at startup."""

    confidence_threshold: float = 0.85
    read_only_threshold: float = 0.5
    max_tool_steps: int = 20
    max_stalls: int = 3
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS
    max_context_chars: int = 32_000

    # Unset root means cwd, not unrestricted filesystem access.
    workspace_root: str | None = None
    allow_create_parent_dirs: bool = False
    read_only: bool = False

    max_search_results: int = 50
    max_matches_per_file: int = 5
    search_context_lines: int = 2
    max_search_file_bytes: int = 2_000_000
    max_search_entries: int = 20_000
    default_timezone: str | None = None

    # Works with llama.cpp, Ollama's /v1 API, and other compatible servers.
    llm_base_url: str = "http://127.0.0.1:8080"
    llm_model: str = "ornith"
    llm_timeout_s: float = 120.0
    llm_max_tokens: int = 1_024
    llm_temperature: float = 0.2
    llm_api_key: str | None = field(default=None, repr=False)
    needle_weights: str | None = None

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "read_only_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        for name in (
            "max_tool_steps",
            "max_stalls",
            "max_tool_output_chars",
            "max_context_chars",
            "max_search_results",
            "max_matches_per_file",
            "max_search_file_bytes",
            "max_search_entries",
            "llm_max_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.search_context_lines) is not int or self.search_context_lines < 0:
            raise ValueError("search_context_lines must be a nonnegative integer.")
        if (
            type(self.llm_timeout_s) not in (int, float)
            or not math.isfinite(self.llm_timeout_s)
            or self.llm_timeout_s <= 0
        ):
            raise ValueError("llm_timeout_s must be positive and finite.")
        if type(self.llm_temperature) not in (int, float) or not 0 <= self.llm_temperature <= 2:
            raise ValueError("llm_temperature must be between 0 and 2.")
