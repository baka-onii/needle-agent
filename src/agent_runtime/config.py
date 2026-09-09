"""Validated runtime values and portable TOML/JSON configuration, using only stdlib."""

from __future__ import annotations

import json
import math
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_CONFIG_BYTES = 250_000
MAX_PROMPT_CHARS = 20_000
MAX_EXPRESSION_CHARS = 1_000
MAX_AST_NODES = 128
MAX_INTEGER_BITS = 4_096
MAX_EXPONENT = 1_000


def default_asset(name: str) -> str:
    """Defaults live outside src in checkouts and are bundled into installed wheels."""
    packaged = files("agent_runtime").joinpath("defaults", name)
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "config" / name).read_text(encoding="utf-8")


_DEFAULT_DOCUMENT = tomllib.loads(default_asset("defaults.toml"))
_DEFAULTS = {key: value for section in _DEFAULT_DOCUMENT.values() for key, value in section.items()}
_PROMPT_NAMES = ("reasoning_prompt", "translator_prompt", "confirmation_prompt")
_DEFAULT_PROMPTS = {name: default_asset(_DEFAULTS[name + "_file"]) for name in _PROMPT_NAMES}
MAX_TOOL_OUTPUT_CHARS = _DEFAULTS["max_tool_output_chars"]


def normalize_model_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in parsed.netloc)
        )
        _ = parsed.port  # Validate invalid or out-of-range port strings too.
    except (ValueError, AttributeError) as exc:
        raise ValueError("Model URL must be a valid HTTP(S) base URL.") from exc
    if not valid:
        raise ValueError("Model URL must be an HTTP(S) base URL without credentials or a query.")
    return value.strip().rstrip("/")


@dataclass(frozen=True)
class AgentConfig:
    """Canonical defaults are in config/defaults.toml and config/prompts/*.md."""

    confidence_threshold: float = _DEFAULTS["confidence_threshold"]
    read_only_threshold: float = _DEFAULTS["read_only_threshold"]
    max_tool_steps: int = _DEFAULTS["max_tool_steps"]
    max_stalls: int = _DEFAULTS["max_stalls"]
    max_repeated_failures: int = _DEFAULTS["max_repeated_failures"]
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS
    max_context_chars: int = _DEFAULTS["max_context_chars"]

    # None means cwd, not unrestricted filesystem access.
    workspace_root: str | None = None
    allow_create_parent_dirs: bool = _DEFAULTS["allow_create_parent_dirs"]
    read_only: bool = _DEFAULTS["read_only"]
    include_workspace_listing: bool = _DEFAULTS["include_workspace_listing"]
    max_directory_entries: int = _DEFAULTS["max_directory_entries"]
    workspace_listing_chars: int = _DEFAULTS["workspace_listing_chars"]

    max_search_results: int = _DEFAULTS["max_search_results"]
    max_matches_per_file: int = _DEFAULTS["max_matches_per_file"]
    search_context_lines: int = _DEFAULTS["search_context_lines"]
    max_search_file_bytes: int = _DEFAULTS["max_search_file_bytes"]
    max_search_entries: int = _DEFAULTS["max_search_entries"]
    max_write_chars: int = _DEFAULTS["max_write_chars"]
    default_timezone: str | None = None

    mode: str = _DEFAULTS["mode"]
    llm_base_url: str = _DEFAULTS["llm_base_url"]
    llm_model: str = _DEFAULTS["llm_model"]
    llm_timeout_s: float = _DEFAULTS["llm_timeout_s"]
    llm_max_tokens: int = _DEFAULTS["llm_max_tokens"]
    llm_temperature: float = _DEFAULTS["llm_temperature"]
    llm_api_key: str | None = field(default=None, repr=False)
    needle_weights: str | None = None
    needle_max_tokens: int = _DEFAULTS["needle_max_tokens"]
    action_model: str = _DEFAULTS["action_model"]
    fg_base_url: str = _DEFAULTS["fg_base_url"]
    fg_timeout_s: float = _DEFAULTS["fg_timeout_s"]
    fg_max_tokens: int = _DEFAULTS["fg_max_tokens"]
    fg_temperature: float = _DEFAULTS["fg_temperature"]
    require_approval_for: tuple[str, ...] = tuple(_DEFAULTS["require_approval_for"])

    llm_stream: bool = _DEFAULTS["llm_stream"]
    stream_buffer_ms: int = _DEFAULTS["stream_buffer_ms"]
    stream_max_lag_ms: int = _DEFAULTS["stream_max_lag_ms"]
    stream_flush_ms: int = _DEFAULTS["stream_flush_ms"]
    max_model_output_chars: int = _DEFAULTS["max_model_output_chars"]
    capture_model_inputs: bool = _DEFAULTS["capture_model_inputs"]

    reasoning_prompt: str = _DEFAULT_PROMPTS["reasoning_prompt"]
    translator_prompt: str = _DEFAULT_PROMPTS["translator_prompt"]
    confirmation_prompt: str = _DEFAULT_PROMPTS["confirmation_prompt"]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"demo", "live"}:
            raise ValueError("mode must be 'demo' or 'live'.")
        if not isinstance(self.action_model, str) or self.action_model not in {
            "needle",
            "functiongemma",
        }:
            raise ValueError("action_model must be 'needle' or 'functiongemma'.")
        if isinstance(self.require_approval_for, list):
            object.__setattr__(self, "require_approval_for", tuple(self.require_approval_for))
        if not isinstance(self.require_approval_for, tuple) or any(
            not isinstance(name, str) or not name.strip() or "\x00" in name
            for name in self.require_approval_for
        ):
            raise ValueError("require_approval_for must be a list of tool names.")
        for name in ("confidence_threshold", "read_only_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        for name in (
            "max_tool_steps",
            "max_stalls",
            "max_repeated_failures",
            "max_tool_output_chars",
            "max_context_chars",
            "max_directory_entries",
            "workspace_listing_chars",
            "max_search_results",
            "max_matches_per_file",
            "max_search_file_bytes",
            "max_search_entries",
            "max_write_chars",
            "llm_max_tokens",
            "needle_max_tokens",
            "fg_max_tokens",
            "max_model_output_chars",
            "stream_max_lag_ms",
            "stream_flush_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.stream_buffer_ms) is not int or not 0 <= self.stream_buffer_ms <= 5000:
            raise ValueError("stream_buffer_ms must be an integer between 0 and 5000.")
        if self.stream_max_lag_ms < self.stream_buffer_ms:
            raise ValueError("stream_max_lag_ms must be at least stream_buffer_ms.")
        for name in (
            "read_only",
            "allow_create_parent_dirs",
            "include_workspace_listing",
            "llm_stream",
            "capture_model_inputs",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        if type(self.search_context_lines) is not int or self.search_context_lines < 0:
            raise ValueError("search_context_lines must be a nonnegative integer.")
        if (
            type(self.llm_timeout_s) not in (int, float)
            or not math.isfinite(self.llm_timeout_s)
            or self.llm_timeout_s <= 0
        ):
            raise ValueError("llm_timeout_s must be positive and finite.")
        if (
            type(self.fg_timeout_s) not in (int, float)
            or not math.isfinite(self.fg_timeout_s)
            or self.fg_timeout_s <= 0
        ):
            raise ValueError("fg_timeout_s must be positive and finite.")
        if type(self.llm_temperature) not in (int, float) or not 0 <= self.llm_temperature <= 2:
            raise ValueError("llm_temperature must be between 0 and 2.")
        if type(self.fg_temperature) not in (int, float) or not 0 <= self.fg_temperature <= 2:
            raise ValueError("fg_temperature must be between 0 and 2.")
        for name in _PROMPT_NAMES:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or len(value) > MAX_PROMPT_CHARS
            ):
                raise ValueError(
                    f"{name} must be nonempty text of at most {MAX_PROMPT_CHARS} characters."
                )
        for name in ("workspace_root", "needle_weights", "default_timezone", "llm_api_key"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or "\x00" in value):
                raise ValueError(f"{name} must be text or null.")
        for name in ("llm_base_url", "llm_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be nonempty text.")
        normalize_model_url(self.llm_base_url)
        if not isinstance(self.fg_base_url, str) or not self.fg_base_url.strip():
            raise ValueError("fg_base_url must be nonempty text.")
        normalize_model_url(self.fg_base_url)
        if self.default_timezone:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            try:
                ZoneInfo(self.default_timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError("default_timezone must be a valid IANA timezone.") from exc


_CONFIG_SECTIONS = {
    name: set(values) for name, values in _DEFAULT_DOCUMENT.items() if name != "prompts"
}
_CONFIG_SECTIONS["models"].add("needle_weights")
_CONFIG_SECTIONS["prompts"] = set(_PROMPT_NAMES) | {name + "_file" for name in _PROMPT_NAMES}
_ENV_FIELDS = {
    "NEEDLE_WORKSPACE": "workspace_root",
    "NEEDLE_LLM_BASE_URL": "llm_base_url",
    "NEEDLE_LLM_MODEL": "llm_model",
    "NEEDLE_LLM_API_KEY": "llm_api_key",
    "NEEDLE_WEIGHTS": "needle_weights",
    "NEEDLE_ACTION_MODEL": "action_model",
    "NEEDLE_FG_BASE_URL": "fg_base_url",
}


def read_prompt(path: str | Path) -> str:
    path = Path(path).expanduser()
    with path.open("rb") as source:
        raw = source.read(MAX_PROMPT_CHARS * 4 + 1)
    if len(raw) > MAX_PROMPT_CHARS * 4:
        raise ValueError(f"Prompt file is too large: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"Prompt file must be UTF-8: {path}") from exc
    if not text.strip() or "\x00" in text or len(text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must contain 1–{MAX_PROMPT_CHARS} characters of text: {path}")
    return text


def parse_config(text: str, *, format: str = "toml", base: Path | None = None) -> dict[str, Any]:
    """Parse strictly. Browser imports use base=None and cannot read server-side files."""
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("Configuration file is too large.")
    try:
        document = json.loads(text) if format == "json" else tomllib.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {format.upper()} configuration: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Configuration must contain named sections.")
    values: dict[str, Any] = {}
    for section, entries in document.items():
        if section not in _CONFIG_SECTIONS or not isinstance(entries, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        for key, value in entries.items():
            if key not in _CONFIG_SECTIONS[section]:
                raise ValueError(f"Unknown configuration key: {section}.{key}")
            values[key] = value
    for name in _PROMPT_NAMES:
        if name + "_file" in values:
            filename = values.pop(name + "_file")
            if name in values:
                raise ValueError(f"Use either {name} or {name}_file, not both.")
            if base is None:
                raise ValueError(
                    "Browser imports must embed prompts; server file paths are not allowed."
                )
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"{name}_file must be a nonempty file path.")
            values[name] = read_prompt(base / Path(filename).expanduser())
    for name in ("workspace_root", "needle_weights"):
        if name in values and base is not None:
            value = values[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path.")
            values[name] = str((base / Path(value).expanduser()).resolve())
    if values.get("default_timezone") == "":
        values["default_timezone"] = None
    return values


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> AgentConfig:
    """Precedence: packaged defaults < explicit config < environment < CLI/overrides.

    No workspace file is auto-loaded: running an agent on an untrusted project
    must not implicitly trust that project's model URLs or prompt instructions.
    """
    env = os.environ if environ is None else environ
    selected = path if path is not None else env.get("NEEDLE_CONFIG")
    values: dict[str, Any] = {}
    if selected:
        filename = Path(selected).expanduser().resolve()
        with filename.open("rb") as source:
            raw = source.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("Configuration file is too large.")
        values = parse_config(
            raw.decode("utf-8"),
            format="json" if filename.suffix.lower() == ".json" else "toml",
            base=filename.parent,
        )
    values.update({field: env[key] for key, field in _ENV_FIELDS.items() if env.get(key)})
    values.update(overrides or {})
    unknown = values.keys() - {item.name for item in fields(AgentConfig)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return AgentConfig(**values)


def export_config(config: AgentConfig, *, include_server: bool = True) -> str:
    """Portable TOML with inline prompts. Never export credentials or machine-only weights."""
    values = asdict(config)
    values.pop("llm_api_key", None)
    values.pop("needle_weights", None)
    values["default_timezone"] = config.default_timezone or ""
    if not include_server:
        values.pop("workspace_root", None)
    lines = ["# Needle configuration. API keys are intentionally excluded.", ""]
    for section, keys in _CONFIG_SECTIONS.items():
        lines.append(f"[{section}]")
        for key in sorted(keys):
            if key not in values or values[key] is None:
                continue
            lines.append(f"{key} = {json.dumps(values[key], ensure_ascii=False, allow_nan=False)}")
        lines.append("")
    return "\n".join(lines)


def init_config(path: str | Path) -> Path:
    """Write an editable config and prompt files; refuse to overwrite existing files."""
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".toml":
        raise ValueError("Use a .toml destination for config init.")
    assets = {destination: default_asset("defaults.toml")}
    for name in _PROMPT_NAMES:
        relative = _DEFAULTS[name + "_file"]
        assets[destination.parent / relative] = default_asset(relative)
    if any(file.exists() for file in assets):
        raise ValueError("A configuration or prompt file already exists; nothing was overwritten.")
    for file, text in assets.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return destination
