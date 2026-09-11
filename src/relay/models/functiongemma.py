"""FunctionGemma translator over llama-server /completion. Stdlib HTTP only.

Wire format follows Google's FunctionGemma docs: a developer turn carrying
<start_function_declaration> blocks, a user turn with the instruction, and a
model turn emitting <start_function_call>call:name{arg:<escape>v<escape>}
<end_function_call>. The prompt is rendered by hand so no server-side chat
template or tool-call parser is required.

FunctionGemma emits no confidence signal. A strict single-call parse yields
1.0; anything else is a retryable ActionOutputError. Calibration therefore
rests on schema validation, intent checks, confidence gates for other
models, and write approval — never on this number alone.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from relay.config import AgentConfig
from relay.models.action import ActionOutputError, NeedleResult, ToolRanking
from relay.protocol.intent import deterministic_write_result
from relay.tools.base import Tool

_DEVELOPER = "You are a model that can do function calling with the following functions"
_CALL = re.compile(r"<start_function_call>(.*?)(?:<end_function_call>|\Z)", re.DOTALL)
_START = "<start_function_call>"
_NAME = re.compile(r"\Acall:([A-Za-z_][A-Za-z0-9_]*)\{(.*)\}\Z", re.DOTALL)
_ITEM = re.compile(r"\A([A-Za-z_][A-Za-z0-9_]*):(.*)\Z", re.DOTALL)
_ESCAPE = "<escape>"

_FG_TYPES = {
    "string": "STRING",
    "integer": "NUMBER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
    "null": "STRING",
}


def _clean(text: str) -> str:
    """Framing tokens inside declarations would corrupt the call format."""
    return text.replace(_ESCAPE, "")


def _fg_type(schema: dict[str, Any]) -> str:
    declared = schema.get("type", "string")
    types = declared if isinstance(declared, list) else [declared]
    for option in types:
        if option in _FG_TYPES and option != "null":
            return _FG_TYPES[option]
    return "STRING"


def render_declarations(tools: list[Tool]) -> str:
    """One declaration block per tool in Google's documented format."""
    blocks = []
    for tool in tools:
        params = tool.parameters or {"type": "object", "properties": {}}
        props = params.get("properties", {})
        required = params.get("required", [])
        fields = "".join(
            f"{name}:{{description:{_ESCAPE}{_clean(schema.get('description', ''))}"
            f"{_ESCAPE},type:{_ESCAPE}{_fg_type(schema)}{_ESCAPE}}},"
            for name, schema in props.items()
        )
        need = "".join(f"{_ESCAPE}{name}{_ESCAPE}," for name in required).rstrip(",")
        blocks.append(
            f"declaration:{tool.name}{{description:{_ESCAPE}{_clean(tool.description)}"
            f"{_ESCAPE},parameters:{{properties:{{{fields.rstrip(',')}}},"
            f"required:[{need}],type:{_ESCAPE}OBJECT{_ESCAPE}}}}}"
        )
    return "".join(
        f"<start_function_declaration>{block}<end_function_declaration>" for block in blocks
    )


def render_prompt(tools: list[Tool], instruction: str, developer: str = _DEVELOPER) -> str:
    return (
        f"<start_of_turn>developer\n{developer}{render_declarations(tools)}<end_of_turn>\n"
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _split_args(body: str) -> list[str]:
    """Split on top-level commas; commas inside <escape> pairs are data."""
    items, part, inside = [], [], False
    i = 0
    while i < len(body):
        if body.startswith(_ESCAPE, i):
            inside = not inside
            part.append(_ESCAPE)
            i += len(_ESCAPE)
        elif body[i] == "," and not inside:
            items.append("".join(part))
            part = []
            i += 1
        else:
            part.append(body[i])
            i += 1
    items.append("".join(part))
    if inside:
        raise ActionOutputError("FunctionGemma returned unbalanced <escape> delimiters.")
    return [item for item in items if item.strip()]


def parse_function_call(text: str, finished: bool = True) -> tuple[str, dict[str, Any]]:
    """Strictly parse exactly one function call. No repair, no guessing.

    The server consumes the <end_function_call> stop token, so a finished
    generation may legitimately lack the end marker; finished=False means the
    model stopped on its own without closing the call and is always an error.
    """
    if text.count(_START) > 1:
        raise ActionOutputError(
            "FunctionGemma returned multiple calls for one action. "
            "Choose one atomic tool operation."
        )
    calls = _CALL.findall(text)
    if not calls:
        raise ActionOutputError("FunctionGemma returned no function call.")
    if not finished and "<end_function_call>" not in text:
        raise ActionOutputError("FunctionGemma did not complete the function call.")
    match = _NAME.match(calls[0].strip())
    if not match:
        raise ActionOutputError("FunctionGemma returned a malformed tool selection.")
    name, body = match.group(1, 2)
    arguments: dict[str, Any] = {}
    if body.strip():
        for item in _split_args(body):
            entry = _ITEM.match(item.strip())
            if not entry:
                raise ActionOutputError("FunctionGemma returned a malformed argument.")
            key, raw = entry.group(1, 2)
            if key in arguments:
                raise ActionOutputError("FunctionGemma returned a duplicate argument.")
            raw = raw.strip()
            if raw.startswith(_ESCAPE) and raw.endswith(_ESCAPE) and len(raw) >= 2 * len(_ESCAPE):
                arguments[key] = raw[len(_ESCAPE) : -len(_ESCAPE)]
            elif raw == "None":
                arguments[key] = None
            else:
                arguments[key] = raw
    return name, arguments


def render_function_call(name: str, arguments: dict[str, Any]) -> str:
    """Render the canonical assistant target. Exact inverse of parse_function_call.

    Shared by training-data generation so the fine-tune sees byte-identical
    call shapes to what the runtime parses. String values are escape-wrapped;
    None/numbers/booleans are bare, matching Google's documented format.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid tool name: {name!r}")
    parts = []
    for key, value in arguments.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid argument name: {key!r}")
        if isinstance(value, str):
            if _ESCAPE in value:
                raise ValueError("Argument values must not contain the <escape> token.")
            parts.append(f"{key}:{_ESCAPE}{value}{_ESCAPE}")
        elif value is None:
            parts.append(f"{key}:None")
        elif isinstance(value, bool):
            parts.append(f"{key}:{'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}:{value}")
        else:
            raise ValueError(f"Unsupported argument value for {key!r}: {type(value).__name__}")
    return f"<start_function_call>call:{name}{{{','.join(parts)}}}<end_function_call>"


def _coerce_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert FunctionGemma's string scalars to declared numeric/boolean types.

    The call format carries every value as text; "10" for an integer argument
    is a transport artifact, not a model error. Anything unparseable is left
    for schema validation to reject.
    """
    props = (tool.parameters or {}).get("properties", {})
    coerced = dict(arguments)
    for key, value in arguments.items():
        schema = props.get(key, {})
        declared = schema.get("type", "string")
        types = declared if isinstance(declared, list) else [declared]
        if not isinstance(value, str):
            continue
        if "integer" in types:
            try:
                coerced[key] = int(value.strip(), 10)
            except ValueError:
                pass
        elif "number" in types:
            try:
                coerced[key] = float(value.strip())
            except ValueError:
                pass
        elif "boolean" in types and value.strip().lower() in ("true", "false"):
            coerced[key] = value.strip().lower() == "true"
    return coerced


class FunctionGemmaActionModel:
    """Stateless translator: each call is one independent /completion request."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        developer: str | None = None,
    ) -> None:
        from relay.config import normalize_model_url

        defaults = AgentConfig()
        self._schemas = [tool.needle_schema() for tool in tools]
        self._signature = json.dumps(self._schemas, sort_keys=True)
        self._base = normalize_model_url(
            defaults.fg_base_url if base_url is None else base_url
        )
        self._timeout_s = defaults.fg_timeout_s if timeout_s is None else timeout_s
        self._max_tokens = defaults.fg_max_tokens if max_tokens is None else max_tokens
        self._temperature = defaults.fg_temperature if temperature is None else temperature
        if type(self._max_tokens) is not int or self._max_tokens <= 0:
            raise ValueError("FunctionGemma max_tokens must be a positive integer.")
        if (
            type(self._timeout_s) not in (int, float)
            or self._timeout_s <= 0
            or type(self._temperature) not in (int, float)
            or not 0 <= self._temperature <= 2
        ):
            raise ValueError("FunctionGemma timeout and temperature are invalid.")
        self._api_key = api_key
        self._developer = developer or _DEVELOPER

    def request_messages(self, action: str) -> list[dict[str, str]]:
        return [
            {"role": "developer", "content": self._developer},
            {"role": "user", "content": action},
        ]

    def reset(self) -> None:
        """No session exists; every translate is already independent."""

    def _complete(self, prompt: str) -> tuple[str, bool]:
        payload = {
            "prompt": prompt,
            "n_predict": self._max_tokens,
            "temperature": self._temperature,
            "top_k": 64,
            "top_p": 0.95,
            "stop": ["<end_function_call>", "<end_of_turn>"],
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._base + "/completion",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                body = json.loads(response.read(2_000_000 + 1))
            content = body.get("content")
            if not isinstance(content, str):
                raise ActionOutputError("FunctionGemma backend returned no text.")
            # The server consumes a hit stop token, so a finished call usually
            # arrives without its end marker. Anything else without the marker
            # means the call was cut off and must not be trusted.
            finished = "<end_function_call>" in content or body.get("stop_type") in {
                "word",
                "eos",
            }
            return content, finished
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"FunctionGemma backend returned HTTP {exc.code}. Check the model server."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "FunctionGemma backend is unreachable or timed out. Start the model "
                "server (e.g. llama-server with a FunctionGemma GGUF on the "
                "configured fg_base_url)."
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise ActionOutputError("FunctionGemma backend returned invalid data.") from exc

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        if json.dumps([tool.needle_schema() for tool in tools], sort_keys=True) != self._signature:
            raise ValueError("FunctionGemmaActionModel is bound to a fixed toolset.")
        # Well-formed writes never reach the model: the runtime attaches the
        # <content> payload after translation (see workflow translate).
        shortcut = deterministic_write_result(action)
        if shortcut is not None:
            return shortcut
        text, finished = self._complete(render_prompt(tools, action, self._developer))
        name, arguments = parse_function_call(text, finished)
        tool = next((item for item in tools if item.name == name), None)
        if tool is None:
            raise ActionOutputError(f"FunctionGemma selected an unknown tool: {name!r}.")
        if not isinstance(arguments, dict):
            raise ActionOutputError("FunctionGemma returned non-object arguments.")
        arguments = _coerce_arguments(tool, arguments)
        return NeedleResult(
            selected_tool=name,
            arguments=arguments,
            confidence=1.0,
            rankings=[ToolRanking(tool_name=name, confidence=1.0)],
        )
