"""Browser console, interactive terminal chat, and a one-shot command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent_runtime import Agent, AgentConfig, ToolCall
from agent_runtime.config import export_config, init_config, load_config, read_prompt
from agent_runtime.tools.base import approval_summary, truncate_text

_REPO_ROOT = Path(
    os.environ.get("NEEDLE_REPO_ROOT", Path(__file__).resolve().parent.parent.parent)
)


def _repo_llama_server() -> str | None:
    """Repo-local binary built by scripts/build-llama-server.bat (.sh)."""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidate = _REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / name
    return str(candidate) if candidate.is_file() else None


def _repo_gguf() -> str | None:
    """First GGUF in the repo models/ directory (weights are never committed)."""
    models = _REPO_ROOT / "models"
    if not models.is_dir():
        return None
    found = sorted(models.glob("*.gguf"))
    return str(found[0]) if found else None


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Load a TOML/JSON config (or set NEEDLE_CONFIG)")
    parser.add_argument("--workspace", dest="workspace_root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        dest="mode",
        action="store_const",
        const="demo",
        help="Simulated models, real sandboxed tools",
    )
    mode.add_argument(
        "--live",
        dest="mode",
        action="store_const",
        const="live",
        help="Use the configured live models",
    )
    for flag, dest, kind in (
        ("base-url", "llm_base_url", str),
        ("model", "llm_model", str),
        ("max-tool-steps", "max_tool_steps", int),
        ("max-stalls", "max_stalls", int),
        ("confidence-threshold", "confidence_threshold", float),
        ("read-only-threshold", "read_only_threshold", float),
        ("max-context-chars", "max_context_chars", int),
        ("llm-max-tokens", "llm_max_tokens", int),
        ("needle-max-tokens", "needle_max_tokens", int),
        ("temperature", "llm_temperature", float),
        ("timeout", "llm_timeout_s", float),
        ("timezone", "default_timezone", str),
    ):
        parser.add_argument("--" + flag, dest=dest, type=kind, default=None)
    parser.add_argument(
        "--stream",
        dest="llm_stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use actual provider streaming when supported",
    )
    parser.add_argument(
        "--capture-model-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include sent model messages in inspectable traces",
    )
    parser.add_argument(
        "--stream-buffer-ms", type=int, default=None, help="Browser display buffer, in milliseconds"
    )
    parser.add_argument(
        "--read-only", action=argparse.BooleanOptionalAction, default=None, help="Block all writes"
    )
    parser.add_argument(
        "--allow-create-parents",
        dest="allow_create_parent_dirs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--workspace-listing",
        dest="include_workspace_listing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    for name in ("reasoning", "translator", "confirmation"):
        parser.add_argument(
            f"--{name}-prompt",
            dest=f"{name}_prompt_file",
            metavar="FILE",
            help=f"Load {name} instructions from a UTF-8 file",
        )
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override any AgentConfig field; JSON values or a plain string",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="needle-agent", description="A local-first agent workspace."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("serve", "Open the browser workspace"),
        ("chat", "Interactive terminal conversation"),
        ("run", "Run a single request"),
    ):
        sub = commands.add_parser(command, help=help_text)
        _add_config_arguments(sub)
        if command == "serve":
            sub.add_argument("--host", default="0.0.0.0")
            sub.add_argument("--port", type=int, default=3000)
        else:
            sub.add_argument(
                "--trace",
                action="store_true",
                help="Print actions, selection reviews, and gate events",
            )
        if command == "run":
            sub.add_argument("request")
            sub.add_argument(
                "--json", action="store_true", help="Print the terminal result as JSON"
            )
    live = commands.add_parser(
        "live",
        help="Start the fine-tuned translator server if needed, then chat or serve",
    )
    _add_config_arguments(live)
    live.add_argument(
        "--ui",
        action="store_true",
        help="Open the browser workspace instead of terminal chat",
    )
    live.add_argument("--host", default="0.0.0.0")
    live.add_argument("--port", type=int, default=3000)
    live.add_argument(
        "--trace",
        action="store_true",
        help="Print actions, selection reviews, and gate events",
    )
    live.add_argument(
        "--fg-gguf",
        default=None,
        help="Fine-tuned translator GGUF (or set FG_GGUF)",
    )
    live.add_argument(
        "--llama-server",
        default=None,
        help="llama-server binary (or set LLAMA_SERVER)",
    )
    live.add_argument("--fg-port", type=int, default=8081)
    live.add_argument(
        "--no-server-start",
        action="store_true",
        help="Use the already-running translator server instead of starting one",
    )
    config = commands.add_parser("config", help="Create or inspect portable configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    init = config_commands.add_parser("init", help="Create a config plus editable prompt files")
    init.add_argument("path", nargs="?", default="needle.toml")
    show = config_commands.add_parser(
        "show", help="Print effective TOML configuration (no secrets)"
    )
    _add_config_arguments(show)
    return parser


def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    from dataclasses import fields

    values = {}
    for setting in args.settings:
        key, separator, value = setting.partition("=")
        if not separator or not key:
            raise ValueError("Use --set NAME=VALUE.")
        try:
            values[key] = json.loads(value)
        except ValueError:
            values[key] = value
    for item in fields(AgentConfig):
        value = getattr(args, item.name, None)
        if value is not None:
            values[item.name] = value
    for name in ("reasoning", "translator", "confirmation"):
        filename = getattr(args, f"{name}_prompt_file", None)
        if filename:
            values[f"{name}_prompt"] = read_prompt(filename)
    return load_config(args.config, overrides=values)


def _approve(call: ToolCall) -> bool:
    print(f"\nApproval requested: {approval_summary(call)}")
    for key in ("content", "code", "command", "patch", "message"):
        if call.arguments.get(key):
            print(truncate_text(str(call.arguments[key]), 2_000))
            break
    try:
        return input("Allow this action? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _trace(event: dict) -> None:
    if event["type"] == "model_start":
        print(
            f"\n  [{event['component']} · {event['model']} · "
            f"{'stream requested' if event['streamed'] else 'buffered'}]",
            file=sys.stderr,
        )
    elif event["type"] == "model_delta":
        print(event["delta"], end="", file=sys.stderr, flush=True)
    elif event["type"] == "model_end":
        print(f"\n  [{event['status']} · {event['duration_ms']} ms]", file=sys.stderr)
    elif event["type"] == "action":
        print(f"  → {event['action']}", file=sys.stderr)
    elif event["type"] == "confidence":
        print(
            f"  {event['tool']}: {event['score']:.2f} / gate {event['threshold']:.2f}",
            file=sys.stderr,
        )
    elif event["type"] == "rejected":
        print(f"  Blocked at {event['stage']}: {event['message']}", file=sys.stderr)
    elif event["type"] == "confirmation":
        candidates = ", ".join(
            f"{item['tool_name']}={item['confidence']:.2f}" for item in event["candidates"]
        )
        print(
            f"  Reasoning review: {event['reason']} "
            f"Suggested tool: {event['suggested_tool'] or 'none'}. "
            f"Candidates: {candidates or 'none supplied'}",
            file=sys.stderr,
        )
    elif event["type"] == "tool_result":
        print(f"  {'✓' if event['success'] else '✕'} {event['tool']}", file=sys.stderr)


def _translator_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _resolve_translator_server(args: argparse.Namespace) -> tuple[str, subprocess.Popen | None]:
    """Return (fg_base_url, owned_process). Starts llama-server unless present/disabled."""
    base_url = f"http://127.0.0.1:{args.fg_port}"
    if _translator_health(base_url):
        print(f"Translator server already up at {base_url}")
        return base_url, None
    if args.no_server_start:
        raise ValueError(
            f"No translator server at {base_url} and --no-server-start was given."
        )
    gguf = args.fg_gguf or os.environ.get("FG_GGUF") or _repo_gguf()
    if gguf is None:
        raise ValueError(
            "No translator model found. Pass --fg-gguf PATH, set FG_GGUF, "
            "or place the fine-tuned GGUF in the repo models/ directory."
        )
    if not os.path.isfile(gguf):
        hint = ""
        if _REPO_ROOT.joinpath("models").is_dir():
            available = sorted(
                path.name for path in _REPO_ROOT.joinpath("models").glob("*.gguf")
            )
            if available:
                hint = f" Available in models/: {', '.join(available)}."
        raise ValueError(f"Translator model not found: {gguf}.{hint}")
    binary = args.llama_server or os.environ.get("LLAMA_SERVER") or _repo_llama_server()
    if binary is None:
        fallback = r"E:\llama.cpp\llama-server.exe"
        try:
            binary = fallback if os.path.isfile(fallback) else "llama-server"
        except OSError:
            binary = "llama-server"
    log_path = os.path.join(tempfile.gettempdir(), "needle-fg-server.log")
    log = open(log_path, "a", encoding="utf-8")  # noqa: PTH123
    try:
        process = subprocess.Popen(
            [
                binary,
                "-m", gguf,
                "-ngl", "all",
                "-fa", "on",
                "-c", "32768",
                "--port", str(args.fg_port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        log.close()
        raise ValueError(f"Cannot start translator server ({binary}): {exc}") from exc
    print(f"Starting translator server ({gguf}) — log: {log_path}")
    for _ in range(90):
        time.sleep(2)
        if process.poll() is not None:
            raise ValueError(
                f"Translator server exited early; see {log_path}."
            )
        if _translator_health(base_url):
            print(f"Translator server up at {base_url}")
            return base_url, process
    process.terminate()
    raise ValueError(f"Translator server did not answer at {base_url}; see {log_path}.")


def _warn_if_reasoning_down(config: AgentConfig) -> None:
    root = config.llm_base_url
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    if not _translator_health(root):
        print(
            f"Warning: no reasoning server at {config.llm_base_url} "
            "(start your chat model there; continuing anyway).",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    owned: subprocess.Popen | None = None
    try:
        if args.command == "config" and args.config_command == "init":
            path = init_config(args.path)
            print(f"Created {path} and editable prompts in {path.parent / 'prompts'}")
            return 0
        if args.command == "live":
            fg_base_url, owned = _resolve_translator_server(args)
            if not any(setting.startswith("fg_base_url=") for setting in args.settings):
                args.settings = [*args.settings, f"fg_base_url={fg_base_url}"]
            if not any(setting.startswith("action_model=") for setting in args.settings):
                args.settings = [*args.settings, "action_model=functiongemma"]
            args.mode = "live"
        config = _config_from_args(args)
        if args.command == "config":
            print(export_config(config))
            return 0
        if args.command == "live":
            _warn_if_reasoning_down(config)
        if args.command == "serve" or (args.command == "live" and args.ui):
            from agent_runtime.server import serve

            serve(config, demo=config.mode == "demo", host=args.host, port=args.port)
            return 0
        with Agent(config, approve_fn=_approve) as agent:
            if args.command == "run":
                result = agent.run(args.request, on_event=_trace if args.trace else None)
                if args.json:
                    print(
                        json.dumps(
                            {key: result[key] for key in ("status", "step_count", "final_answer")}
                        )
                    )
                else:
                    print(result["final_answer"])
                    print(
                        f"[{result['status']}; {result['step_count']} tool steps]", file=sys.stderr
                    )
                return 0 if result["status"] == "COMPLETED" else 1
            print(
                "Needle · "
                + (
                    "offline demo (simulated models, real tools)"
                    if config.mode == "demo"
                    else "live models"
                )
            )
            print("/new resets context · /tools lists tools · /exit quits\n")
            history = []
            while True:
                try:
                    request = input("You > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if request in {"/exit", "/quit"}:
                    break
                if request in {"/new", "/reset"}:
                    history = []
                    print("Started a new conversation.\n")
                    continue
                if request == "/tools":
                    print("\n".join(tool.reasoning_description() for tool in agent.registry.list()))
                    continue
                if not request:
                    continue
                try:
                    result = agent.run(
                        request, history=history, on_event=_trace if args.trace else None
                    )
                except KeyboardInterrupt:
                    print("\nRun interrupted.\n")
                    continue
                history = result["messages"]
                print(f"\nNeedle > {result['final_answer']}\n")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Needle: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned is not None and owned.poll() is None:
            print("Stopping the translator server started by this command.")
            owned.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
