"""Browser console, interactive terminal chat, and a one-shot command."""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent_runtime import Agent, AgentConfig, ToolCall
from agent_runtime.models.demo import DemoActionModel, DemoReasoningModel
from agent_runtime.tools.base import truncate_text


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
        sub.add_argument("--workspace", default=os.getenv("NEEDLE_WORKSPACE", "."))
        sub.add_argument(
            "--demo", action="store_true", help="Simulated models, real sandboxed tools"
        )
        sub.add_argument(
            "--base-url", default=os.getenv("NEEDLE_LLM_BASE_URL", "http://127.0.0.1:8080")
        )
        sub.add_argument("--model", default=os.getenv("NEEDLE_LLM_MODEL", "ornith"))
        sub.add_argument("--max-tool-steps", type=int, default=20)
        sub.add_argument("--read-only", action="store_true", help="Block all writes")
        sub.add_argument("--allow-create-parents", action="store_true")
        if command == "serve":
            sub.add_argument("--host", default="0.0.0.0")
            sub.add_argument("--port", type=int, default=3000)
        else:
            sub.add_argument("--trace", action="store_true", help="Print action and gate events")
        if command == "run":
            sub.add_argument("request")
            sub.add_argument(
                "--json", action="store_true", help="Print the terminal result as JSON"
            )
    return parser


def _approve(call: ToolCall) -> bool:
    print(f"\nWrite approval: {call.arguments['path']}")
    print(truncate_text(call.arguments.get("content", ""), 2_000))
    try:
        return input("Allow this write? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _trace(event: dict) -> None:
    if event["type"] == "action":
        print(f"  → {event['action']}", file=sys.stderr)
    elif event["type"] == "confidence":
        print(
            f"  {event['tool']}: {event['score']:.2f} / gate {event['threshold']:.2f}",
            file=sys.stderr,
        )
    elif event["type"] == "rejected":
        print(f"  Blocked at {event['stage']}: {event['message']}", file=sys.stderr)
    elif event["type"] == "tool_result":
        print(f"  {'✓' if event['success'] else '✕'} {event['tool']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = AgentConfig(
            workspace_root=args.workspace,
            llm_base_url=args.base_url,
            llm_model=args.model,
            llm_api_key=os.getenv("NEEDLE_LLM_API_KEY"),
            needle_weights=os.getenv("NEEDLE_WEIGHTS"),
            max_tool_steps=args.max_tool_steps,
            read_only=args.read_only,
            allow_create_parent_dirs=args.allow_create_parents,
        )
        if args.command == "serve":
            from agent_runtime.server import serve

            serve(config, demo=args.demo, host=args.host, port=args.port)
            return 0
        models = (
            {"reasoning": DemoReasoningModel(), "action": DemoActionModel()} if args.demo else {}
        )
        with Agent(config, **models, approve_fn=_approve) as agent:
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
                + ("offline demo (simulated models, real tools)" if args.demo else "live models")
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


if __name__ == "__main__":
    raise SystemExit(main())
