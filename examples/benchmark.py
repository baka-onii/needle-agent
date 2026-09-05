"""Compare native structured calling with reasoning → Needle on identical tools.

Requires real models; never manufactures benchmark scores from the offline demo.
Each run gets a fresh TemporaryDirectory. Existing workspaces are never cleaned
or overwritten. Results are JSONL; the archived V0 report is docs/benchmark-v0.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

from agent_runtime import Agent, AgentConfig, ToolCall, ToolError
from agent_runtime.execution.executor import execute
from agent_runtime.execution.validator import validate
from agent_runtime.models.reasoning import OpenAICompatibleReasoningModel
from agent_runtime.tools.registry import create_default_registry

TASKS = {
    "read": {"prompt": "Read config.py and tell me the auth backend.", "evidence": "local"},
    "list": {"prompt": "List the top-level directory of the project.", "evidence": "src"},
    "search": {"prompt": "Find which file defines authenticate_user.", "evidence": "auth.py"},
    "calc": {"prompt": "What is 2 * (15 + 3)?", "evidence": "36"},
    "time": {"prompt": "What time is it in UTC right now? Return ISO-8601 with timezone."},
    "write": {"prompt": "Write exactly 'hello benchmark' to note.txt, then read it back to me."},
    "multi": {
        "prompt": "Find which file defines authenticate_user and show me its contents.",
        "evidence": "authenticate_user",
    },
    "refusal": {"prompt": "Write a poem about the sea."},
}


def _setup_task(root: Path, task_id: str) -> None:
    if task_id in {"read", "list", "search", "multi"}:
        (root / "src").mkdir()
        (root / "main.py").write_text("from src.auth import authenticate_user\n", encoding="utf-8")
        (root / "config.py").write_text("AUTH_BACKEND = 'local'\n", encoding="utf-8")
        (root / "src" / "auth.py").write_text(
            "def authenticate_user(user):\n    return True\n",
            encoding="utf-8",
        )


def _check(task_id: str, final: str, root: Path) -> bool:
    if task_id == "write":
        note = root / "note.txt"
        return (
            note.is_file()
            and note.read_text(encoding="utf-8") == "hello benchmark"
            and "hello" in final
        )
    if task_id == "refusal":
        return not any(root.iterdir()) and bool(final.strip())
    if task_id == "time":
        return bool(
            re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", final)
        )
    return TASKS[task_id]["evidence"] in final


def no_human(question: str) -> str:
    raise ToolError(
        "No human is available in this benchmark. Complete the task without clarification."
    )


def _record(task: str, path: str, state: dict, root: Path, started: float, **metrics) -> dict:
    final = state.get("final_answer") or ""
    return {
        "task": task,
        "path": path,
        "success": state["status"] == "COMPLETED" and _check(task, final, root),
        "status": state["status"],
        "steps": state.get("step_count", 0),
        "seconds": round(time.monotonic() - started, 3),
        "final": final[:500],
        **metrics,
    }


def run_lifted(task_id: str, config: AgentConfig) -> dict:
    counts = {"invalid": 0, "confidence_retries": 0, "safety_blocks": 0}

    def count(event):
        if event["type"] == "rejected":
            stage = event["stage"]
            key = (
                "confidence_retries"
                if stage == "confidence"
                else ("safety_blocks" if stage == "safety" else "invalid")
            )
            counts[key] += 1

    started = time.monotonic()
    try:
        with Agent(config, ask_fn=no_human) as agent:
            state = agent.run(TASKS[task_id]["prompt"], on_event=count)
    except Exception as exc:
        state = {"status": "ERROR", "final_answer": str(exc), "step_count": 0}
    return _record(task_id, "lifted", state, Path(config.workspace_root), started, **counts)


def run_native(task_id: str, config: AgentConfig) -> dict:
    """Only the benchmark baseline gives JSON schemas to the reasoning model."""
    registry = create_default_registry(config, ask_fn=no_human)
    schemas = [{"type": "function", "function": tool.needle_schema()} for tool in registry.list()]
    model = OpenAICompatibleReasoningModel(
        config.llm_base_url,
        config.llm_model,
        timeout_s=config.llm_timeout_s,
        max_tokens=config.llm_max_tokens,
        api_key=config.llm_api_key,
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with tools. Use one tool per turn.",
        },
        {"role": "user", "content": TASKS[task_id]["prompt"]},
    ]
    started = time.monotonic()
    steps, invalid, stalls = 0, 0, 0
    status, final = "MAX_STEPS_REACHED", "Tool step limit reached."
    try:
        for _ in range((config.max_tool_steps + 1) * (config.max_stalls + 1)):
            body = model._request(
                "/chat/completions",
                {
                    "model": config.llm_model,
                    "messages": messages,
                    "tools": schemas,
                    "tool_choice": "auto",
                    "temperature": config.llm_temperature,
                    "max_tokens": config.llm_max_tokens,
                    "stream": False,
                },
            )
            message = body["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                status, final = "COMPLETED", message.get("content") or ""
                break
            if steps >= config.max_tool_steps:
                break
            raw = calls[0]
            messages.append(
                {"role": "assistant", "content": message.get("content"), "tool_calls": [raw]}
            )
            try:
                call = validate(
                    ToolCall(
                        name=raw["function"]["name"],
                        arguments=json.loads(raw["function"]["arguments"]),
                    ),
                    registry,
                )
            except (ToolError, ValueError, TypeError, KeyError) as exc:
                invalid += 1
                stalls += 1
                observation = f"Invalid tool call: {exc}. Correct it or answer directly."
            else:
                result = execute(call, registry, config)
                steps += 1
                stalls = 0
                observation = result.output if result.success else f"Tool error: {result.error}"
            messages.append(
                {"role": "tool", "tool_call_id": raw.get("id", "0"), "content": observation}
            )
            if stalls >= config.max_stalls:
                status, final = "STALLED", "Repeated invalid native calls."
                break
    except Exception as exc:
        status, final = "ERROR", str(exc)
    state = {"status": status, "final_answer": final, "step_count": steps}
    return _record(task_id, "native", state, Path(config.workspace_root), started, invalid=invalid)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task IDs")
    parser.add_argument("--out", default=".cache/bench_results.jsonl")
    parser.add_argument("--workroot", default=None, help="Parent for new temporary workspaces")
    parser.add_argument(
        "--base-url", default=os.getenv("NEEDLE_LLM_BASE_URL", "http://127.0.0.1:8080")
    )
    parser.add_argument("--model", default=os.getenv("NEEDLE_LLM_MODEL", "ornith"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    task_ids = [task.strip() for task in args.tasks.split(",")]
    if not task_ids or any(task not in TASKS for task in task_ids):
        parser.error(f"Tasks must be chosen from: {', '.join(TASKS)}")
    if args.repeats < 1 or args.max_steps < 1:
        parser.error("Repeats and max-steps must be positive.")
    if args.workroot:
        Path(args.workroot).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for task_id in task_ids:
            for path, runner in (("native", run_native), ("lifted", run_lifted)):
                with tempfile.TemporaryDirectory(
                    prefix=f"needle-{task_id}-{path}-", dir=args.workroot
                ) as tmp:
                    root = Path(tmp)
                    _setup_task(root, task_id)
                    config = AgentConfig(
                        workspace_root=str(root),
                        llm_base_url=args.base_url,
                        llm_model=args.model,
                        llm_api_key=os.getenv("NEEDLE_LLM_API_KEY"),
                        max_tool_steps=args.max_steps,
                        llm_max_tokens=512,
                        needle_weights=os.getenv("NEEDLE_WEIGHTS"),
                    )
                    print(f"[bench] {task_id}/{path} (repeat {repeat + 1})", flush=True)
                    record = runner(task_id, config)
                record.update(repeat=repeat + 1, model=args.model)
                with out.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(
                    f"  {record['status']} · success={record['success']} · "
                    f"{record['steps']} steps · {record['invalid']} invalid · {record['seconds']}s",
                    flush=True,
                )
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
