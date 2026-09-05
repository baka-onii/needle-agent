"""V0 benchmark: native LLM tool calling vs Needle-lifted calling.

Same model (llama-server), same 7 tools, same tasks. The only difference is
who produces structured output: the reasoning LLM itself (native `tools`
parameter) or Needle 2 (lifted `<tool>` NL actions).

Usage (server must be up):
    .venv\\Scripts\\python.exe examples\\benchmark.py [--tasks read,list] [--out results.jsonl]

Results: one JSON record per task×path, plus a human-readable summary.
The committed report lives in docs/benchmark-v0.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_runtime.agent import Agent  # noqa: E402
from agent_runtime.config import AgentConfig  # noqa: E402
from agent_runtime.execution.executor import execute  # noqa: E402
from agent_runtime.execution.validator import validate  # noqa: E402
from agent_runtime.tools.base import ToolCall, ToolError, ToolResult  # noqa: E402
from agent_runtime.tools.registry import create_default_registry  # noqa: E402

BASE_URL = "http://127.0.0.1:8080"
MAX_STEPS = 8


def _chat(messages: list[dict], tools: list[dict] | None, max_tokens: int = 256) -> dict:
    payload: dict = {
        "model": "ornith",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def _setup_task(root: Path, task_id: str) -> None:
    if task_id in ("read", "list", "search", "multi"):
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "main.py").write_text("from src.auth import authenticate_user\n")
        (root / "config.py").write_text("AUTH_BACKEND = 'local'\n")
        (root / "src" / "auth.py").write_text("def authenticate_user(user):\n    return True\n")
    else:
        root.mkdir(parents=True, exist_ok=True)


TASKS: dict[str, dict] = {
    "read": {
        "prompt": "Read the file config.py and tell me the auth backend.",
        "evidence": "local",
    },
    "list": {
        "prompt": "List the top-level directory of the project.",
        "evidence": "src",
    },
    "search": {
        "prompt": "Find which file defines authenticate_user.",
        "evidence": "auth.py",
    },
    "calc": {
        "prompt": "What is 2 * (15 + 3)?",
        "evidence": "36",
    },
    "time": {
        "prompt": "What time is it in UTC right now?",
        "evidence": ":",
    },
    "write": {
        "prompt": "Write exactly 'hello benchmark' to note.txt, then read it back to me.",
        "evidence": "hello benchmark",
    },
    "multi": {
        "prompt": "Find which file defines authenticate_user and show me its contents.",
        "evidence": "authenticate_user",
    },
    "refusal": {
        "prompt": "Write a poem about the sea.",
        "evidence": "",
    },
}


def _check(task_id: str, final: str, workdir: Path) -> bool:
    final = final or ""
    if task_id == "write":
        note = workdir / "note.txt"
        return note.exists() and note.read_text() == "hello benchmark" and "hello" in final
    if task_id == "refusal":
        leftovers = [p for p in workdir.iterdir()]
        return not leftovers and len(final.strip()) > 0
    return TASKS[task_id]["evidence"] in final


def run_lifted(task_id: str, workdir: Path) -> dict:
    def no_human(question: str) -> str:
        raise ToolError("No human available in benchmark.")

    config = AgentConfig(workspace_root=str(workdir), max_tool_steps=MAX_STEPS)
    agent = Agent(config=config, ask_fn=no_human)
    started = time.time()
    try:
        state = agent.run(TASKS[task_id]["prompt"])
        ok = _check(task_id, state["final_answer"] or "", workdir)
        return {
            "task": task_id,
            "path": "lifted",
            "success": ok,
            "status": state["status"],
            "steps": state["step_count"],
            "invalid": 0,
            "seconds": round(time.time() - started, 1),
            "final": (state["final_answer"] or "")[:300],
        }
    except Exception as exc:
        return {
            "task": task_id,
            "path": "lifted",
            "success": False,
            "status": f"EXCEPTION: {type(exc).__name__}: {exc}",
            "steps": -1,
            "invalid": 0,
            "seconds": round(time.time() - started, 1),
            "final": "",
        }


def run_native(task_id: str, workdir: Path) -> dict:
    config = AgentConfig(workspace_root=str(workdir))
    registry = create_default_registry(config)
    schemas = [{"type": "function", "function": t.needle_schema()} for t in registry.list()]
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": TASKS[task_id]["prompt"]},
    ]
    steps = 0
    invalid = 0
    final_text = ""
    started = time.time()
    try:
        for _ in range(MAX_STEPS):
            body = _chat(messages, schemas)
            msg = body["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if msg.get("content"):
                final_text = msg["content"]
            if not calls:
                break
            call = calls[0]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                tool_call = validate(
                    ToolCall(name=call["function"]["name"], arguments=args), registry
                )
            except (ToolError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                invalid += 1
                messages.append({"role": "assistant", "content": None, "tool_calls": calls[:1]})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", "0"),
                        "content": f"Invalid tool call: {exc}. Fix it or answer directly.",
                    }
                )
                continue
            result: ToolResult = execute(tool_call, registry, config)
            steps += 1
            messages.append({"role": "assistant", "content": None, "tool_calls": calls[:1]})
            text = result.output if result.success else f"Tool error: {result.error}"
            messages.append({"role": "tool", "tool_call_id": call.get("id", "0"), "content": text})
        else:
            final_text = final_text or ""
        if not final_text:
            body = _chat(
                [*messages, {"role": "user", "content": "Summarize the result for the user."}],
                None,
                max_tokens=128,
            )
            final_text = body["choices"][0]["message"].get("content") or ""
        ok = _check(task_id, final_text, workdir)
        return {
            "task": task_id,
            "path": "native",
            "success": ok,
            "status": "COMPLETED",
            "steps": steps,
            "invalid": invalid,
            "seconds": round(time.time() - started, 1),
            "final": final_text[:300],
        }
    except Exception as exc:
        return {
            "task": task_id,
            "path": "native",
            "success": False,
            "status": f"EXCEPTION: {type(exc).__name__}: {exc}",
            "steps": steps,
            "invalid": invalid,
            "seconds": round(time.time() - started, 1),
            "final": final_text[:300],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task ids")
    parser.add_argument("--out", default="bench_results.jsonl")
    parser.add_argument("--workroot", default=r"C:\Users\acer\AppData\Local\Temp\opencode\bench")
    args = parser.parse_args()

    task_ids = [t for t in args.tasks.split(",") if t in TASKS]
    out_path = Path(args.out)
    for task_id in task_ids:
        for path in ("native", "lifted"):
            workdir = Path(args.workroot) / f"{task_id}_{path}"
            if workdir.exists():
                for child in sorted(workdir.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
            _setup_task(workdir, task_id)
            print(f"[bench] {task_id}/{path} ...", flush=True)
            if path == "native":
                record = run_native(task_id, workdir)
            else:
                record = run_lifted(task_id, workdir)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            print(
                f"[bench] {task_id}/{path} success={record['success']} "
                f"steps={record['steps']} invalid={record['invalid']} "
                f"status={record['status']} {record['seconds']}s",
                flush=True,
            )
    print("[bench] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
