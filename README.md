# Needle · Agent workspace

A local-first agent that **reasons in natural language and acts through validated tools**.
Your reasoning model emits `<tool>` / `<final>` text. Needle 2 translates an action into
one tool call. This runtime—not either model—owns validation, confidence, permissions,
execution, and the next reasoning turn.

**Interact through the browser, a terminal conversation, or the Python API.**
Full architecture: [V0 specification](docs/spec-v0.md).

## Try it now — no model or API key needed

Requires Python **3.11–3.13** and [uv](https://docs.astral.sh/uv/).
No Node.js, frontend build, or extra web framework is required.

```sh
uv sync --locked
uv run needle-agent serve --demo --workspace examples/workspace
```

Open **http://localhost:3000**. Try:

- **Explore this workspace** — list files, then read the README.
- **Find the authentication implementation** — search, read, and report the matching file.
- **Calculate 24 * 18 + 120** — use the restricted arithmetic tool.
- **Create a note** — answer a question, approve the write, and see it read back.

> **Demo is not live AI.** Its deterministic planner and translator simulate the model
> interfaces and use explicitly synthetic confidence scores. Filesystem operations,
> arithmetic, time, the LangGraph loop, validation, approvals, and cancellation are real.
> Use live mode below for open-ended reasoning and actual Needle inference. There is
> **no silent fallback** from live mode to the demo.

Prefer pip? Create and activate a virtual environment, then run `python -m pip install -e .`.
Use `python -m agent_runtime` in place of `uv run needle-agent`. On Windows, activate with
`.venv\Scripts\Activate.ps1`; on macOS/Linux, use `source .venv/bin/activate`.

## Use real models

### 1. Start an OpenAI-compatible reasoning server

Ollama example, on the **same machine as the Python runtime**:

```sh
ollama pull qwen2.5:3b
# Start `ollama serve` if Ollama is not already running.
uv run needle-agent serve --workspace examples/workspace \
  --base-url http://127.0.0.1:11434/v1 --model qwen2.5:3b
```

Or use your existing llama.cpp / Ornith server:

```sh
uv run needle-agent serve --workspace /path/to/project \
  --base-url http://127.0.0.1:8080 --model ornith
```

Both bare server URLs and URLs ending in `/v1` are supported. A hosted compatible
provider also works. Its API key belongs in the **server's environment**, not in a
browser setting or repository file:

| Environment variable | Purpose |
| --- | --- |
| `NEEDLE_WORKSPACE` | Default filesystem root; otherwise the current directory |
| `NEEDLE_LLM_BASE_URL` | Reasoning server base URL |
| `NEEDLE_LLM_MODEL` | Model ID accepted by that server |
| `NEEDLE_LLM_API_KEY` | Optional bearer API key; never sent to the browser |
| `NEEDLE_WEIGHTS` | Optional custom `.cact` weights |
| `NEEDLE_LIB_PATH` | Needle 2 shared library for offline installation |

If an API key is configured, browser settings cannot redirect it to a different
origin. Set `NEEDLE_LLM_BASE_URL` on the server and restart to switch providers.
The HTTP adapter also rejects redirects rather than forwarding credentials.

In the browser, **Settings → Live models → Test connection** checks `/v1/models`
and initializes Needle. Save the settings, then send a message. Changing settings
while a run is active is blocked. Settings and histories are isolated per browser session.

**Remote previews:** `127.0.0.1` means the machine hosting the Python server, *not your
laptop*. Use a reasoning endpoint that machine can reach. The frontend only calls
same-origin relative `/api/...` URLs; it never directly connects to a model provider.

### 2. Let Needle initialize

`cactus-needle` downloads its small platform-specific inference engine from Hugging Face
on first use and caches it. Internet access is needed for that first download. Subsequent
inference is local. This runtime disables Needle telemetry by default; an explicitly set
`NEEDLE_TELEMETRY` environment variable is respected.

For an air-gapped machine, install the official engine on a connected machine and copy
the platform-matching `libneedle.so`, `libneedle.dylib`, or `libneedle.dll`. Set
`NEEDLE_LIB_PATH` to its absolute path. Follow the package's official offline setup
instructions; keep large model artifacts outside Git.

Custom Needle weights currently return uncalibrated confidence. Missing confidence is
treated as zero rather than invented; default gates therefore prevent execution. Do not
lower confidence gates just to hide model failures.

**This checkout's development environment could not download Hugging Face model files.**
The adapters and their HTTP/single-turn contracts are tested with local mock servers;
the live-model benchmark has not been rerun as part of this implementation.

## What you can interact with

- **Playground:** multi-turn conversations, streamed action/gate/result events, collapsible
  tool cards, `ask_user` pause/resume, per-write approval, and cooperative cancellation.
- **Files:** browse the configured workspace and inspect bounded text previews.
- **Tools:** inspect all seven canonical definitions and their validation schemas.
- **Run history:** real outcomes, step counts, durations, and downloadable JSON traces.
- **Settings:** demo/live mode, model endpoint, confidence thresholds, step limits,
  read-only mode, and explicit permission to create parent directories.

Use **Enter** to send, **Shift+Enter** for a newline, and **Ctrl/Cmd+K** for a new conversation.
Reloading the browser reconnects to active runs, including pending questions/approvals.

### Terminal

```sh
uv run needle-agent chat --demo --workspace examples/workspace --trace
uv run needle-agent run --demo 'Calculate 2 * (15 + 3)' --json

# Live terminal conversation
uv run needle-agent chat --workspace /path/to/project \
  --base-url http://127.0.0.1:11434/v1 --model qwen2.5:3b --trace
```

Terminal commands: `/new` (reset conversation), `/tools`, `/exit`. Both terminal and web
interfaces ask permission before every write. Library callers can inject their own
approval policy. Recoverable tool errors become observations; internal errors stop the run.

### Python API

```python
from agent_runtime import Agent, AgentConfig

config = AgentConfig(workspace_root="examples/workspace")
with Agent(config, approve_fn=lambda call: False) as agent:  # deny writes in this example
    first = agent.run("Find the authentication implementation.")
    print(first["status"], first["final_answer"])

    followup = agent.run("Show me that file.", history=first["messages"])
    print(followup["final_answer"])
```

Supply `reasoning=` and `action=` to replace either model without changing the graph.
`ReasoningModel.generate(messages)` returns text; `ActionModel.translate(action, tools)`
returns `NeedleResult`. `ask_fn` abstracts human input. Dependencies stay outside the
plain-data `AgentState`.

For streaming, iterate `agent.stream(request, history=...)`. It yields `phase`, `action`,
`translation`, `validated`, `confidence`, `safety`, `tool_start`, `tool_result`, and
`rejected` events, then a final `complete` event containing `state`. Alternatively, pass
`on_event=` to `run()`. Pass a thread-safe `cancelled=` predicate to stop at node boundaries.
Closing a stream alone does not constitute cancellation.

## Runtime guarantees and limits

```text
REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE
                                                    ├─ low → CONFIRM → REASON
                                                    └─ high → SAFETY → EXECUTE
                                                               → OBSERVE → UPDATE_CONTEXT → REASON
```

- Only a well-formed `<tool>` block contains executable intent. Final/tagless answers
  never execute arbitrary text. Multiple actions use the first, then return to reasoning.
- Invalid calls do not reach confidence or execution, regardless of their score.
- Read/search/calculate/time clear **0.50** by default; write/ask clear **0.85**.
- Defaults: **20 tool steps**, **3 consecutive non-executing turns**, **20,000 characters**
  per tool output, and **32,000 characters** of model context.
- The full system prompt, canonical tool descriptions, and original/current requests stay
  pinned. Older observations are dropped first; oversized recent output is truncated.
  Requests that cannot fit the pinned budget are rejected rather than exceeding it.
- Filesystem tools reject traversal, outside absolute paths, symlink escapes, and `.git`
  metadata. Reads are bounded; search skips generated directories, NUL-byte binaries,
  and files over **2 MB**, returning at most **50 matches** with line numbers/context.
- Calculator input is a restricted, resource-bounded AST; it never uses `eval`.
- No shell, terminal, arbitrary Python execution, append, binary write, or delete tool.
- Parent-directory creation is **off** unless explicitly enabled. `--read-only` enforces
  a server-side floor which browser settings cannot remove.
- Needle calls are serialized around its process-global C engine. Each translation resets
  its session and uses **`complete()` only**, never Needle's agent loop.

### Safety scope

This is a **personal development console**, not a production multi-tenant service. It
binds to `0.0.0.0` for remote previews; only expose it to trusted users or put it behind an
authenticated proxy. Anyone allowed to open the console can obtain a session and operate
its configured workspace. Start it against the included sample workspace when sharing a demo.
Opaque session tokens prevent cross-session access, but do not authenticate people.

Paths are checked before operations, not protected by an OS sandbox against a hostile
concurrent process replacing directory entries. Custom tool handlers are trusted Python
code and must enforce their own safety rules. Confidence scores are not proof that a call
matches user intent; review write approvals. Cancellation prevents subsequent operations,
but cannot undo a write or instantly abort an in-flight HTTP/C-engine call.

Conversations/runs live **in server memory**, expire after two hours of inactivity, and
reset on server restart. Histories are bounded (24 conversations / 60 runs per session).
Export traces to retain them. Browser tokens are stored locally; model keys are not.

## Development and tests

```sh
uv sync --locked --group dev
uv run pytest tests/unit
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Optional real-browser tests (otherwise skipped):

```sh
uv sync --locked --group dev --group browser
uv run --group browser playwright install chromium
uv run --group browser pytest tests/e2e/test_browser.py
```

`NEEDLE_BROWSER_EXECUTABLE` can point to an existing Chromium installation. The tests cover
real tool loops, strict Needle result parsing, schema validation, path/symlink safety,
context budgets, long runs, SSE recovery, session isolation, approval/denial, cancellation,
CLI interaction, mobile layout, untrusted-content rendering, and explicit live-backend errors.
No test requires model weights or an API key.

`uv.lock` records the resolved dependency versions. The frontend ships as static package
assets. Core dependencies remain LangGraph, Pydantic, and cactus-needle (plus timezone data
on Windows). HTTP, filesystem/search, arithmetic, and server orchestration use the stdlib.

### Benchmark

```sh
uv run python examples/benchmark.py --base-url http://127.0.0.1:8080 \
  --model ornith --tasks read,list,search,calc,time,write,multi,refusal --repeats 3
```

Both paths use the same tool registry, validator, execution safety, and step/stall limits.
The baseline intentionally permits native structured calling **only inside the benchmark**.
The lifted path counts actual invalid calls, confidence retries, and safety blocks from
runtime events. Each cell uses a fresh temporary workspace; existing directories are
never wiped. Output goes to ignored `.cache/bench_results.jsonl` by default.

[The original benchmark report](docs/benchmark-v0.md) is an archived, small-sample result,
not a claim about the newly extended runtime or your selected model.
