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
uv run needle-agent serve --workspace /path/to/project --base-url http://127.0.0.1:8080 --model ornith
```

Both bare server URLs and URLs ending in `/v1` are supported. A hosted compatible
provider also works. Its API key belongs in the **server's environment**, not in a
browser setting or repository file:

| Environment variable | Purpose |
| --- | --- |
| `NEEDLE_CONFIG` | Explicit TOML/JSON configuration file |
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

- **Playground:** streamed model reasoning, draft tool requests, and answers; collapsible,
  resizable reasoning panels with inspectable model messages; tool cards, `ask_user`
  pause/resume, approval for severe actions, and cancellation.
- **Files:** browse the configured workspace and inspect bounded text previews.
- **Tools:** inspect all thirty-three canonical definitions and their validation schemas.
- **Run history:** real outcomes, step counts, durations, and downloadable JSON traces.
- **Settings:** editable reasoning/translator/confirmation prompts, model generation budgets,
  confidence thresholds, retry/context/search limits, read-only mode, and config import/export.
- **Appearance:** light/dark switch in the top bar; follows your system until you choose a theme,
  then remembers that choice across reloads.
- **Chat management:** delete a conversation with its sidebar trash button. Confirmation is
  required; its messages and run traces are removed, **not workspace files**. Active runs must
  finish or be stopped first.

Use **Enter** to send, **Shift+Enter** for a newline, and **Ctrl/Cmd+K** for a new conversation.
Reloading the browser reconnects to active runs, including pending questions/approvals.

### Terminal

```sh
uv run needle-agent chat --demo --workspace examples/workspace --trace
uv run needle-agent run --demo 'Calculate 2 * (15 + 3)' --json

# Live terminal conversation
uv run needle-agent chat --workspace /path/to/project --base-url http://127.0.0.1:11434/v1 --model qwen2.5:3b --trace

# One command: translator server (fine-tuned GGUF) + live harness
uv run needle-agent live --workspace /path/to/project --fg-gguf /path/to/fg-tools.gguf
uv run needle-agent live --workspace /path/to/project --fg-gguf /path/to/fg-tools.gguf --ui  # browser GUI
```

Terminal commands: `/new` (reset conversation), `/tools`, `/exit`. Both terminal and web
interfaces ask permission before severe actions (deletion, code execution, text
edits, commits); simple writes run freely. Library callers can inject their own
approval policy. Recoverable tool errors become observations; internal errors stop the run.

### Configuration files, prompts, and CLI overrides

Defaults now live in **`config/defaults.toml`** and **`config/prompts/*.md`**, not embedded
system-prompt strings. They are also bundled into the installed package. Create your own copy:

```sh
uv run needle-agent config init needle.toml
# Edit needle.toml and prompts/{reasoning,translator,confirmation}.md.
uv run needle-agent serve --config needle.toml --workspace examples/workspace --demo

# Individual prompt files and numeric overrides work with run, chat, and serve.
uv run needle-agent chat --config needle.toml --reasoning-prompt prompts/reasoning.md \
  --llm-max-tokens 4096 --needle-max-tokens 4096 --set max_search_results=25

# Resolve file references into a portable config with inline prompts (no API keys).
uv run needle-agent config show --config needle.toml > portable.toml
```

Precedence: **packaged defaults → explicit config → environment → CLI overrides**.
File-relative paths resolve beside the config file. Workspace config files are **not**
auto-discovered or silently trusted; use `--config` or `NEEDLE_CONFIG`. File changes take
effect when reloaded/restarted. UI changes apply to the next run and stay session-local;
export them to survive a server restart. The UI can import the portable file above but
cannot read arbitrary server-side prompt paths or change the server's workspace/weights.
See [Configuration reference](docs/configuration.md) for fields, bounds, and examples.

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

For streaming, iterate `agent.stream(request, history=...)`. Model calls emit `model_start`,
`model_status`, `model_delta`, and `model_end`, alongside the existing `phase`, `action`,
`translation`, `validated`, `confidence`, `confirmation`, `safety`, `tool_start`, `tool_result`,
and `rejected` events. The last `complete` event contains the terminal `state`. Alternatively,
pass `on_event=` to `run()`. A thread-safe `cancelled=` predicate stops an active HTTP stream
and prevents subsequent tools. Closing a stream alone does not constitute cancellation.

### Live model output in the chat

Expand a **Reasoning** card to see provider-exposed reasoning or model commentary as it
arrives. **Expand / Shrink** changes the viewport; the lower resize handle is also draggable.
**Model conversation** reveals the messages sent to the adapter and its raw reply. Both
panel disclosure and scroll position are preserved while streaming. Translations appear
inside the corresponding tool card, including the exact request and structured result.

`<tool>` and `<final>` tags are recognized incrementally across token boundaries. Draft tool
cards are visibly **not executed**; final answers render as streaming Markdown, including
unfinished code blocks. The full response must finish successfully before the normal parser,
translator, validation, confidence, and approval gates run. A late final answer still wins
over earlier tool drafts. Interrupted streams retain visible partial output, not a successful
answer or executable partial call.

The browser uses a **1,000 ms initial buffer**, measured delivery rate, and bounded catch-up
to smooth uneven token delivery. Short/completed responses catch up promptly. This is a
presentation buffer, not an execution delay. Change it under **Settings → Streaming & model
conversation**; `0` displays incoming text immediately. Reduced-motion preferences skip typing
animation. Reconnecting restores the received prefix without replaying it token by token.

Streaming is enabled by default for compatible reasoning servers. If a server returns JSON
instead of SSE, it is labelled **buffered**. If it rejects streaming altogether, disable it
with **Settings** or `--no-stream`; requests are not silently retried. Needle's current
single-turn `complete()` API returns a whole translation, so that output is shown honestly
as buffered rather than inventing token events. The offline demo explicitly simulates delivery.
No unavailable/private reasoning is fabricated or requested; only text returned by the adapter
is displayed. See [Streaming reference](docs/streaming.md) for the protocol, limits, and adapters.

### Concrete writes, workspace paths, and selection review

The reasoning model must **compose the actual file content**, not tell the small translator
to generate it. This is one atomic write, expressed in natural language:

````text
<tool>Use write_file to write the file "test.txt" with this exact text:
```text
1. print(): Display values.
2. len(): Count items.
```
</tool>
````

For a request for ten functions, the payload must contain all ten. The runtime rejects writes
with no explicit literal payload and rejects translated content that differs from that payload.
Fences preserve indentation, blank lines, and literal protocol tags. A longer fence can contain
triple backticks. The newline immediately before the closing fence is a delimiter; include an
extra newline if the file should end with one. A small output budget may truncate long payloads:
reasoning now defaults to **4,096** output tokens and Needle to **2,048**, both configurable.

Both models receive the real workspace root and a bounded directory snapshot. Use **`.`** for
that root, never guessed aliases such as `root` or `user/home`. Nested directory observations
show workspace-relative paths. Explicit tool names and quoted paths are checked against the
translator's proposal; wrong paths are sent back for correction, not silently remapped.

On low confidence or recoverable failure, the **reasoning model** receives the original action,
proposed tool and arguments, sorted available candidates, and an explicit question about the
highest-ranked tool. It must confirm/correct the choice with a new atomic `<tool>Use NAME to
...</tool>` action or finish. The `confirmation` event appears in expanded tool cards and CLI
traces. Native Needle may supply only the selected tool's score; alternatives are never invented.
As in spec §24, confirmation **does not bypass** retranslation, validation, confidence, or safety.

Approval for severe actions belongs to the runtime, never `ask_user`. Identical
successful mutations, further actions on a path denied in the same run, repeated
answered questions, and common permission questions are blocked before another
interaction. Repeated identical failed calls have a bounded
budget. Records live separately from trimmed model context, so context eviction cannot reset these
per-run guards. A new user turn starts a fresh run, allowing an intentional new request.

## Runtime guarantees and limits

```text
REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE
                                                    ├─ low → CONFIRM → REASON
                                                    └─ high → SAFETY → EXECUTE
                                                               → OBSERVE → UPDATE_CONTEXT → REASON
```

- Only a well-formed `<tool>` block contains executable intent. Final/tagless answers
  never execute arbitrary text. Multiple blocks use the first, then return to reasoning.
  Common compound instructions within a block are rejected; multiple translated calls are
  rejected outright. Natural-language checks are conservative, not a proof of semantic intent.
- Invalid calls do not reach confidence or execution, regardless of their score.
- Read/search/calculate/time clear **0.50** by default; write/ask clear **0.85**.
- Defaults: **20 tool steps**, **3 consecutive non-executing turns**, **20,000 characters**
  per tool output, **32,000 characters** of model context, and at most **2 attempts** at an
  identical failing tool call before it is blocked. These limits are configurable.
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
but cannot undo a write. Connected HTTP streams are interrupted on Stop; initial connection
work, generate-only adapters, and Needle's blocking C-engine call remain cooperative.

Conversations/runs live **in server memory**, expire after two hours of inactivity, and
reset on server restart. Histories are bounded (24 conversations / 60 runs per session).
Export traces to retain them. Deleting a chat also deletes its stored runs and traces.
Browser tokens and your theme choice are stored locally; model keys are not. Model input
traces can contain workspace/file content. Disable `capture_model_inputs` for new runs if you
do not want their sent messages retained. Each run caps additional model trace details at
4 MB / 16,000 events, with a visible notice; tool results and the terminal answer remain.

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
uv run --group browser pytest tests/e2e
```

`NEEDLE_BROWSER_EXECUTABLE` can point to an existing Chromium installation. The tests cover
real tool loops, strict Needle result parsing, schema validation, path/symlink safety,
context budgets, long runs, SSE recovery, session isolation, approval/denial, cancellation,
CLI configuration, exact write payloads, selection reviews, repeated-interaction guards,
configuration import/export, chat deletion, light/dark/mobile layouts, untrusted-content
rendering, true HTTP/SSE delivery, split tags, partial code blocks, stream cancellation,
replay, adaptive pacing, stable reasoning panels, and explicit live-backend errors.
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
