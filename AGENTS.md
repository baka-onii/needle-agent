# AGENTS.md

## Source of truth
- `docs/final-spec.md` is the full architecture spec; `docs/spec-v0.md` covers the
  original V0 slice. Trust the specs over any other prose.

## Starting the harness
- Terminal with the fine-tuned translator: `python -m relay live
  --workspace <dir> --fg-gguf <model.gguf>` (add `--ui` for the browser GUI).
  The command starts `llama-server` with the GGUF if port 8081 is free and
  stops only servers it started itself. Reasoning model must already serve
  OpenAI-compatible chat on the configured `llm_base_url` (default 8080).
- Fresh-clone translator setup: build with `scripts/build-llama-server.bat`
  (Windows) or `scripts/build-llama-server.sh` (Linux) — VS Build Tools/gcc +
  CUDA toolkit + cmake + ninja; only the `llama-server` target builds into
  `third_party/llama.cpp/build/bin/` (git-ignored; `live` finds it there
  without flags). Drop the fine-tuned GGUF in `models/` (git-ignored;
  `--fg-gguf` / `FG_GGUF` override). Never commit weights, sources, or logs.
- Never start heavy GPU processes (model servers, training) while a
  fine-tune is running in WSL — check `nvidia-smi` and running processes first.

## Stack
- Product identity is **Relay** (`relay` package, CLI, `RELAY_*` env). **Needle**
  denotes only the external action-model engine (`cactus-needle`, `NeedleResult`,
  `needle_schema`, `needle_weights`/`needle_max_tokens`, `NEEDLE_*` engine vars).
  Keep the distinction in code, docs, and UI copy.
- Python `>=3.11,<3.14`. `pyproject.toml` with loose pins: `langgraph>=1.2,<1.3`,
  `pydantic>=2.13,<3`, `cactus-needle>=2.0,<3`. Dev: `pytest>=8,<9`,
  `pytest-asyncio>=1,<2`, `ruff>=0.12,<1`.
- Do NOT add LangChain (LangGraph is standalone) or extra deps for filesystem/search/datetime/AST/logging — use stdlib.
- Do NOT use Needle's full agent loop (`agent.run()`). Use single-turn `complete()`; framework owns the loop.
- The FunctionGemma adapter (`models/functiongemma.py`) is stdlib HTTP against
  llama-server `/completion` — no torch/transformers dependency in this repo.

## Architecture invariants (do not violate)
- Reasoning LLM never emits JSON or executes tools. It emits `<tool>instruction
  + payload blocks</tool>` / `<final>answer</final>`. Only `<tool>` content is
  executable intent.
- Bulk text travels in payload blocks, never in prose: one `<content>` block
  for single-payload tools, ordered `<text-1>`, `<text-2>`, … for multi-payload
  tools (`replace_text`). Blocks are masked before tag scanning; mixed, gapped,
  duplicated, or stray blocks fail closed. Which args are payloads is declared
  once per tool via `Tool.payload_args`; the runtime attaches them positionally
  after translation and discards the translator's own payload output.
- The action model (Needle or FunctionGemma) only does `nl instruction → tool
  selection + small args`. Runtime owns sanitize → validate → confidence →
  safety → execution.
- `AgentState` is a `TypedDict` (`messages, current_action, current_payloads,
  needle_result, tool_call, last_tool_result, step_count, model_turn,
  max_tool_steps, stall_count, action_records, final_answer, status`). Never put
  model instances, registries, executors, or config in state.
- Context is budgeted twice: `max_context_chars` (always) and
  `max_context_tokens` (when the server tokenizer in `models/tokens.py`
  resolves; llama.cpp/Ollama backends, heuristic fallback). Past ~80%,
  `context/summarize.py` replaces history with a validated structured summary
  (summary + current request); malformed summaries fall back to trim, never install.
- Multi-step work lives in workspace-root `tasks.md`, maintained by the model
  via file tools; `workspace_description` notes its presence when it exists.
- One tool action per reasoning turn. After every successful execution go
  `OBSERVE → UPDATE_CONTEXT → REASON`. Never chain tools without reasoning.
- No `<tool>` and no `<final>` → treat response as final answer (prevents failures on tag-less models).
- Multiple `<tool>` blocks: first action wins, then return to reasoning, not concurrent.

## Pipeline order (fixed)
`REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE → SAFETY → EXECUTE → OBSERVE → UPDATE_CONTEXT → REASON`
- Malformed/invalid calls never reach confidence/execution. High-confidence invalid is still invalid — never execute.
- Gates: `confidence_threshold = 0.85` default, `read_only_threshold = 0.5` for `READ_ONLY_TOOLS`. `< gate` → do not execute; send candidates back to reasoning via `CONFIRM`. `max_tool_steps = 20`, `max_stalls = 3` (consecutive non-executing turns → `STALLED`); check before every execution, terminate with `MAX_STEPS_REACHED`.
- Sanitizer: normalize name/args, reject malformed — do not "repair" untrusted translator output into a dangerous command.
- Single canonical `Tool(name, description, parameters, handler, payload_args)`; reasoning descriptions and translator schemas derive from it. Never maintain both by hand.

## Safety (tiered approvals)
- `require_approval_for` in `[safety]` config (default: delete_file, run_python,
  run_process, run_powershell, git_commit, git_checkout, replace/insert/delete_text,
  apply_patch). Everything else — reads, `write_file`, move/copy/create, `git_stage`,
  `ask_user` — runs without asking. Tiers are config, not code; keep them there.
- All filesystem paths: resolve against optional `workspace_root`, containment-check
  (block `..`, outside absolutes, symlink escape) before any op. `write_file` creates
  parents only if config allows; no append/binary.
- `calculator`: restricted `ast` parser (`+ - * / ** % ()` + numeric literals only). Never `eval()`.
- `search_files`: stdlib `Path.rglob`, skip `.git __pycache__ node_modules .venv venv dist build` + NUL-byte binaries; limits as config constants, not magic numbers. Return matches with line numbers, not whole files.
- Execution tools run argv-only (never shell), cwd-locked to the workspace, with timeouts
  and truncated output. Git checkout only switches to existing local branches.
- Output caps: `MAX_TOOL_OUTPUT_CHARS`; context trim keeps system prompt + tool descs +
  original request + recent messages, drops old observations first. No summarization models.

## Structure
- `src/relay/{agent.py,cli.py,server.py,store.py,config.py,state.py,models/{reasoning,action,needle,functiongemma,demo,streaming,tokens}.py,protocol/{parser,stream,intent}.py,tools/{base,registry,filesystem,editing,execution,git,web,environment,utility,interaction,preview}.py,execution/{sanitizer,validator,confidence,executor}.py,context/{manager,summarize}.py,graph/workflow.py,web/}`, `tests/{unit,integration,e2e}/`, `examples/basic.py`, `config/{defaults.toml,prompts/}`.
- Browser sessions persist write-through to SQLite (`store.py`, default
  `~/.relay/sessions.db`, `RELAY_SESSIONS_DB` override); per-token model
  traces are never stored. Storage failures never break runs.
- Do not split files further without concrete reason. Pydantic for external/model data (`ToolCall, ToolResult, NeedleResult+ToolRanking`); dataclasses for internal runtime objects.

## Build / test
- Focused: `pytest tests/unit` (parser, registry, path safety, sanitizer, validator, confidence, context trim). Full loop without real LLM via mocked `ReasoningModel.generate()` + `ActionModel.translate()` (`tests/integration/test_loop.py`).
- `ruff check src tests`. Browser e2e needs Playwright + browsers (not installed here).
- Heavy local verification (live llama-server runs, GGUF evals, fine-tunes) lives outside the repo in `E:\finetune`; never commit weights, logs, or datasets.
