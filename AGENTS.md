# AGENTS.md

## Source of truth
- `docs/spec-v0.md` is the full V0 spec. Trust it over any other prose. Repo is greenfield — only `docs/` exists until scaffolded.

## Stack
- Python `>=3.11,<3.14`. `pyproject.toml` with loose pins: `langgraph>=1.2,<1.3`, `pydantic>=2.13,<3`, `cactus-needle>=2.0,<3`. Dev: `pytest>=8,<9`, `pytest-asyncio>=1,<2`, `ruff>=0.12,<1`.
- Do NOT add LangChain (LangGraph is standalone) or extra deps for filesystem/search/datetime/AST/logging — use stdlib.
- Do NOT use Needle's full agent loop (`agent.run()`). Use single-turn `complete()`; framework owns the loop.

## Architecture invariants (do not violate)
- Reasoning LLM never emits JSON or executes tools. It emits only `<tool>nl action</tool>` / `<final>answer</final>`. Only `<tool>` content is executable intent.
- Needle only does `nl action → tool selection + args`. Runtime owns sanitize → validate → confidence → safety → execution.
- `AgentState` is a `TypedDict` (`messages, current_action, last_tool_result, step_count, max_tool_steps, final_answer, status`). Never put model instances, registries, executors, or config in state.
- One tool action per reasoning turn. After every successful execution go `OBSERVE → UPDATE_CONTEXT → REASON`. Never chain tools without reasoning.
- No `<tool>` and no `<final>` → treat response as final answer (prevents failures on tag-less models).
- Multiple `<tool>` blocks: execute sequentially; preferred is first action then return to reasoning, not concurrent.

## Pipeline order (fixed)
`REASON → PARSE → TRANSLATE(Needle) → SANITIZE → VALIDATE → CONFIDENCE → SAFETY → EXECUTE → OBSERVE → UPDATE_CONTEXT → REASON`
- Malformed/invalid calls never reach confidence/execution. High-confidence invalid is still invalid — never execute.
- Gates: `confidence_threshold = 0.85` default, `read_only_threshold = 0.5` for `READ_ONLY_TOOLS` (read/search/calc/time). `< gate` → do not execute; send candidates back to reasoning via `CONFIRM`. `max_tool_steps = 20`, `max_stalls = 3` (consecutive non-executing turns → `STALLED`); check before every execution, terminate with `MAX_STEPS_REACHED`.
- Sanitizer: normalize name/args, reject malformed — do not "repair" untrusted Needle output into a dangerous command.
- Single canonical `Tool(name, description, parameters, handler)`; generate reasoning-model description and Needle schema from it. Never maintain both by hand.

## Safety / V0 scope
- No `shell` / `terminal` / `execute_command` / `run_python` tool in V0.
- All filesystem paths: resolve against optional `workspace_root`, containment-check (block `..`, outside absolutes, symlink escape) before any op. `write_file` creates parents only if config allows; no append/binary/delete.
- `calculator`: restricted `ast` parser (`+ - * / ** % ()` + numeric literals only). Never `eval()`.
- `search_files`: stdlib `Path.rglob`, skip `.git __pycache__ node_modules .venv venv dist build` + NUL-byte binaries; limits `max_results=50, context_lines=2, max_file_size=2MB` as config constants, not magic numbers. Return matches with line numbers, not whole files.
- Output caps: `MAX_TOOL_OUTPUT_CHARS = 20_000`; context trim keeps system prompt + tool descs + original request + recent messages, drops old observations first. No summarization models in V0.

## Structure (scaffold per spec §3 when starting)
- `src/agent_runtime/{agent.py,config.py,state.py,models/{reasoning,action,needle}.py,protocol/parser.py,tools/{base,registry,filesystem,utility,interaction}.py,execution/{sanitizer,validator,confidence,executor}.py,context/manager.py,graph/workflow.py}`, `tests/{unit,integration,e2e}/`, `examples/basic.py`.
- Do not split files further without concrete reason. Pydantic for external/model data (`ToolCall, ToolResult, NeedleResult+ToolRanking`); dataclasses for internal runtime objects.

## Build / test
- No `pyproject.toml`, CI, or lint config exists yet — create `pyproject.toml` first per spec §2.
- Once scaffolded: `pytest tests/unit` for focused checks (parser, registry, path safety, sanitizer, validator, confidence, context trim); full loop without real LLM via mocked `ReasoningModel.generate()` + `ActionModel.translate()` integration test; `ruff check src tests`.
- Implementation order (§39): models → tools → protocol → execution pipeline → model interfaces → context → LangGraph → mocked integration tests → real models last → benchmark. Do not implement everything in one pass.
