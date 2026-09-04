# V0 Benchmark: Native vs Needle-Lifted Tool Calling

Date: 2026-09-05. Harness: `examples/benchmark.py`. Raw records: 16 runs
(8 tasks × 2 paths), one shot each — treat numbers as indicative, not statistical.

## Setup

- Reasoning model: `ornith-1.0-9b-Q4_K_M.gguf` via llama.cpp server
  (`-ngl all -fa on -c 65536`, q8_0/turbo4 KV, `--reasoning off`,
  temp 0.3 / top_p 0.9 / top_k 40 / min_p 0.05), temp 0.2 per request.
- Action model: Needle 2 (`cactus-needle` 2.0.12, base weights, CPU).
- Same 7 tools, same fixture workspaces both paths.
- Gates (per user decision): read-only tools 0.5, mutating tools 0.85,
  `max_tool_steps` 8, `max_stalls` 3.
- Native path: model emits `tool_calls` via the server `tools` parameter;
  calls run through the **same** validator + executor as lifted, results fed
  back as `tool` messages. Only the structured-output producer differs.

## Results

| task    | native (steps) | lifted (steps) | note |
|---------|----------------|----------------|------|
| read    | ✓ (1) | ✓ (1) | identical answers |
| list    | ✓ (1) | ✓ (2) | lifted needed one CONFIRM retry, then executed |
| search  | ✓ (1) | ✓ (1) | |
| calc    | ✓ (1, tool) | ✓ (0) | lifted answered `36` directly, no tool needed |
| time    | ✓ (1) | ✓ (1) | |
| write   | ✓ (2) | ✗ STALLED (0) | Needle chose `write_file` correctly 3× but never cleared the 0.85 strict gate |
| multi   | ✓ (2) | ✓ (2) | search → read chains on both paths |
| refusal | ✓ (poem, 0 tools) | ✓ (poem, 0 tools) | neither path touched the filesystem |

Score: **native 8/8, lifted 7/8.** Malformed-output count: **0 on both paths.**
Per-run latency: native 1.4–4.0 s, lifted 1.3–4.8 s (one extra LLM turn per
CONFIRM retry is the only systematic cost).

## Findings

1. **The strict gate fails safe, and visibly.** The single lifted failure is
   not a wrong action — Needle selected `write_file` with right args every
   time but scored < 0.85, so the run STALLED instead of executing an
   under-confident mutation. That is the designed behavior; it just also
   marks the current capability ceiling for mutating tools.
2. **CONFIRM-retry self-heals.** `list/lifted` failed the gate once, the
   reasoning model rephrased, the retry cleared it. Low-confidence turns are
   recoverable, not fatal.
3. **No output-discipline gap on this model.** Ornith 9B emits valid native
   `tool_calls` (0 invalid both paths), so lifting buys little *here*. The
   architecture's thesis targets smaller/weaker reasoning models that cannot
   do structured output reliably — rerunning this suite against a 1–3B
   reasoning model is the obvious follow-up.
4. **Read-only 0.5 gate is load-bearing.** With the original flat 0.85,
   `search`/`read_directory` translations (0.49–0.84 when correct) would stall
   instead of execute. The per-tool split is what makes the lifted path work
   at all on this tool surface.

## Limitations

Single run per cell; one capable reasoning model; Needle base weights (no
LoRA fine-tune, deferred by decision); toy fixture workspaces. The `time`
checker (substring `:`) is weak by construction.
