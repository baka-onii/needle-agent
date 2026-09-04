# Agent Runtime (V0)

Local-first natural-language agent runtime. A small reasoning LLM emits
`<tool>` / `<final>` intents; Needle 2 translates NL actions into validated
tool calls; the runtime owns sanitize → validate → confidence → safety → execution.

Full spec: `docs/spec-v0.md`.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"  # or: pip install -e . ; pip install --group dev
```

Serve the reasoning model (llama.cpp, OpenAI-compatible). Example for ~8 GB VRAM
(Ornith 9B Q4_K_M, 64k context with quantized KV cache, thinking off):

```powershell
E:\llama.cpp\llama-server.exe -m "E:\llm models\ornith-1.0-9b-Q4_K_M.gguf" `
  -ngl all -fa on -c 65536 --top_p 0.9 --top_k 40 --min_p 0.05 --temp 0.3 `
  --repeat-penalty 1 --presence-penalty 0 --frequency-penalty 0 `
  --cont-batching --defrag-thold 0.2 -np 1 --fit-target 256 `
  --cache-type-k q8_0 --cache-type-v turbo4 --reasoning off --port 8080
```

Useful while it runs (any PowerShell window):

```powershell
Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing  # alive?
Stop-Process -Name llama-server -Force  # stop it (not mid-run)
```

Note: if started with stdout redirected to a file, its console window stays
empty by design — check `/health`, not the window.

Run the example:

```powershell
python examples\basic.py
```

## Checks

```powershell
ruff check src tests
pytest tests/unit
pytest
```
