# Models directory (weights are never committed)

Place the fine-tuned FunctionGemma translator GGUF here, e.g.
`models/fg-tools.gguf`. The `live` command picks up the first `*.gguf`
in this directory automatically (`--fg-gguf` / `FG_GGUF` override it).

Produce the weights with the fine-tune pipeline (see `E:\finetune` on the
dev machine) and export to GGUF, or copy an existing export here.
