# Configuration and prompt reference

## Create and load settings

```sh
uv run needle-agent config init needle.toml
uv run needle-agent serve --config needle.toml --demo
uv run needle-agent chat --config needle.toml --live
uv run needle-agent run --config needle.toml --demo 'Calculate 6*7'
```

`config init` copies the canonical `config/defaults.toml` and the three Markdown prompt
files into your chosen directory. It refuses to overwrite existing config/prompt files.
The same assets are bundled in wheels and source distributions; an installed command
works without a repository checkout.

Loading is explicit (`--config FILE` or `NEEDLE_CONFIG`). The runtime never auto-loads a
config merely because it is present in an untrusted workspace. TOML and JSON are accepted
with the same named sections. Unknown sections/keys, invalid types, non-finite values,
invalid timezones, unreadable prompt files, and overlarge files fail before inference.

Priority is **defaults < config file < environment < CLI**. File paths for the workspace,
weights, and prompt files resolve relative to the config's directory; CLI/environment
paths resolve relative to the working directory. Paths are not guessed or rewritten.
A workspace must already exist. Relative tool paths always resolve *inside that workspace*,
not the config directory or a fabricated home directory.

## File format

Partial files are supported; unspecified fields keep their defaults:

```toml
[workspace]
workspace_root = "./project"
read_only = false
allow_create_parent_dirs = false
include_workspace_listing = true

[models]
mode = "live" # or "demo" (explicit simulated models, real tools)
action_model = "needle" # or "functiongemma" (llama-server GGUF translator)
llm_base_url = "http://127.0.0.1:11434/v1"
llm_model = "qwen2.5:3b"
llm_max_tokens = 4096
needle_max_tokens = 256
fg_base_url = "http://127.0.0.1:8081"
fg_max_tokens = 256
llm_temperature = 0.2
llm_timeout_s = 120.0

[runtime]
confidence_threshold = 0.85
read_only_threshold = 0.5
max_tool_steps = 20
max_stalls = 3
max_repeated_failures = 2
max_context_chars = 32000
max_tool_output_chars = 20000

[tools]
max_search_results = 50
max_matches_per_file = 5
search_context_lines = 2
max_search_file_bytes = 2000000
max_search_entries = 20000
max_write_chars = 100000
default_timezone = "Asia/Kolkata" # "" uses the system timezone

[safety]
require_approval_for = ["delete_file", "run_python", "run_process"] # + run_powershell,
  # git_commit, git_checkout, replace_text, insert_text, delete_text, apply_patch.
  # Tools outside this list (reads, write_file, move/copy/create, stage) run freely.

[prompts]
reasoning_prompt_file = "prompts/reasoning.md"
translator_prompt_file = "prompts/translator.md"
confirmation_prompt_file = "prompts/confirmation.md"
```

The `[workspace]` section also accepts `max_directory_entries` (default 200) and
`workspace_listing_chars` (default 3000). The root snapshot is bounded, skips generated
folders and symlinks, and reads names only, not file contents. It is refreshed for each
run. Disable it to have the reasoning model explicitly use `read_directory` instead.

The `[models]` section also accepts `llm_stream` (default true). Disable it for an endpoint
that cannot return SSE. The `[streaming]` settings are described in the
[streaming reference](streaming.md) and can be edited in the UI or with CLI overrides.

The `[models]` section optionally accepts `needle_weights`, resolved relative to the file.
Custom Needle weights currently have no calibrated confidence; missing scores fail closed.
Keep credentials in `NEEDLE_LLM_API_KEY`. API keys are not file settings, browser settings,
or config exports. URLs containing credentials, queries, or fragments are rejected.

See `config/defaults.toml` for the canonical complete list. Basic validity and safety
checks always apply. The browser additionally caps resource values (for example, 100
steps, 262,144 context characters, and 32,768 output tokens per model). `--help` lists
common flags; use `--set` for the remaining `AgentConfig` fields.

## Prompt files and inline instructions

- **Reasoning:** role, planning, one-tool/one-target actions, actual content drafting,
  paths, and how to interpret observations.
- **Translator:** selecting the explicitly named tool and extracting exact arguments.
  It must not compose missing content or change the supplied file payload.
- **Confirmation:** how the reasoning model reviews the highest-ranked available tool,
  proposed arguments, and errors, and selects a corrected atomic action.

The runtime appends actual workspace metadata and canonical tool descriptions; there are
no template placeholders to maintain. Prompts are text, not executable code. Each prompt
is limited to 20,000 characters and must be nonempty. The assembled reasoning system
prompt and tool descriptions must fit the context budget. Customization cannot remove
schema validation, confidence gates, path containment, repeat-action guards, or approval.
The `<tool>` / `<final>` protocol and literal-write checks remain runtime contracts.

Instead of a file reference, each prompt can be supplied inline:

```toml
[prompts]
confirmation_prompt = """
Is the highest-ranked tool correct? Check the arguments too.
Reissue one explicit <tool>Use TOOL_NAME to ...</tool> action, or finish with <final>.
Do not use ask_user to choose a tool or request write permission.
"""
```

Use `reasoning_prompt` **or** `reasoning_prompt_file`, not both; the same applies to the
other prompts. Shortening prompts may reduce model reliability; the supplied versions
include writing/path/approval examples. The offline demo does not interpret custom prose
as a live language model would. Use live mode to exercise model instruction changes.

## CLI overrides

All run/chat/serve commands and `config show` support:

```sh
needle-agent chat --config needle.toml \
  --reasoning-prompt my-reasoning.md --translator-prompt my-translator.md \
  --confirmation-prompt my-review.md \
  --confidence-threshold 0.85 --read-only-threshold 0.5 \
  --max-stalls 4 --max-context-chars 48000 \
  --llm-max-tokens 4096 --needle-max-tokens 4096 \
  --temperature 0.1 --timeout 180 --timezone Asia/Kolkata \
  --set max_search_results=25 --set max_repeated_failures=2
```

`--set NAME=VALUE` accepts JSON values (numbers, booleans, null, quoted strings) or a plain
string. Dedicated flags take priority over `--set` for the same field. Boolean options
include `--read-only` / `--no-read-only`, `--allow-create-parents` /
`--no-allow-create-parents`, and `--workspace-listing` / `--no-workspace-listing`.
`--demo` / `--live` override the file's selected mode. `--stream` / `--no-stream` controls
provider streaming; `--capture-model-inputs` / `--no-capture-model-inputs` controls input
trace retention. `--stream-buffer-ms 0` disables the browser's cosmetic delay. `--trace`
prints arriving model text as well as tool/gate events to stderr; JSON results stay on stdout.

`config show` prints the effective configuration with inline prompt text and without
credentials or machine-specific Needle weights:

```sh
needle-agent config show --config needle.toml > portable.toml
```

The CLI export can include the resolved workspace path; override `--workspace` when
moving to a different machine. Programmatic users can call
`agent_runtime.config.load_config(path, overrides=...)` and pass the result to `Agent`.

## Browser settings and persistence

**Settings** exposes prompts, mode/model URL/name, generation limits, timeouts, confidence
thresholds, retry limits, context/output budgets, directory/search limits, timezone, and
workspace write policy. Saving applies to the next run; stop the current run before
changing settings. **Restore server instructions** restores only the three prompt fields
in the form; click Save to apply. It restores the operator's startup values, including
any prompt files loaded by the server, not unrelated browser defaults.

**Export saved settings** downloads portable TOML with inline prompts. It exports the
last saved values, not unsaved edits in the form. **Import config** validates and applies
a TOML/JSON file atomically. Unspecified values in an imported file retain the current
session's settings. Exporting embeds all prompts and the default timezone, so complete
exports can reproduce those values in another session.

Browser imports cannot read server files through `*_prompt_file`. Convert a file-based
config with `config show` first, or embed the prompt text. The server's workspace and
Needle weights are operator-owned: imported values for them are ignored and reported.
The server's read-only floor and API-key origin binding also apply to imported settings.

Settings and chat histories are session-local and held in server memory. They expire
with the session and do not survive a restart. To persist changes, export settings and
restart with `--config` pointing to that file. Files are loaded at startup, not watched.
The light/dark preference is separate and persists in the browser's local storage.

## Selection reviews and repeated actions

Low-confidence valid calls enter `CONFIRM` before anything executes. The review contains:

1. The requested action and reason for review.
2. The selected tool and proposed arguments.
3. Actual available candidates sorted by score; no made-up alternatives.
4. An explicit question asking whether the highest-ranked tool is correct.

The reasoning model chooses with another natural-language action naming its intended
tool. That action is retranslated and must clear all normal gates. A bare "yes" is not
an executable action, and confirmation is not a confidence override. Tool execution
failures get the same grounded review with their error. `--trace` and expanded browser
tool cards expose the `confirmation` events.

Successful severe actions and denied targets are tracked independently of context
trimming. An exact successful mutation is not executed again in that run; an answered
question is not asked again; a denied target cannot be reapproved by rephrasing it.
Common permission questions are rejected because approval is a runtime operation, not
an `ask_user` task. Repeated identical tool failures are bounded by
`max_repeated_failures`; consecutive blocked attempts are bounded by `max_stalls`.
These records reset on a new user turn, allowing a genuinely new request.

Chat deletion removes a finished conversation and its runs, including pending-free
traces. It is session-scoped, requires UI confirmation, and never removes workspace
files. Stop and await any active run before deleting its conversation.
