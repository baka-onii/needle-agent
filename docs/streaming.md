# Model streaming and the chat timeline

## What is visible

Each reasoning-model invocation creates a collapsible **Reasoning** card. Its live output
contains text the model actually returned: `reasoning_content` / `reasoning` when the
provider exposes either field, inline `<think>` text, and commentary outside tool/final
blocks. It does not invent hidden reasoning or make a second request to obtain it.
Models that return no separate reasoning show an explanatory placeholder.

Open a card, use **Expand / Shrink**, or drag its resize handle. **Model conversation** shows
sent messages by role and the raw response, with individual messages folded by default.
The translator's request and normalized adapter result are nested inside its tool card. Registered tool schemas
remain inspectable in the Tools view. An externally supplied native client may have
unknown private system context; only known adapter inputs are displayed in that case.

Native Needle currently exposes blocking `complete()`, not token callbacks. Its result is
labelled **buffered response**. Generate-only reasoning adapters and compatible servers
that answer a streaming request with `application/json` are labelled buffered too. The demo
explicitly marks its short simulated token-delivery sequence as such.

## Display pacing

```toml
[models]
llm_stream = true

[streaming]
stream_buffer_ms = 1000
stream_max_lag_ms = 1800
stream_flush_ms = 50
max_model_output_chars = 128000
capture_model_inputs = true
```

- **`stream_buffer_ms`**: initial browser presentation buffer (0–5000 ms). Zero means immediate
  updates. Reduced-motion preferences also skip the typing animation.
- **`stream_max_lag_ms`**: maximum target lag/catch-up window. Must be at least the initial
  buffer. Delivery-rate smoothing follows real incoming text, accelerating bursts and
  draining completed responses within 250 ms. Grapheme boundaries prevent split emoji.
- **`stream_flush_ms`**: maximum normal backend coalescing interval. A scoped timer flushes
  small pending batches even if the provider pauses; a 1024-character batch flushes early.
- **`max_model_output_chars`**: combined reasoning/content bound per model generation.
  Exceeding it aborts the generation before any partial tool action can execute.
- **`capture_model_inputs`**: include the messages sent to each adapter in new traces. This
  can contain workspace files and prior observations, so treat exported traces as private.
  Provider HTTP headers/API keys are not added to traces.

These fields are available in **Settings → Streaming & model conversation**, configuration
files, and CLI overrides. Examples:

```sh
relay serve --config relay.toml --stream-buffer-ms 500
relay chat --config relay.toml --stream --trace
relay run --no-stream --no-capture-model-inputs 'Explain the project'
```

The buffer is **cosmetic**. Validation, tools, Stop, questions, and write approval never wait
for a typing animation. Pending text is caught up when an action/approval needs to be shown.
Only live content nodes are patched during animation, not the whole conversation. Open
panels, focus, manual sizing, selected text, and readers who scroll away from the bottom
are not reset on each token. Markdown code blocks render before their closing fence arrives.

## Detection is not execution

The display parser recognizes `<tool>` / `<final>` even when a tag spans many packets.
Markdown fences protect literal tags. Inline thoughts and provider reasoning fields are
never executable, including thought text that contains a complete-looking `<tool>`.

Tool requests appear as **drafts** while being generated. A closed tool block still does not
authorize a call: the entire model generation must complete successfully, after which the
normal parser and every gate run. A later complete final block overrides earlier tool drafts.
Multiple tool blocks retain the existing first-action-then-reason rule.

Missing terminal stream markers, malformed SSE/JSON, token-limit/content-filter finishes,
network errors, cancellation, and output-limit failures do not execute accumulated tool
text. Partial output remains inspectable and is labelled incomplete/interrupted rather than
reported as a successful answer. Partial generation is not committed to follow-up history.

## Backend/API contract

The OpenAI-compatible client requests `/v1/chat/completions` with `stream: true` and usage
reporting. It handles UTF-8 fragments, CRLF, SSE comments, multiline `data:`, usage-only
packets, and `[DONE]` / finish markers. It does not follow redirects or forward credentials
to another origin. There is no automatic second request on a provider error; use
`llm_stream = false` if the endpoint does not support the streaming request/options.

Connected HTTP reads (including waiting for response headers) can be interrupted on Stop.
The configured `llm_timeout_s` also acts as an overall streaming deadline, so endless
heartbeats do not keep a run alive forever. Initial DNS/connect/TLS work and non-streaming
native inference may still need to finish before cooperative cancellation returns.

The existing `generate(messages) -> str` adapter interface remains valid. A streaming
adapter may implement `stream(messages, *, cancelled=None)`, yielding `ModelDelta` or
plain string content deltas. A custom adapter owns detecting failures in its own transport;
normal iterator exhaustion means successful completion. Raise an error for incomplete
output, or yield a non-success `finish_reason` (such as `length`), which the runtime rejects.

```python
from relay import ModelDelta
from relay.models.streaming import check_cancelled

class MyStreamingAdapter:
    def stream(self, messages, *, cancelled=None):
        # Replace these illustrative chunks with real data from your provider.
        for text in ("<final>", "A streamed response.", "</final>"):
            check_cancelled(cancelled)
            yield ModelDelta(text=text)
        yield ModelDelta(kind="metadata", finish_reason="stop")
```

`ModelDelta(kind="reasoning", text=...)` carries provider-visible reasoning separately;
`kind="content"` is assistant response text. Metadata can include `streamed`, `usage`, and
`finish_reason`. Only documented nonnegative integer usage counters are retained. Raw
chunk sizes are not labelled as exact token counts; usage totals are shown when provided.

New events in `Agent.stream()` and the web SSE endpoint:

| Event | Purpose |
| --- | --- |
| `model_start` | Stable model/turn ID, component, optional sent messages, display settings |
| `model_status` | Actual delivery mode, including JSON-buffered fallback |
| `model_delta` | Raw returned text plus append-only display section updates |
| `model_end` | Completion/error/cancellation, duration, usage, and final parse decision |
| `model_trace_limited` | Web-only notice when retained optional details exceed their bound |

Display section updates have stable `index`, `kind`, `text`, and optional `complete` fields.
Their kinds are `reasoning`, `text`, `tool`, and `final`; provider reasoning uses the stable
`provider` index. **Section completion is not a tool execution event.** Consumers should
use `action`, `validated`, `tool_start`, `tool_result`, and terminal `complete` for that.
The existing runtime phase/gate/result events remain intact. `--trace` prints model traffic
as it arrives; normal CLI output and `--json` retain their existing result contracts.

## Replay and bounds

The browser receives same-origin, replayable SSE. Reload restores the received prefix
immediately and resumes after the last retained event ID—no duplicate text or slow replay
of old tokens. Server memory remains session-scoped. Deleting a chat deletes its traces,
not workspace files.

Each model generation is bounded by `max_model_output_chars`. SSE transport is bounded to
16 MB, with a 1 MB line limit; non-streaming JSON remains capped at 2 MB. A browser run
retains at most 4 MB / 16,000 events of additional model details. On reaching this limit,
subsequent token/input details are omitted with a visible notice while tool results and
the terminal answer remain available. It is not an execution failure.

Tests exercise real HTTP/SSE with scripted provider packets, live LangGraph delivery,
interruption, early tool previews, late-final precedence, cross-session isolation, reload,
DOM stability, Unicode, and deterministic pacing. This verifies the transport/runtime/UI
contract, not live-model reasoning quality or unavailable model inference.
