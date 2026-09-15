"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icon = (name, tiny = false) =>
  `<svg class="icon${tiny ? " tiny" : ""}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const { PacedText, reconcile } = RelayStreaming;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const toolIcons = {
  read_file: "file",
  read_directory: "folder",
  search_files: "search",
  write_file: "tools",
  file_info: "info",
  create_directory: "folder",
  move_file: "file",
  copy_file: "copy",
  delete_file: "trash",
  replace_text: "tools",
  insert_text: "tools",
  delete_text: "tools",
  apply_patch: "tools",
  run_python: "code",
  run_process: "code",
  run_powershell: "code",
  git_status: "history",
  git_diff: "history",
  git_log: "history",
  git_show: "history",
  git_branch_list: "history",
  git_stage: "history",
  git_commit: "history",
  git_checkout: "history",
  web_search: "search",
  web_open: "external",
  web_extract: "file",
  get_working_directory: "folder",
  find_executable: "search",
  process_info: "chip",
  calculator: "calculator",
  get_time: "clock",
  ask_user: "chat",
};
const toolNames = {
  read_file: "Read file",
  read_directory: "Read directory",
  search_files: "Search files",
  write_file: "Write file",
  file_info: "File info",
  create_directory: "Create directory",
  move_file: "Move file",
  copy_file: "Copy file",
  delete_file: "Delete file",
  replace_text: "Replace text",
  insert_text: "Insert text",
  delete_text: "Delete text",
  apply_patch: "Apply patch",
  run_python: "Python run",
  run_process: "Command run",
  run_powershell: "PowerShell run",
  git_status: "Git status",
  git_diff: "Git diff",
  git_log: "Git log",
  git_show: "Git show",
  git_branch_list: "Git branches",
  git_stage: "Git stage",
  git_commit: "Git commit",
  git_checkout: "Git checkout",
  web_search: "Web search",
  web_open: "Web open",
  web_extract: "Web extract",
  get_working_directory: "Working directory",
  find_executable: "Find executable",
  process_info: "Process info",
  calculator: "Calculator",
  get_time: "Current time",
  ask_user: "Ask user",
};
const toolLabel = (name) =>
  (toolNames[name] || name || "action").toLowerCase();
const PERMISSION_PRESETS = {
  commands: { label: "Run commands", tools: ["run_process", "run_powershell"] },
  python: { label: "Run Python", tools: ["run_python"] },
  git: { label: "Git commands", tools: ["git_commit", "git_checkout"] },
};
function autoApproved() {
  return new Set(S.settings?.auto_approve || []);
}
function renderPermissions() {
  const approved = autoApproved();
  const busy = Boolean(activeRun());
  $$("#permission-popover input[data-preset]").forEach((input) => {
    const tools = PERMISSION_PRESETS[input.dataset.preset].tools;
    const on = tools.filter((tool) => approved.has(tool)).length;
    input.checked = on === tools.length;
    input.indeterminate = on > 0 && on < tools.length;
    input.disabled = busy;
  });
  $("#permission-dot").classList.toggle("hidden", approved.size === 0);
  $("#permission-chip").title = approved.size
    ? `Auto-approved this session: ${[...approved].join(", ")}`
    : "No session auto-approvals. Severe actions ask first.";
}
async function saveAutoApprove(preset, enabled) {
  const { label, tools } = PERMISSION_PRESETS[preset];
  const approved = autoApproved();
  for (const tool of tools)
    if (enabled) approved.add(tool);
    else approved.delete(tool);
  try {
    const result = await api("/api/settings", {
      method: "POST",
      body: { ...S.settings, auto_approve: [...approved] },
    });
    S.settings = result.settings;
    renderSettings();
    toast(
      enabled
        ? `${label} auto-approved for this session.`
        : `${label} will ask for approval again.`,
    );
  } catch (error) {
    toast(error.message, true);
  }
  renderPermissions();
}
const phases = [
  "reason",
  "parse",
  "translate",
  "sanitize",
  "validate",
  "confidence",
  "confirm",
  "safety",
  "execute",
  "observe",
  "update_context",
];
const phaseNames = {
  reason: "Reasoning",
  parse: "Parse intent",
  translate: "Translate action",
  sanitize: "Sanitize call",
  validate: "Validate arguments",
  confidence: "Confidence gate",
  confirm: "Request selection review",
  safety: "Safety & permissions",
  execute: "Execute tool",
  observe: "Observe result",
  update_context: "Update context",
};
const statusNames = {
  RUNNING: "Running",
  WAITING_FOR_INPUT: "Waiting for you",
  CANCELLING: "Stopping",
  COMPLETED: "Completed",
  CANCELLED: "Stopped",
  ERROR: "Error",
  STALLED: "Stalled",
  MAX_STEPS_REACHED: "Step limit",
};
const pageInfo = {
  playground: ["Playground", "A little reasoning. A world of possibilities."],
  workspace: ["Files", "Your project, inside a clear boundary."],
  tools: ["Tools", "Seven capabilities. One carefully controlled runtime."],
  history: [
    "Run history",
    "A record of what happened, not just what was said.",
  ],
};
const stored = (key, value) => {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {}
  return null;
};
const S = {
  token: stored("relay-session"),
  settings: null,
  workspace: null,
  tools: [],
  payloadArgs: {},
  edits: {},
  conversations: [],
  runs: new Map(),
  conversation: null,
  currentRun: null,
  view: "playground",
  inspector: "setup",
  stream: null,
  streamRun: null,
  streamError: false,
  starting: false,
  ready: false,
  filePath: ".",
  entries: [],
  file: null,
  fileLoading: false,
  fileFilter: "",
  fileLoadSequence: 0,
  conversationLoadSequence: 0,
  deletedConversations: new Set(),
  deleteTarget: null,
  animating: new Set(),
  streamDirty: new Set(),
};

const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
function applyTheme(theme, persist = true) {
  document.documentElement.dataset.theme = theme;
  if (persist) stored("relay-theme", theme);
  const next = theme === "dark" ? "light" : "dark";
  const button = $("#theme-toggle");
  button.innerHTML = icon(next === "light" ? "sun" : "moon");
  button.setAttribute("aria-label", `Switch to ${next} mode`);
  button.title = `Switch to ${next} mode`;
  $('meta[name="theme-color"]').content =
    theme === "dark" ? "#17231d" : "#f8f9f6";
}
systemTheme.addEventListener("change", (event) => {
  if (!["light", "dark"].includes(stored("relay-theme")))
    applyTheme(event.matches ? "dark" : "light", false);
});
window.addEventListener("storage", (event) => {
  if (event.key === "relay-theme")
    applyTheme(
      ["light", "dark"].includes(event.newValue)
        ? event.newValue
        : systemTheme.matches
          ? "dark"
          : "light",
      false,
    );
});

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (S.token) headers["X-Relay-Session"] = S.token;
  const response = await fetch(path, {
    ...options,
    headers: { ...headers, ...options.headers },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The server returned an unexpected response (${response.status}).`,
    );
  }
  if (!response.ok)
    throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function toast(message, error = false) {
  const item = document.createElement("div");
  item.className = "toast" + (error ? " error" : "");
  item.innerHTML =
    icon(error ? "info" : "check") + `<span>${esc(message)}</span>`;
  $("#toasts").append(item);
  setTimeout(() => item.remove(), error ? 6500 : 3500);
}
function duration(ms) {
  return ms < 1000 ? `${ms || 0} ms` : `${(ms / 1000).toFixed(1)} s`;
}
function statusBadge(status) {
  const style = ["ERROR", "STALLED", "MAX_STEPS_REACHED"].includes(status)
    ? "error"
    : status === "WAITING_FOR_INPUT"
      ? "waiting"
      : ["RUNNING", "CANCELLING"].includes(status)
        ? "running"
        : "";
  return `<span class="status-badge ${style}">${esc(statusNames[status] || status)}</span>`;
}
function activeRun() {
  return [...S.runs.values()].find((run) => !run.done);
}
function currentRun() {
  return S.runs.get(S.currentRun);
}
function inlineMarkdown(text) {
  return esc(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
}
function markdown(text, streaming = false) {
  const blocks = [];
  let prose = [],
    fence = null,
    code = [],
    language = "text";
  const flushProse = () => {
    if (prose.length) blocks.push({ prose: prose.join("\n") });
    prose = [];
  };
  for (const line of String(text).split("\n")) {
    const opening = line.match(/^[ \t]{0,3}(`{3,}|~{3,})(.*)$/);
    if (!fence && opening) {
      flushProse();
      fence = opening[1];
      language = opening[2].trim() || "text";
      code = [];
    } else if (
      fence &&
      line.trim().length >= fence.length &&
      [...line.trim()].every((char) => char === fence[0])
    ) {
      blocks.push({ code: code.join("\n"), language, closed: true });
      fence = null;
    } else if (fence) code.push(line);
    else prose.push(line);
  }
  if (fence) blocks.push({ code: code.join("\n"), language, closed: false });
  flushProse();
  return blocks
    .map((block, index) => {
      if (block.code !== undefined)
        return `<div class="code-block${streaming && !block.closed ? " code-streaming" : ""}" data-render-key="code-${index}"><div class="code-block-header"><span>${esc(block.language)}</span><button class="icon-button copy-code" aria-label="Copy code">${icon("copy")}</button></div><pre><code>${esc(block.code)}</code></pre></div>`;
      let html = "",
        paragraph = [],
        list = [],
        ordered = false;
      const flushParagraph = () => {
        if (paragraph.length)
          html += `<p>${inlineMarkdown(paragraph.join("\n"))}</p>`;
        paragraph = [];
      };
      const flushList = () => {
        if (list.length) {
          const tag = ordered ? "ol" : "ul";
          html += `<${tag}>${list.map((line) => `<li>${inlineMarkdown(line)}</li>`).join("")}</${tag}>`;
        }
        list = [];
      };
      for (const line of block.prose.split("\n")) {
        const heading = line.match(/^(#{1,4})\s+(.*)/);
        const item = line.match(/^\s*(?:([-*])|\d+[.)])\s+(.*)/);
        if (heading) {
          flushParagraph();
          flushList();
          const tag = heading[1].length < 2 ? "h2" : "h3";
          html += `<${tag}>${inlineMarkdown(heading[2])}</${tag}>`;
        } else if (item) {
          flushParagraph();
          if (list.length && ordered !== !item[1]) flushList();
          ordered = !item[1];
          list.push(item[2]);
        } else if (!line.trim()) {
          flushParagraph();
          flushList();
        } else {
          flushList();
          paragraph.push(line);
        }
      }
      flushParagraph();
      flushList();
      return html;
    })
    .join("");
}
async function copyText(text, button) {
  try {
    if (navigator.clipboard && window.isSecureContext)
      await navigator.clipboard.writeText(text);
    else {
      const area = document.createElement("textarea");
      area.className = "sr-only";
      area.value = text;
      document.body.append(area);
      area.select();
      if (!document.execCommand("copy"))
        throw new Error("Clipboard unavailable");
      area.remove();
    }
    if (button) {
      const old = button.innerHTML;
      button.innerHTML = icon("check", true);
      setTimeout(() => {
        if (button.isConnected) button.innerHTML = old;
      }, 1600);
    } else toast("Copied to clipboard");
  } catch {
    toast(
      "Could not access the clipboard. Select the text and copy it manually.",
      true,
    );
  }
}
function setRuntimeStatus() {
  const run = activeRun(),
    node = $("#runtime-status");
  node.classList.toggle(
    "offline",
    !S.ready || run?.status === "WAITING_FOR_INPUT",
  );
  node.innerHTML = `<i class="status-dot"></i>${!S.ready ? "Disconnected" : run ? esc(statusNames[run.status] || "Working") : "Runtime ready"}`;
}
function renderSettings() {
  if (!S.settings) return;
  const demo = S.settings.mode === "demo";
  $("#mode-badge").classList.toggle("live", !demo);
  $("#mode-badge span:nth-child(2)").textContent = demo
    ? "Offline demo"
    : "Live models";
  $("#model-chip span").textContent = demo ? "Offline demo" : S.settings.model;
  $("#model-chip").title = demo
    ? "Demo adapters, not live models"
    : S.settings.model;
  $("#demo-notice").classList.toggle("hidden", !demo);
  $("#reasoning-name").textContent = demo ? "Demo planner" : S.settings.model;
  $("#reasoning-name").title = demo
    ? "Deterministic demonstration planner"
    : S.settings.model;
  $("#reasoning-detail").textContent = demo
    ? "Deterministic · offline"
    : "OpenAI-compatible server";
  $("#reasoning-status").textContent = demo ? "DEMO" : "LIVE";
  $("#action-name").textContent = demo ? "Demo translator" : "Needle 2";
  $("#action-detail").textContent = demo
    ? "Simulated confidence"
    : "Single-turn · on-device";
  $("#read-gate").textContent =
    `≥ ${S.settings.read_only_threshold.toFixed(2)}`;
  $("#write-gate").textContent =
    `≥ ${S.settings.confidence_threshold.toFixed(2)}`;
  $("#step-limit").textContent = `${S.settings.max_tool_steps} steps`;
  $("#write-policy").textContent = S.settings.read_only
    ? "Writes disabled"
    : (S.settings.auto_approve || []).length
      ? `Auto: ${(S.settings.auto_approve || []).slice(0, 3).join(", ")}${(S.settings.auto_approve || []).length > 3 ? ` +${(S.settings.auto_approve || []).length - 3}` : ""}`
      : "Required";
  $("#workspace-name").textContent = S.workspace?.name || "workspace";
  $("#workspace-chip").title = S.workspace?.path || "";
  renderPermissions();
  $("#tool-list").innerHTML = S.tools
    .map(
      (tool) =>
        `<button class="tool-list-item" data-tool="${esc(tool.name)}">${icon(toolIcons[tool.name] || "code")}<span>${esc(tool.name)}</span><i class="tool-status-dot"></i></button>`,
    )
    .join("");
  if (S.view === "tools") renderTools();
}
function renderSidebar() {
  const conversations = [...S.conversations]
    .filter((c) => c.message_count > 0)
    .sort((a, b) => b.updated_at - a.updated_at);
  $("#conversation-count").textContent = conversations.length;
  $("#recent-list").innerHTML = conversations.length
    ? conversations
        .map(
          (c) =>
            `<div class="recent-row"><button class="recent-item${c.id === S.conversation?.id ? " selected" : ""}" data-conversation="${esc(c.id)}" title="${esc(c.title)}">${icon("chat")}<span>${esc(c.title)}</span></button><button class="recent-delete" data-delete-conversation="${esc(c.id)}" aria-label="Delete conversation: ${esc(c.title)}" ${[...S.runs.values()].some((run) => run.conversation_id === c.id && !run.done) ? 'disabled title="Stop the run before deleting"' : 'title="Delete conversation"'}>${icon("trash", true)}</button></div>`,
        )
        .join("")
    : '<div class="recent-empty">A fresh start.<br>Your conversations will appear here.</div>';
}
function formatChars(value) {
  return value >= 1000
    ? `${(value / 1000).toFixed(1)}k`
    : `${value}`;
}
function updateContextMeter() {
  const node = $("#context-meter");
  if (!node) return;
  // Real tokens once a run reports them, else the char budget.
  const tokens = currentRun()?.context_tokens ?? null;
  const total =
    tokens !== null
      ? S.settings?.max_context_tokens || 0
      : S.settings?.max_context_chars || 0;
  const used = tokens !== null ? tokens : currentRun()?.context_chars || 0;
  node.textContent = total
    ? `${formatChars(used)} / ${formatChars(total)} ${tokens !== null ? "tokens" : "chars"}`
    : "";
  node.classList.toggle("warn", total > 0 && used / total >= 0.8 && used / total < 0.95);
  node.classList.toggle("critical", total > 0 && used / total >= 0.95);
}
function updateComposer() {
  const run = currentRun(),
    busy = activeRun();
  const answering =
    run &&
    run.status === "WAITING_FOR_INPUT" &&
    run.pending?.kind === "question";
  const hasText = Boolean($("#message-input").value.trim());
  $("#message-input").disabled = !S.ready;
  $("#message-input").placeholder = answering
    ? "Your answer… the agent is waiting for you."
    : "Give Relay something to work on…";
  $("#composer").classList.toggle("waiting", Boolean(answering));
  $("#send-message").disabled =
    !S.ready || !hasText || S.starting || (Boolean(busy) && !answering);
  $("#send-message").classList.toggle("hidden", Boolean(busy) && !answering);
  $("#stop-run").classList.toggle("hidden", !busy);
  $("#stop-run").disabled = busy?.status === "CANCELLING";
  $("#stop-run").innerHTML =
    icon("stop", true) + (busy?.status === "CANCELLING" ? "Stopping…" : "Stop");
  $("#send-hint").classList.toggle("hidden", Boolean(busy) || S.starting);
  $$("[data-prompt]").forEach((button) => {
    button.disabled = !S.ready || S.starting || Boolean(busy);
  });
  updateContextMeter();
  setRuntimeStatus();
}
function selectInspector(tab) {
  S.inspector = tab;
  for (const name of ["setup", "activity"]) {
    $(`#${name}-tab`).classList.toggle("active", name === tab);
    $(`#${name}-tab`).setAttribute("aria-selected", String(name === tab));
    $(`#${name}-panel`).classList.toggle("hidden", name !== tab);
  }
  if (tab === "activity") renderActivity();
}
function modelsFor(run) {
  if (!run.models) {
    run.models = new Map();
    for (const event of run.events || []) reduceModelEvent(run, event, true);
  }
  return run.models;
}
function reduceModelEvent(run, event, replay = false) {
  run.models ||= new Map();
  const now = performance.now();
  if (event.context_chars !== undefined && event.context_chars !== null)
    run.context_chars = event.context_chars;
  if (event.context_tokens !== undefined && event.context_tokens !== null)
    run.context_tokens = event.context_tokens;
  if (event.type === "model_start") {
    run.models.set(event.model_id, {
      ...event,
      startId: event.id,
      status: "receiving",
      parts: new Map(),
      raw: "",
      providerReasoning: "",
      expanded: false,
      inspect: false,
      done: false,
    });
    return;
  }
  if (event.type === "model_trace_limited") {
    run.trace_limited = true;
    for (const model of run.models.values()) {
      model.trace_limited = true;
      for (const part of model.parts.values()) part.pacer.finish(now, true);
    }
    return;
  }
  const model = run.models.get(event.model_id);
  if (!model) return;
  model.dirty = true;
  if (event.type === "model_status") {
    model.streamed = event.streamed;
    if (!model.streamed)
      for (const part of model.parts.values()) {
        part.pacer.smooth = false;
        part.pacer.flush();
      }
  }
  if (event.type === "model_delta") {
    if (event.channel === "reasoning") model.providerReasoning += event.delta;
    else model.raw += event.delta;
    for (const update of event.parts || []) {
      const index = String(update.index);
      let part = model.parts.get(index);
      if (!part) {
        part = {
          index,
          kind: update.kind,
          complete: false,
          pacer: new PacedText({
            bufferMs: model.buffer_ms,
            maxLagMs: model.max_lag_ms,
            smooth: model.streamed && !reducedMotion.matches,
          }),
        };
        model.parts.set(index, part);
      }
      part.pacer.append(update.text, now, event.elapsed_ms);
      if (Object.hasOwn(update, "complete")) {
        part.complete = update.complete;
        part.pacer.finish(now, replay);
      }
      if (replay) {
        part.pacer.flush();
        part.pacer.firstAt = now - part.pacer.bufferMs;
      }
    }
  }
  if (event.type === "model_end") {
    Object.assign(model, event, { done: true });
    for (const part of model.parts.values())
      part.pacer.finish(
        now,
        replay || !model.streamed || model.status !== "completed",
      );
  }
  if (!replay) {
    S.animating.add(run.id);
    S.streamDirty.add(run.id);
  }
}
function flushModel(run, id) {
  const model = modelsFor(run).get(id);
  if (model) {
    for (const part of model.parts.values()) part.pacer.flush();
    model.dirty = true;
  }
}
function actionGroups(run) {
  if (run.groupCache && run.groupVersion === (run.toolVersion || 0))
    return run.groupCache;
  const groups = [];
  let group;
  for (const event of run.events || []) {
    if (event.type === "action") {
      group = { id: event.id, action: event.action, modelId: event.model_id };
      groups.push(group);
    }
    if (!group) continue;
    if (event.type === "translation") group.translation = event;
    if (event.type === "validated") group.validated = true;
    if (event.type === "confidence") group.confidence = event;
    if (event.type === "tool_start") group.call = event;
    if (event.type === "tool_result") group.result = event;
    if (event.type === "rejected") group.rejected = event;
    if (event.type === "confirmation") group.review = event;
    if (event.type === "user_answer" && event.kind === "approval")
      group.decision = event;
  }
  run.groupCache = groups;
  run.groupVersion = run.toolVersion || 0;
  return groups;
}
function modelConversation(model) {
  if (!model.inspect) return "";
  const messages = model.input_messages || [];
  return `<div class="model-conversation" data-render-key="${esc(model.model_id)}-conversation"><p class="model-note">Exact text sent through this adapter. Request headers and provider credentials are not included.</p>${model.inputs_captured && !model.trace_limited ? messages.map((message, index) => `<details class="model-message" data-render-key="${esc(model.model_id)}-input-${index}"><summary><span>${esc(message.role)}</span><small>${message.content.length.toLocaleString()} characters</small></summary><pre>${esc(message.content)}</pre></details>`).join("") : '<p class="model-note">Input capture is disabled or the trace limit was reached.</p>'}<details class="model-message" data-render-key="${esc(model.model_id)}-raw"><summary><span>${model.component === "translator" ? "Translation result" : "Raw response"}</span><small>${model.raw.length.toLocaleString()} characters</small></summary><pre class="model-raw">${esc(model.raw || "No response text received yet.")}</pre></details>${model.providerReasoning ? `<details class="model-message" data-render-key="${esc(model.model_id)}-provider-reasoning"><summary><span>Provider reasoning text</span></summary><pre class="model-provider-reasoning">${esc(model.providerReasoning)}</pre></details>` : ""}</div>`;
}
function modelOutputHTML(model) {
  const parts = [...model.parts.values()];
  const structured = parts.some((part) =>
    ["tool", "final"].includes(part.kind),
  );
  const text = parts
    .filter(
      (part) =>
        part.kind === "reasoning" ||
        (part.kind === "text" &&
          (model.component === "translator" || structured)),
    )
    .map((part) => part.pacer.text)
    .join("\n\n");
  if (text)
    return model.component === "translator"
      ? `<pre class="model-output">${esc(text)}</pre>`
      : `<div class="markdown model-output">${markdown(text, !model.done)}</div>`;
  return `<p class="model-note">${!model.done ? (model.component === "translator" ? "This translator returns its result in one piece. Waiting for completion…" : "Waiting for model commentary or provider-exposed reasoning…") : "No separate reasoning text was returned. Inspect the conversation or response below."}</p>`;
}
function modelStats(model) {
  const tokens = model.usage?.completion_tokens;
  const outputSize = (
    model.output_chars ?? model.raw.length + model.providerReasoning.length
  ).toLocaleString();
  return `${tokens !== undefined ? `${tokens.toLocaleString()} tokens` : `${outputSize} characters`}${model.duration_ms !== undefined ? ` · ${duration(model.duration_ms)}` : ""}`;
}
function renderModelCard(model, run) {
  const translator = model.component === "translator";
  const busy = !model.done && !model.trace_limited;
  const status = model.trace_limited
    ? "Trace limited"
    : busy
      ? model.streamed
        ? "Streaming"
        : "Waiting"
      : model.status === "completed"
        ? "Complete"
        : model.status === "cancelled"
          ? "Stopped"
          : "Interrupted";
  return `<details class="model-card ${translator ? "translator-card" : "reasoning-card"}" data-render-key="${esc(run.id)}-${esc(model.model_id)}" data-model-run="${esc(run.id)}" data-model-id="${esc(model.model_id)}"><summary>${busy ? '<span class="model-stream-dot"></span>' : icon(translator ? "chip" : "spark", true)}<span class="model-card-title">${translator ? "Translator" : "Reasoning"}${translator ? "" : ` <small>· ${model.turn}</small>`}</span><span class="model-card-state">${status}</span>${icon("chevron", true)}</summary><div class="model-card-body"><div class="model-toolbar"><span>${esc(model.model)} · ${model.streamed ? (run.mode === "demo" ? "simulated stream" : "live text") : "buffered response"}</span><button class="text-button model-resize" data-model-id="${esc(model.model_id)}" data-model-run="${esc(run.id)}" aria-expanded="${Boolean(model.expanded)}">${model.expanded ? "Shrink" : "Expand"}</button></div><div class="model-scroll${model.expanded ? " expanded" : ""}" data-render-key="${esc(model.model_id)}-scroll"><div class="model-output-content">${modelOutputHTML(model)}</div>${model.error ? `<p class="model-error">${esc(model.error)}</p>` : ""}${model.trace_limited ? '<p class="model-note">The model trace was capped. Tool results and the final answer remain available.</p>' : ""}</div><div class="model-footer"><button class="text-button model-inspect" data-model-id="${esc(model.model_id)}" data-model-run="${esc(run.id)}" aria-expanded="${Boolean(model.inspect)}">${icon("chat", true)}${model.inspect ? "Hide conversation" : "Model conversation"}</button><span class="model-stats">${modelStats(model)}</span></div>${modelConversation(model)}</div></details>`;
}
function draftTool(model, part, run) {
  const interrupted = model.done && model.status !== "completed";
  const ignored =
    model.done &&
    (model.decision === "final" ||
      part !== [...model.parts.values()].find((item) => item.kind === "tool"));
  const label =
    interrupted || ignored
      ? "not executed"
      : model.done
        ? "awaiting validation"
        : "drafting";
  return `<details class="tool-card draft-tool" data-event-key="${esc(run.id)}-tool-${esc(model.model_id)}-${part.index}"><summary>${icon("tools")}<span class="tool-card-title">Tool request</span><span class="tool-card-status">${label}</span>${icon("chevron")}</summary><div class="tool-card-body"><div class="tool-detail-label">Draft intent · not an executable call</div><pre data-draft-model="${esc(model.model_id)}" data-draft-part="${part.index}">${esc(part.pacer.text || "Receiving the instruction…")}</pre><p class="model-note">${interrupted ? "The response was interrupted. This draft was not executed." : ignored ? "This draft was not selected for execution." : "The complete response must finish before translation, validation, confidence, and permission checks."}</p></div></details>`;
}
function renderTimeline(run) {
  const models = [...modelsFor(run).values()].filter(
    (model) => model.component === "reasoning",
  );
  const groups = actionGroups(run);
  if (!models.length)
    return groups.map((group) => renderToolCard(group, run)).join("");
  const rendered = new Set();
  let html = "";
  for (const model of models) {
    html += renderModelCard(model, run);
    const related = groups.filter((group) => group.modelId === model.model_id);
    const drafts = [...model.parts.values()].filter(
      (part) => part.kind === "tool",
    );
    for (const part of drafts) {
      const group = related.find(
        (item) =>
          !rendered.has(item) &&
          item.action.trim() === part.pacer.received.trim(),
      );
      if (group) {
        group.partIndex = part.index;
        rendered.add(group);
        html += renderToolCard(group, run);
      } else html += draftTool(model, part, run);
    }
    for (const group of related)
      if (!rendered.has(group)) {
        rendered.add(group);
        html += renderToolCard(group, run);
      }
  }
  for (const group of groups)
    if (!rendered.has(group)) html += renderToolCard(group, run);
  return html;
}
function answerPreview(run, message) {
  const model = [...modelsFor(run).values()]
    .filter((item) => item.component === "reasoning")
    .at(-1);
  if (!model) return { text: message.content, streaming: false };
  if (model.trace_limited && message.content)
    return { text: message.content, streaming: false };
  const parts = [...model.parts.values()];
  const final = parts.find((part) => part.kind === "final");
  const plain = !parts.some((part) => ["tool", "final"].includes(part.kind));
  const sources = final
    ? [final]
    : plain
      ? parts.filter((part) => part.kind === "text")
      : [];
  const text = sources
    .map((part) => part.pacer.text)
    .join("")
    .trimStart();
  const pending = sources.some((part) => part.pacer.pending);
  if (run.done && run.status !== "COMPLETED")
    return { text: message.content, partial: text, streaming: false };
  if (model.done && model.decision === "final" && !pending)
    return { text: message.content || model.answer || text, streaming: false };
  if (!model.done || pending) return { text, streaming: sources.length > 0 };
  return { text: message.content || "", streaming: false };
}
function renderSelectionReview(review) {
  if (!review) return "";
  return `<div class="selection-review"><div class="tool-detail-label">Reasoning model review</div><p>${review.suggested_tool ? `Is <strong>${esc(review.suggested_tool)}</strong> the correct tool?` : "Choose the correct registered tool."}</p>${review.candidates.length ? `<ul>${review.candidates.map((candidate) => `<li>${esc(candidate.tool_name)} <span>${candidate.confidence.toFixed(2)}</span></li>`).join("")}</ul>` : ""}<p>The reasoning model must confirm or correct the selection in its next action. All gates still apply.</p></div>`;
}
function renderToolCard(group, run) {
  const name = group.call?.tool || group.translation?.selected_tool;
  const score = group.confidence?.score ?? group.translation?.confidence;
  const failed = Boolean(group.rejected) || group.result?.success === false;
  const label = group.rejected
    ? "blocked"
    : group.result
      ? group.result.success
        ? "done"
        : "error"
      : "in progress";
  const args = group.call?.arguments || group.translation?.arguments;
  const key = group.modelId
    ? `${run.id}-tool-${group.modelId}-${group.partIndex ?? 0}`
    : `${run.id}-${group.id}`;
  const translator = [...modelsFor(run).values()].find(
    (model) =>
      model.component === "translator" && model.parent_id === group.modelId,
  );
  return `<details class="tool-card" data-event-key="${esc(key)}"><summary>${icon(toolIcons[name] || "spark")}<span class="tool-card-title">${esc(toolNames[name] || "Action requested")}</span><span class="tool-card-status${failed ? " error" : ""}">${esc(label)}</span>${icon("chevron")}</summary><div class="tool-card-body"><p>${esc(group.action)}</p>${args ? `<div class="tool-detail-label">${group.validated ? "Validated arguments" : "Proposed arguments"}${group.rejected ? " · not executed" : ""}</div><pre>${esc(JSON.stringify(args, null, 2))}</pre>` : ""}${group.result ? `<div class="tool-detail-label">${group.result.success ? "Tool observation" : "Tool error"}</div><pre>${esc(group.result.success ? group.result.output : group.result.error)}</pre>` : ""}${group.rejected ? `<div class="tool-detail-label">Blocked at ${esc(group.rejected.stage)}</div><pre>${esc(group.rejected.message)}</pre>` : ""}${renderDecision(group.decision)}${renderSelectionReview(group.review)}${translator ? renderModelCard(translator, run) : ""}${score !== undefined ? `<div class="confidence-line">${run.mode === "demo" ? "Synthetic demo score" : "Translator confidence"}: ${Number(score).toFixed(2)}${group.confidence ? ` · gate ≥ ${Number(group.confidence.threshold).toFixed(2)}` : ""}</div>` : ""}</div></details>`;
}
function renderDecision(decision) {
  if (!decision) return "";
  const approved = Boolean(decision.answer);
  const label = decision.tool
    ? toolLabel(decision.tool)
    : "action";
  const edited = decision.edited ? "edited " : "";
  return `<div class="tool-decision ${approved ? "approved" : "declined"}">${icon(approved ? "check" : "close", true)}<span>You ${approved ? "approved" : "declined"} the ${edited}${esc(label)}.</span></div>`;
}
function approvalPreview(call) {
  if (!call || !call.arguments) return "(no details)";
  const args = call.arguments;
  const preview =
    args.content ?? args.code ?? args.command ?? args.patch ?? args.message ?? args.question ?? args.path ?? args.source ?? args.branch ?? "";
  const text = String(preview);
  return text ? text : "(no details)";
}
function renderDiff(diff) {
  if (!diff || !diff.text) return "";
  const lines = String(diff.text)
    .split("\n")
    .map((line) => {
      const cls =
        line.startsWith("+") && !line.startsWith("+++")
          ? "diff-add"
          : line.startsWith("-") && !line.startsWith("---")
            ? "diff-del"
            : line.startsWith("@@")
              ? "diff-hunk"
              : "";
      return `<span class="${cls}">${esc(line) || " "}</span>`;
    })
    .join("\n");
  return `<div class="tool-detail-label">${esc(diff.label || "Proposed change")}</div><pre class="pending-diff">${lines}</pre>`;
}
function stringifyArg(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}
function approvalEditor(run, call) {
  const draft = S.edits[run.id] || { open: false, values: {} };
  const tool = (S.tools || []).find((item) => item.name === call.name);
  const props = tool?.parameters?.properties || {};
  const required = tool?.parameters?.required || [];
  const payloads = new Set((S.payloadArgs || {})[call.name] || []);
  const names = [...Object.keys(call.arguments || {})];
  for (const key of Object.keys(props))
    if (!names.includes(key)) names.push(key);
  const rows = names
    .map((key) => {
      const schema = props[key] || {};
      const current =
        draft.values[key] !== undefined
          ? draft.values[key]
          : stringifyArg(call.arguments?.[key]);
      const field = payloads.has(key)
        ? `<textarea data-edit-arg="${esc(key)}" rows="5">${esc(current)}</textarea>`
        : `<input data-edit-arg="${esc(key)}" value="${esc(current)}" spellcheck="false">`;
      const hint = [required.includes(key) ? "required" : null, schema.type]
        .filter(Boolean)
        .join(" · ");
      return `<label class="approval-field"><span>${esc(key)}${hint ? `<small>${esc(hint)}</small>` : ""}</span>${field}</label>`;
    })
    .join("");
  return `<button type="button" class="text-button approval-edit-toggle" data-edit-toggle="${esc(run.id)}" aria-expanded="${Boolean(draft.open)}">${icon("tools", true)}${draft.open ? "Hide editor" : "Edit arguments"}</button><div class="approval-editor${draft.open ? "" : " hidden"}">${rows}<p class="pending-hint">Edited arguments are re-validated and re-gated before anything runs.</p></div>`;
}
function coerceArg(schema, value) {
  const text = String(value ?? "");
  const types = Array.isArray(schema?.type) ? schema.type : [schema?.type];
  if (types.includes("integer") && /^-?\d+$/.test(text.trim()))
    return Number(text.trim());
  if (
    types.includes("number") &&
    text.trim() !== "" &&
    Number.isFinite(Number(text))
  )
    return Number(text);
  if (types.includes("boolean")) {
    const lowered = text.trim().toLowerCase();
    if (lowered === "true") return true;
    if (lowered === "false") return false;
  }
  return text;
}
function gatherEditedArgs(run, card) {
  const call = run.pending?.call;
  const tool = (S.tools || []).find((item) => item.name === call?.name);
  const props = tool?.parameters?.properties || {};
  const args = {};
  $$("[data-edit-arg]", card).forEach((field) => {
    args[field.dataset.editArg] = coerceArg(props[field.dataset.editArg], field.value);
  });
  return args;
}
function renderPending(run) {
  if (!run.pending || run.done) return "";
  const pending = run.pending,
    approval = pending.kind === "approval";
  const disabled =
    run.status === "CANCELLING" || pending.submitted ? "disabled" : "";
  const allowLabel = pending.call?.name === "write_file" ? "Allow write" : "Allow action";
  const diff = approval ? renderDiff(pending.diff) : "";
  const editor = approval ? approvalEditor(run, pending.call || { arguments: {} }) : "";
  return `<div class="pending-card"><div class="pending-title">${icon(approval ? "shield" : "chat")}${approval ? "Your permission is needed" : "A quick question for you"}</div><p>${esc(pending.question)}</p>${approval ? `${diff || `<pre>${esc(approvalPreview(pending.call))}</pre>`}${editor}<p class="pending-hint">Nothing runs until you approve.</p><div class="pending-actions"><button class="button primary small" data-answer-run="${esc(run.id)}" data-approved="true" ${disabled}>${icon("check", true)}${allowLabel}</button><button class="button secondary small" data-answer-run="${esc(run.id)}" data-approved="edited" ${disabled}>${icon("tools", true)}Approve edited</button><button class="button secondary small" data-answer-run="${esc(run.id)}" data-approved="false" ${disabled}>Deny</button></div>` : '<p class="pending-hint">Type your answer in the message box below to continue.</p>'}</div>`;
}
function renderAssistant(message) {
  const run = S.runs.get(message.run_id);
  if (!run)
    return `<article class="message assistant" data-render-key="assistant-${esc(message.run_id)}"><div class="assistant-label"><span class="assistant-mark">${icon("relay")}</span>Relay</div><div class="assistant-body"><div class="markdown">${markdown(message.content)}</div></div></article>`;
  const timeline = renderTimeline(run);
  const answers = (run.events || []).filter(
    (event) => event.type === "user_answer",
  );
  // Approval outcomes now live inside their own tool card; only question
  // answers and legacy approvals (recorded without a tool name) stay here.
  const looseAnswers = answers.filter(
    (event) => event.kind !== "approval" || !event.tool,
  );
  const pending = renderPending(run);
  const preview = answerPreview(run, message);
  const working =
    !run.done && !run.pending && !preview.text
      ? `<div class="working" data-render-key="working"><span class="spinner"></span><span>${S.streamError && S.streamRun === run.id ? "Connection interrupted. Reconnecting…" : run.status === "CANCELLING" ? "Stopping the current model stream…" : `${esc(phaseNames[run.phase] || "Starting the agent")}…`}</span></div>`
      : "";
  return `<article class="message assistant" data-render-key="assistant-${esc(run.id)}" data-message-run="${esc(run.id)}"><div class="assistant-label"><span class="assistant-mark">${icon("relay")}</span>Relay<span class="label-mode">${run.mode === "demo" ? "DEMO" : "LIVE"}</span></div><div class="assistant-body">${timeline ? `<div class="tool-stack model-timeline" data-render-key="timeline">${timeline}</div>` : ""}${looseAnswers.map((event) => `<div class="answered-note" data-render-key="answer-${event.id}">${event.kind === "approval" ? `You ${event.answer ? "approved" : "declined"} an action.` : `You: ${esc(event.answer)}`}</div>`).join("")}${pending}${working}${run.trace_limited ? '<p class="model-note trace-warning" data-render-key="trace-warning">Model trace limit reached. Further model text is omitted; tool results and the final answer are still retained.</p>' : ""}${preview.partial ? `<div class="partial-response" data-render-key="partial"><p class="model-note">Partial response · generation did not complete</p><div class="markdown">${markdown(preview.partial)}</div></div>` : ""}${preview.text || preview.streaming ? `<div class="markdown${preview.streaming ? " streaming-answer" : ""}" data-render-key="answer-output" aria-busy="${preview.streaming}">${markdown(preview.text, preview.streaming)}</div>` : ""}${run.done ? `<div class="message-meta" data-render-key="meta">${statusBadge(run.status)}<span>${run.steps} tool ${run.steps === 1 ? "step" : "steps"}</span><span>·</span><span>${duration(run.elapsed_ms)}</span><span class="meta-spacer"></span><button class="icon-button inspect-run" data-run="${esc(run.id)}" aria-label="Inspect run">${icon("history")}</button><button class="icon-button copy-answer" data-run="${esc(run.id)}" aria-label="Copy answer">${icon("copy")}</button></div>` : ""}</div></article>`;
}
function preserveScroll(render, force = false) {
  const scroll = $("#chat-scroll");
  const following =
    scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
  const modelScrolls = $$(".model-scroll", $("#conversation")).filter(
    (node) => node.scrollHeight - node.scrollTop - node.clientHeight < 30,
  );
  render();
  for (const node of modelScrolls)
    if (node.isConnected) node.scrollTop = node.scrollHeight;
  if (following || force) scroll.scrollTop = scroll.scrollHeight;
}
function renderConversation(forceScroll = false) {
  const messages = S.conversation?.messages || [];
  $("#welcome").classList.toggle("hidden", messages.length > 0);
  preserveScroll(
    () =>
      reconcile(
        $("#conversation"),
        messages
          .map((message, index) =>
            message.role === "user"
              ? `<article class="message user" data-render-key="user-${esc(message.run_id || index)}"><div class="user-bubble">${esc(message.content)}</div></article>`
              : renderAssistant(message),
          )
          .join(""),
      ),
    forceScroll,
  );
  updateComposer();
}
let framePending = false;
let streamFrame = null;
let lastStreamPaint = 0;
function scheduleRender() {
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    renderConversation();
    renderActivity();
    if (S.view === "history") renderHistory();
  });
}
function paintStreamingRun(run) {
  const article = $(`[data-message-run="${run.id}"]`);
  if (!article) return;
  for (const model of modelsFor(run).values()) {
    if (!model.dirty) continue;
    model.dirty = false;
    const card = $(`.model-card[data-model-id="${model.model_id}"]`, article);
    if (card) {
      reconcile($(".model-output-content", card), modelOutputHTML(model));
      $(".model-stats", card).textContent = modelStats(model);
      const raw = $(".model-raw", card);
      if (raw) {
        reconcile(raw, esc(model.raw || "No response text received yet."));
        $("summary small", raw.parentElement).textContent =
          `${model.raw.length.toLocaleString()} characters`;
      }
      const reasoningRaw = $(".model-provider-reasoning", card);
      if (reasoningRaw) reconcile(reasoningRaw, esc(model.providerReasoning));
    }
    for (const part of model.parts.values())
      if (part.kind === "tool") {
        const draft = $(
          `[data-draft-model="${model.model_id}"][data-draft-part="${part.index}"]`,
          article,
        );
        if (draft)
          reconcile(
            draft,
            esc(part.pacer.text || "Receiving the instruction…"),
          );
      }
  }
  const message = S.conversation?.messages.find(
    (item) => item.role === "assistant" && item.run_id === run.id,
  );
  if (!message) return;
  const preview = answerPreview(run, message);
  const answer = $('[data-render-key="answer-output"]', article);
  if (answer) {
    answer.classList.toggle("streaming-answer", preview.streaming);
    answer.setAttribute("aria-busy", String(preview.streaming));
    reconcile(answer, markdown(preview.text || "", preview.streaming));
  } else if (preview.text || preview.streaming) scheduleRender();
}
function animateStreams(now) {
  streamFrame = null;
  for (const id of S.animating) {
    const run = S.runs.get(id);
    if (!run) {
      S.animating.delete(id);
      S.streamDirty.delete(id);
      continue;
    }
    let pending = false;
    for (const model of modelsFor(run).values())
      for (const part of model.parts.values()) {
        if (reducedMotion.matches) part.pacer.smooth = false;
        if (part.pacer.tick(now)) {
          model.dirty = true;
          S.streamDirty.add(id);
        }
        pending ||= part.pacer.pending;
      }
    if (!pending) S.animating.delete(id);
  }
  if (S.streamDirty.size && now - lastStreamPaint >= 32) {
    preserveScroll(() => {
      for (const id of S.streamDirty) {
        const run = S.runs.get(id);
        if (run) paintStreamingRun(run);
      }
    });
    S.streamDirty.clear();
    lastStreamPaint = now;
  }
  if (S.animating.size || S.streamDirty.size)
    streamFrame = requestAnimationFrame(animateStreams);
}
function wakeStreamRenderer() {
  if (streamFrame === null) streamFrame = requestAnimationFrame(animateStreams);
}
function renderActivity() {
  const run = currentRun();
  $("#activity-count").textContent = run
    ? (run.events || []).filter((e) => e.type === "action").length
    : 0;
  if (!run) {
    $("#activity-panel").innerHTML =
      `<div class="activity-empty"><span class="starter-icon sage">${icon("history")}</span><h3>Every step, in the open.</h3><p>Send a message to follow the agent’s actions, safety checks, and tool results here.</p></div>`;
    return;
  }
  const events = run.events || [],
    seen = new Set(events.filter((e) => e.type === "phase").map((e) => e.node));
  const actions = events.filter((e) => e.type === "action");
  const reviewRequested = events.some((e) => e.type === "confirmation");
  if (reviewRequested) seen.add("confirm");
  const latestPhase = [...events]
    .reverse()
    .find((e) => e.type === "phase")?.node;
  const rejected = [...events].reverse().find((e) => e.type === "rejected");
  $("#activity-panel").innerHTML =
    `<div class="activity-summary">${statusBadge(run.status)}<span class="activity-mode">${run.mode === "demo" ? "DEMO RUN" : "LIVE RUN"}</span></div><div class="activity-metrics"><div class="activity-metric"><strong>${run.steps || 0}</strong><span>tool steps</span></div><div class="activity-metric"><strong>${duration(run.elapsed_ms || 0)}</strong><span>elapsed time</span></div></div><h4 class="activity-section-title">RUNTIME PIPELINE</h4><ol class="phase-list">${phases
      .filter((phase) => phase !== "confirm" || reviewRequested)
      .map((phase) => {
        const active = !run.done && latestPhase === phase;
        const done = seen.has(phase) && !active;
        return `<li class="${active ? "current" : done ? "done" : ""}"><span class="phase-icon">${done ? icon("check") : ""}</span><span>${phaseNames[phase]}</span></li>`;
      })
      .join(
        "",
      )}</ol>${rejected ? `<p class="activity-note warn">An action was blocked at ${esc(rejected.stage)}. It did not reach tool execution.</p>` : ""}${run.mode === "demo" ? '<p class="activity-note">Demo adapters simulate reasoning and confidence. Files, calculations, permissions, and every pipeline step are real.</p>' : ""}${actions.length ? `<h4 class="activity-section-title">REQUESTED ACTIONS</h4><div class="activity-actions">${actions.map((event, i) => `<div class="activity-action"><p>${esc(event.action)}</p><small>ACTION ${i + 1} · ${duration(event.elapsed_ms)}</small></div>`).join("")}</div>` : ""}${run.done ? `<button class="button secondary small export-trace" data-run="${esc(run.id)}">${icon("code", true)}Export trace</button>` : ""}`;
}
function applyEvent(run, event) {
  if (event.id <= (run.events?.at(-1)?.id || 0)) return;
  modelsFor(run);
  run.events ||= [];
  run.events.push(event);
  const previousParts = event.model_id
    ? run.models.get(event.model_id)?.parts.size || 0
    : 0;
  reduceModelEvent(run, event);
  const newPart =
    event.model_id &&
    (run.models.get(event.model_id)?.parts.size || 0) !== previousParts;
  if (
    [
      "action",
      "translation",
      "validated",
      "confidence",
      "tool_start",
      "tool_result",
      "rejected",
      "confirmation",
      "user_answer",
    ].includes(event.type)
  )
    run.toolVersion = (run.toolVersion || 0) + 1;
  if (event.type === "action") flushModel(run, event.model_id);
  if (event.type === "question")
    for (const model of modelsFor(run).values())
      flushModel(run, model.model_id);
  run.elapsed_ms = event.elapsed_ms || run.elapsed_ms;
  if (event.type === "phase") run.phase = event.node;
  if (event.type === "tool_result") run.steps = event.step;
  if (event.type === "question") {
    run.pending = event;
    run.status = "WAITING_FOR_INPUT";
  }
  if (event.type === "question_expired" || event.type === "user_answer") {
    run.pending = null;
    run.status = "RUNNING";
    delete S.edits[run.id];
  }
  if (event.type === "cancelling") run.status = "CANCELLING";
  if (event.type === "complete") {
    run.status = event.status;
    run.steps = event.steps;
    run.done = true;
    run.pending = null;
    const message = S.conversation?.messages.find(
      (m) => m.role === "assistant" && m.run_id === run.id,
    );
    if (message) {
      message.content = event.final_answer;
      message.status = event.status;
    }
    refreshSummaries();
  }
  if (event.type === "model_delta") {
    wakeStreamRenderer();
    // A new section must appear even before its initial display buffer drains.
    if (
      newPart ||
      (event.parts || []).some((part) => Object.hasOwn(part, "complete"))
    )
      scheduleRender();
  } else {
    updateComposer();
    scheduleRender();
    wakeStreamRenderer();
  }
}
async function connectStream(runId, attempt = 0) {
  if (S.stream) S.stream.abort();
  const controller = new AbortController();
  S.stream = controller;
  S.streamRun = runId;
  S.streamError = false;
  const run = S.runs.get(runId);
  if (!run) return;
  const after = run.events?.at(-1)?.id || 0;
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/events?after=${after}`,
      { headers: { "X-Relay-Session": S.token }, signal: controller.signal },
    );
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Could not connect to the event stream.");
    }
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6))
          .join("\n");
        if (data) applyEvent(run, JSON.parse(data));
      }
    }
    if (!run.done)
      throw new Error("Event stream ended before the run finished.");
  } catch (error) {
    if (error.name === "AbortError" || controller.signal.aborted) return;
    S.streamError = true;
    scheduleRender();
    if (attempt < 3 && S.currentRun === runId)
      setTimeout(
        () => {
          if (S.stream === controller) connectStream(runId, attempt + 1);
        },
        1000 * (attempt + 1),
      );
    else
      toast(
        `${error.message} Open this conversation again to reconnect.`,
        true,
      );
  }
}
async function refreshSummaries() {
  try {
    const data = await api("/api/session");
    if (data.session_token !== S.token) return;
    S.conversations = data.conversations.filter(
      (c) => !S.deletedConversations.has(c.id),
    );
    for (const snapshot of data.runs) {
      if (S.deletedConversations.has(snapshot.conversation_id)) continue;
      const existing = S.runs.get(snapshot.id);
      if (!existing) S.runs.set(snapshot.id, snapshot);
      // In-flight events are authoritative; never overwrite a pending reply with an old snapshot.
    }
    renderSidebar();
    if (S.view === "history") renderHistory();
  } catch {
    /* The streaming connection reports reconnect errors separately. */
  }
}
async function openConversation(id) {
  const sequence = ++S.conversationLoadSequence;
  try {
    const conversation = await api(
      `/api/conversations/${encodeURIComponent(id)}`,
    );
    if (
      sequence !== S.conversationLoadSequence ||
      S.deletedConversations.has(id)
    )
      return;
    S.conversation = conversation;
    for (const run of conversation.runs) S.runs.set(run.id, run);
    for (const run of conversation.runs) modelsFor(run);
    S.currentRun = conversation.runs.at(-1)?.id || null;
    stored("relay-conversation", id);
    showView("playground");
    renderSidebar();
    renderConversation(true);
    selectInspector(S.currentRun ? "activity" : "setup");
    const run = currentRun();
    if (S.stream) S.stream.abort();
    if (run && !run.done) connectStream(run.id);
  } catch (error) {
    toast(error.message, true);
  }
}
function requestDeleteConversation(id) {
  const chat = S.conversations.find((item) => item.id === id);
  if (!chat) return;
  if (
    [...S.runs.values()].some((run) => run.conversation_id === id && !run.done)
  ) {
    toast(
      "Stop the run and wait for it to finish before deleting this conversation.",
    );
    return;
  }
  S.deleteTarget = id;
  $("#delete-chat-title").textContent = chat.title;
  $("#delete-error").classList.add("hidden");
  $("#delete-dialog").showModal();
}
async function deleteConversation() {
  const id = S.deleteTarget;
  if (!id) return;
  const button = $("#confirm-delete");
  button.disabled = true;
  try {
    await api(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    S.deletedConversations.add(id);
    S.conversations = S.conversations.filter((chat) => chat.id !== id);
    for (const [runId, run] of S.runs) {
      if (run.conversation_id !== id) continue;
      if (S.streamRun === runId) {
        S.stream?.abort();
        S.stream = null;
        S.streamRun = null;
      }
      S.runs.delete(runId);
    }
    if (S.conversation?.id === id) {
      S.conversationLoadSequence++;
      S.conversation = null;
      S.currentRun = null;
      stored("relay-conversation", null);
      $("#message-input").value = "";
      resizeInput();
      const busy = activeRun();
      if (busy) await openConversation(busy.conversation_id);
      else selectInspector("setup");
    }
    S.deleteTarget = null;
    $("#delete-dialog").close();
    renderSidebar();
    renderConversation();
    renderActivity();
    if (S.view === "history") renderHistory();
    toast(
      "Conversation and run history deleted. Workspace files are unchanged.",
    );
  } catch (error) {
    $("#delete-error").textContent = error.message;
    $("#delete-error").classList.remove("hidden");
  } finally {
    button.disabled = false;
  }
}
function startNew() {
  if (activeRun()) {
    toast("Finish or stop the active run before starting a new conversation.");
    return;
  }
  if (S.stream) S.stream.abort();
  S.conversationLoadSequence++;
  S.conversation = null;
  S.currentRun = null;
  stored("relay-conversation", null);
  $("#message-input").value = "";
  resizeInput();
  showView("playground");
  renderSidebar();
  renderConversation();
  selectInspector("setup");
  $("#message-input").focus();
}
async function sendMessage(text = $("#message-input").value) {
  text = text.trim();
  if (!text || !S.ready || S.starting) return;
  if (text.length > 8000) {
    toast("Please keep messages under 8,000 characters.", true);
    return;
  }
  const run = currentRun();
  if (run?.status === "WAITING_FOR_INPUT" && run.pending?.kind === "question") {
    try {
      await answerRun(run.id, { answer: text });
      $("#message-input").value = "";
      resizeInput();
      updateComposer();
    } catch (error) {
      toast(error.message, true);
    }
    return;
  }
  if (activeRun()) {
    toast("The agent is still working. Stop it or wait for its response.");
    return;
  }
  S.starting = true;
  updateComposer();
  showView("playground");
  try {
    if (!S.conversation) {
      S.conversation = {
        ...(await api("/api/conversations", { method: "POST", body: {} })),
        messages: [],
        runs: [],
      };
      stored("relay-conversation", S.conversation.id);
    }
    const run = await api(`/api/conversations/${S.conversation.id}/runs`, {
      method: "POST",
      body: { message: text },
    });
    run.events = [];
    S.runs.set(run.id, run);
    S.currentRun = run.id;
    S.conversation.messages.push(
      { role: "user", content: text, run_id: run.id },
      { role: "assistant", content: "", run_id: run.id },
    );
    if (S.conversation.messages.length === 2)
      S.conversation.title = text.slice(0, 48);
    $("#message-input").value = "";
    resizeInput();
    renderConversation(true);
    selectInspector("activity");
    refreshSummaries();
    connectStream(run.id);
  } catch (error) {
    toast(error.message, true);
  } finally {
    S.starting = false;
    updateComposer();
  }
}
async function answerRun(runId, answer) {
  const run = S.runs.get(runId);
  if (!run?.pending || run.pending.submitted) return;
  const questionId = run.pending.question_id;
  run.pending.submitted = true;
  renderConversation();
  try {
    await api(`/api/runs/${runId}/answer`, {
      method: "POST",
      body: { question_id: questionId, ...answer },
    });
  } catch (error) {
    if (run.pending) run.pending.submitted = false;
    renderConversation();
    throw error;
  }
}
async function stopRun() {
  const run = activeRun();
  if (!run) return;
  try {
    await api(`/api/runs/${run.id}/cancel`, { method: "POST", body: {} });
    if (!run.done) run.status = "CANCELLING";
    updateComposer();
    renderConversation();
  } catch (error) {
    toast(error.message, true);
  }
}
function resizeInput() {
  const input = $("#message-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}
function showView(view) {
  if (!pageInfo[view]) view = "playground";
  S.view = view;
  for (const name of Object.keys(pageInfo))
    $(`#${name}-view`).classList.toggle("hidden", name !== view);
  $$(".nav-item").forEach((item) => {
    const active = item.dataset.view === view;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  $("#breadcrumb-page").textContent = pageInfo[view][0];
  $("#page-title").textContent = pageInfo[view][0];
  $("#page-description").textContent = pageInfo[view][1];
  $("#sidebar").classList.remove("open");
  if (view === "workspace") loadFiles(S.filePath);
  if (view === "tools") renderTools();
  if (view === "history") {
    renderHistory();
    refreshSummaries();
  }
}
function renderTools() {
  if (!S.settings) return;
  $("#tools-view").innerHTML =
    `<div class="view-subheading"><p>One canonical definition powers the model description, schema, and validation.</p><span class="status-badge">${S.tools.length} tools registered</span></div><div class="tool-catalog">${S.tools
      .map((tool) => {
        const category =
          tool.name === "write_file"
            ? "APPROVAL REQUIRED"
            : tool.name === "ask_user"
              ? "INTERACTIVE"
              : "READ ONLY";
        const gate = ["write_file", "ask_user"].includes(tool.name)
          ? S.settings.confidence_threshold
          : S.settings.read_only_threshold;
        const count = Object.keys(tool.parameters.properties || {}).length;
        return `<button class="tool-catalog-card" data-tool="${esc(tool.name)}"><div class="tool-catalog-top"><span class="starter-icon ${tool.name === "write_file" ? "peach" : tool.name === "ask_user" ? "lavender" : "sage"}">${icon(toolIcons[tool.name] || "code")}</span><small>${category}</small></div><strong>${esc(tool.name)}</strong><p>${esc(tool.description)}</p><div class="tool-catalog-footer"><span>${count} ${count === 1 ? "parameter" : "parameters"} · gate ≥ ${gate.toFixed(2)}</span>${icon("arrow")}</div></button>`;
      })
      .join("")}</div>`;
}
function showTool(name) {
  const tool = S.tools.find((item) => item.name === name);
  if (!tool) return;
  const required = tool.parameters.required || [];
  const properties = Object.entries(tool.parameters.properties || {});
  $("#detail-title").textContent = tool.name;
  $("#detail-content").innerHTML =
    `<p class="detail-description">${esc(tool.description)}</p><table class="parameter-table"><thead><tr><th>PARAMETER</th><th>TYPE</th><th>DESCRIPTION</th></tr></thead><tbody>${properties.map(([name, schema]) => `<tr><td>${esc(name)}${required.includes(name) ? " *" : ""}</td><td>${esc(Array.isArray(schema.type) ? schema.type.join(" / ") : schema.type)}</td><td>${esc(schema.description || "")}</td></tr>`).join("")}</tbody></table><p class="field-help">* Required. Unexpected arguments and unsupported types are rejected before confidence is checked.</p><details class="schema-details"><summary>View canonical tool schema</summary><pre>${esc(JSON.stringify(tool, null, 2))}</pre></details>`;
  $("#detail-dialog").showModal();
}
function fileSize(size) {
  return size === null
    ? "Folder"
    : size < 1000
      ? `${size} B`
      : `${(size / 1000).toFixed(1)} KB`;
}
async function loadFiles(path) {
  const sequence = ++S.fileLoadSequence;
  S.filePath = path;
  S.fileLoading = true;
  S.fileFilter = "";
  renderWorkspace();
  try {
    const data = await api(`/api/files?path=${encodeURIComponent(path)}`);
    if (sequence !== S.fileLoadSequence) return;
    S.entries = data.entries;
  } catch (error) {
    if (sequence === S.fileLoadSequence) {
      S.entries = [];
      toast(error.message, true);
    }
  } finally {
    if (sequence === S.fileLoadSequence) {
      S.fileLoading = false;
      renderWorkspace();
    }
  }
}
async function openFile(path) {
  S.file = { path, loading: true };
  renderWorkspace();
  try {
    const data = await api(`/api/file?path=${encodeURIComponent(path)}`);
    if (S.file?.path === path) S.file = data;
  } catch (error) {
    if (S.file?.path === path) S.file = { path, error: error.message };
    toast(error.message, true);
  }
  renderWorkspace();
}
function fileRows() {
  const entries = S.entries.filter((entry) =>
    entry.name.toLowerCase().includes(S.fileFilter.toLowerCase()),
  );
  if (S.fileLoading)
    return '<div class="file-placeholder"><span class="spinner"></span><p>Reading directory…</p></div>';
  if (!entries.length)
    return '<div class="file-placeholder"><p>No files match this view.</p></div>';
  return entries
    .map(
      (entry) =>
        `<button class="file-row ${entry.type}${S.file?.path === entry.path ? " selected" : ""}" data-file-path="${esc(entry.path)}" data-file-type="${entry.type}">${icon(entry.type === "directory" ? "folder" : entry.name.endsWith(".py") ? "code" : "file")}<span>${esc(entry.name)}</span><small>${fileSize(entry.size)}</small></button>`,
    )
    .join("");
}
function renderWorkspace() {
  const root = S.workspace?.name || "workspace";
  const segments = S.filePath === "." ? [] : S.filePath.split("/");
  const breadcrumbs =
    `<button data-directory=".">${esc(root)}</button>` +
    segments
      .map(
        (part, i) =>
          `<span>/</span><button data-directory="${esc(segments.slice(0, i + 1).join("/"))}">${esc(part)}</button>`,
      )
      .join("");
  let preview = `<div class="file-placeholder">${icon("file")}<h3>A closer look.</h3><p>Select a file to see what the agent sees. Everything here lives inside your workspace.</p></div>`;
  if (S.file) {
    const file = S.file;
    preview = `<div class="file-preview-header"><span>${icon("file", true)}${esc(file.path)}</span><div class="file-preview-actions">${!file.loading && !file.error ? `<button class="button secondary small" data-ask-file="${esc(file.path)}">${icon("spark", true)}Ask Relay</button><button class="icon-button" id="copy-file" aria-label="Copy file contents">${icon("copy")}</button>` : ""}</div></div>${file.loading ? '<div class="file-placeholder"><span class="spinner"></span><p>Reading file…</p></div>' : file.error ? `<div class="file-placeholder"><p>${esc(file.error)}</p></div>` : `<pre class="file-content">${esc(file.content)}</pre>`}`;
  }
  $("#workspace-view").innerHTML =
    `<div class="view-subheading"><p>A read-only view of <strong>${esc(root)}</strong>. Agent writes require permission.</p><button class="button secondary small" id="refresh-files">${icon("refresh", true)}Refresh</button></div><div class="workspace-layout"><div class="files-card"><div class="files-card-header"><div class="file-breadcrumbs">${breadcrumbs}</div>${segments.length ? `<button class="icon-button" data-directory="${esc(segments.slice(0, -1).join("/") || ".")}" aria-label="Parent directory">${icon("up")}</button>` : ""}</div><label class="file-search">${icon("search")}<input id="file-filter" placeholder="Filter this directory…" value="${esc(S.fileFilter)}" aria-label="Filter files"></label><div class="file-list" id="file-list">${fileRows()}</div></div><div class="file-preview">${preview}</div></div>`;
}
function renderHistory() {
  const runs = [...S.runs.values()].sort((a, b) => b.created_at - a.created_at);
  const done = runs.filter((run) => run.status === "COMPLETED").length;
  const tools = runs.reduce((sum, run) => sum + (run.steps || 0), 0);
  $("#history-view").innerHTML =
    `<div class="history-stats"><div class="history-stat"><strong>${runs.length}</strong><span>Total runs</span></div><div class="history-stat"><strong>${done}</strong><span>Completed</span></div><div class="history-stat"><strong>${tools}</strong><span>Tool executions</span></div></div>${
      runs.length
        ? `<div class="history-table"><div class="history-row table-header"><span>CONVERSATION</span><span>STATUS</span><span>MODE</span><span>TOOLS</span><span>DURATION</span><span></span></div>${runs
            .map((run) => {
              const conversation = S.conversations.find(
                (c) => c.id === run.conversation_id,
              );
              return `<button class="history-row" data-conversation="${esc(run.conversation_id)}"><span>${esc(conversation?.title || "Conversation")}</span><span>${statusBadge(run.status)}</span><span>${run.mode === "demo" ? "Demo" : "Live"}</span><span>${run.steps || 0} steps</span><span>${duration(run.elapsed_ms || 0)}</span>${icon("chevron")}</button>`;
            })
            .join("")}</div>`
        : `<div class="empty-state">${icon("history")}<h3>A clean slate.</h3><p>Your runs, tool activity, and execution outcomes will appear here as you work.</p><button class="button secondary" data-view="playground">Start a conversation${icon("arrow", true)}</button></div>`
    }`;
}
function showSettings() {
  if (!S.settings) return;
  const s = S.settings;
  $(`input[name="mode"][value="${s.mode}"]`).checked = true;
  $("#base-url").value = s.base_url;
  $("#model-name").value = s.model;
  // The server never returns the key; the field always starts empty.
  $("#api-key").value = "";
  $("#api-key").placeholder = S.apiKeySet
    ? "A session key is set — enter a new one to replace it, or save blank to clear"
    : "Use a hosted model? Paste a key — kept in server memory for this session only";
  $("#max-steps").value = s.max_tool_steps;
  $("#read-threshold").value = s.read_only_threshold;
  $("#write-threshold").value = s.confidence_threshold;
  $("#read-only").checked = s.read_only;
  $("#create-parents").checked = s.allow_create_parent_dirs;
  $("#read-only").disabled = Boolean(S.readOnlyEnforced);
  $$("[data-setting]").forEach((input) => {
    const value = s[input.dataset.setting];
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value ?? "";
  });
  $("#settings-workspace").textContent = S.workspace?.path || ".";
  $("#connection-result").classList.add("hidden");
  updateModeExplanation();
  if (!$("#settings-dialog").open) $("#settings-dialog").showModal();
}
function updateModeExplanation() {
  const demo = $('input[name="mode"]:checked').value === "demo";
  $("#live-settings").classList.toggle("hidden", demo);
  $("#base-url").disabled = demo;
  $("#model-name").disabled = demo;
  $("#api-key").disabled = demo;
  $("#mode-explanation").textContent = demo
    ? "A deterministic demo, not a language model. Try search, reading, arithmetic, time, and approved writes through the real runtime. Confidence scores are simulated."
    : "Your reasoning model emits natural-language intents. Needle 2 translates them into tool calls. The runtime validates and gates every action. Live mode never silently falls back to the demo.";
}
function settingsFromForm() {
  const settings = {
    ...S.settings,
    mode: $('input[name="mode"]:checked').value,
    base_url: $("#base-url").value.trim(),
    model: $("#model-name").value.trim(),
    api_key: $("#api-key").value.trim(),
    max_tool_steps: Number($("#max-steps").value),
    read_only_threshold: Number($("#read-threshold").value),
    confidence_threshold: Number($("#write-threshold").value),
    read_only: $("#read-only").checked,
    allow_create_parent_dirs: $("#create-parents").checked,
  };
  $$("[data-setting]").forEach((input) => {
    settings[input.dataset.setting] =
      input.type === "checkbox"
        ? input.checked
        : input.type === "number"
          ? Number(input.value)
          : input.value;
  });
  if (!settings.default_timezone) settings.default_timezone = null;
  return settings;
}
async function saveSettings(event) {
  event.preventDefault();
  const button = $("#save-settings");
  button.disabled = true;
  try {
    const result = await api("/api/settings", {
      method: "POST",
      body: settingsFromForm(),
    });
    S.settings = result.settings;
    $("#api-key").value = "";
    renderSettings();
    $("#settings-dialog").close();
    toast(
      S.settings.mode === "demo"
        ? "Offline demo is ready. Real tools, simulated models."
        : "Live mode selected. Make sure your model server is running.",
    );
  } catch (error) {
    showConnectionResult(error.message, false);
  } finally {
    button.disabled = false;
  }
}
function showConnectionResult(message, ok) {
  const element = $("#connection-result");
  element.textContent = message;
  element.classList.remove("hidden");
  element.classList.toggle("error", !ok);
}
async function testConnection() {
  if (!$("#settings-form").reportValidity()) return;
  const button = $("#test-connection");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Checking…';
  showConnectionResult(
    "Checking the selected adapters. A first-time engine download can take a moment.",
    true,
  );
  try {
    const result = await api("/api/connection", {
      method: "POST",
      body: settingsFromForm(),
    });
    showConnectionResult(result.message, result.ok);
  } catch (error) {
    showConnectionResult(error.message, false);
  } finally {
    button.disabled = false;
    button.innerHTML = icon("refresh", true) + "Test connection";
  }
}
function downloadText(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function exportSettings() {
  try {
    const data = await api("/api/settings/export");
    downloadText(data.filename, data.content, "application/toml");
    toast("Saved settings exported without API keys or machine-only paths.");
  } catch (error) {
    showConnectionResult(error.message, false);
  }
}
async function importSettings(file) {
  if (!file) return;
  try {
    if (file.size > 250000) throw new Error("Keep config files under 250 KB.");
    const data = await api("/api/settings/import", {
      method: "POST",
      body: {
        content: await file.text(),
        format: file.name.toLowerCase().endsWith(".json") ? "json" : "toml",
      },
    });
    S.settings = data.settings;
    renderSettings();
    showSettings();
    showConnectionResult(
      "Configuration imported and applied." +
        (data.ignored.length
          ? ` Server-only settings left unchanged: ${data.ignored.join(", ")}.`
          : ""),
      true,
    );
  } catch (error) {
    showConnectionResult(error.message, false);
  } finally {
    $("#config-file").value = "";
  }
}
async function resetPrompts() {
  try {
    const data = await api("/api/settings/defaults");
    for (const name of [
      "reasoning_prompt",
      "translator_prompt",
      "confirmation_prompt",
    ])
      $(`[data-setting="${name}"]`).value = data.settings[name];
    showConnectionResult(
      "Server instructions restored in the form. Save settings to apply.",
      true,
    );
  } catch (error) {
    showConnectionResult(error.message, false);
  }
}
function showGuide() {
  $("#detail-title").textContent = "A small guide to Relay";
  $("#detail-content").innerHTML =
    `<div class="guide-section"><span class="guide-number">01</span><div><h3>Start with the workspace</h3><p>Try “Explore this workspace”, “Find the authentication implementation”, or “Calculate 24 * 18 + 120”. The offline demo supports these concrete tasks with real tools, not live AI reasoning.</p></div></div><div class="guide-section"><span class="guide-number">02</span><div><h3>Bring your reasoning model</h3><p>Start an OpenAI-compatible server, then choose <strong>Live models</strong> in Settings. For Ollama, an example is <code>ollama run qwen2.5:3b</code> with server URL <code>http://127.0.0.1:11434/v1</code>. The server must be reachable from the machine hosting this console. The action model downloads its small inference engine on first use; offline installation is described in the README.</p></div></div><div class="guide-section"><span class="guide-number">03</span><div><h3>Stay in the loop</h3><p>The agent can ask you questions. Reply in the composer to continue. Each write asks for your approval and displays the exact content. Stop a run anytime; an in-flight model call may finish, but no subsequent tool will execute.</p></div></div><div class="guide-section"><span class="guide-number">04</span><div><h3>Trust the boundary, inspect the work</h3><p>All paths stay in your configured workspace. No shell, Python execution, append, binary writes, or delete tools. Low-confidence or invalid calls do not execute. Use the Activity panel or expand an action card to see exactly what happened.</p></div></div><div class="guide-pipeline">REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE → SAFETY → EXECUTE → OBSERVE → UPDATE CONTEXT → REASON</div><p class="field-help">Conversations are isolated to your browser session and kept in server memory. They expire after two hours of inactivity and reset when the server restarts. Export a run trace if you want to keep it.</p>`;
  $("#detail-dialog").showModal();
}
async function exportRun(id) {
  try {
    const run = await api(`/api/runs/${id}`);
    const blob = new Blob([JSON.stringify(run, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = `relay-run-${id}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Run trace exported");
  } catch (error) {
    toast(error.message, true);
  }
}

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});
$("#message-input").addEventListener("input", () => {
  resizeInput();
  updateComposer();
});
$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage();
  }
});
$("#new-session").addEventListener("click", startNew);
$("#theme-toggle").addEventListener("click", () =>
  applyTheme(
    document.documentElement.dataset.theme === "dark" ? "light" : "dark",
  ),
);
$("#confirm-delete").addEventListener("click", deleteConversation);
$("#export-settings").addEventListener("click", exportSettings);
$("#import-settings").addEventListener("click", () =>
  $("#config-file").click(),
);
$("#config-file").addEventListener("change", (event) =>
  importSettings(event.target.files[0]),
);
$("#reset-prompts").addEventListener("click", resetPrompts);
$("#stop-run").addEventListener("click", stopRun);
function applyPanelState() {
  const root = document.documentElement.style;
  const sidebarWidth = Number(stored("relay-sidebar-w"));
  if (sidebarWidth >= 140 && sidebarWidth <= 320)
    root.setProperty("--sidebar", `${sidebarWidth}px`);
  const inspectorWidth = Number(stored("relay-inspector-w"));
  if (inspectorWidth >= 220 && inspectorWidth <= 480)
    root.setProperty("--inspector", `${inspectorWidth}px`);
  document.body.classList.toggle(
    "sidebar-hidden",
    stored("relay-sidebar-hidden") === "1",
  );
  document.body.classList.toggle(
    "inspector-hidden",
    stored("relay-inspector-hidden") === "1",
  );
  syncSidebarCompact();
  syncCollapseIcon();
  applyCustomColors();
}
const DEFAULT_UI_COLORS = { base: "#f8f9f6", accent: "#d36b48" };
function applyCustomColors() {
  const root = document.documentElement.style;
  const base = stored("relay-ui-base"),
    accent = stored("relay-ui-accent");
  if (base) {
    root.setProperty("--paper", base);
    // Cards, inputs, and dialogs follow the base in both themes. Dark
    // bases lift subtly toward white (dark elevation); light bases mix
    // further toward white. The mixes reproduce the default light palette
    // when the default base is picked, so there is no jump there.
    // A blind white mix would turn dark bases into washed-out gray.
    const dark = (baseLuminance(base) ?? 1) < 0.35;
    const mix = dark ? [88, 78, 68] : [30, 45, 60];
    root.setProperty("--surface", `color-mix(in srgb, ${base} ${mix[0]}%, white)`);
    root.setProperty(
      "--surface-soft",
      `color-mix(in srgb, ${base} ${mix[1]}%, white)`,
    );
    root.setProperty(
      "--surface-hover",
      `color-mix(in srgb, ${base} ${mix[2]}%, white)`,
    );
    // Sidebar stays a touch darker than the workspace, topbar a touch
    // lighter, so the panels keep their visual hierarchy in any color.
    root.setProperty("--sidebar-bg", `color-mix(in srgb, ${base} 80%, black)`);
    root.setProperty(
      "--topbar-custom",
      `color-mix(in srgb, ${base} 90%, white)`,
    );
  } else {
    for (const name of [
      "--paper",
      "--surface",
      "--surface-soft",
      "--surface-hover",
      "--sidebar-bg",
      "--topbar-custom",
    ])
      root.removeProperty(name);
  }
  if (accent) {
    root.setProperty("--accent", accent);
    root.setProperty("--accent-hover", `color-mix(in srgb, ${accent} 88%, black)`);
  } else {
    root.removeProperty("--accent");
    root.removeProperty("--accent-hover");
  }
  const baseInput = $("#ui-base-color"),
    accentInput = $("#ui-accent-color");
  if (baseInput) baseInput.value = base || DEFAULT_UI_COLORS.base;
  if (accentInput) accentInput.value = accent || DEFAULT_UI_COLORS.accent;
}
function baseLuminance(hex) {
  const match = /^#([0-9a-f]{6})$/i.exec(hex || "");
  if (!match) return null;
  const [r, g, b] = [0, 2, 4].map((i) =>
    parseInt(match[1].slice(i, i + 2), 16),
  );
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}
function checkThemeFit() {
  const luminance = baseLuminance(stored("relay-ui-base"));
  if (luminance === null) return;
  const wantsDark = luminance < 0.35;
  const isDark = document.documentElement.dataset.theme === "dark";
  // A dark base under light text colors (or vice versa) is unreadable, so
  // match the theme to the pick. The toggle stays available to override.
  if (wantsDark !== isDark) {
    applyTheme(wantsDark ? "dark" : "light");
    toast(
      `Switched to the ${wantsDark ? "dark" : "light"} theme to match this base color.`,
    );
  }
}
function shellRect() {
  return $(".app-shell").getBoundingClientRect();
}
const SIDEBAR_COMPACT_BELOW = 150;
function sidebarWidth() {
  return (
    parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue("--sidebar"),
    ) || 167
  );
}
function syncSidebarCompact() {
  document.body.classList.toggle(
    "sidebar-compact",
    window.innerWidth > 760 && sidebarWidth() < SIDEBAR_COMPACT_BELOW,
  );
}
function syncCollapseIcon() {
  const hidden = document.body.classList.contains("sidebar-hidden");
  const button = $("#sidebar-collapse");
  if (button) {
    button.innerHTML = `<span class="${hidden ? "" : "flip"}">${icon("chevron")}</span>`;
    button.setAttribute(
      "aria-label",
      hidden ? "Show navigation panel" : "Collapse navigation panel",
    );
  }
}
function setSidebarHidden(hidden) {
  document.body.classList.toggle("sidebar-hidden", hidden);
  stored("relay-sidebar-hidden", hidden ? "1" : null);
  syncCollapseIcon();
}
function dragSplitter(node, min, max, apply, persist) {
  const clamp = (value) => Math.min(max, Math.max(min, Math.round(value)));
  const set = (value, save) => {
    const width = clamp(value);
    document.documentElement.style.setProperty(apply, `${width}px`);
    if (apply === "--sidebar") syncSidebarCompact();
    if (save) stored(persist, String(width));
  };
  node.addEventListener("pointerdown", (event) => {
    if (node.id === "splitter-left" && window.innerWidth <= 760) return;
    if (node.id === "splitter-right" && window.innerWidth <= 1060) return;
    event.preventDefault();
    node.classList.add("dragging");
    node.setPointerCapture(event.pointerId);
    const rect = shellRect();
    const move = (e) => {
      set(
        node.id === "splitter-left"
          ? e.clientX - rect.left
          : rect.right - e.clientX,
        false,
      );
    };
    const up = (e) => {
      node.classList.remove("dragging");
      set(
        node.id === "splitter-left"
          ? e.clientX - rect.left
          : rect.right - e.clientX,
        true,
      );
      node.removeEventListener("pointermove", move);
      node.removeEventListener("pointerup", up);
      node.removeEventListener("pointercancel", up);
    };
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", up);
    node.addEventListener("pointercancel", up);
  });
  node.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const current = parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue(apply),
    );
    const delta = event.key === "ArrowLeft" ? -12 : 12;
    set((node.id === "splitter-left" ? current + delta : current - delta), true);
  });
}
$("#menu-button").addEventListener("click", () => {
  if (window.innerWidth <= 760) $("#sidebar").classList.toggle("open");
  else
    setSidebarHidden(!document.body.classList.contains("sidebar-hidden"));
});
$("#sidebar-collapse").addEventListener("click", (event) => {
  event.stopPropagation();
  setSidebarHidden(!document.body.classList.contains("sidebar-hidden"));
});
window.addEventListener("resize", syncSidebarCompact);
$("#sidebar-scrim").addEventListener("click", () =>
  $("#sidebar").classList.remove("open"),
);
$("#inspector-toggle").addEventListener("click", () => {
  if (window.innerWidth <= 1060) {
    document.body.classList.remove("inspector-hidden");
    $("#inspector").classList.toggle("visible");
  } else {
    const hidden = document.body.classList.toggle("inspector-hidden");
    stored("relay-inspector-hidden", hidden ? "1" : null);
  }
});
$("#permission-chip").addEventListener("click", (event) => {
  event.stopPropagation();
  const popover = $("#permission-popover");
  popover.classList.toggle("hidden");
  $("#permission-chip").setAttribute(
    "aria-expanded",
    String(!popover.classList.contains("hidden")),
  );
});
document.addEventListener("click", (event) => {
  if (
    !event.target.closest("#permission-popover") &&
    !event.target.closest("#permission-chip")
  )
    $("#permission-popover").classList.add("hidden");
});
$("#permission-popover").addEventListener("change", (event) => {
  const preset = event.target.dataset?.preset;
  if (preset && PERMISSION_PRESETS[preset])
    saveAutoApprove(preset, event.target.checked);
});
$("#ui-base-color").addEventListener("input", (event) => {
  stored("relay-ui-base", event.target.value);
  applyCustomColors();
});
$("#ui-base-color").addEventListener("change", checkThemeFit);
$("#ui-accent-color").addEventListener("input", (event) => {
  stored("relay-ui-accent", event.target.value);
  applyCustomColors();
});
$("#reset-appearance").addEventListener("click", () => {
  stored("relay-ui-base", null);
  stored("relay-ui-accent", null);
  applyCustomColors();
  toast("Default colors restored.");
});
dragSplitter($("#splitter-left"), 64, 320, "--sidebar", "relay-sidebar-w");
dragSplitter($("#splitter-right"), 220, 480, "--inspector", "relay-inspector-w");
$("#setup-tab").addEventListener("click", () => selectInspector("setup"));
$("#activity-tab").addEventListener("click", () => selectInspector("activity"));
[
  "open-settings",
  "mode-badge",
  "model-chip",
  "connect-models",
  "connect-inline",
].forEach((id) => $(`#${id}`).addEventListener("click", showSettings));
$("#workspace-chip").addEventListener("click", () => showView("workspace"));
$("#open-guide").addEventListener("click", showGuide);
$("#settings-form").addEventListener("submit", saveSettings);
$$('input[name="mode"]').forEach((input) =>
  input.addEventListener("change", updateModeExplanation),
);
$("#test-connection").addEventListener("click", testConnection);
$(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  showView("playground");
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNew();
  }
  if ((event.metaKey || event.ctrlKey) && event.key === ",") {
    event.preventDefault();
    showSettings();
  }
});
document.addEventListener("input", (event) => {
  if (event.target.id === "file-filter") {
    S.fileFilter = event.target.value;
    $("#file-list").innerHTML = fileRows();
  }
  if (event.target.matches?.("[data-edit-arg]")) {
    const article = event.target.closest("[data-message-run]");
    const runId = article?.dataset.messageRun;
    if (runId) {
      const draft = S.edits[runId] || { open: true, values: {} };
      draft.open = true;
      draft.values[event.target.dataset.editArg] = event.target.value;
      S.edits[runId] = draft;
    }
    if (event.target.tagName === "TEXTAREA") {
      event.target.style.height = "auto";
      event.target.style.height = `${event.target.scrollHeight}px`;
    }
  }
});
document.addEventListener("click", async (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  try {
    if (
      target.classList.contains("model-resize") ||
      target.classList.contains("model-inspect")
    ) {
      const run = S.runs.get(target.dataset.modelRun);
      const model = run && modelsFor(run).get(target.dataset.modelId);
      if (model) {
        if (target.classList.contains("model-resize")) {
          model.expanded = !model.expanded;
          const scroller = target
            .closest(".model-card")
            .querySelector(".model-scroll");
          scroller.style.height = "";
        } else model.inspect = !model.inspect;
        renderConversation();
      }
    }
    if (target.dataset.view) showView(target.dataset.view);
    if (target.dataset.prompt) await sendMessage(target.dataset.prompt);
    if (target.dataset.conversation)
      await openConversation(target.dataset.conversation);
    if (target.dataset.deleteConversation)
      requestDeleteConversation(target.dataset.deleteConversation);
    if (target.dataset.tool) showTool(target.dataset.tool);
    if (target.dataset.directory) await loadFiles(target.dataset.directory);
    if (target.dataset.filePath)
      target.dataset.fileType === "directory"
        ? await loadFiles(target.dataset.filePath)
        : await openFile(target.dataset.filePath);
    if (target.dataset.askFile) {
      showView("playground");
      await sendMessage(`Read the file ${target.dataset.askFile}`);
    }
    if (target.dataset.answerRun) {
      const mode = target.dataset.approved;
      if (mode === "edited") {
        const run = S.runs.get(target.dataset.answerRun);
        const card = target.closest(".pending-card");
        const args = run && card ? gatherEditedArgs(run, card) : {};
        await answerRun(target.dataset.answerRun, {
          approved: true,
          arguments: args,
        });
      } else
        await answerRun(target.dataset.answerRun, {
          approved: mode === "true",
        });
    }
    if (target.dataset.editToggle) {
      const draft = S.edits[target.dataset.editToggle] || {
        open: false,
        values: {},
      };
      draft.open = !draft.open;
      S.edits[target.dataset.editToggle] = draft;
      renderConversation();
    }
    if (target.classList.contains("dialog-close"))
      target.closest("dialog").close();
    if (target.classList.contains("copy-code"))
      await copyText(
        target.closest(".code-block").querySelector("pre").textContent,
        target,
      );
    if (target.classList.contains("copy-answer")) {
      const message = S.conversation?.messages.find(
        (m) => m.role === "assistant" && m.run_id === target.dataset.run,
      );
      if (message) await copyText(message.content, target);
    }
    if (target.classList.contains("inspect-run")) {
      S.currentRun = target.dataset.run;
      selectInspector("activity");
      if (window.innerWidth <= 1060) {
        document.body.classList.remove("inspector-hidden");
        $("#inspector").classList.add("visible");
      }
    }
    if (target.classList.contains("export-trace"))
      await exportRun(target.dataset.run);
    if (target.id === "refresh-files") await loadFiles(S.filePath);
    if (target.id === "copy-file" && S.file?.content !== undefined)
      await copyText(S.file.content, target);
  } catch (error) {
    toast(error.message, true);
  }
});
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    )
      dialog.close();
  }),
);

async function boot() {
  try {
    const data = await api("/api/session");
    S.token = data.session_token;
    stored("relay-session", S.token);
    S.readOnlyEnforced = data.read_only_enforced;
    S.apiKeySet = Boolean(data.api_key_set);
    S.settings = data.settings;
    S.workspace = data.workspace;
    S.tools = data.tools;
    S.payloadArgs = data.payload_args || {};
    S.conversations = data.conversations;
    for (const run of data.runs) S.runs.set(run.id, run);
    S.ready = true;
    renderSettings();
    renderSidebar();
    updateComposer();
    const previous = stored("relay-conversation");
    if (previous && S.conversations.some((c) => c.id === previous))
      await openConversation(previous);
    else {
      const busy = activeRun();
      if (busy) await openConversation(busy.conversation_id);
    }
  } catch (error) {
    S.ready = false;
    updateComposer();
    toast(`${error.message} Reload this page to reconnect.`, true);
  }
}
applyTheme(document.documentElement.dataset.theme || "light", false);
applyPanelState();
boot();
