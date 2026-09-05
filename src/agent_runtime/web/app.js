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
const toolIcons = {
  read_file: "file",
  read_directory: "folder",
  search_files: "search",
  write_file: "tools",
  calculator: "calculator",
  get_time: "clock",
  ask_user: "chat",
};
const toolNames = {
  read_file: "Read file",
  read_directory: "Read directory",
  search_files: "Search files",
  write_file: "Write file",
  calculator: "Calculator",
  get_time: "Current time",
  ask_user: "Ask user",
};
const phases = [
  "reason",
  "parse",
  "translate",
  "sanitize",
  "validate",
  "confidence",
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
  token: stored("needle-session"),
  settings: null,
  workspace: null,
  tools: [],
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
};

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (S.token) headers["X-Needle-Session"] = S.token;
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
function markdown(text) {
  const chunks = String(text).split(/(```[^\n]*\n[\s\S]*?```)/g);
  return chunks
    .map((chunk) => {
      if (chunk.startsWith("```")) {
        const newline = chunk.indexOf("\n");
        const language = chunk.slice(3, newline).trim() || "text";
        const code = chunk.slice(newline + 1, -3).replace(/\n$/, "");
        return `<div class="code-block"><div class="code-block-header"><span>${esc(language)}</span><button class="icon-button copy-code" aria-label="Copy code">${icon("copy")}</button></div><pre><code>${esc(code)}</code></pre></div>`;
      }
      let html = "",
        paragraph = [],
        list = [];
      const flushParagraph = () => {
        if (paragraph.length)
          html += `<p>${inlineMarkdown(paragraph.join("\n"))}</p>`;
        paragraph = [];
      };
      const flushList = () => {
        if (list.length)
          html += `<ul>${list.map((line) => `<li>${inlineMarkdown(line)}</li>`).join("")}</ul>`;
        list = [];
      };
      for (const line of chunk.split("\n")) {
        if (/^#{1,4}\s/.test(line)) {
          flushParagraph();
          flushList();
          const heading = line.match(/^(#{1,4})\s+(.*)/);
          const tag = heading[1].length < 2 ? "h2" : "h3";
          html += `<${tag}>${inlineMarkdown(heading[2])}</${tag}>`;
        } else if (/^\s*[-*]\s/.test(line)) {
          flushParagraph();
          list.push(line.replace(/^\s*[-*]\s+/, ""));
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
    : "Required";
  $("#workspace-name").textContent = S.workspace?.name || "workspace";
  $("#workspace-chip").title = S.workspace?.path || "";
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
            `<button class="recent-item${c.id === S.conversation?.id ? " selected" : ""}" data-conversation="${esc(c.id)}" title="${esc(c.title)}">${icon("chat")}<span>${esc(c.title)}</span></button>`,
        )
        .join("")
    : '<div class="recent-empty">A fresh start.<br>Your conversations will appear here.</div>';
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
    : "Give Needle something to work on…";
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
function actionGroups(run) {
  const groups = [];
  let group;
  for (const event of run.events || []) {
    if (event.type === "action") {
      group = { id: event.id, action: event.action };
      groups.push(group);
    }
    if (!group) continue;
    if (event.type === "translation") group.translation = event;
    if (event.type === "validated") group.validated = true;
    if (event.type === "confidence") group.confidence = event;
    if (event.type === "tool_start") group.call = event;
    if (event.type === "tool_result") group.result = event;
    if (event.type === "rejected") group.rejected = event;
  }
  return groups;
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
  return `<details class="tool-card" data-event-key="${esc(run.id)}-${group.id}"><summary>${icon(toolIcons[name] || "spark")}<span class="tool-card-title">${esc(toolNames[name] || "Action requested")}</span><span class="tool-card-status${failed ? " error" : ""}">${esc(label)}</span>${icon("chevron")}</summary><div class="tool-card-body"><p>${esc(group.action)}</p>${args ? `<div class="tool-detail-label">${group.validated ? "Validated arguments" : "Proposed arguments"}${group.rejected ? " · not executed" : ""}</div><pre>${esc(JSON.stringify(args, null, 2))}</pre>` : ""}${group.result ? `<div class="tool-detail-label">${group.result.success ? "Tool observation" : "Tool error"}</div><pre>${esc(group.result.success ? group.result.output : group.result.error)}</pre>` : ""}${group.rejected ? `<div class="tool-detail-label">Blocked at ${esc(group.rejected.stage)}</div><pre>${esc(group.rejected.message)}</pre>` : ""}${score !== undefined ? `<div class="confidence-line">${run.mode === "demo" ? "Synthetic demo score" : "Needle confidence"}: ${Number(score).toFixed(2)}${group.confidence ? ` · gate ≥ ${Number(group.confidence.threshold).toFixed(2)}` : ""}</div>` : ""}</div></details>`;
}
function renderPending(run) {
  if (!run.pending || run.done) return "";
  const pending = run.pending,
    approval = pending.kind === "approval";
  const disabled =
    run.status === "CANCELLING" || pending.submitted ? "disabled" : "";
  return `<div class="pending-card"><div class="pending-title">${icon(approval ? "shield" : "chat")}${approval ? "Your permission is needed" : "A quick question for you"}</div><p>${esc(pending.question)}</p>${approval ? `<pre>${esc(pending.call?.arguments?.content || "(empty file)")}</pre><p class="pending-hint">This replaces the file’s contents. Nothing is written until you approve.</p><div class="pending-actions"><button class="button primary small" data-answer-run="${esc(run.id)}" data-approved="true" ${disabled}>${icon("check", true)}Allow write</button><button class="button secondary small" data-answer-run="${esc(run.id)}" data-approved="false" ${disabled}>Deny</button></div>` : '<p class="pending-hint">Type your answer in the message box below to continue.</p>'}</div>`;
}
function renderAssistant(message) {
  const run = S.runs.get(message.run_id);
  if (!run)
    return `<article class="message assistant"><div class="assistant-label"><span class="assistant-mark">${icon("needle")}</span>Needle</div><div class="assistant-body"><div class="markdown">${markdown(message.content)}</div></div></article>`;
  const groups = actionGroups(run);
  const answers = (run.events || []).filter((e) => e.type === "user_answer");
  const pending = renderPending(run);
  const working =
    !run.done && !run.pending
      ? `<div class="working"><span class="spinner"></span><span>${S.streamError && S.streamRun === run.id ? "Connection interrupted. Reconnecting…" : run.status === "CANCELLING" ? "Stopping after the current model call…" : `${esc(phaseNames[run.phase] || "Starting the agent")}…`}</span></div>`
      : "";
  return `<article class="message assistant" data-message-run="${esc(run.id)}"><div class="assistant-label"><span class="assistant-mark">${icon("needle")}</span>Needle<span class="label-mode">${run.mode === "demo" ? "DEMO" : "LIVE"}</span></div><div class="assistant-body">${groups.length ? `<div class="tool-stack">${groups.map((g) => renderToolCard(g, run)).join("")}</div>` : ""}${answers.map((e) => `<div class="answered-note">${e.kind === "approval" ? `You ${e.answer ? "approved" : "declined"} the write.` : `You: ${esc(e.answer)}`}</div>`).join("")}${pending}${working}${message.content ? `<div class="markdown">${markdown(message.content)}</div>` : ""}${run.done ? `<div class="message-meta">${statusBadge(run.status)}<span>${run.steps} tool ${run.steps === 1 ? "step" : "steps"}</span><span>·</span><span>${duration(run.elapsed_ms)}</span><span class="meta-spacer"></span><button class="icon-button inspect-run" data-run="${esc(run.id)}" aria-label="Inspect run">${icon("history")}</button><button class="icon-button copy-answer" data-run="${esc(run.id)}" aria-label="Copy answer">${icon("copy")}</button></div>` : ""}</div></article>`;
}
function renderConversation(forceScroll = false) {
  const container = $("#conversation"),
    scroll = $("#chat-scroll");
  const nearBottom =
    scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 110;
  const openCards = new Set(
    $$("details[open]", container).map((detail) => detail.dataset.eventKey),
  );
  const messages = S.conversation?.messages || [];
  $("#welcome").classList.toggle("hidden", messages.length > 0);
  container.innerHTML = messages
    .map((message) =>
      message.role === "user"
        ? `<article class="message user"><div class="user-bubble">${esc(message.content)}</div></article>`
        : renderAssistant(message),
    )
    .join("");
  $$("details", container).forEach((detail) => {
    if (openCards.has(detail.dataset.eventKey)) detail.open = true;
  });
  if (nearBottom || forceScroll) scroll.scrollTop = scroll.scrollHeight;
  updateComposer();
}
let framePending = false;
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
  const latestPhase = [...events]
    .reverse()
    .find((e) => e.type === "phase")?.node;
  const rejected = [...events].reverse().find((e) => e.type === "rejected");
  $("#activity-panel").innerHTML =
    `<div class="activity-summary">${statusBadge(run.status)}<span class="activity-mode">${run.mode === "demo" ? "DEMO RUN" : "LIVE RUN"}</span></div><div class="activity-metrics"><div class="activity-metric"><strong>${run.steps || 0}</strong><span>tool steps</span></div><div class="activity-metric"><strong>${duration(run.elapsed_ms || 0)}</strong><span>elapsed time</span></div></div><h4 class="activity-section-title">RUNTIME PIPELINE</h4><ol class="phase-list">${phases
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
  if ((run.events || []).some((previous) => previous.id === event.id)) return;
  run.events ||= [];
  run.events.push(event);
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
  updateComposer();
  scheduleRender();
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
      { headers: { "X-Needle-Session": S.token }, signal: controller.signal },
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
    S.conversations = data.conversations;
    for (const snapshot of data.runs) {
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
  try {
    const conversation = await api(
      `/api/conversations/${encodeURIComponent(id)}`,
    );
    S.conversation = conversation;
    for (const run of conversation.runs) S.runs.set(run.id, run);
    S.currentRun = conversation.runs.at(-1)?.id || null;
    stored("needle-conversation", id);
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
function startNew() {
  if (activeRun()) {
    toast("Finish or stop the active run before starting a new conversation.");
    return;
  }
  if (S.stream) S.stream.abort();
  S.conversation = null;
  S.currentRun = null;
  stored("needle-conversation", null);
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
      stored("needle-conversation", S.conversation.id);
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
    preview = `<div class="file-preview-header"><span>${icon("file", true)}${esc(file.path)}</span><div class="file-preview-actions">${!file.loading && !file.error ? `<button class="button secondary small" data-ask-file="${esc(file.path)}">${icon("spark", true)}Ask Needle</button><button class="icon-button" id="copy-file" aria-label="Copy file contents">${icon("copy")}</button>` : ""}</div></div>${file.loading ? '<div class="file-placeholder"><span class="spinner"></span><p>Reading file…</p></div>' : file.error ? `<div class="file-placeholder"><p>${esc(file.error)}</p></div>` : `<pre class="file-content">${esc(file.content)}</pre>`}`;
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
  $("#max-steps").value = s.max_tool_steps;
  $("#read-threshold").value = s.read_only_threshold;
  $("#write-threshold").value = s.confidence_threshold;
  $("#read-only").checked = s.read_only;
  $("#create-parents").checked = s.allow_create_parent_dirs;
  $("#read-only").disabled = Boolean(S.readOnlyEnforced);
  $("#connection-result").classList.add("hidden");
  updateModeExplanation();
  $("#settings-dialog").showModal();
}
function updateModeExplanation() {
  const demo = $('input[name="mode"]:checked').value === "demo";
  $("#live-settings").classList.toggle("hidden", demo);
  $("#base-url").disabled = demo;
  $("#model-name").disabled = demo;
  $("#mode-explanation").textContent = demo
    ? "A deterministic demo, not a language model. Try search, reading, arithmetic, time, and approved writes through the real runtime. Confidence scores are simulated."
    : "Your reasoning model emits natural-language intents. Needle 2 translates them into tool calls. The runtime validates and gates every action. Live mode never silently falls back to the demo.";
}
function settingsFromForm() {
  return {
    mode: $('input[name="mode"]:checked').value,
    base_url: $("#base-url").value.trim(),
    model: $("#model-name").value.trim(),
    max_tool_steps: Number($("#max-steps").value),
    read_only_threshold: Number($("#read-threshold").value),
    confidence_threshold: Number($("#write-threshold").value),
    read_only: $("#read-only").checked,
    allow_create_parent_dirs: $("#create-parents").checked,
  };
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
    "Checking the selected adapters. A first-time Needle download can take a moment.",
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
function showGuide() {
  $("#detail-title").textContent = "A small guide to Needle";
  $("#detail-content").innerHTML =
    `<div class="guide-section"><span class="guide-number">01</span><div><h3>Start with the workspace</h3><p>Try “Explore this workspace”, “Find the authentication implementation”, or “Calculate 24 * 18 + 120”. The offline demo supports these concrete tasks with real tools, not live AI reasoning.</p></div></div><div class="guide-section"><span class="guide-number">02</span><div><h3>Bring your reasoning model</h3><p>Start an OpenAI-compatible server, then choose <strong>Live models</strong> in Settings. For Ollama, an example is <code>ollama run qwen2.5:3b</code> with server URL <code>http://127.0.0.1:11434/v1</code>. The server must be reachable from the machine hosting this console. Needle downloads its small inference engine on first use; offline installation is described in the README.</p></div></div><div class="guide-section"><span class="guide-number">03</span><div><h3>Stay in the loop</h3><p>The agent can ask you questions. Reply in the composer to continue. Each write asks for your approval and displays the exact content. Stop a run anytime; an in-flight model call may finish, but no subsequent tool will execute.</p></div></div><div class="guide-section"><span class="guide-number">04</span><div><h3>Trust the boundary, inspect the work</h3><p>All paths stay in your configured workspace. No shell, Python execution, append, binary writes, or delete tools. Low-confidence or invalid calls do not execute. Use the Activity panel or expand an action card to see exactly what happened.</p></div></div><div class="guide-pipeline">REASON → PARSE → TRANSLATE → SANITIZE → VALIDATE → CONFIDENCE → SAFETY → EXECUTE → OBSERVE → UPDATE CONTEXT → REASON</div><p class="field-help">Conversations are isolated to your browser session and kept in server memory. They expire after two hours of inactivity and reset when the server restarts. Export a run trace if you want to keep it.</p>`;
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
    link.download = `needle-run-${id}.json`;
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
$("#stop-run").addEventListener("click", stopRun);
$("#menu-button").addEventListener("click", () =>
  $("#sidebar").classList.toggle("open"),
);
$("#sidebar-scrim").addEventListener("click", () =>
  $("#sidebar").classList.remove("open"),
);
$("#inspector-toggle").addEventListener("click", () =>
  $("#inspector").classList.toggle("visible"),
);
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
});
document.addEventListener("click", async (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  try {
    if (target.dataset.view) showView(target.dataset.view);
    if (target.dataset.prompt) await sendMessage(target.dataset.prompt);
    if (target.dataset.conversation)
      await openConversation(target.dataset.conversation);
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
    if (target.dataset.answerRun)
      await answerRun(target.dataset.answerRun, {
        approved: target.dataset.approved === "true",
      });
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
      if (window.innerWidth <= 1060) $("#inspector").classList.add("visible");
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
    stored("needle-session", S.token);
    S.readOnlyEnforced = data.read_only_enforced;
    S.settings = data.settings;
    S.workspace = data.workspace;
    S.tools = data.tools;
    S.conversations = data.conversations;
    for (const run of data.runs) S.runs.set(run.id, run);
    S.ready = true;
    renderSettings();
    renderSidebar();
    updateComposer();
    const previous = stored("needle-conversation");
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
boot();
