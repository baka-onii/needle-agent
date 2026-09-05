"""Single-user development web console, using only the standard library.

Same-origin JSON + SSE, opaque browser session tokens (works in preview iframes),
bounded requests/history, and isolated conversations. The workspace is set by the
server operator, never by browser input. Do not expose this console to untrusted
users: they are intentionally allowed to operate tools in that workspace.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_runtime import Agent, AgentConfig, ToolCall, ToolError
from agent_runtime.models.demo import DemoActionModel, DemoReasoningModel
from agent_runtime.models.needle import NeedleActionModel
from agent_runtime.models.reasoning import OpenAICompatibleReasoningModel, api_base_url
from agent_runtime.tools.filesystem import SKIP_DIRS, resolve_safe_path
from agent_runtime.tools.registry import create_default_registry

MAX_BODY_BYTES = 150_000
MAX_SESSIONS = 24
MAX_CONVERSATIONS = 24
MAX_RUNS = 60
SESSION_TTL_SECONDS = 7_200
HUMAN_TIMEOUT_SECONDS = 600


class WebError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["demo", "live"] = "live"
    base_url: str = Field(default="http://127.0.0.1:8080", max_length=2_048)
    model: str = Field(default="ornith", min_length=1, max_length=200)
    confidence_threshold: float = Field(default=0.85, ge=0.05, le=1, allow_inf_nan=False)
    read_only_threshold: float = Field(default=0.5, ge=0.05, le=1, allow_inf_nan=False)
    max_tool_steps: int = Field(default=20, ge=1, le=50)
    read_only: bool = False
    allow_create_parent_dirs: bool = False

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        api_base_url(value)
        return value.strip().rstrip("/")


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1, max_length=8_000)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank.")
        return value.strip()


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_id: str
    answer: str | None = Field(default=None, max_length=8_000)
    approved: bool | None = None


@dataclass
class Run:
    id: str
    conversation_id: str
    mode: str
    created_at: float = field(default_factory=time.time)
    status: str = "RUNNING"
    events: list[dict] = field(default_factory=list)
    pending: dict | None = None
    reply: str | bool | None = None
    done: bool = False
    steps: int = 0
    elapsed_ms: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)

    def push(self, event: dict) -> None:
        with self.condition:
            event = {**event, "id": len(self.events) + 1}
            event.setdefault("elapsed_ms", round((time.time() - self.created_at) * 1_000))
            self.events.append(event)
            if event["type"] == "tool_result":
                self.steps = event["step"]
            self.elapsed_ms = event["elapsed_ms"]
            self.condition.notify_all()

    def wait_for_user(self, kind: str, question: str, call: ToolCall | None = None) -> str | bool:
        with self.condition:
            if self.cancel.is_set():
                raise ToolError("Run cancelled.")
            self.pending = {
                "question_id": secrets.token_hex(12),
                "kind": kind,
                "question": question,
                "call": call.model_dump() if call else None,
            }
            self.reply = None
            self.status = "WAITING_FOR_INPUT"
            self.push({"type": "question", **self.pending})
            answered = self.condition.wait_for(
                lambda: self.reply is not None or self.cancel.is_set(),
                HUMAN_TIMEOUT_SECONDS,
            )
            self.pending = None
            if self.cancel.is_set():
                raise ToolError("Run cancelled while waiting for input.")
            self.status = "RUNNING"
            if not answered:
                self.push({"type": "question_expired"})
                raise ToolError("The question expired after ten minutes without a response.")
            return self.reply

    def stop(self) -> None:
        with self.condition:
            if not self.done and not self.cancel.is_set():
                self.cancel.set()
                self.status = "CANCELLING"
                self.push({"type": "cancelling"})
                self.condition.notify_all()

    def snapshot(self, include_events: bool = False) -> dict:
        with self.condition:
            data = {
                "id": self.id,
                "conversation_id": self.conversation_id,
                "mode": self.mode,
                "created_at": self.created_at,
                "status": self.status,
                "pending": self.pending,
                "done": self.done,
                "steps": self.steps,
                "elapsed_ms": self.elapsed_ms,
            }
            if include_events:
                data["events"] = list(self.events)
            return data


@dataclass
class Conversation:
    id: str
    title: str = "New conversation"
    messages: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }


@dataclass
class BrowserSession:
    token: str
    settings: Settings
    touched: float = field(default_factory=time.monotonic)
    conversations: dict[str, Conversation] = field(default_factory=dict)
    runs: dict[str, Run] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def busy(self) -> bool:
        return any(not run.done for run in self.runs.values())


class WorkspaceService:
    def __init__(self, config: AgentConfig, *, demo: bool = False, agent_factory=None):
        root = Path(config.workspace_root or Path.cwd()).resolve()
        if not root.is_dir():
            raise ValueError("Workspace must be an existing directory.")
        self.config = replace(config, workspace_root=str(root))
        self.default_settings = Settings(
            mode="demo" if demo else "live",
            base_url=config.llm_base_url,
            model=config.llm_model,
            confidence_threshold=config.confidence_threshold,
            read_only_threshold=config.read_only_threshold,
            max_tool_steps=config.max_tool_steps,
            read_only=config.read_only,
            allow_create_parent_dirs=config.allow_create_parent_dirs,
        )
        self.sessions: dict[str, BrowserSession] = {}
        self.lock = threading.RLock()
        self.agent_factory = agent_factory or self._make_agent

    def session(self, token: str | None, *, create: bool = False) -> BrowserSession:
        with self.lock:
            now = time.monotonic()
            for key, existing in list(self.sessions.items()):
                if now - existing.touched > SESSION_TTL_SECONDS:
                    for run in existing.runs.values():
                        run.stop()
                    del self.sessions[key]
            if token and token in self.sessions:
                self.sessions[token].touched = now
                return self.sessions[token]
            if not create:
                raise WebError(401, "Session expired. Reload the workspace to reconnect.")
            if len(self.sessions) >= MAX_SESSIONS:
                raise WebError(503, "Too many browser sessions. Try again later.")
            token = secrets.token_urlsafe(32)
            session = BrowserSession(token, self.default_settings.model_copy())
            self.sessions[token] = session
            return session

    def run_config(self, settings: Settings) -> AgentConfig:
        # A browser may not redirect a server-owned API key to a different provider.
        if self.config.llm_api_key and settings.mode == "live":
            configured = urlsplit(api_base_url(self.config.llm_base_url))
            requested = urlsplit(api_base_url(settings.base_url))
            if (configured.scheme, configured.netloc) != (requested.scheme, requested.netloc):
                raise WebError(
                    403,
                    "The API key is bound to the server-configured model origin. "
                    "Change NEEDLE_LLM_BASE_URL on the server to switch providers.",
                )
        return replace(
            self.config,
            llm_base_url=settings.base_url,
            llm_model=settings.model,
            confidence_threshold=settings.confidence_threshold,
            read_only_threshold=settings.read_only_threshold,
            max_tool_steps=settings.max_tool_steps,
            read_only=self.config.read_only or settings.read_only,
            allow_create_parent_dirs=settings.allow_create_parent_dirs,
        )

    def _make_agent(self, settings: Settings, run: Run) -> Agent:
        models = (
            {"reasoning": DemoReasoningModel(), "action": DemoActionModel()}
            if settings.mode == "demo"
            else {}
        )
        return Agent(
            self.run_config(settings),
            **models,
            ask_fn=lambda question: str(run.wait_for_user("question", question)),
            approve_fn=lambda call: (
                run.wait_for_user(
                    "approval",
                    f"Allow writing to {call.arguments['path']}?",
                    call,
                )
                is True
            ),
        )

    def bootstrap(self, session: BrowserSession) -> dict:
        registry = create_default_registry(self.run_config(session.settings))
        with session.lock:
            return {
                "session_token": session.token,
                "settings": session.settings.model_dump(),
                "workspace": {
                    "name": Path(self.config.workspace_root).name,
                    "path": self.config.workspace_root,
                },
                "api_key_configured": bool(self.config.llm_api_key),
                "read_only_enforced": self.config.read_only,
                "tools": [tool.needle_schema() for tool in registry.list()],
                "conversations": [c.summary() for c in session.conversations.values()],
                "runs": [run.snapshot() for run in session.runs.values()],
            }

    def conversation(self, session: BrowserSession, conversation_id: str) -> Conversation:
        try:
            return session.conversations[conversation_id]
        except KeyError as exc:
            raise WebError(404, "Conversation not found.") from exc

    def get_run(self, session: BrowserSession, run_id: str) -> Run:
        try:
            return session.runs[run_id]
        except KeyError as exc:
            raise WebError(404, "Run not found.") from exc

    def new_conversation(self, session: BrowserSession) -> Conversation:
        with session.lock:
            if len(session.conversations) >= MAX_CONVERSATIONS:
                oldest = next(iter(session.conversations))
                if any(r.conversation_id == oldest and not r.done for r in session.runs.values()):
                    raise WebError(409, "Finish the active conversation first.")
                del session.conversations[oldest]
            conversation = Conversation(secrets.token_hex(12))
            session.conversations[conversation.id] = conversation
            return conversation

    def start_run(self, session: BrowserSession, conversation_id: str, message: str) -> Run:
        with session.lock:
            if session.busy():
                raise WebError(409, "Finish or stop the active run before sending another message.")
            conversation = self.conversation(session, conversation_id)
            run = Run(secrets.token_hex(12), conversation_id, session.settings.mode)
            session.runs[run.id] = run
            while len(session.runs) > MAX_RUNS:
                del session.runs[next(iter(session.runs))]
            if not conversation.messages:
                conversation.title = message[:48] + ("…" if len(message) > 48 else "")
            conversation.updated_at = time.time()
            conversation.messages.extend(
                [
                    {"role": "user", "content": message, "run_id": run.id},
                    {"role": "assistant", "content": "", "run_id": run.id},
                ]
            )
            conversation.messages = conversation.messages[-80:]
            settings = session.settings.model_copy()
            thread = threading.Thread(
                target=self._work,
                args=(session, conversation, run, settings, message),
                daemon=True,
            )
            thread.start()
            return run

    def _work(self, session, conversation, run, settings, message) -> None:
        def receive(event: dict) -> None:
            if event["type"] != "complete":
                run.push(event)
                return
            state = event["state"]
            with session.lock:
                conversation.history = state["messages"]
                for item in reversed(conversation.messages):
                    if item["role"] == "assistant" and item["run_id"] == run.id:
                        item["content"] = state["final_answer"] or "(no response)"
                        item["status"] = state["status"]
                        break
                conversation.updated_at = time.time()
            with run.condition:
                run.status, run.steps = state["status"], state["step_count"]
                run.pending, run.done = None, True
                run.push(
                    {
                        "type": "complete",
                        "status": run.status,
                        "final_answer": state["final_answer"] or "(no response)",
                        "steps": run.steps,
                        "elapsed_ms": event["elapsed_ms"],
                    }
                )

        agent = None
        try:
            agent = self.agent_factory(settings, run)
            agent.run(
                message, history=conversation.history, on_event=receive, cancelled=run.cancel.is_set
            )
        except Exception as exc:
            receive(
                {
                    "type": "complete",
                    "elapsed_ms": round((time.time() - run.created_at) * 1000),
                    "state": {
                        "messages": conversation.history,
                        "step_count": run.steps,
                        "status": "ERROR",
                        "final_answer": f"Could not start the agent: {exc}",
                    },
                }
            )
        finally:
            if agent is not None:
                agent.close()

    def answer(self, run: Run, answer: Answer) -> None:
        with run.condition:
            pending = run.pending
            if run.done or run.cancel.is_set() or not pending or run.reply is not None:
                raise WebError(409, "This run is no longer waiting for an answer.")
            if not secrets.compare_digest(pending["question_id"], answer.question_id):
                raise WebError(409, "That question is no longer active.")
            if pending["kind"] == "approval":
                if answer.approved is None or answer.answer is not None:
                    raise WebError(400, "A write approval requires a boolean approved value.")
                run.reply = answer.approved
            else:
                if (
                    answer.answer is None
                    or not answer.answer.strip()
                    or answer.approved is not None
                ):
                    raise WebError(400, "A nonempty text answer is required.")
                run.reply = answer.answer.strip()
            run.push(
                {
                    "type": "user_answer",
                    "question_id": answer.question_id,
                    "answer": run.reply,
                    "kind": pending["kind"],
                }
            )
            run.condition.notify_all()

    def list_files(self, path: str) -> dict:
        root = resolve_safe_path(path, self.config.workspace_root)
        if not root.is_dir():
            raise WebError(404, "Directory not found.")
        children = []
        for entry in sorted(root.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if entry.name in SKIP_DIRS or entry.is_symlink():
                continue
            try:
                safe = resolve_safe_path(str(entry), self.config.workspace_root)
                rel = entry.relative_to(self.config.workspace_root).as_posix()
                children.append(
                    {
                        "name": entry.name,
                        "path": rel,
                        "type": "directory" if safe.is_dir() else "file",
                        "size": safe.stat().st_size if safe.is_file() else None,
                    }
                )
            except (ToolError, OSError):
                continue
            if len(children) >= 200:
                break
        children.sort(key=lambda item: (item["type"] != "directory", item["name"].casefold()))
        return {"path": path, "entries": children, "limited": len(children) == 200}

    def check_connection(self, settings: Settings) -> dict:
        if settings.mode == "demo":
            return {
                "ok": True,
                "message": "Offline demo is ready. Simulated planning and confidence; real tools.",
            }
        config = self.run_config(settings)
        reasoning = OpenAICompatibleReasoningModel(
            config.llm_base_url,
            config.llm_model,
            api_key=config.llm_api_key,
        )
        try:
            models = reasoning.check_connection()
        except Exception as exc:
            return {"ok": False, "message": str(exc)}
        needle = NeedleActionModel(
            create_default_registry(config).list(), weights=config.needle_weights
        )
        try:
            needle.prepare()
        except Exception:
            return {
                "ok": False,
                "models": models,
                "message": (
                    "Reasoning is reachable, but Needle could not initialize. Check the server "
                    "logs, network access to Hugging Face, or the NEEDLE_LIB_PATH offline setup."
                ),
            }
        finally:
            needle.close()
        return {"ok": True, "models": models, "message": "Reasoning server and Needle are ready."}

    def close(self) -> None:
        with self.lock:
            for session in self.sessions.values():
                for run in session.runs.values():
                    run.stop()


def make_server(service: WorkspaceService, host: str = "0.0.0.0", port: int = 3000):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Needle/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # No request bodies, headers, tokens, or conversations in HTTP logs.
            pass

        def headers_for(self, status: int, content_type: str, length: int | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; "
                "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors *",
            )
            if length is not None:
                self.send_header("Content-Length", str(length))

        def json(self, data: dict, status: int = 200) -> None:
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.headers_for(status, "application/json; charset=utf-8", len(raw))
            self.end_headers()
            self.wfile.write(raw)

        def body(self) -> dict:
            if self.headers.get("Transfer-Encoding"):
                raise WebError(400, "Chunked request bodies are not supported.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise WebError(400, "Invalid Content-Length.") from exc
            if not 0 <= length <= MAX_BODY_BYTES:
                raise WebError(413, "Request body is too large.")
            if length and self.headers.get_content_type() != "application/json":
                raise WebError(415, "Use application/json.")
            try:
                data = json.loads(self.rfile.read(length)) if length else {}
            except (ValueError, UnicodeError) as exc:
                raise WebError(400, "Invalid JSON.") from exc
            if not isinstance(data, dict):
                raise WebError(400, "Expected a JSON object.")
            return data

        def browser_session(self, create: bool = False) -> BrowserSession:
            # No cookies: third-party-cookie restrictions would break embedded previews.
            return service.session(self.headers.get("X-Needle-Session"), create=create)

        def do_GET(self) -> None:  # noqa: N802
            self.dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self.dispatch("POST")

        def dispatch(self, method: str) -> None:
            try:
                parsed = urlsplit(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if method == "GET" and path == "/health":
                    self.json({"status": "ok"})
                    return
                if method == "GET" and path in {"/", "/app.js", "/style.css", "/favicon.svg"}:
                    name = "index.html" if path == "/" else path[1:]
                    data = files("agent_runtime").joinpath("web", name).read_bytes()
                    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    self.headers_for(200, content_type + "; charset=utf-8", len(data))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if method == "GET" and path == "/api/session":
                    self.json(service.bootstrap(self.browser_session(create=True)))
                    return
                if not path.startswith("/api/"):
                    raise WebError(404, "Not found.")
                # Token + JSON content type + no permissive CORS protects browser mutations.
                session = self.browser_session()
                if method == "POST" and self.headers.get("Sec-Fetch-Site") == "cross-site":
                    raise WebError(403, "Cross-site requests are not allowed.")
                if path == "/api/conversations" and method == "POST":
                    self.body()
                    self.json(service.new_conversation(session).summary(), 201)
                    return
                if path == "/api/settings" and method == "POST":
                    settings = Settings.model_validate(self.body())
                    service.run_config(settings)
                    with session.lock:
                        if session.busy():
                            raise WebError(409, "Stop the active run before changing settings.")
                        if service.config.read_only and not settings.read_only:
                            raise WebError(403, "The server was started in read-only mode.")
                        session.settings = settings
                    self.json({"settings": settings.model_dump()})
                    return
                if path == "/api/connection" and method == "POST":
                    settings = Settings.model_validate(self.body())
                    self.json(service.check_connection(settings))
                    return
                if path == "/api/files" and method == "GET":
                    self.json(service.list_files(query.get("path", ["."])[0]))
                    return
                if path == "/api/file" and method == "GET":
                    filename = query.get("path", [""])[0]
                    tool = create_default_registry(service.config).get("read_file")
                    self.json({"path": filename, "content": tool.handler(path=filename)})
                    return
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "conversations"]:
                    conversation = service.conversation(session, parts[2])
                    if len(parts) == 3 and method == "GET":
                        with session.lock:
                            self.json(
                                {
                                    **conversation.summary(),
                                    "messages": conversation.messages,
                                    "runs": [
                                        r.snapshot(True)
                                        for r in session.runs.values()
                                        if r.conversation_id == conversation.id
                                    ],
                                }
                            )
                        return
                    if len(parts) == 4 and parts[3] == "runs" and method == "POST":
                        prompt = Prompt.model_validate(self.body())
                        run = service.start_run(session, conversation.id, prompt.message)
                        self.json(run.snapshot(), 202)
                        return
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    run = service.get_run(session, parts[2])
                    if len(parts) == 3 and method == "GET":
                        self.json(run.snapshot(True))
                        return
                    if len(parts) == 4:
                        if parts[3] == "events" and method == "GET":
                            try:
                                after = int(query.get("after", ["0"])[0])
                            except ValueError as exc:
                                raise WebError(400, "Invalid event cursor.") from exc
                            if after < 0 or after > len(run.events):
                                raise WebError(400, "Event cursor is outside this run.")
                            self.events(run, after)
                            return
                        if parts[3] == "cancel" and method == "POST":
                            self.body()
                            run.stop()
                            self.json({"ok": True, "status": run.status})
                            return
                        if parts[3] == "answer" and method == "POST":
                            service.answer(run, Answer.model_validate(self.body()))
                            self.json({"ok": True})
                            return
                raise WebError(404, "Not found.")
            except WebError as exc:
                self.json({"error": exc.message}, exc.status)
            except ValidationError as exc:
                error = exc.errors(include_input=False, include_url=False)[0]
                self.json({"error": f"{'.'.join(map(str, error['loc']))}: {error['msg']}"}, 400)
            except ToolError as exc:
                self.json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A disconnected viewer does not implicitly cancel its run.
            except OSError:
                self.json({"error": "The filesystem operation failed."}, 400)
            except Exception:
                self.json({"error": "Internal server error."}, 500)

        def events(self, run: Run, cursor: int) -> None:
            self.headers_for(HTTPStatus.OK, "text/event-stream; charset=utf-8")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                with run.condition:
                    run.condition.wait_for(
                        lambda cursor=cursor: len(run.events) > cursor or run.done, timeout=10
                    )
                    batch, done = run.events[cursor:], run.done
                if batch:
                    for event in batch:
                        data = json.dumps(event, ensure_ascii=False, allow_nan=False)
                        self.wfile.write(f"id: {event['id']}\ndata: {data}\n\n".encode())
                    cursor += len(batch)
                else:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                if done:
                    break

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(
    config: AgentConfig, *, demo: bool = False, host: str = "0.0.0.0", port: int = 3000
) -> None:
    service = WorkspaceService(config, demo=demo)
    server = make_server(service, host, port)
    print(f"Needle workspace: http://{host}:{server.server_port}", flush=True)
    print(
        f"Mode: {'offline demo (simulated models, real tools)' if demo else 'live models'}",
        flush=True,
    )
    print(f"Workspace: {service.config.workspace_root}", flush=True)
    print("Personal development console — only expose it to trusted users.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        server.server_close()
