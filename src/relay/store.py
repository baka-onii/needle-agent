"""Write-through SQLite backup for browser sessions.

Conversations survive server restarts; per-token model deltas are deliberately
excluded (capped at megabytes in memory and useless on reload). Storage
failures never break a run: callers treat the store as best-effort backup
while memory stays the hot path. Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SESSION_TTL_SECONDS = 7_200
MAX_STORED_RUNS_PER_CONVERSATION = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    touched REAL NOT NULL,
    settings_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_token TEXT NOT NULL,
    title TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    history_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_token);
CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id);
"""


def default_db_path() -> str:
    override = os.environ.get("RELAY_SESSIONS_DB")
    if override:
        return override
    return str(Path.home() / ".relay" / "sessions.db")


class ConversationStore:
    def __init__(self, path: str | None = None) -> None:
        self._path = path if path is not None else default_db_path()
        self._lock = threading.RLock()
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def save_session(self, token: str, settings_json: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions(token, touched, settings_json) VALUES(?,?,?) "
                "ON CONFLICT(token) DO UPDATE SET touched=excluded.touched, "
                "settings_json=excluded.settings_json",
                (token, time.time(), settings_json),
            )
            self._db.commit()

    def delete_session(self, token: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM sessions WHERE token=?", (token,))
            self._db.commit()

    def save_conversation(
        self,
        conversation_id: str,
        session_token: str,
        title: str,
        messages: list[dict[str, Any]],
        history: list[dict[str, Any]],
        updated_at: float,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO conversations(id, session_token, title, messages_json, "
                "history_json, updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET session_token=excluded.session_token, "
                "title=excluded.title, messages_json=excluded.messages_json, "
                "history_json=excluded.history_json, updated_at=excluded.updated_at",
                (
                    conversation_id,
                    session_token,
                    title,
                    json.dumps(messages),
                    json.dumps(history),
                    updated_at,
                ),
            )
            self._db.commit()

    def save_run(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO runs(id, conversation_id, snapshot_json, updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "snapshot_json=excluded.snapshot_json, updated_at=excluded.updated_at",
                (
                    snapshot["id"],
                    snapshot["conversation_id"],
                    json.dumps(snapshot),
                    time.time(),
                ),
            )
            self._db.execute(
                "DELETE FROM runs WHERE conversation_id=? AND id NOT IN "
                "(SELECT id FROM runs WHERE conversation_id=? "
                "ORDER BY updated_at DESC LIMIT ?)",
                (
                    snapshot["conversation_id"],
                    snapshot["conversation_id"],
                    MAX_STORED_RUNS_PER_CONVERSATION,
                ),
            )
            self._db.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM runs WHERE conversation_id=?", (conversation_id,))
            self._db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            self._db.commit()

    def load(self) -> dict[str, Any]:
        """Sessions with settings, conversations, and run snapshots.

        Expired sessions (by wall clock) are dropped with their data.
        Corrupt rows are skipped, never fatal.
        """
        now = time.time()
        with self._lock:
            sessions: dict[str, Any] = {}
            expired: list[str] = []
            for token, touched, settings_json in self._db.execute(
                "SELECT token, touched, settings_json FROM sessions"
            ):
                if now - touched > SESSION_TTL_SECONDS:
                    expired.append(token)
                    continue
                try:
                    sessions[token] = json.loads(settings_json)
                except ValueError:
                    expired.append(token)
            for token in expired:
                self._db.execute("DELETE FROM sessions WHERE token=?", (token,))
            if expired:
                self._db.execute(
                    "DELETE FROM conversations WHERE session_token NOT IN "
                    "(SELECT token FROM sessions)"
                )
                self._db.execute(
                    "DELETE FROM runs WHERE conversation_id NOT IN "
                    "(SELECT id FROM conversations)"
                )
                self._db.commit()
            valid_tokens = set(sessions)
            conversations = []
            for row in self._db.execute(
                "SELECT id, session_token, title, messages_json, history_json, "
                "updated_at FROM conversations"
            ):
                conv_id, token, title, messages_json, history_json, updated = row
                if token not in valid_tokens:
                    continue
                try:
                    conversations.append(
                        {
                            "id": conv_id,
                            "session_token": token,
                            "title": title,
                            "messages": json.loads(messages_json),
                            "history": json.loads(history_json),
                            "updated_at": updated,
                        }
                    )
                except ValueError:
                    continue
            runs = []
            conv_ids = {conv["id"] for conv in conversations}
            for run_id, conv_id, snapshot_json in self._db.execute(
                "SELECT id, conversation_id, snapshot_json FROM runs"
            ):
                if conv_id not in conv_ids:
                    continue
                try:
                    snapshot = json.loads(snapshot_json)
                except ValueError:
                    continue
                if isinstance(snapshot, dict) and snapshot.get("id") == run_id:
                    runs.append(snapshot)
            return {"sessions": sessions, "conversations": conversations, "runs": runs}
