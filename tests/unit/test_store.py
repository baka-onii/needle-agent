"""SQLite conversation persistence: round-trips, expiry, deletes, corruption."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from relay import AgentConfig
from relay.server import Run, WorkspaceService
from relay.store import SESSION_TTL_SECONDS


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# project\n")
    return tmp_path


def _service(root: Path, db: str) -> WorkspaceService:
    return WorkspaceService(AgentConfig(workspace_root=str(root)), demo=True, store_path=db)


def test_round_trip_restores_conversations_and_runs(root: Path, tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.db")
    service = _service(root, db)
    session = service.session(None, create=True)
    conversation = service.new_conversation(session)
    conversation.messages.append({"role": "user", "content": "hi", "run_id": "r"})
    conversation.history.append({"role": "user", "content": "hi"})
    run = Run("r", conversation.id, "demo")
    run.status, run.steps, run.done = "COMPLETED", 2, True
    session.runs[run.id] = run
    service._remember(session)
    assert service.store is not None
    service.store.save_run(run.snapshot())
    service.close()

    revived = _service(root, db)
    assert revived.sessions.keys() == {session.token}
    restored = revived.sessions[session.token]
    assert restored.conversations[conversation.id].messages == conversation.messages
    assert restored.conversations[conversation.id].history == conversation.history
    saved = restored.runs["r"]
    assert saved.done and saved.status == "COMPLETED" and saved.steps == 2
    revived.close()


def test_expired_and_corrupt_sessions_are_dropped(root: Path, tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.db")
    service = _service(root, db)
    session = service.session(None, create=True)
    service.new_conversation(session)
    service._remember(session)
    service.close()

    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE sessions SET touched=?", (time.time() - SESSION_TTL_SECONDS - 10,)
    )
    raw.execute(
        "INSERT INTO sessions(token, touched, settings_json) VALUES(?,?,?)",
        ("broken", time.time(), "{not json"),
    )
    raw.commit()
    raw.close()

    revived = _service(root, db)
    assert session.token not in revived.sessions
    assert "broken" not in revived.sessions
    revived.close()


def test_delete_conversation_cascades(root: Path, tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.db")
    service = _service(root, db)
    session = service.session(None, create=True)
    conversation = service.new_conversation(session)
    service._remember(session)
    raw = sqlite3.connect(db)
    raw.execute(
        "INSERT INTO runs(id, conversation_id, snapshot_json, updated_at) VALUES(?,?,?,?)",
        ("run-1", conversation.id, json.dumps({"id": "run-1"}), time.time()),
    )
    raw.commit()
    service.delete_conversation(session, conversation.id)
    remaining = raw.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    remaining += raw.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    raw.close()
    assert remaining == 0
    service.close()


def test_unusable_store_path_disables_backup(root: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    service = _service(root, str(blocker / "sessions.db"))
    assert service.store is None
    session = service.session(None, create=True)
    assert session.token
    service.close()
