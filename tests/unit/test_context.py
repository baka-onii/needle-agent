"""P6: context manager."""

from agent_runtime.config import AgentConfig
from agent_runtime.context.manager import ContextManager, total_chars


def _manager(budget: int = 100) -> ContextManager:
    return ContextManager(AgentConfig(max_context_chars=budget), system_prompt="SYS")


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_short_history_untouched() -> None:
    manager = _manager()
    transcript = [_msg("user", "hello"), _msg("assistant", "hi")]
    assert manager.build(transcript)[1:] == transcript


def test_trim_keeps_system_original_and_recent() -> None:
    manager = _manager(budget=40)
    transcript = [_msg("user", "request")] + [_msg("user", f"obs-{i}-xx") for i in range(6)]
    built = manager.build(transcript)
    assert built[0] == {"role": "system", "content": "SYS"}
    assert built[1] == {"role": "user", "content": "request"}
    assert built[-1]["content"] == "obs-5-xx"
    assert total_chars(built) <= 40
    assert len(built) < 1 + len(transcript)  # something was dropped


def test_trim_always_keeps_most_recent() -> None:
    manager = _manager(budget=13)
    transcript = [_msg("user", "request"), _msg("user", "huge-old-message"), _msg("user", "new")]
    built = manager.build(transcript)
    assert built[-1] == {"role": "user", "content": "new"}


def test_oversized_recent_message_is_truncated_without_mutation() -> None:
    manager = _manager(budget=40)
    transcript = [
        _msg("user", "request"),
        {"role": "user", "kind": "observation", "content": "x" * 500},
    ]
    built = manager.build(transcript)
    assert total_chars(built) <= 40
    assert built[0]["content"] == "SYS"
    assert built[1]["content"] == "request"
    assert "truncated" in built[-1]["content"]
    assert len(transcript[-1]["content"]) == 500


def test_observations_drop_before_old_reasoning() -> None:
    manager = _manager(budget=43)
    built = manager.build(
        [
            _msg("user", "request"),
            _msg("assistant", "plan"),
            {"role": "user", "content": "x" * 40, "kind": "observation"},
            _msg("assistant", "latest"),
        ]
    )
    assert [m["content"] for m in built] == ["SYS", "request", "plan", "latest"]


def test_impossible_pinned_context_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="system prompt"):
        ContextManager(AgentConfig(max_context_chars=2), "SYS")
    with pytest.raises(ValueError, match="Original request"):
        _manager(budget=10).build([_msg("user", "long user request")])


def test_current_followup_is_preserved_alongside_original() -> None:
    manager = _manager(budget=80)
    built = manager.build(
        [
            _msg("user", "Original request"),
            _msg("assistant", "Previous answer"),
            _msg("user", "The current follow-up"),
            _msg("assistant", "<tool>Read file</tool>"),
            {"role": "user", "kind": "observation", "content": "x" * 1000},
        ]
    )
    assert total_chars(built) <= 80
    assert built[1]["content"] == "Original request"
    assert any(m["content"] == "The current follow-up" for m in built)
