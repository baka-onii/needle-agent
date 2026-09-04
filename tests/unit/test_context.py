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
    manager = _manager(budget=10)
    transcript = [_msg("user", "request"), _msg("user", "huge-old-message"), _msg("user", "new")]
    built = manager.build(transcript)
    assert built[-1] == {"role": "user", "content": "new"}
