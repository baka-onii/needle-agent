"""Structured compression: shape validation and thresholds."""

from relay.context.summarize import (
    parse_summary,
    render_summary,
    should_compress,
)

VALID = """task:
  objective: "Add OAuth authentication"

constraints:
  - "Keep existing JWT authentication"

completed:
  - "Refactored token validation"

current:
  subtask: "Implement Google provider"

files:
  primary:
    - src/auth/google.py
  related:
    - src/auth/auth.py

decisions:
  - "Use existing JWT implementation"

blockers: []
"""


def test_parse_and_render_round_trip():
    data = parse_summary(VALID)
    assert data is not None
    assert data["objective"] == "Add OAuth authentication"
    assert data["constraints"] == ["Keep existing JWT authentication"]
    assert data["completed"] == ["Refactored token validation"]
    assert data["subtask"] == "Implement Google provider"
    assert data["files_primary"] == ["src/auth/google.py"]
    assert data["files_related"] == ["src/auth/auth.py"]
    assert data["decisions"] == ["Use existing JWT implementation"]
    assert data["blockers"] == []
    rendered = render_summary(data)
    assert parse_summary(rendered) == data


def test_parse_rejects_unusable_shapes():
    assert parse_summary("Just some prose about work.") is None
    assert parse_summary("task:\n  objective: \"\"\ncurrent:\n  subtask: \"x\"") is None
    assert parse_summary("task:\n  objective: \"x\"\ncurrent:\n  subtask: \"\"") is None
    # Missing sections default to empty but the core must exist.
    minimal = "task:\n  objective: \"x\"\ncurrent:\n  subtask: \"y\"\n"
    data = parse_summary(minimal)
    assert data is not None and data["constraints"] == []


def test_should_compress_thresholds():
    assert should_compress(100, 9000, 32000, 10000)
    assert not should_compress(100, 7999, 32000, 10000)
    assert should_compress(26000, None, 32000, 10000)
    assert not should_compress(100, None, 32000, 10000)
