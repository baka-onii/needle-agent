"""Executable entrypoint smoke tests with the offline demo adapters."""

import json
import subprocess
import sys


def test_one_shot_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "run",
            "--demo",
            "--json",
            "Calculate 24 * 18 + 120",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["status"] == "COMPLETED"
    assert state["step_count"] == 1
    assert "552" in state["final_answer"]


def test_interactive_cli():
    result = subprocess.run(
        [sys.executable, "-m", "agent_runtime", "chat", "--demo"],
        input="Calculate 2+2\n/new\n/tools\n/exit\n",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "**4**" in result.stdout
    assert "Started a new conversation" in result.stdout
    assert "read_file" in result.stdout


def test_invalid_workspace_has_friendly_error():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "run",
            "--demo",
            "--workspace",
            "/nonexistent-needle-workspace",
            "hello",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Workspace does not exist" in result.stderr
    assert "Traceback" not in result.stderr
