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


def test_live_refuses_without_server_when_start_disabled(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "live",
            "--workspace",
            str(tmp_path),
            "--fg-port",
            "8999",
            "--no-server-start",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "8999" in result.stderr and "no-server-start" in result.stderr


def test_live_requires_a_model_path(tmp_path, monkeypatch):
    monkeypatch.delenv("FG_GGUF", raising=False)
    monkeypatch.setenv("NEEDLE_REPO_ROOT", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "live",
            "--workspace",
            str(tmp_path),
            "--fg-port",
            "8999",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "--fg-gguf" in result.stderr


def test_live_finds_repo_server_and_model():
    import pytest

    from agent_runtime import cli

    server, gguf = cli._repo_llama_server(), cli._repo_gguf()
    if not server or not gguf:
        pytest.skip("translator binary/weights not built here")
    assert server.endswith("llama-server.exe") and gguf.endswith(".gguf")


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


def test_config_init_show_and_run(tmp_path):
    from agent_runtime.config import parse_config

    path = tmp_path / "needle.toml"
    result = subprocess.run(
        [sys.executable, "-m", "agent_runtime", "config", "init", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "prompts" / "confirmation.md").is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "config",
            "show",
            "--config",
            str(path),
            "--demo",
            "--max-tool-steps",
            "7",
            "--needle-max-tokens",
            "3000",
            "--set",
            "max_search_results=9",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    values = parse_config(result.stdout)
    assert values["max_tool_steps"] == 7 and values["max_search_results"] == 9
    assert values["mode"] == "demo" and values["needle_max_tokens"] == 3000
    portable = tmp_path / "portable.toml"
    portable.write_text(result.stdout)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "run",
            "--config",
            str(portable),
            "--json",
            "Calculate 6*7",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0 and "42" in json.loads(result.stdout)["final_answer"]


def test_cli_prompt_files_and_bad_keys(tmp_path):
    prompt = tmp_path / "custom.md"
    prompt.write_text("CUSTOM CLI INSTRUCTIONS")
    for option in ("--reasoning-prompt", "--translator-prompt", "--confirmation-prompt"):
        result = subprocess.run(
            [sys.executable, "-m", "agent_runtime", "config", "show", option, str(prompt)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0 and "CUSTOM CLI INSTRUCTIONS" in result.stdout
    result = subprocess.run(
        [sys.executable, "-m", "agent_runtime", "run", "--demo", "--set", "max_stallz=4", "Hello"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1 and "Unknown settings" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_write_runs_without_approval(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "run",
            "--demo",
            "--workspace",
            str(tmp_path),
            'Write "hello from CLI" to result.txt',
        ],
        input="y\n",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Allow this action?" not in result.stdout
    assert "Allow this write?" not in result.stdout
    assert (tmp_path / "result.txt").read_text() == "hello from CLI"
