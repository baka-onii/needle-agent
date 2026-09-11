"""Bounded subprocess execution inside the workspace. No shell, ever.

Commands run with argv lists (run_process) or fixed interpreters (run_python,
run_powershell), a configurable timeout, and truncated output. Anything
outside the workspace root must be reached through explicit absolute paths;
the working directory is always the workspace.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from relay.config import AgentConfig
from relay.tools.base import Tool, ToolError, truncate_text

MAX_EXEC_TIMEOUT_S = 300


def _run(argv: list[str], root: Path, timeout_s: int, limit: int) -> str:
    if type(timeout_s) is not int or not 1 <= timeout_s <= MAX_EXEC_TIMEOUT_S:
        raise ToolError(f"Timeout must be 1-{MAX_EXEC_TIMEOUT_S} seconds.")
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"Executable not found: {argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"Command timed out after {timeout_s}s.") from exc
    except OSError as exc:
        raise ToolError(f"Cannot start process: {exc.strerror}") from exc
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or output or "no output"
        raise ToolError(f"Exit {completed.returncode}: {truncate_text(detail, 2000)}")
    if completed.stderr and completed.stderr.strip():
        output = (output + "\n[stderr]\n" + completed.stderr.strip()).strip()
    return truncate_text(output or "(no output)", limit)


def _timeout_schema() -> dict:
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_EXEC_TIMEOUT_S,
        "description": "Timeout in seconds (default 30).",
    }


def make_run_python_tool(config: AgentConfig) -> Tool:
    def run_python(code: str, timeout_s: int = 30) -> str:
        if not code.strip():
            raise ToolError("Code must not be empty.")
        root = Path(config.workspace_root or Path.cwd()).resolve()
        return _run(
            [sys.executable, "-c", code], root, timeout_s, config.max_tool_output_chars
        )

    return Tool(
        name="run_python",
        description="Execute inline Python source code with the workspace interpreter.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Python source to run, e.g. 'print(2 + 2)'.",
                },
                "timeout_s": _timeout_schema(),
            },
            "required": ["code"],
        },
        handler=run_python,
        payload_args=("code",),
    )


def make_run_powershell_tool(config: AgentConfig) -> Tool:
    def run_powershell(command: str, timeout_s: int = 30) -> str:
        if not command.strip():
            raise ToolError("Command must not be empty.")
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            raise ToolError("No PowerShell interpreter found on PATH.")
        root = Path(config.workspace_root or Path.cwd()).resolve()
        return _run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", command],
            root,
            timeout_s,
            config.max_tool_output_chars,
        )

    return Tool(
        name="run_powershell",
        description="Run one PowerShell command in the workspace (no profiles, non-interactive).",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Command line, e.g. 'Get-ChildItem src'.",
                },
                "timeout_s": _timeout_schema(),
            },
            "required": ["command"],
        },
        handler=run_powershell,
        payload_args=("command",),
    )


def make_run_process_tool(config: AgentConfig) -> Tool:
    def run_process(command: str, timeout_s: int = 30) -> str:
        if not command.strip():
            raise ToolError("Command must not be empty.")
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise ToolError(f"Cannot parse command: {exc}") from exc
        if not argv:
            raise ToolError("Command must not be empty.")
        if os.path.dirname(argv[0]) in ("", ".") and shutil.which(argv[0]) is None:
            raise ToolError(f"Executable not found on PATH: {argv[0]!r}")
        root = Path(config.workspace_root or Path.cwd()).resolve()
        return _run(argv, root, timeout_s, config.max_tool_output_chars)

    return Tool(
        name="run_process",
        description="Run one executable with arguments in the workspace. No shell parsing.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Quoted command line, e.g. 'pytest tests/unit -q'.",
                },
                "timeout_s": _timeout_schema(),
            },
            "required": ["command"],
        },
        handler=run_process,
        payload_args=("command",),
    )


def execution_tools(config: AgentConfig) -> list[Tool]:
    return [
        make_run_python_tool(config),
        make_run_powershell_tool(config),
        make_run_process_tool(config),
    ]
