"""Workspace environment facts: directory, executables, and process liveness."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import Tool, ToolError


def make_get_working_directory_tool(config: AgentConfig) -> Tool:
    def get_working_directory() -> str:
        return str(Path(config.workspace_root or Path.cwd()).resolve())

    return Tool(
        name="get_working_directory",
        description="Return the absolute workspace root. Paths stay relative to it.",
        parameters={"type": "object", "properties": {}},
        handler=get_working_directory,
    )


def make_find_executable_tool(config: AgentConfig) -> Tool:
    def find_executable(name: str) -> str:
        if not name.strip() or "\x00" in name or "/" in name or "\\" in name:
            raise ToolError("Give a bare executable name, e.g. 'git' or 'python'.")
        found = shutil.which(name.strip())
        return found if found else f"{name.strip()} was not found on PATH."

    return Tool(
        name="find_executable",
        description="Locate an executable on PATH, e.g. 'git', 'python', 'pytest'.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Bare executable name, e.g. 'git'.",
                }
            },
            "required": ["name"],
        },
        handler=find_executable,
    )


def make_process_info_tool(config: AgentConfig) -> Tool:
    def process_info(pid: int | None = None) -> str:
        if pid is None:
            current = os.getpid()
            exe = shutil.which("python") or "python"
            return f"pid: {current} (this runtime)\nexecutable: {exe}"
        if type(pid) is not int or pid <= 0:
            raise ToolError("Pid must be a positive integer.")
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        except OSError:
            alive = False
        return f"pid: {pid}\nalive: {'yes' if alive else 'no'}"

    return Tool(
        name="process_info",
        description="Check whether a process id is alive; omit pid for this runtime.",
        parameters={
            "type": "object",
            "properties": {
                "pid": {"type": ["integer", "null"], "description": "Process id, e.g. 1234."}
            },
            "required": [],
        },
        handler=process_info,
    )


def environment_tools(config: AgentConfig) -> list[Tool]:
    return [
        make_get_working_directory_tool(config),
        make_find_executable_tool(config),
        make_process_info_tool(config),
    ]
