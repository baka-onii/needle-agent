"""Read-only-first git inspection plus staged, explicit mutations.

Status, diff, log, show, and branch listing never change the repository.
Stage, commit, and checkout mutate and therefore use the strict gate; checkout
only switches to an existing local branch, never restores paths.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from relay.config import AgentConfig
from relay.tools.base import Tool, ToolError, truncate_text
from relay.tools.filesystem import _path_schema

_GIT_TIMEOUT_S = 30


def _git(root: Path, limit: int, *args: str) -> str:
    if shutil.which("git") is None:
        raise ToolError("The git executable was not found on PATH.")
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError("git timed out after 30s.") from exc
    except OSError as exc:
        raise ToolError(f"Cannot start git: {exc.strerror}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or "no output"
        raise ToolError(f"git failed: {truncate_text(detail, 1000)}")
    return truncate_text((completed.stdout or "").strip() or "(no output)", limit)


def _root(config: AgentConfig) -> Path:
    return Path(config.workspace_root or Path.cwd()).resolve()


def make_git_status_tool(config: AgentConfig) -> Tool:
    def git_status() -> str:
        return _git(_root(config), config.max_tool_output_chars, "status", "--short", "--branch")

    return Tool(
        name="git_status",
        description="Show branch and short working-tree status. Read-only.",
        parameters={"type": "object", "properties": {}},
        handler=git_status,
    )


def make_git_diff_tool(config: AgentConfig) -> Tool:
    def git_diff(path: str = ".", revision: str = "") -> str:
        argv = ["diff"]
        if revision.strip():
            argv.append(revision.strip())
        argv += ["--", path or "."]
        return _git(_root(config), config.max_tool_output_chars, *argv)

    return Tool(
        name="git_diff",
        description="Show the unstaged diff, optionally against REVISION, under PATH.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Path to diff. Use '.' for everything.", "."),
                "revision": {
                    "type": "string",
                    "description": "Optional revision to diff against, e.g. 'HEAD~1'.",
                },
            },
            "required": [],
        },
        handler=git_diff,
    )


def make_git_log_tool(config: AgentConfig) -> Tool:
    def git_log(limit: int = 10, path: str = ".") -> str:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ToolError("Limit must be 1-100.")
        return _git(
            _root(config),
            config.max_tool_output_chars,
            "log",
            "--oneline",
            "-n",
            str(limit),
            "--",
            path or ".",
        )

    return Tool(
        name="git_log",
        description="Show the commit history as one-line summaries. Read-only.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "path": _path_schema("Path to filter by. Use '.' for everything.", "."),
            },
            "required": [],
        },
        handler=git_log,
    )


def make_git_show_tool(config: AgentConfig) -> Tool:
    def git_show(revision: str, path: str = "") -> str:
        if not revision.strip():
            raise ToolError("Revision must not be empty.")
        argv = ["show", "--stat", revision.strip()]
        if path.strip():
            argv += ["--", path.strip()]
        return _git(_root(config), config.max_tool_output_chars, *argv)

    return Tool(
        name="git_show",
        description="Show a commit summary (files changed), optionally for one PATH.",
        parameters={
            "type": "object",
            "properties": {
                "revision": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Commit, e.g. 'HEAD' or a hash.",
                },
                "path": {"type": "string", "description": "Optional path to narrow the view."},
            },
            "required": ["revision"],
        },
        handler=git_show,
    )


def make_git_branch_list_tool(config: AgentConfig) -> Tool:
    def git_branch_list() -> str:
        return _git(_root(config), config.max_tool_output_chars, "branch", "--list")

    return Tool(
        name="git_branch_list",
        description="List local branches; the current one is marked with *. Read-only.",
        parameters={"type": "object", "properties": {}},
        handler=git_branch_list,
    )


def make_git_stage_tool(config: AgentConfig) -> Tool:
    def git_stage(paths: str) -> str:
        entries = [entry.strip() for entry in paths.split(",") if entry.strip()]
        if not entries:
            raise ToolError("Give at least one path to stage.")
        for entry in entries:
            if "\x00" in entry or entry.startswith("-"):
                raise ToolError(f"Invalid path: {entry!r}")
        return _git(_root(config), config.max_tool_output_chars, "add", "--", *entries)

    return Tool(
        name="git_stage",
        description="Stage files for commit. Comma-separated workspace paths.",
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Comma-separated paths, e.g. 'src/a.py, README.md'.",
                }
            },
            "required": ["paths"],
        },
        handler=git_stage,
    )


def make_git_commit_tool(config: AgentConfig) -> Tool:
    def git_commit(message: str) -> str:
        if not message.strip():
            raise ToolError("Commit message must not be empty.")
        return _git(
            _root(config), config.max_tool_output_chars, "commit", "-m", message.strip()
        )

    return Tool(
        name="git_commit",
        description="Record already-staged changes as a new commit with a message.",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4_000,
                    "description": "Commit message, e.g. 'Add file info tool'.",
                }
            },
            "required": ["message"],
        },
        handler=git_commit,
    )


def make_git_checkout_tool(config: AgentConfig) -> Tool:
    def git_checkout(branch: str) -> str:
        wanted = branch.strip()
        if not wanted:
            raise ToolError("Branch must not be empty.")
        branches = _git(_root(config), config.max_tool_output_chars, "branch", "--list")
        names = {
            line.strip().lstrip("* ").strip()
            for line in branches.splitlines()
            if line.strip() and "->" not in line
        }
        if wanted not in names:
            raise ToolError(
                f"Unknown local branch {wanted!r}. List branches with git_branch_list first."
            )
        return _git(_root(config), config.max_tool_output_chars, "checkout", wanted)

    return Tool(
        name="git_checkout",
        description="Switch to an existing local branch. Never restores file paths.",
        parameters={
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Existing branch name, e.g. 'main'.",
                }
            },
            "required": ["branch"],
        },
        handler=git_checkout,
    )


def git_tools(config: AgentConfig) -> list[Tool]:
    return [
        make_git_status_tool(config),
        make_git_diff_tool(config),
        make_git_log_tool(config),
        make_git_show_tool(config),
        make_git_branch_list_tool(config),
        make_git_stage_tool(config),
        make_git_commit_tool(config),
        make_git_checkout_tool(config),
    ]
