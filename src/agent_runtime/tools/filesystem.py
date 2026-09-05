"""Filesystem tools: read_file, read_directory, search_files, write_file.

All paths resolve against the configured workspace root with a containment
check (spec §7, §35). Never trust model-generated paths.
"""

from __future__ import annotations

from pathlib import Path

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import Tool, ToolError

SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"})


def resolve_safe_path(user_path: str, workspace_root: str | None) -> Path:
    """Resolve ``user_path`` against ``workspace_root`` (or cwd if unset).

    Raises ToolError if the resolved path escapes the workspace via ``..``,
    an outside absolute path, or a symlink.
    """
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
    candidate = (root / user_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"Path escapes workspace: {user_path!r}")
    return candidate


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated to {limit} chars]"


def make_read_file_tool(config: AgentConfig) -> Tool:
    def read_file(path: str) -> str:
        resolved = resolve_safe_path(path, config.workspace_root)
        if not resolved.is_file():
            raise ToolError(f"File does not exist: {path!r}")
        try:
            text = resolved.read_bytes().decode("utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"Cannot read file {path!r}: {exc}") from exc
        return _truncate(text, config.max_tool_output_chars)

    return Tool(
        name="read_file",
        description="Read a text file inside the workspace and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read, e.g. 'main.py' or 'src/auth.py'.",
                }
            },
            "required": ["path"],
        },
        handler=read_file,
    )


def make_read_directory_tool(config: AgentConfig) -> Tool:
    def read_directory(path: str) -> str:
        resolved = resolve_safe_path(path, config.workspace_root)
        if not resolved.is_dir():
            raise ToolError(f"Directory does not exist: {path!r}")
        try:
            children = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            raise ToolError(f"Cannot list directory {path!r}: {exc}") from exc
        files = [c.name for c in children if c.is_file()]
        dirs = [c.name for c in children if c.is_dir()]
        lines = [f"Directory: {path}"]
        if files:
            lines.append("\nFiles:")
            lines.extend(f"- {name}" for name in files)
        if dirs:
            lines.append("\nDirectories:")
            lines.extend(f"- {name}" for name in dirs)
        if not files and not dirs:
            lines.append("(empty)")
        return _truncate("\n".join(lines), config.max_tool_output_chars)

    return Tool(
        name="read_directory",
        description="List the direct children of a workspace directory. Not recursive.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Use '.' for the top level.",
                    "default": ".",
                }
            },
            "required": ["path"],
        },
        handler=read_directory,
    )


def _search_file(
    entry: Path, query: str, context_lines: int, max_file_bytes: int, max_matches: int
) -> list[tuple[int, list[str]]]:
    """Return [(line_number, context_block)] matches for one file."""
    try:
        if entry.stat().st_size > max_file_bytes:
            return []
    except OSError:
        return []
    try:
        raw = entry.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw[:8192]:
        return []  # binary
    lines = raw.decode("utf-8", errors="replace").splitlines()
    lowered = query.lower()
    hits: list[tuple[int, list[str]]] = []
    for i, line in enumerate(lines):
        if lowered in line.lower():
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            hits.append((i + 1, lines[start:end]))
            if len(hits) >= max_matches:
                break
    return hits


def make_search_files_tool(config: AgentConfig) -> Tool:
    def search_files(query: str, path: str = ".") -> str:
        if not query.strip():
            raise ToolError("Search query must not be empty.")
        root = resolve_safe_path(path, config.workspace_root)
        if not root.is_dir():
            raise ToolError(f"Search path is not a directory: {path!r}")
        matches: list[tuple[str, int, list[str]]] = []
        for entry in sorted(root.rglob("*")):
            if len(matches) >= config.max_search_results:
                break
            if entry.is_symlink():
                continue
            parts = set(entry.relative_to(root).parts[:-1]) | {entry.name}
            if parts & set(SKIP_DIRS):
                continue
            if not entry.is_file():
                continue
            try:
                resolved = entry.resolve()
            except OSError:
                continue
            if resolved != root and root not in resolved.parents:
                continue
            for lineno, context in _search_file(
                entry,
                query,
                config.search_context_lines,
                config.max_search_file_bytes,
                config.max_matches_per_file,
            ):
                rel = entry.relative_to(root).as_posix()
                matches.append((rel, lineno, context))
                if len(matches) >= config.max_search_results:
                    break
        if not matches:
            return f"No matches found for {query!r} under {path!r}."
        lines = [f"Found {len(matches)} matches."]
        for rel, lineno, context in matches:
            lines.append(f"\n{rel}:{lineno}")
            lines.extend(f"    {c}" for c in context)
        return _truncate("\n".join(lines), config.max_tool_output_chars)

    return Tool(
        name="search_files",
        description="Search workspace files recursively. Returns matches with line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Single word or exact phrase to find, e.g. 'authenticate_user'.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search under. Use '.' for everywhere.",
                    "default": ".",
                },
            },
            "required": ["query"],
        },
        handler=search_files,
    )


def make_write_file_tool(config: AgentConfig) -> Tool:
    def write_file(path: str, content: str) -> str:
        resolved = resolve_safe_path(path, config.workspace_root)
        if resolved.is_dir():
            raise ToolError(f"Path is a directory: {path!r}")
        if not resolved.parent.exists():
            if not config.allow_create_parent_dirs:
                raise ToolError(f"Parent directory does not exist: {path!r}")
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ToolError(f"Cannot create parent dirs for {path!r}: {exc}") from exc
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Cannot write file {path!r}: {exc}") from exc
        return f"Successfully wrote {len(content)} characters to {path}."

    return Tool(
        name="write_file",
        description="Write text to a workspace file. No append, binary, or delete modes.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full UTF-8 text content to write.",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_file,
    )


def filesystem_tools(config: AgentConfig) -> list[Tool]:
    return [
        make_read_file_tool(config),
        make_read_directory_tool(config),
        make_search_files_tool(config),
        make_write_file_tool(config),
    ]


__all__: list[str] = [
    "SKIP_DIRS",
    "filesystem_tools",
    "make_read_directory_tool",
    "make_read_file_tool",
    "make_search_files_tool",
    "make_write_file_tool",
    "resolve_safe_path",
]
