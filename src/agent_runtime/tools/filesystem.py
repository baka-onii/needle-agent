"""Workspace-restricted, bounded text-file tools. No shell or arbitrary mutation."""

from __future__ import annotations

from pathlib import Path

from agent_runtime.config import AgentConfig
from agent_runtime.tools.base import Tool, ToolError, truncate_text

SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".cache",
    }
)


def resolve_safe_path(user_path: str, workspace_root: str | None) -> Path:
    """Containment-check both lexical and resolved paths before any file operation.

    The workspace defaults to cwd. Reject traversal, NUL bytes, symlink escapes,
    and repository metadata. This is not an OS sandbox against concurrent hostile
    processes that can replace directories between a check and an operation.
    """
    if not isinstance(user_path, str) or not user_path or "\x00" in user_path:
        raise ToolError("Invalid filesystem path.")
    path = Path(user_path)
    if ".." in path.parts:
        raise ToolError(f"Path escapes workspace: {user_path!r}")
    try:
        root = Path(workspace_root or Path.cwd()).resolve()
        lexical = root / path
        if not lexical.is_relative_to(root):
            raise ToolError(f"Path escapes workspace: {user_path!r}")
        if ".git" in lexical.relative_to(root).parts:
            raise ToolError("Repository metadata (.git) is protected.")
        candidate = lexical.resolve()
        if not candidate.is_relative_to(root):
            raise ToolError(f"Path escapes workspace: {user_path!r}")
        if ".git" in candidate.relative_to(root).parts:
            raise ToolError("Repository metadata (.git) is protected.")
        return candidate
    except (OSError, ValueError, RuntimeError) as exc:
        raise ToolError(f"Cannot resolve path {user_path!r}.") from exc


def _path_schema(description: str, default: str | None = None) -> dict:
    schema = {"type": "string", "minLength": 1, "maxLength": 4_096, "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def make_read_file_tool(config: AgentConfig) -> Tool:
    def read_file(path: str) -> str:
        resolved = resolve_safe_path(path, config.workspace_root)
        try:
            if not resolved.is_file():
                raise ToolError(f"File does not exist or is not a regular file: {path!r}")
            with resolved.open(encoding="utf-8", errors="replace") as file:
                text = file.read(config.max_tool_output_chars + 1)
            if "\x00" in text:
                raise ToolError(f"Cannot read binary file as text: {path!r}")
        except OSError as exc:
            raise ToolError(f"Cannot read file {path!r}: {exc.strerror}") from exc
        return truncate_text(text, config.max_tool_output_chars)

    return Tool(
        name="read_file",
        description="Read a UTF-8 text file inside the workspace. Large output is truncated.",
        parameters={
            "type": "object",
            "properties": {"path": _path_schema("File to read, e.g. 'main.py' or 'src/auth.py'.")},
            "required": ["path"],
        },
        handler=read_file,
    )


def make_read_directory_tool(config: AgentConfig) -> Tool:
    def read_directory(path: str = ".") -> str:
        resolved = resolve_safe_path(path, config.workspace_root)
        try:
            if not resolved.is_dir():
                raise ToolError(f"Directory does not exist: {path!r}")
            children = sorted(resolved.iterdir(), key=lambda p: (p.name.casefold(), p.name))
            files, dirs = [], []
            for child in children:
                try:
                    safe = resolve_safe_path(str(child), config.workspace_root)
                except ToolError:
                    continue
                if safe.is_file():
                    files.append(child.name)
                elif safe.is_dir():
                    dirs.append(child.name)
        except OSError as exc:
            raise ToolError(f"Cannot list directory {path!r}: {exc.strerror}") from exc
        lines = [f"Directory: {path}"]
        if files:
            lines.extend(["\nFiles:", *(f"- {name}" for name in files)])
        if dirs:
            lines.extend(["\nDirectories:", *(f"- {name}" for name in dirs)])
        if not files and not dirs:
            lines.append("(empty)")
        return truncate_text("\n".join(lines), config.max_tool_output_chars)

    return Tool(
        name="read_directory",
        description="List a workspace directory's direct children, sorted by name. Not recursive.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Directory to list. Use '.' for the top level.", ".")
            },
            "required": [],
        },
        handler=read_directory,
    )


def _search_file(
    entry: Path, query: str, context_lines: int, max_file_bytes: int, max_matches: int
) -> list[tuple[int, list[str]]]:
    try:
        if entry.stat().st_size > max_file_bytes:
            return []
        with entry.open("rb") as file:
            raw = file.read(max_file_bytes + 1)
    except OSError:
        return []
    if len(raw) > max_file_bytes or b"\x00" in raw:
        return []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    hits: list[tuple[int, list[str]]] = []
    for i, line in enumerate(lines):
        if query.casefold() in line.casefold():
            start, end = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
            hits.append((i + 1, [f"{j + 1}: {lines[j]}" for j in range(start, end)]))
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
        limited = False
        try:
            # Lazy rglob: never materialize the whole repository before limiting it.
            for visited, entry in enumerate(root.rglob("*"), start=1):
                if visited > config.max_search_entries or len(matches) >= config.max_search_results:
                    limited = True
                    break
                if set(entry.relative_to(root).parts) & SKIP_DIRS or entry.is_symlink():
                    continue
                try:
                    safe = resolve_safe_path(str(entry), config.workspace_root)
                except ToolError:
                    continue
                if not safe.is_file():
                    continue
                for lineno, context in _search_file(
                    safe,
                    query,
                    config.search_context_lines,
                    config.max_search_file_bytes,
                    config.max_matches_per_file,
                ):
                    # Workspace-relative paths can be passed straight to read_file.
                    workspace = Path(config.workspace_root or Path.cwd()).resolve()
                    matches.append((entry.relative_to(workspace).as_posix(), lineno, context))
                    if len(matches) >= config.max_search_results:
                        limited = True
                        break
        except OSError as exc:
            raise ToolError(f"Cannot search {path!r}: {exc.strerror}") from exc
        lines = [
            f"Found {len(matches)} matches."
            if matches
            else f"No matches found for {query!r} under {path!r}."
        ]
        if limited:
            lines.append("Search limit reached; narrow the query or search path for more results.")
        for rel, lineno, context in sorted(matches):
            lines.extend([f"\n{rel}:{lineno}", *(f"    {line}" for line in context)])
        return truncate_text("\n".join(lines), config.max_tool_output_chars)

    return Tool(
        name="search_files",
        description="Search workspace text files recursively. Return matching lines and paths.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1_000,
                    "description": "Word or exact phrase to find, e.g. 'authenticate_user'.",
                },
                "path": _path_schema("Directory to search under. Use '.' for everywhere.", "."),
            },
            "required": ["query"],
        },
        handler=search_files,
    )


def make_write_file_tool(config: AgentConfig) -> Tool:
    def write_file(path: str, content: str) -> str:
        if config.read_only:
            raise ToolError("Workspace is read-only; writing is disabled.")
        if "\x00" in content:
            raise ToolError("Only UTF-8 text writes are supported; NUL bytes are not allowed.")
        resolved = resolve_safe_path(path, config.workspace_root)
        try:
            if resolved.exists() and not resolved.is_file():
                raise ToolError(f"Path is not a regular file: {path!r}")
            if resolved.exists() and resolved.stat().st_nlink > 1:
                raise ToolError("Writing to hard-linked files is not allowed.")
            if not resolved.parent.exists():
                if not config.allow_create_parent_dirs:
                    raise ToolError(f"Parent directory does not exist: {path!r}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
            # Check again after parent creation, before opening the destination.
            resolved = resolve_safe_path(path, config.workspace_root)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Cannot write file {path!r}: {exc.strerror}") from exc
        return f"Successfully wrote {len(content)} characters to {path}."

    return Tool(
        name="write_file",
        description="Write UTF-8 text to a file, replacing its contents. No append/delete.",
        parameters={
            "type": "object",
            "properties": {
                "path": _path_schema("Workspace-relative path of the file to write."),
                "content": {
                    "type": "string",
                    "maxLength": 100_000,
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
