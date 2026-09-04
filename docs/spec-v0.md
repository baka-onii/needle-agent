# Natural-Language Agent Runtime

## V0 Implementation Specification

**Language:** Python 3.11+
**Orchestration:** LangGraph
**Action model:** Needle 2
**Architecture:** Local-first, modular
**Goal:** Separate reasoning from structured tool calling.

---

## 1. Core Architecture

The agent follows this loop:

```text
User Request
    ↓
Reasoning LLM
    ↓
Parse response
    ↓
Natural-language <tool> action
    ↓
Needle
    ↓
Sanitize + validate
    ↓
Confidence check
    ↓
Tool execution
    ↓
Natural-language observation
    ↓
Reasoning LLM
    ↓
...
    ↓
<final>
```

The reasoning LLM never directly produces or executes a structured tool call.

Needle is responsible only for:

```text
natural-language action
        ↓
tool selection + argument extraction
```

The runtime remains responsible for validation, permissions, and execution.

---

# 2. Dependencies

Use a `pyproject.toml`.

Loose dependency policy:

```toml
requires-python = ">=3.11,<3.14"

dependencies = [
    "langgraph>=1.2,<1.3",
    "pydantic>=2.13,<3",
    "cactus-needle>=2.0,<3",
]
```

The exact latest patch versions should be resolved into the project's lock file when the environment is created.

Current references:

* LangGraph 1.2.11 is the current PyPI release.
* Needle's package is currently 2.x and exposes the Python `Needle` API.
* Pydantic 2.13.5 is current at the time of this specification.

Do **not** add LangChain unless an implementation specifically requires it. LangGraph can be used independently.

Development dependencies:

```toml
[dependency-groups]
dev = [
    "pytest>=8,<9",
    "pytest-asyncio>=1,<2",
    "ruff>=0.12,<1",
]
```

Keep the dependency list small.

---

# 3. Project Structure

```text
agent-runtime/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── agent_runtime/
│       ├── __init__.py
│       ├── agent.py
│       ├── config.py
│       ├── state.py
│       │
│       ├── models/
│       │   ├── reasoning.py
│       │   ├── action.py
│       │   └── needle.py
│       │
│       ├── protocol/
│       │   └── parser.py
│       │
│       ├── tools/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── filesystem.py
│       │   ├── utility.py
│       │   └── interaction.py
│       │
│       ├── execution/
│       │   ├── sanitizer.py
│       │   ├── validator.py
│       │   ├── confidence.py
│       │   └── executor.py
│       │
│       ├── context/
│       │   └── manager.py
│       │
│       └── graph/
│           └── workflow.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── examples/
    └── basic.py
```

Do not split files further unless there is a concrete reason.

---

# 4. Data Models

Use Pydantic for external/model-generated data and simple dataclasses where appropriate for internal runtime objects.

## Agent State

```python
class AgentState(TypedDict):
    messages: list
    current_action: str | None
    last_tool_result: ToolResult | None
    step_count: int
    max_tool_steps: int
    final_answer: str | None
    status: str
```

Do not put model instances, tool registries, executors, or configuration into state.

---

## ToolCall

```python
class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
```

---

## ToolResult

```python
class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str | None = None
```

---

## NeedleResult

```python
class ToolRanking(BaseModel):
    tool_name: str
    confidence: float


class NeedleResult(BaseModel):
    selected_tool: str | None
    arguments: dict[str, Any] = {}
    confidence: float
    rankings: list[ToolRanking] = []
```

---

# 5. Tool Definition

Every tool has one canonical definition.

```python
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
```

The framework generates:

```text
Tool
 ├── reasoning-model description
 └── Needle JSON schema
```

Never manually maintain both versions.

---

# 6. Tool Registry

The registry maintains available tools:

```python
class ToolRegistry:
    def register(self, tool: Tool): ...
    def get(self, name: str) -> Tool: ...
    def list(self) -> list[Tool]: ...
```

At startup:

```text
ToolRegistry
    ├── read_file
    ├── read_directory
    ├── search_files
    ├── write_file
    ├── calculator
    ├── get_time
    └── ask_user
```

The registry is passed to the graph/runtime as a dependency.

---

# 7. Filesystem Tools

Filesystem tools must operate relative to an optional configured workspace root.

Example:

```text
workspace_root = /home/user/project
```

or on Windows:

```text
workspace_root = E:\project
```

The implementation must resolve paths and prevent escaping the workspace using `..`, absolute paths outside the workspace, or symlink traversal where applicable.

---

## 7.1 `read_file`

### Input

```python
path: str
```

### Implementation

1. Resolve path against workspace root.
2. Verify it remains inside workspace.
3. Verify it is a regular file.
4. Open using UTF-8.
5. Return contents.
6. Handle decoding and filesystem errors.
7. Limit output size.

Conceptually:

```python
def read_file(path: str) -> str:
    resolved = resolve_safe_path(path)

    if not resolved.is_file():
        raise ToolError("File does not exist.")

    return resolved.read_text(
        encoding="utf-8",
        errors="replace"
    )
```

Do not return arbitrarily huge files directly into the LLM context.

The tool/runtime should enforce a maximum read size.

---

# 8. `read_directory`

### Input

```python
path: str
```

### Implementation

1. Resolve safe path.
2. Verify directory exists.
3. Enumerate direct children.
4. Return name + basic type.
5. Sort deterministically.

Example output:

```text
Directory: src/

Files:
- main.py
- config.py

Directories:
- models
- tools
```

Do not recursively traverse the entire directory.

Recursion belongs to `search_files`.

---

# 9. `search_files`

This is the most important filesystem tool for agent exploration.

### Inputs

```python
query: str
path: str = "."
```

### Behavior

Search recursively under `path`.

For V0, use Python's standard library rather than adding a search dependency.

Preferred implementation:

```python
Path.rglob("*")
```

Algorithm:

```text
resolve search root
        ↓
walk recursive entries
        ↓
skip directories/files that should not be searched
        ↓
inspect text files
        ↓
search query
        ↓
collect matches
        ↓
limit results
        ↓
return concise result
```

---

## Search exclusions

V0 should skip common generated/binary directories:

```text
.git
__pycache__
node_modules
.venv
venv
dist
build
```

Also skip obvious binary files.

Do not attempt to decode every file blindly.

A simple approach:

1. Read a limited byte sample.
2. Detect NUL bytes.
3. Treat files containing NUL bytes as binary.
4. Otherwise decode as UTF-8 with replacement.

---

## Search output

Do not return entire matching files.

Return:

```text
Found 3 matches.

src/auth/login.py:42
    authenticate_user(user)

src/api/routes.py:18
    from auth.login import authenticate_user

tests/test_auth.py:11
    def test_authentication():
```

Use:

```text
max_results
max_matches_per_file
context_lines
```

internally/configurably.

V0 can keep these values fixed.

Recommended initial limits:

```text
max_results = 50
context_lines = 2
max_file_size = 2 MB
```

The exact values should be configuration constants rather than scattered magic numbers.

---

# 10. `write_file`

### Inputs

```python
path: str
content: str
```

### Behavior

1. Resolve safe path.
2. Verify path is inside workspace.
3. Create parent directories only if explicitly allowed by configuration.
4. Write UTF-8 text.
5. Return a concise success message.

Example:

```text
Successfully wrote 183 characters to src/config.py.
```

V0 should not implement append mode, binary writing, file deletion, or arbitrary filesystem mutation.

---

# 11. `calculator`

Do **not** use unrestricted:

```python
eval(expression)
```

Implement a restricted mathematical parser using Python's `ast` module.

Allow:

```text
+
-
*
/
**
%
()
```

and numeric literals.

Optionally allow a small fixed set of mathematical functions later.

Any other AST node must be rejected.

Example:

```text
2 * (15 + 3)
```

→

```text
36
```

---

# 12. `get_time`

Use Python's standard library:

```python
datetime
zoneinfo
```

Input:

```python
timezone: str | None
```

If omitted, use the configured/default local timezone.

Invalid timezone names must return a tool error.

---

# 13. `ask_user`

Input:

```python
question: str
```

The tool pauses the execution flow and obtains user input.

The exact UI/CLI mechanism should be abstracted behind the tool handler.

The answer becomes a normal observation for the reasoning model.

---

# 14. Reasoning Model Interface

The reasoning model should be abstracted:

```python
class ReasoningModel(Protocol):

    def generate(
        self,
        messages: list
    ) -> str:
        ...
```

The V0 framework should not require a particular reasoning model provider.

An adapter can later support:

* local Transformers models
* llama.cpp
* Ollama
* OpenAI-compatible local servers
* other backends

The graph only sees `generate()`.

---

# 15. Action Model Interface

```python
class ActionModel(Protocol):

    def translate(
        self,
        action: str,
        tools: list[Tool]
    ) -> NeedleResult:
        ...
```

Needle implements this interface.

The rest of the framework must not call Needle directly.

---

# 16. Needle Adapter

Use the official `cactus-needle` Python package.

Needle accepts tool definitions and produces structured tool calls; its current API also exposes confidence and supports custom weights.

The adapter should conceptually:

```text
framework Tool
      ↓
Needle-compatible schema
      ↓
Needle
      ↓
raw result
      ↓
NeedleResult
```

Do not use Needle's own full agent loop (`agent.run()`) because the framework itself owns the reasoning/action/execution loop.

Use the lower-level API such as `complete()` or the appropriate current single-turn interface instead. Needle's documentation explicitly provides `complete()` for applications where the caller executes the call and feeds the result back itself.

---

# 17. Action Protocol

The reasoning model uses:

```text
<tool>
natural language action
</tool>
```

and:

```text
<final>
final answer
</final>
```

Example:

```text
I need to inspect the project first.

<tool>
Read the project directory.
</tool>
```

Parser output:

```python
ParsedResponse(
    reasoning="I need to inspect the project first.",
    actions=["Read the project directory."],
    final_answer=None
)
```

---

# 18. Protocol Parser

Implement the parser independently from LangGraph.

Use a regular expression for the initial implementation, but do not make the regex itself the architecture.

It must support:

* multiline content
* multiple `<tool>` blocks
* `<final>`
* whitespace
* malformed tags
* empty blocks

Important rule:

**Do not execute arbitrary text as a tool action.**

Only content inside `<tool>` is executable intent.

---

# 19. Final Handling

If `<final>` exists:

```text
final_answer = contents
status = COMPLETED
```

If there is no `<tool>` and no `<final>`:

V0 should treat the model's response as a final answer.

This prevents unnecessary failures with models that omit the final tag.

---

# 20. Multiple Actions

If the response contains:

```text
<tool>A</tool>

<tool>B</tool>
```

V0 executes them sequentially.

Do not execute them concurrently.

Each action goes independently through:

```text
Needle
 → sanitize
 → validate
 → confidence
 → execute
```

The result of each action is added to context before the next reasoning cycle.

For V0, the preferred behavior is actually to execute **one tool action per reasoning turn**.

If multiple actions are detected, process the first action and preserve the others as pending actions only if the implementation needs this. Otherwise, return to reasoning after each action.

This keeps the state simple and avoids stale plans.

---

# 21. Sanitization

Needle's output is considered untrusted.

Sanitization should:

1. Extract the expected tool call.
2. Normalize the tool name.
3. Normalize arguments.
4. Reject malformed structures.
5. Produce a `ToolCall`.

Do not silently "repair" arbitrary malformed model output into a potentially dangerous command.

Prefer:

```text
valid → continue
invalid → error/ask reasoning model to retry
```

over aggressive guessing.

---

# 22. Validation

After sanitization:

```text
ToolCall
   ↓
ToolRegistry
   ↓
Tool definition
   ↓
parameter validation
```

Check:

* tool exists
* required arguments exist
* argument types are valid
* unexpected arguments are rejected
* constraints are satisfied

Only then may execution occur.

Pydantic can be used to implement argument validation.

---

# 23. Confidence Routing

Configuration:

```python
confidence_threshold = 0.85
```

If:

```text
confidence >= threshold
```

then:

```text
validate
 ↓
safety check
 ↓
execute
```

If:

```text
confidence < threshold
```

then:

```text
do not execute
 ↓
send candidates to reasoning model
```

Needle 2 explicitly provides a confidence score and is designed around confidence-gated tool execution, making this a natural use of the model.

---

# 24. Low-Confidence Reasoning

Provide the reasoning model with:

```text
The action translator is uncertain.

Requested action:
<original action>

Candidate tools:
1. search_files — 0.52
2. read_directory — 0.37
3. read_file — 0.11

Decide whether to clarify the action, choose another action, or continue without using a tool.
```

The reasoning model then generates another response.

It may produce:

```text
<tool>
Search the project files for "authentication".
</tool>
```

or:

```text
I don't have enough information to perform that action.
```

The latter becomes a final response under the normal no-tool fallback.

---

# 25. Confidence and Validation Order

The exact order must be:

```text
Reasoning
   ↓
Parse
   ↓
Needle
   ↓
Sanitize
   ↓
Validate
   ↓
Confidence
   ↓
Safety
   ↓
Execute
```

A malformed or invalid call must never reach the confidence/execution stage.

A high-confidence invalid call is still invalid.

---

# 26. Context Manager

The context manager is deliberately simple in V0.

It maintains the reasoning model's conversation.

The context contains:

```text
System prompt
Tool descriptions
User request
Assistant reasoning/actions
Tool observations
Confirmation messages
```

Do not include:

```text
Needle internal implementation details
raw runtime logs
unnecessary JSON schemas
debug information
```

---

# 27. Context Limits

Implement a configurable character/token budget.

When the context becomes too large:

1. Keep system prompt.
2. Keep tool descriptions.
3. Keep original user request.
4. Keep recent messages.
5. Remove/truncate older observations first.

Do not introduce summarization models in V0.

---

# 28. Tool Output Limits

Every tool must have an output limit.

For example:

```python
MAX_TOOL_OUTPUT_CHARS = 20_000
```

The context manager can further reduce it.

This prevents a single large file or search operation from consuming the entire reasoning context.

---

# 29. LangGraph State Machine

The graph should be:

```text
START
  ↓
REASON
  ↓
PARSE
  ├──────── FINAL ───────→ END
  │
  └──────── TOOL ────────→ TRANSLATE
                              ↓
                           SANITIZE
                              ↓
                           VALIDATE
                              ↓
                          CONFIDENCE
                           /       \
                        HIGH       LOW
                         ↓          ↓
                      SAFETY     CONFIRM
                         ↓          ↓
                      EXECUTE    REASON
                         ↓
                      OBSERVE
                         ↓
                 UPDATE_CONTEXT
                         ↓
                       REASON
```

---

# 30. Graph Node Responsibilities

Each node should have one responsibility.

### `reason`

Call reasoning model.

### `parse`

Parse response.

### `translate`

Call ActionModel.

### `sanitize`

Normalize/reject action-model output.

### `validate`

Validate ToolCall.

### `confidence`

Select high/low branch.

### `confirm`

Add low-confidence information to the reasoning context.

### `safety`

Perform permission checks.

### `execute`

Run tool.

Increment `step_count`.

### `observe`

Convert ToolResult into natural-language observation.

### `update_context`

Append observation and trim context.

---

# 31. Graph Routing

The conditional edges should be explicit.

Pseudo-code:

```python
def route_after_parse(state):
    if state["final_answer"] is not None:
        return "end"

    if state["current_action"]:
        return "translate"

    return "end"
```

After confidence:

```python
def route_after_confidence(state):
    if state["needle_result"].confidence >= threshold:
        return "safety"

    return "confirm"
```

After execution:

```python
def route_after_execution(state):
    if state["step_count"] >= state["max_tool_steps"]:
        return "max_steps"

    return "observe"
```

---

# 32. Important Graph Loop Rule

The graph must return to `REASON` after every successful tool execution.

Never do:

```text
Reason
 → Tool
 → Tool
 → Tool
```

without allowing the reasoning model to interpret the result.

The intended architecture is:

```text
Reason
 → Action
 → Tool
 → Observation
 → Reason
```

This is what gives the reasoning model control over the next action.

---

# 33. Max-Step Protection

Before every execution:

```python
if state["step_count"] >= state["max_tool_steps"]:
    state["status"] = "MAX_STEPS_REACHED"
```

Terminate the graph.

Default:

```text
max_tool_steps = 20
```

This should be configurable.

---

# 34. Error Recovery

Recoverable tool errors should become observations.

Example:

```text
Tool error:
FileNotFoundError
```

becomes:

```text
The file "main.py" does not exist.
You may want to inspect the directory first.
```

The reasoning model can then recover.

Internal framework errors should terminate the run.

---

# 35. Safety

V0 filesystem operations are workspace restricted.

For every filesystem path:

```text
user/model path
      ↓
resolve
      ↓
workspace containment check
      ↓
filesystem operation
```

Do not trust paths generated by either LLM.

Symlinks should be handled carefully so they cannot be used to escape the workspace.

---

# 36. No Shell Tool in V0

Do not add:

```text
shell
terminal
execute_command
run_python
```

to V0.

These dramatically increase the security surface and are unnecessary for proving the core architecture.

They can be added later behind a sandbox.

---

# 37. Testing

Tests should be written alongside implementation.

## Unit tests

Test:

```text
parser
tool registry
path safety
file reading
directory listing
search
calculator
Needle result parsing
sanitizer
validator
confidence routing
context trimming
```

---

## Integration test

Use mocked reasoning and action models:

```text
User
 ↓
Reasoning mock
 ↓
Parser
 ↓
Action mock
 ↓
Validator
 ↓
Filesystem tool
 ↓
Observation
 ↓
Reasoning mock
 ↓
Final
```

This test should run without an actual LLM.

---

# 38. End-to-End Example

Workspace:

```text
project/
├── main.py
├── config.py
└── src/
    └── auth.py
```

User:

```text
Find the project's authentication implementation.
```

Reasoning:

```text
<tool>
Search the project for authentication-related code.
</tool>
```

Needle:

```text
search_files
query="authentication"
path="."
confidence=0.94
```

Runtime validates and executes.

Observation:

```text
Found references in src/auth.py and config.py.
```

Reasoning:

```text
<tool>
Read src/auth.py.
</tool>
```

Needle:

```text
read_file
path="src/auth.py"
confidence=0.99
```

Tool executes.

Reasoning:

```text
<final>
The authentication implementation is located in src/auth.py.
</final>
```

Graph ends.

---

# 39. Implementation Order

Do not ask the coding AI to implement everything simultaneously.

Implement in this order:

### Phase 1 — Core models

```text
state
Tool
ToolCall
ToolResult
NeedleResult
configuration
```

### Phase 2 — Tools

Implement and test:

```text
read_file
read_directory
search_files
write_file
calculator
get_time
ask_user
```

### Phase 3 — Protocol

Implement:

```text
<tool>
<final>
parser
```

### Phase 4 — Tool execution pipeline

Implement:

```text
sanitizer
validator
confidence
executor
```

### Phase 5 — Model interfaces

Implement:

```text
ReasoningModel
ActionModel
NeedleActionModel
```

### Phase 6 — Context manager

Implement simple context construction and limits.

### Phase 7 — LangGraph

Connect all components into the graph.

### Phase 8 — Integration tests

Test complete multi-step loops.

### Phase 9 — Real models

Only after mocked execution works:

```text
real reasoning model
+
real Needle
```

### Phase 10 — Benchmark

Compare native tool calling against the proposed architecture.

---

# 40. Dependency Philosophy

Keep the core runtime lightweight.

Use the Python standard library for:

```text
filesystem
search
datetime
timezone
AST calculator
logging
path handling
```

Use external packages only where they provide core architectural functionality:

```text
LangGraph       → orchestration
Pydantic        → validation/data models
cactus-needle   → action model
```

Do not add a framework simply because it is popular.

---

# 41. V0 Success Criteria

The first usable version is complete when this works reliably:

```text
User request
    ↓
Small reasoning model
    ↓
Natural-language <tool>
    ↓
Needle
    ↓
Validated ToolCall
    ↓
Filesystem/calculator/etc.
    ↓
Natural-language observation
    ↓
Reasoning
    ↓
Final answer
```

And specifically:

* reasoning model never needs to emit JSON
* Needle never performs the overall agent loop
* invalid calls never execute
* low-confidence calls return to reasoning
* filesystem access is sandboxed to the workspace
* context remains bounded
* the LangGraph loop terminates correctly
* the action model can eventually be replaced without rewriting the agent
* tests can run with mocked models
* the architecture can be benchmarked against conventional tool calling
