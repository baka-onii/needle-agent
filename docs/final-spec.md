# Agent Runtime

**Status:** Final Product Specification
**Target:** Open-source Python agent framework
**Primary orchestration:** LangGraph
**Primary action model:** Needle 2
**Architecture:** Reasoning LLM + specialized action model + deterministic runtime
**Initial release:** V0 / first public OSS release

---

# 1. Product Definition

Agent Runtime is a modular agent framework built around a strict separation between:

1. **Reasoning**

   * Performed by a general-purpose reasoning LLM.
   * Responsible for understanding the user, deciding what should happen, interpreting observations, and deciding when the task is complete.

2. **Action translation**

   * Performed by a small specialized action model.
   * Initial implementation uses Needle 2.
   * Converts natural-language action intent into a structured tool call.

3. **Runtime execution**

   * Deterministic Python code.
   * Validates, sanitizes, authorizes, executes, and reports tool calls.

The framework must not depend on the reasoning LLM having native structured tool-calling capabilities.

The central design principle is:

> The reasoning model decides **what should happen**.
> The action model translates that decision into **how to call a tool**.
> The runtime decides **whether and how that call may execute**.

Conceptually:

```text
Reasoning LLM
     │
     │ natural-language action
     ▼
Action Model
     │
     │ structured ToolCall
     ▼
Sanitizer
     │
     ▼
Validator
     │
     ▼
Confidence Gate
     │
     ▼
Safety / Authorization
     │
     ▼
Tool Executor
     │
     ▼
Tool Result
     │
     ▼
Observation
     │
     ▼
Reasoning LLM
```

---

# 2. Design Goals

The first public version must optimize for:

### 2.1 Separation of concerns

The framework must not couple reasoning, action selection, tool implementation, and orchestration.

A future action model must be replaceable without rewriting the runtime.

A future reasoning model must be replaceable without rewriting the tool system.

### 2.2 Small local models

The architecture must support small/local reasoning models and particularly small action models.

Needle 2 is appropriate for the first implementation because it is specifically designed for structured tool calling, exposes confidence, and can retrieve among larger tool catalogs.

### 2.3 Deterministic execution

Models may propose actions.

Models must never directly execute Python functions.

Every model-produced action must pass through:

```text
parse
→ sanitize
→ validate
→ confidence
→ safety
→ execute
```

### 2.4 Minimal core

The framework should have as few dependencies as practical.

Do not introduce LangChain simply because LangGraph is being used.

LangGraph can be used independently of LangChain.

### 2.5 Extensibility

New tools, reasoning models, action models, safety policies, and context strategies must be implementable through explicit interfaces.

### 2.6 Public OSS quality

The repository must be structured like a real open-source project:

* typed interfaces
* documentation
* tests
* examples
* predictable error handling
* no hard-coded machine-specific paths
* no hidden model dependencies
* clear licensing
* reproducible development setup

---

# 3. Non-Goals for V0

The following are explicitly outside the first release:

* arbitrary shell execution
* arbitrary Python execution
* browser automation
* GUI automation
* web search
* databases
* MCP
* image generation
* computer vision
* remote tool execution
* vector databases
* RAG
* long-term memory
* autonomous background execution
* multi-agent coordination
* parallel tool execution
* complex planning frameworks
* model-generated executable code
* automatic destructive-action approval
* unrestricted filesystem access

These may be added later as extensions.

---

# 4. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │        USER         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   REASONING MODEL   │
                         └──────────┬──────────┘
                                    │
                         natural-language output
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      PROTOCOL       │
                         │       PARSER        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
                 <final>                         <tool>
                    │                               │
                    ▼                               ▼
                   END                       ACTION MODEL
                                                (Needle)
                                                  │
                                                  ▼
                                             ToolCall
                                                  │
                                                  ▼
                                             SANITIZER
                                                  │
                                                  ▼
                                             VALIDATOR
                                                  │
                                                  ▼
                                           CONFIDENCE GATE
                                             /         \
                                            /           \
                                         HIGH           LOW
                                          │              │
                                          ▼              ▼
                                       SAFETY         REASONING
                                          │           clarification
                                          ▼
                                       EXECUTE
                                          │
                                          ▼
                                     TOOL RESULT
                                          │
                                          ▼
                                      OBSERVER
                                          │
                                          ▼
                                   CONTEXT MANAGER
                                          │
                                          ▼
                                  REASONING MODEL
```

---

# 5. Core Runtime Principle

The reasoning model does not directly emit JSON.

Instead, it speaks naturally.

Example:

```text
I should inspect the project directory first.

<tool>
Read the directory "E://project//new//"
</tool>
```

The text inside `<tool>` is extracted and passed to the action model.

Needle then translates that into something conceptually equivalent to:

```json
{
  "name": "read_directory",
  "arguments": {
    "path": "E://project//new//"
  }
}
```

The runtime then validates and executes this call.

The reasoning model receives the result as a natural-language observation:

```text
Tool `read_directory` succeeded.

Directories:
- src
- tests

Files:
- pyproject.toml
- README.md
- main.py
```

The reasoning model then decides what happens next.

---

# 6. Agent Protocol

The reasoning model uses two explicit control constructs.

## Tool block

```text
<tool>
natural language action
</tool>
```

Example:

```text
<tool>
Search the project for references to authentication
</tool>
```

## Final block

```text
<final>
final response
</final>
```

Example:

```text
<final>
The authentication logic is located in src/auth/login.py.
</final>
```

---

# 7. Protocol Rules

The parser must implement the following rules.

### Rule 1 — Only tagged tool content is executable

Text outside `<tool>` blocks must never be passed to the action model.

### Rule 2 — Final blocks terminate the agent

A valid `<final>` block transitions the graph to `END`.

### Rule 3 — Empty tool blocks are invalid

```text
<tool>
</tool>
```

must produce a protocol error rather than an empty action.

### Rule 4 — Multiline actions are supported

```text
<tool>
Look through the source tree
and find where user authentication
is implemented.
</tool>
```

### Rule 5 — Multiple tool blocks are parsed

The parser should return all discovered blocks.

However, V0 execution should default to **one action per reasoning turn**.

If multiple tool blocks appear, the runtime must not silently execute all of them.

Preferred V0 behavior:

* execute the first valid tool block
* retain the remaining blocks for diagnostic handling or reject the response as a multi-action response

The recommended implementation is to reject multiple actions with a protocol error and ask the reasoning model to issue one action at a time.

This avoids stale plans.

### Rule 6 — `<final>` has priority over malformed trailing prose

The parser should identify valid control blocks rather than attempting to interpret arbitrary prose.

### Rule 7 — Unclosed tags are errors

Example:

```text
<tool>
read the project
```

must not be executed.

### Rule 8 — Arbitrary XML/HTML is not interpreted

Only the exact supported control tags are recognized.

---

# 8. Protocol Parser Interface

```python
class ParsedResponse(BaseModel):
    reasoning_text: str
    tool_actions: list[str]
    final_answer: str | None
```

Parser interface:

```python
class ResponseParser(Protocol):
    def parse(self, text: str) -> ParsedResponse:
        ...
```

Implementation requirements:

* multiline support
* whitespace normalization
* empty block detection
* duplicate block detection
* unclosed block detection
* safe extraction
* deterministic behavior

Parser tests must include:

* normal tool
* normal final
* reasoning + tool
* reasoning + final
* multiline tool
* multiple tools
* empty tool
* unclosed tool
* unclosed final
* unknown tags
* tool-like text in ordinary prose
* nested tags
* extra whitespace

---

# 9. Canonical Tool Definition

There must be exactly one canonical representation of a tool.

```python
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
```

The canonical definition must generate all other tool representations.

From the same definition generate:

```text
Tool Registry representation
Reasoning-model tool description
Needle JSON schema
Validation schema
Documentation metadata
```

Do not maintain separate manually written descriptions.

That would eventually cause the reasoning model, Needle, and validator to disagree.

---

# 10. Tool Interface

A tool is an executable capability.

Conceptually:

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable
```

Recommended additional metadata fields:

```python
requires_confirmation: bool = False
read_only: bool = True
category: str | None = None
```

These fields should be supported internally even if V0 only makes limited use of them.

---

# 11. Tool Registry

The registry owns the canonical set of available tools.

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool:
        ...

    def all(self) -> list[Tool]:
        ...

    def contains(self, name: str) -> bool:
        ...
```

Requirements:

* duplicate names rejected
* empty names rejected
* invalid schemas rejected during registration
* tool names must be deterministic
* registry must not execute tools
* registry must not contain model-specific logic

---

# 12. Human-Readable Tool Descriptions

The reasoning model must receive a concise action space.

Example:

```text
Available actions:

read_file(path)
  Read the contents of a text file.

read_directory(path)
  List the direct contents of a directory.

search_files(query, path=".")
  Recursively search text files under a directory.

write_file(path, content)
  Write text content to a file.

calculator(expression)
  Evaluate a mathematical expression.

get_time(timezone=None)
  Get the current time.

ask_user(question)
  Ask the user for additional information.
```

This representation is deliberately simpler than the complete JSON schema.

---

# 13. Needle Tool Representation

Needle receives the complete structured tool schemas.

Needle 2 supports structured tool calling and schema-constrained output, and supports retrieval when larger tool catalogs are declared.

The framework must adapt canonical tools into Needle's expected tool format.

The runtime must not otherwise depend on Needle's internal implementation.

---

# 14. ActionModel Abstraction

Needle must be an implementation of a generic action-model interface.

```python
class ActionModel(Protocol):
    def translate(
        self,
        action: str,
        tools: list[Tool],
    ) -> NeedleResult:
        ...
```

The rest of the framework must never call `needle.Needle` directly.

Instead:

```text
Agent
  ↓
ActionModel
  ↓
NeedleActionModel
  ↓
Needle
```

This allows later implementations such as:

```text
NeedleActionModel
FunctionGemmaActionModel
CustomFineTunedActionModel
RemoteActionModel
RuleBasedActionModel
```

without changing the graph.

---

# 15. ReasoningModel Abstraction

The reasoning model must use an equally small abstraction.

```python
class ReasoningModel(Protocol):
    def generate(
        self,
        messages: list,
    ) -> str:
        ...
```

The reasoning model is responsible for:

* interpreting the user
* generating natural-language reasoning
* deciding when a tool is necessary
* choosing what to ask the user
* interpreting observations
* deciding when the task is finished

It is not responsible for:

* executing tools
* validating paths
* deciding authorization
* parsing arbitrary JSON
* directly invoking Python functions

---

# 16. Runtime Models

Use Pydantic models for structured boundaries and validation.

```python
class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
```

```python
class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str | None = None
```

```python
class ToolRanking(BaseModel):
    tool_name: str
    confidence: float
```

```python
class NeedleResult(BaseModel):
    selected_tool: str | None
    arguments: dict[str, Any] = {}
    confidence: float
    rankings: list[ToolRanking] = []
```

Optional internal metadata may include:

```python
raw_response
latency_ms
model_name
```

but these must not be required by the core runtime.

---

# 17. Needle Adapter

The Needle adapter translates the framework's generic action request into Needle's API.

Current Needle documentation supports caller-owned loops through `complete()`, and returns structured function calls plus confidence; the framework should therefore use that style rather than handing the complete agent loop to Needle.

Conceptual implementation:

```python
class NeedleActionModel:
    def __init__(self, ...):
        ...

    def translate(self, action: str, tools: list[Tool]) -> NeedleResult:
        ...
```

It should:

1. convert tools to Needle schemas
2. invoke Needle
3. extract function calls
4. normalize the result
5. return `NeedleResult`

It must not:

* execute the returned tool
* manage the reasoning loop
* maintain the AgentState
* call LangGraph
* perform authorization
* modify files

---

# 18. Needle Output Sanitization

Raw action-model output is untrusted data.

Pipeline:

```text
Needle
  ↓
Sanitize
  ↓
Normalize
  ↓
Validate
  ↓
Confidence
  ↓
Safety
  ↓
Execute
```

Sanitization must:

* extract expected function-call information
* normalize tool name
* normalize argument container
* reject malformed structures
* reject unknown fields where appropriate
* reject missing required information
* reject impossible values where schema validation can detect them

It must not perform aggressive guessing.

For example, this must not silently become valid:

```json
{
  "name": "read_file",
  "pathh": "main.py"
}
```

The sanitizer should reject it rather than guessing that `"pathh"` means `"path"`.

The same principle applies to unsafe malformed paths or arguments.

---

# 19. Tool Schema Validation

After sanitization, validate:

```text
tool exists
+
arguments match schema
+
required parameters exist
+
types are correct
+
additional parameters obey policy
```

Validation must occur before execution.

Pydantic is the preferred validation layer for internal structured models; current Pydantic 2.13.x remains compatible with the planned design.

---

# 20. Confidence Routing

Needle confidence is a routing signal.

Example configuration:

```python
confidence_threshold = 0.85
```

Routing:

```text
confidence >= threshold
        │
        ▼
 schema valid?
        │
        ▼
 safety valid?
        │
        ▼
 execute
```

For low confidence:

```text
confidence < threshold
        │
        ▼
DO NOT EXECUTE
        │
        ▼
Reasoning model receives:
- original action
- selected candidate
- confidence
- ranked candidates
        │
        ▼
Reason again
```

Needle documents confidence as a mechanism for escalating low-confidence decisions; the framework should use it as such rather than treating confidence as proof of correctness.

---

# 21. Confidence Is Not Authorization

This distinction is mandatory.

A high confidence score does **not** mean:

```text
authorized
safe
correct
non-destructive
inside workspace
```

Therefore:

```text
confidence
≠
permission
```

The framework must always run safety checks even for high-confidence actions.

---

# 22. Low-Confidence Reasoning Prompt

When confidence is too low, the reasoning model receives a structured observation such as:

```text
The action translator is uncertain.

Requested action:
"Look through the project for anything related to authentication."

Candidate actions:
1. search_files — confidence 0.52
2. read_directory — confidence 0.37
3. read_file — confidence 0.11

No action was executed.

Decide whether to:
- clarify the intended action,
- issue a better action,
- ask the user for information,
- or continue without using a tool.
```

The reasoning model must not be forced to choose the highest-confidence result.

---

# 23. Safety Layer

The safety layer sits between validation and execution.

```text
validated ToolCall
       ↓
safety policy
       ↓
authorized ToolCall
       ↓
executor
```

Safety policies include:

* filesystem root restrictions
* path traversal protection
* symlink restrictions
* destructive-operation confirmation
* size limits
* recursion limits
* allowed tool set
* future permission policies

---

# 24. Workspace Model

Filesystem tools operate only inside a configured workspace.

Example:

```python
workspace_root = Path("/home/user/project")
```

Every filesystem operation must resolve its target against the workspace.

The runtime must prevent:

```text
../outside.txt
```

absolute paths outside the workspace:

```text
/etc/passwd
C:\Windows\...
```

and symlink traversal outside the workspace.

---

# 25. Safe Path Resolution

Every filesystem tool should call one shared function:

```python
resolve_safe_path(path, workspace_root)
```

The function must:

1. normalize the user/model path
2. resolve it relative to the workspace when appropriate
3. resolve symlinks where necessary for authorization
4. verify the final path remains inside the workspace
5. reject traversal outside the workspace

Conceptually:

```python
resolved = candidate.resolve()

if not resolved.is_relative_to(workspace_root):
    raise ToolError("Path is outside the workspace")
```

The implementation must also account for the platform-specific behavior of missing paths.

Never duplicate path-security logic across individual tools.

---

# 26. V0 Tools

The first release contains exactly seven built-in tools:

```text
read_file
read_directory
search_files
write_file
calculator
get_time
ask_user
```

---

# 27. read_file

Signature:

```python
read_file(path: str) -> str
```

Behavior:

1. resolve path through workspace safety layer
2. verify target is a regular file
3. enforce maximum file size
4. open as UTF-8
5. use `errors="replace"`
6. return text
7. return a concise error on failure

Example implementation concept:

```python
resolved = resolve_safe_path(path)

if not resolved.is_file():
    raise ToolError("Not a file")

return resolved.read_text(
    encoding="utf-8",
    errors="replace",
)
```

The tool must not:

* read outside workspace
* return unlimited data
* execute the file
* guess encodings indefinitely
* follow unsafe external symlinks

---

# 28. read_file Output Limit

Define one configuration constant:

```python
MAX_READ_FILE_BYTES
```

Default recommendation:

```text
1–2 MB
```

If a file exceeds the limit, return a bounded error/message rather than loading the entire file into context.

Future versions may support:

```text
offset
limit
line range
```

but V0 should keep the interface simple.

---

# 29. read_directory

Signature:

```python
read_directory(path: str = ".") -> str
```

Behavior:

1. resolve safe path
2. verify directory
3. enumerate direct children
4. classify entries
5. sort deterministically
6. return bounded output

It must not recursively walk the directory.

Example output:

```text
Directory: src

Directories:
- auth
- api
- utils

Files:
- main.py
- config.py
```

Deterministic sorting is required so tests and model observations remain stable.

---

# 30. search_files

Signature:

```python
search_files(
    query: str,
    path: str = ".",
) -> str
```

Purpose:

Recursively search text files inside the workspace.

Recommended implementation:

```python
Path.rglob("*")
```

with filtering and safety checks.

---

# 31. search_files Directory Filtering

Skip common generated or dependency directories:

```text
.git
__pycache__
node_modules
.venv
venv
dist
build
```

This list should be configurable.

Do not scatter these names throughout the implementation.

Use a constant:

```python
DEFAULT_IGNORED_DIRECTORIES = {...}
```

---

# 32. search_files Binary Detection

Do not blindly decode every file.

For each candidate file:

1. check file size
2. inspect a small byte sample
3. detect obvious binary data
4. skip binary files
5. decode text as UTF-8 with replacement

A simple NUL-byte check is sufficient for the initial implementation.

Do not build an elaborate MIME-detection system into V0.

---

# 33. search_files File Limits

Initial defaults:

```text
max_results = 50
context_lines = 2
max_file_size = 2 MB
```

These must be configuration values.

The search tool must never return entire matching files.

---

# 34. search_files Output

Example:

```text
Found 3 matches.

src/auth/login.py:42
    authenticate_user(user)

src/api/routes.py:18
    from auth.login import authenticate_user

tests/test_auth.py:11
    authenticate_user(test_user)
```

Each match should contain:

* relative file path
* line number
* limited context

The output must be deterministic.

---

# 35. Symlinks During Search

Search must not follow symlinks that can escape the workspace.

A symlink should be:

* skipped
* or resolved and verified before use

The simpler V0 behavior is to skip symlink directories/files.

---

# 36. write_file

Signature:

```python
write_file(
    path: str,
    content: str,
) -> str
```

Behavior:

1. validate workspace path
2. validate destination
3. optionally validate parent directory
4. write UTF-8 content
5. return concise success result

Example:

```text
Successfully wrote 183 bytes to src/config.py.
```

V0 must not support:

* append
* binary writes
* deletion
* chmod
* rename
* arbitrary directory mutation

---

# 37. write_file Confirmation

`write_file` is modifying and potentially destructive.

The architecture must therefore support:

```python
requires_confirmation = True
```

or an equivalent safety-policy check.

For the initial interactive implementation, confirmation should be possible before execution.

The graph status should support:

```text
WAITING_FOR_CONFIRMATION
```

The framework must never assume that model confidence authorizes a write.

---

# 38. calculator

Signature:

```python
calculator(expression: str) -> str
```

Never use:

```python
eval(expression)
```

Implement through Python's AST.

Allowed:

```text
numbers
+
-
*
/
**
%
()
```

Potentially unary:

```text
+x
-x
```

All other AST nodes must be rejected.

Examples of rejected input:

```python
__import__("os")
open("file")
[x for x in range(10)]
foo()
```

The calculator must be deterministic and side-effect free.

---

# 39. get_time

Signature:

```python
get_time(timezone: str | None = None) -> str
```

Use:

```python
datetime
zoneinfo
```

Default behavior:

* use configured application timezone
* if none exists, use system/local timezone

Invalid timezone names must return tool errors.

Example:

```text
Current time:
2026-09-05 15:10:23 IST
```

---

# 40. ask_user

Signature:

```python
ask_user(question: str) -> str
```

The tool is an abstraction over user interaction.

The core tool must not hard-code CLI assumptions.

Conceptually:

```python
class InteractionHandler(Protocol):
    def ask(self, question: str) -> str:
        ...
```

V0 can provide:

```text
CLIInteractionHandler
```

Future interfaces may include:

```text
WebInteractionHandler
GUIInteractionHandler
APIInteractionHandler
```

---

# 41. Tool Errors

Tools should raise internal `ToolError` exceptions for expected operational failures.

The executor converts these to:

```python
ToolResult(
    success=False,
    error="..."
)
```

Tool failures must normally become observations rather than crashing the entire agent.

Example:

```text
Tool `read_file` failed.

Reason:
The requested file does not exist.
```

The reasoning model can then recover.

---

# 42. Agent State

Keep LangGraph state deliberately small.

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

Allowed statuses:

```text
RUNNING
WAITING_FOR_CONFIRMATION
COMPLETED
MAX_STEPS_REACHED
ERROR
```

Do not put these in state:

* model instances
* tool registry
* Needle instance
* executor
* configuration objects
* context manager
* logging system
* filesystem root
* arbitrary service objects

Those are runtime dependencies, not conversational state.

---

# 43. LangGraph Workflow

The canonical graph is:

```text
START
  ↓
REASON
  ↓
PARSE
  ├────────────── FINAL ──────────────→ END
  │
  ▼ TOOL
TRANSLATE
  ↓
SANITIZE
  ↓
VALIDATE
  ↓
CONFIDENCE
  ├──────── LOW ───────→ CONFIRM
  │                         │
  │                         ▼
  │                       REASON
  │
  ▼ HIGH
SAFETY
  ↓
EXECUTE
  ↓
OBSERVE
  ↓
UPDATE_CONTEXT
  ↓
REASON
```

This is the core control loop.

---

# 44. REASON Node

Responsibilities:

1. retrieve the current reasoning context
2. invoke the `ReasoningModel`
3. store the generated response
4. increment no tool counter yet

The node must not parse the response itself beyond basic storage.

---

# 45. PARSE Node

Responsibilities:

1. parse the reasoning response
2. detect `<final>`
3. detect `<tool>`
4. reject malformed protocol
5. determine next graph branch

Possible outcomes:

```text
FINAL
TOOL
ERROR
```

If neither `<tool>` nor `<final>` is present, V0 should treat the response as a final answer rather than inventing a tool action.

---

# 46. TRANSLATE Node

Input:

```text
natural-language tool action
```

Output:

```text
NeedleResult
```

Example:

```text
Input:
"Search the src directory for authenticate_user"

Output:
selected_tool = search_files
arguments = {
    "query": "authenticate_user",
    "path": "src"
}
confidence = 0.94
```

No execution occurs here.

---

# 47. SANITIZE Node

Input:

```text
NeedleResult
```

Output:

```text
normalized ToolCall
```

Failures go to the reasoning recovery path.

Example:

```text
The action translator produced an invalid tool call.

No action was executed.

Please issue a clearer action.
```

---

# 48. VALIDATE Node

Validate the normalized call against the registered canonical tool schema.

Validation must verify:

```text
tool exists
required arguments exist
argument types are correct
arguments are acceptable
```

No execution occurs here.

---

# 49. CONFIDENCE Node

Logic:

```python
if result.confidence >= config.confidence_threshold:
    route("high")
else:
    route("low")
```

Do not make this a tool-specific hard-coded rule.

The threshold belongs in configuration.

---

# 50. CONFIRM Node

Despite the name, this node does not necessarily mean asking the human.

For low-confidence action-model output, it means:

> Return the ambiguity to the reasoning layer.

The node should construct a message containing:

* original requested action
* candidate selected tool
* confidence
* ranked alternatives
* statement that nothing was executed

Then route back to `REASON`.

For destructive tools, actual human confirmation is handled by the safety layer and `ask_user`.

---

# 51. SAFETY Node

The safety node checks:

```text
tool policy
path policy
confirmation policy
resource limits
permission policy
```

For example:

```text
write_file
→ workspace check
→ destructive-operation check
→ confirmation required
```

A rejected action must not reach `EXECUTE`.

---

# 52. EXECUTE Node

The executor is the only component allowed to invoke `tool.handler`.

Conceptually:

```python
result = executor.execute(tool_call)
```

The executor must:

1. look up tool
2. ensure the tool remains registered
3. enforce safety assumptions
4. invoke handler
5. catch known tool errors
6. catch unexpected exceptions
7. convert result into `ToolResult`
8. increment the tool-step counter

Unexpected exceptions should be recorded while returning a controlled failure to the graph.

---

# 53. Maximum Tool Steps

Every tool execution increments:

```python
step_count += 1
```

Before executing a new tool:

```python
if step_count >= max_tool_steps:
    terminate
```

Default:

```text
20 tool executions
```

The value must be configurable.

When reached:

```text
status = MAX_STEPS_REACHED
```

The agent must terminate honestly rather than pretending the task completed.

---

# 54. OBSERVE Node

Tool output should be converted into a compact natural-language observation.

Example:

```text
Tool: search_files
Status: success

The search found 3 matching locations:

src/auth/login.py:42
src/api/routes.py:18
tests/test_auth.py:11
```

The observation should be understandable to the reasoning model without exposing internal implementation details unnecessarily.

---

# 55. UPDATE_CONTEXT Node

Append the observation to the reasoning conversation.

Then invoke the context manager.

The context manager must trim content when the configured context limit is exceeded.

---

# 56. Context Manager

V0 uses a simple bounded sliding-window strategy.

No:

* vector database
* embeddings
* retrieval system
* summarization model
* persistent semantic memory

The context manager receives:

```text
system prompt
original user request
reasoning history
tool observations
clarification messages
```

and constructs a bounded model context.

---

# 57. Context Priority

When trimming context, preserve in this order:

1. system instructions
2. original user request
3. most recent interaction
4. most recent tool observation
5. previous reasoning/action messages
6. older observations

Do not delete the original request merely because the context is large.

---

# 58. Tool Output Limits

Every tool must produce bounded output.

This protects both:

* memory
* reasoning context

Do not pass multi-megabyte files or search results directly into the reasoning model.

Tool-level limits are preferred over a global last-minute truncation because the tool can produce better summaries.

---

# 59. Observability

Logging must be separate from AgentState.

The framework should support structured events such as:

```text
agent.started
reasoning.generated
protocol.parsed
action.translated
action.rejected
action.validated
confidence.checked
safety.checked
tool.started
tool.completed
tool.failed
context.trimmed
agent.completed
agent.failed
```

Each event may include:

```text
timestamp
run_id
step
node
tool_name
latency
status
```

Do not log:

* API keys
* credentials
* arbitrary secrets
* full sensitive file contents by default

---

# 60. Run Identification

Every agent invocation should have a `run_id`.

This should be used for:

* logging
* debugging
* tracing
* benchmark correlation

It does not need to be stored as conversational state unless the chosen LangGraph execution design requires it.

---

# 61. Configuration

Create one configuration layer.

Example:

```python
@dataclass
class AgentConfig:
    workspace_root: Path
    max_tool_steps: int = 20
    confidence_threshold: float = 0.85
    max_context_chars: int = ...
    max_file_size_bytes: int = ...
    search_max_results: int = 50
    search_context_lines: int = 2
    default_timezone: str | None = None
```

Do not scatter constants throughout the project.

---

# 62. Dependency Policy

The dependency philosophy is:

```text
stdlib whenever practical
+
LangGraph
+
Pydantic
+
Needle
```

Current reference releases verified for this specification include LangGraph 1.2.11, Pydantic 2.13.5, and cactus-needle 2.0.12.

Do not hard-code these exact patch versions in the main compatibility constraints.

Use loose compatible bounds and lock exact resolved versions in the project's lockfile.

Recommended:

```toml
[project]
requires-python = ">=3.11,<3.14"

dependencies = [
    "langgraph>=1.2,<1.3",
    "pydantic>=2.13,<3",
    "cactus-needle>=2.0,<3",
]
```

Development dependencies:

```toml
[dependency-groups]
dev = [
    "pytest>=8,<9",
    "pytest-asyncio>=1,<2",
    "ruff>=0.12,<1",
]
```

Python 3.11 is recommended as the project's minimum despite Needle itself supporting older Python versions, because the framework should use one clean modern baseline and LangGraph currently requires Python >=3.10.

---

# 63. LangGraph Dependency Boundary

LangGraph should be used strictly for orchestration.

Do not let application code become:

```text
LangChain → tool definitions → model-specific abstraction → LangGraph
```

Instead:

```text
Agent Runtime Interfaces
        ↓
LangGraph implementation
```

LangGraph is an implementation detail of the workflow layer.

This keeps the core architecture understandable.

---

# 64. Project Structure

Recommended repository:

```text
agent-runtime/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
│
├── src/
│   └── relay/
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
│       │   ├── safety.py
│       │   └── executor.py
│       │
│       ├── context/
│       │   └── manager.py
│       │
│       ├── graph/
│       │   └── workflow.py
│       │
│       └── observability/
│           └── logging.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── examples/
│   └── basic.py
│
└── docs/
    ├── architecture.md
    ├── tools.md
    ├── models.md
    └── extending.md
```

Do not split modules further unless implementation complexity requires it.

---

# 65. `agent.py`

This is the public high-level entrypoint.

Conceptually:

```python
class Agent:
    def __init__(
        self,
        reasoning_model: ReasoningModel,
        action_model: ActionModel,
        tools: ToolRegistry,
        config: AgentConfig,
    ):
        ...
```

Public operation:

```python
result = agent.run("Find where authentication is implemented.")
```

The `Agent` class owns:

* runtime dependencies
* graph compilation
* invocation
* final result normalization

It should not contain the implementation of individual tools.

---

# 66. Agent Result

Recommended:

```python
class AgentResult(BaseModel):
    success: bool
    final_answer: str
    status: str
    steps: int
    run_id: str
```

Optional debugging information should be separately exposed through tracing rather than bloating the normal result.

---

# 67. Public API

The first public API should be intentionally small.

Potential package surface:

```python
from relay import Agent
from relay.config import AgentConfig
from relay.models import ReasoningModel, ActionModel
from relay.tools import Tool, ToolRegistry
```

Avoid exposing internal graph nodes as primary public APIs.

---

# 68. Model Loading

Model loading should not be hidden inside `Agent`.

The caller should construct models and inject them.

Example conceptually:

```python
reasoning_model = MyLocalReasoningModel(...)
action_model = NeedleActionModel(...)

agent = Agent(
    reasoning_model=reasoning_model,
    action_model=action_model,
    tools=registry,
    config=config,
)
```

This makes testing trivial.

---

# 69. Mock Models

Testing must use mock models.

Example reasoning mock:

```python
class MockReasoningModel:
    def generate(self, messages):
        return "<tool>Read the directory '.'</tool>"
```

Example action mock:

```python
class MockActionModel:
    def translate(self, action, tools):
        return NeedleResult(
            selected_tool="read_directory",
            arguments={"path": "."},
            confidence=0.99,
        )
```

No real model should be required for unit or integration tests.

---

# 70. Testing Strategy

Tests are mandatory at three levels.

## Unit

Test individual components.

## Integration

Test graph interaction with mocked models.

## E2E

Test with actual reasoning and action models.

---

# 71. Required Unit Tests

### Parser

* tool extraction
* final extraction
* malformed tags
* multiline
* multiple blocks

### Registry

* register
* duplicate rejection
* lookup
* schema validation

### Filesystem

* relative path
* valid absolute path within workspace
* `..` traversal
* external absolute path
* symlink escape
* missing file
* missing directory
* permission failure

### read_file

* UTF-8
* invalid byte sequence
* size limit

### read_directory

* file/directories
* sorting
* hidden entries
* nonexistent directory

### search_files

* recursive matching
* line numbers
* context
* ignored directories
* binary files
* size limits
* result limits

### calculator

* valid arithmetic
* unary operations
* precedence
* division
* invalid AST
* attempted code execution

### get_time

* default timezone
* valid timezone
* invalid timezone

### sanitizer

* valid Needle output
* missing fields
* unknown tool
* malformed arguments

### validator

* valid arguments
* missing argument
* wrong type
* unknown argument

### confidence

* above threshold
* below threshold
* boundary condition

### context manager

* normal context
* trimming
* system preservation
* user request preservation
* tool observation preservation

---

# 72. Integration Test

Mandatory integration flow:

```text
User
 ↓
Reasoning Mock
 ↓
Parser
 ↓
Action Mock
 ↓
Sanitizer
 ↓
Validator
 ↓
Filesystem Tool
 ↓
Tool Result
 ↓
Observation
 ↓
Context Manager
 ↓
Reasoning Mock
 ↓
Final
```

The test should prove the full loop works without using real LLMs.

---

# 73. Recovery Integration Tests

Also test:

```text
low confidence
→ reasoning retry
```

```text
invalid tool arguments
→ reasoning retry
```

```text
tool failure
→ reasoning recovery
```

```text
maximum step count
→ clean termination
```

```text
unsafe filesystem path
→ execution blocked
```

---

# 74. End-to-End Scenarios

At minimum:

### Scenario A — Directory exploration

User:

```text
Inspect this project and tell me what the main entry point is.
```

Expected behavior:

```text
reason
→ read_directory
→ reason
→ read_file
→ reason
→ final
```

### Scenario B — Search

```text
Find every place authentication is referenced.
```

Expected:

```text
reason
→ search_files
→ reason
→ final
```

### Scenario C — Write

```text
Create a file called hello.txt containing "Hello".
```

Expected:

```text
reason
→ write_file
→ safety/confirmation
→ execute
→ reason
→ final
```

### Scenario D — Invalid action

The action model returns a malformed call.

Expected:

```text
no execution
→ reasoning recovery
```

### Scenario E — Low confidence

Expected:

```text
no execution
→ ranked alternatives returned to reasoning
```

---

# 75. Error Philosophy

Errors are data whenever possible.

A bad tool call should not automatically crash the entire agent.

The preferred pattern is:

```text
failure
→ ToolResult(success=False)
→ observation
→ reasoning recovery
```

Only unrecoverable runtime failures should move the agent to:

```text
ERROR
```

Examples:

```text
model unavailable
graph failure
invalid configuration
broken runtime dependency
```

---

# 76. No Silent Recovery

The framework must never silently change model-produced intent.

Bad:

```text
Model:
read_file("auth.py")

Runtime:
auth.py doesn't exist.
I guessed src/auth.py.
```

Good:

```text
Tool call failed because auth.py does not exist.
```

The reasoning model is responsible for deciding what to try next.

---

# 77. One Action Per Reasoning Turn

This rule is important enough to be explicit.

V0 operates as:

```text
Reason
→ one action
→ execute
→ observe
→ Reason again
```

not:

```text
Reason
→ five actions
→ execute all five
```

Reasons:

* observations may invalidate future actions
* one action keeps the reasoning loop simple
* failures can be interpreted before continuing
* tool side effects become easier to control
* debugging becomes much easier

Parallel or batched execution can be added later.

---

# 78. Tool Results as Observations

The action model sees:

```text
natural-language action
```

The reasoning model sees:

```text
natural-language observation
```

Neither model needs to know the internal graph.

This is deliberate.

The graph is the deterministic boundary between cognition and environment.

---

# 79. Fine-Tuning Strategy

Fine-tuning Needle is an extension of the action layer, not a requirement of the runtime.

The training target is:

```text
natural-language action
→ correct tool
→ correct arguments
```

not:

```text
user request
→ complete reasoning
→ tool
→ execution
```

Needle's published tooling supports JSONL training examples and LoRA-based fine-tuning, so the planned dataset can follow that intent-to-call structure.

---

# 80. Fine-Tuning Dataset Categories

The dataset should contain:

### Direct commands

```text
Read main.py
```

### Paraphrases

```text
Open main.py and show me what's inside.
```

### Conversational language

```text
Can you have a look at main.py?
```

### Implicit intent

```text
I want to know what files are in src.
```

### Path variations

```text
src
./src/
"src"
```

### Ambiguous requests

```text
Look through the project for authentication.
```

### Missing arguments

```text
Read the file.
```

### Multiple plausible tools

```text
Check the project structure.
```

### Irrelevant requests

```text
Tell me a joke.
```

Expected action:

```text
no tool
```

### Malformed phrasing

The model should learn robust interpretation rather than brittle keyword matching.

---

# 81. Fine-Tuning Evaluation

Do not measure only tool-selection accuracy.

Measure separately:

```text
tool selection accuracy
argument accuracy
argument grounding
invalid-call rate
off-topic rejection
confidence calibration
end-to-end task success
```

This distinction matters because selecting the correct tool does not guarantee that the arguments are correct. A recent independent Needle 2 evaluation specifically observed that tool-selection improvements did not necessarily translate into equally strong argument grounding.

---

# 82. Benchmark Hypothesis

The project's main experimental hypothesis is:

> Separating natural-language reasoning from structured tool calling allows a small reasoning model to spend more of its capacity on cognition rather than formatting and tool-call syntax, while a tiny specialized action model handles structured action translation.

The project must measure whether this actually improves agent performance.

This is a hypothesis, not an assumption.

---

# 83. Benchmark Baselines

At minimum compare:

### A — Native tool-calling baseline

Small reasoning model directly emits structured calls.

### B — Proposed architecture

```text
small reasoning model
+
Needle
```

### C — Optional larger baseline

A larger model with native tool calling.

---

# 84. Benchmark Metrics

Measure:

```text
task completion rate
tool selection accuracy
argument accuracy
invalid calls
hallucinated tools
multi-step success
average latency
p95 latency
RAM
VRAM
token usage
number of reasoning turns
number of tool executions
failure recovery rate
```

The experiment should also report:

```text
Needle inference latency
reasoning-model latency
total end-to-end latency
```

---

# 85. Benchmark Task Set

Tasks should cover:

```text
directory exploration
file reading
text search
simple calculation
time query
ambiguous requests
multi-step investigation
tool failures
invalid paths
write operations
user clarification
```

Do not benchmark only trivial one-step calls.

The architecture exists primarily to improve agentic behavior.

---

# 86. Security Requirements

The first release must consider the model untrusted.

Never assume:

```text
LLM output is safe
Needle output is safe
confidence means authorization
tool description means permission
```

All execution must be controlled by deterministic runtime code.

---

# 87. Sensitive Information Handling

Logging must avoid exposing:

* API keys
* tokens
* credentials
* secrets
* unnecessary file contents

Tool outputs should only be logged at full detail when explicitly configured.

Default logs should prefer metadata.

---

# 88. Resource Safety

Filesystem tools must enforce:

```text
maximum file size
maximum search results
maximum context size
maximum tool steps
```

Future tools should follow the same principle.

Every potentially unbounded operation should have a limit.

---

# 89. Future Permission System

The architecture should leave room for:

```python
ToolPermission
```

or:

```python
SafetyPolicy
```

with decisions such as:

```text
ALLOW
DENY
REQUIRE_CONFIRMATION
```

V0 only needs enough implementation to protect filesystem operations and writes.

Do not build an elaborate policy engine prematurely.

---

# 90. Future Tool Categories

The eventual architecture should support:

```text
Filesystem
Web
Browser
Database
Shell
Python
GUI
OS
Remote APIs
MCP
```

Each should become a `Tool`.

The runtime should not need to know what a tool actually does beyond:

```text
schema
metadata
handler
safety policy
```

---

# 91. Why Shell Is Not V0

A shell tool fundamentally changes the security model.

This:

```text
run("ls")
```

is simple.

This:

```text
run("rm -rf ...")
```

is not.

Adding shell execution would require:

* sandboxing
* command policies
* environment control
* process limits
* timeout controls
* output limits
* filesystem isolation
* user confirmation

Therefore it is intentionally deferred.

---

# 92. Why Python Execution Is Not V0

Unrestricted Python execution is effectively arbitrary code execution.

A proper implementation would require a sandbox.

Therefore:

```text
run_python
```

must not exist in the first release.

---

# 93. Action Model Replacement

Replacing Needle should require only implementing:

```python
class ActionModel(Protocol):
    def translate(
        self,
        action: str,
        tools: list[Tool],
    ) -> NeedleResult:
        ...
```

The implementation may internally use:

```text
Needle
FunctionGemma
custom transformer
API
rule-based parser
```

The graph and tools remain unchanged.

---

# 94. Reasoning Model Replacement

Likewise:

```python
class ReasoningModel(Protocol):
    def generate(self, messages: list) -> str:
        ...
```

Possible implementations:

```text
local GGUF model
Transformers model
Ollama adapter
llama.cpp adapter
OpenAI-compatible API
OpenRouter adapter
custom inference engine
```

The runtime must not care.

---

# 95. Custom Model Metadata

Model-specific settings should not leak into AgentState.

Example:

```python
NeedleActionModel(
    weights="..."
)
```

or:

```python
LocalReasoningModel(
    model_path="..."
)
```

These belong in the model adapter.

---

# 96. Installation Experience

The user should be able to install the framework with something conceptually like:

```bash
pip install agent-runtime
```

Then:

```python
from relay import Agent
```

Development installation:

```bash
pip install -e ".[dev]"
```

The repository should document any platform-specific Needle setup separately.

Needle currently distributes its inference package through `cactus-needle` and provides platform-specific runtime engines; its package is Python >=3.9 while the framework itself intentionally uses the newer Python baseline described above.

---

# 97. README Requirements

README must explain:

1. what the project is
2. why the architecture exists
3. architecture diagram
4. installation
5. minimal example
6. adding a tool
7. replacing the reasoning model
8. replacing the action model
9. safety model
10. benchmark philosophy

The README should not require the reader to understand LangGraph first.

Explain the architecture before implementation details.

---

# 98. Extension Example

Documentation should show something like:

```python
registry.register(
    Tool(
        name="my_tool",
        description="...",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"],
        },
        handler=my_handler,
    )
)
```

The developer should not need to modify:

```text
graph
parser
executor
Needle adapter
```

to add a normal tool.

---

# 99. Graph Compilation

The workflow should be built once from injected runtime dependencies.

Conceptually:

```python
workflow = build_workflow(
    reasoning_model=reasoning_model,
    action_model=action_model,
    tool_registry=registry,
    context_manager=context_manager,
    safety_policy=safety_policy,
    config=config,
)
```

The compiled graph becomes the execution engine.

---

# 100. Dependency Injection

Every major subsystem should be injectable:

```text
ReasoningModel
ActionModel
ToolRegistry
ContextManager
SafetyPolicy
InteractionHandler
```

This enables:

* testing
* alternative implementations
* downstream customization
* future plugins

---

# 101. Default Implementations

V0 provides:

```text
NeedleActionModel
SimpleContextManager
DefaultSafetyPolicy
CLIInteractionHandler
built-in tools
LangGraph workflow
```

A user can replace any of them independently.

---

# 102. Thread / Session Model

A single agent invocation represents one task/session.

The architecture should allow a future session identifier but V0 does not need persistent memory.

The simplest conceptual flow is:

```python
agent.run(user_input)
```

Each invocation begins with fresh conversational state unless the caller explicitly provides a conversation mechanism later.

---

# 103. Concurrency

V0 should not run multiple tool actions in parallel.

The runtime should remain sequential:

```text
reason
→ action
→ execute
→ observe
→ reason
```

This produces a clean causal chain.

Parallel tools can be added after correctness and benchmarks are established.

---

# 104. Determinism

Where possible, deterministic runtime behavior is required.

The following must be deterministic:

* tool registration
* tool ordering
* directory sorting
* search result ordering
* schema generation
* validation
* path authorization
* graph routing

Model outputs are inherently nondeterministic, but the runtime itself should not introduce unnecessary nondeterminism.

---

# 105. Tool Ordering

When producing human-readable or structured tool lists, sort consistently.

Recommended:

```python
sorted(tools, key=lambda t: t.name)
```

This ensures stable prompts, stable Needle schema ordering, and reproducible tests.

---

# 106. API Error Handling

Public methods should distinguish:

```text
configuration error
model error
protocol error
tool error
safety denial
agent termination
```

Do not throw arbitrary raw exceptions from deep implementation layers when a typed error can be returned.

---

# 107. Status Semantics

### RUNNING

Agent currently executing.

### WAITING_FOR_CONFIRMATION

A user decision is required before execution.

### COMPLETED

A final answer was produced.

### MAX_STEPS_REACHED

Execution stopped because the configured tool-step limit was reached.

### ERROR

An unrecoverable runtime failure occurred.

---

# 108. Completion Requirement

An agent is considered successful only when:

```text
<final>
...
</final>
```

is produced and accepted by the protocol parser.

The runtime must not infer successful completion merely because:

* a tool succeeded
* the model stopped talking
* the model generated an ordinary paragraph

Ordinary untagged text may be treated as final for V0 convenience, but explicit `<final>` should be the preferred protocol.

---

# 109. Example Full Run

User:

```text
Find where the application starts.
```

Reasoning model:

```text
I should inspect the project structure.

<tool>
Read the project directory.
</tool>
```

Parser:

```text
TOOL
Read the project directory.
```

Needle:

```json
{
  "name": "read_directory",
  "arguments": {
    "path": "."
  },
  "confidence": 0.97
}
```

Validator:

```text
valid
```

Safety:

```text
allowed
```

Executor:

```text
read_directory(".")
```

Observation:

```text
The project contains:

Directories:
- src
- tests

Files:
- pyproject.toml
- README.md
```

Reasoning model:

```text
The main application is likely under src.

<tool>
Read the source directory.
</tool>
```

Needle:

```json
{
  "name": "read_directory",
  "arguments": {
    "path": "src"
  },
  "confidence": 0.98
}
```

Observation:

```text
src contains:

Files:
- main.py
- config.py
- agent.py
```

Reasoning model:

```text
<tool>
Read src/main.py.
</tool>
```

Observation:

```text
The file defines the application's main entrypoint.
```

Reasoning model:

```text
<final>
The main application entry point is src/main.py.
</final>
```

Graph:

```text
END
```

---

# 110. Example Low-Confidence Run

Reasoning:

```text
<tool>
Look through the project for authentication.
</tool>
```

Needle:

```text
search_files 0.52
read_directory 0.37
read_file 0.11
```

Confidence threshold:

```text
0.85
```

No tool executes.

Reasoning receives:

```text
The action translator is uncertain.

Requested action:
"Look through the project for authentication."

Candidates:
search_files — 0.52
read_directory — 0.37
read_file — 0.11

No action was executed.
```

Reasoning may then generate:

```text
I'll search the project for authentication-related references.

<tool>
Search the project recursively for "authentication".
</tool>
```

Needle now produces:

```text
search_files(...)
```

The runtime continues.

---

# 111. Example Unsafe Run

Reasoning:

```text
<tool>
Read /etc/passwd.
</tool>
```

Needle:

```json
{
  "name": "read_file",
  "arguments": {
    "path": "/etc/passwd"
  },
  "confidence": 0.99
}
```

Validation:

```text
valid schema
```

Confidence:

```text
high
```

Safety:

```text
DENY
```

Execution:

```text
never called
```

Observation:

```text
The requested path is outside the configured workspace and cannot be accessed.
```

Reasoning continues.

This demonstrates why confidence is not a security mechanism.

---

# 112. Code Quality Requirements

The implementation must use:

* Python type hints
* small functions
* explicit interfaces
* meaningful names
* docstrings on public classes/functions
* no giant monolithic `agent.py`
* no global mutable registries
* no hidden singleton state
* no arbitrary `except Exception: pass`

Ruff should enforce basic style/quality rules.

---

# 113. Type Checking

A type checker should eventually be part of CI.

The initial development dependency set may omit a specific type checker to keep the first iteration small, but the code should be written with static typing in mind.

A later release may add:

```text
mypy
```

or:

```text
pyright
```

after the public interfaces stabilize.

---

# 114. CI

GitHub Actions should eventually run:

```text
lint
unit tests
integration tests
package build
```

against supported Python versions.

At minimum test:

```text
3.11
3.12
3.13
```

LangGraph currently publishes support across these Python versions.

---

# 115. Release Structure

The repository should use semantic versioning:

```text
0.x
```

during rapid API evolution.

For example:

```text
0.1.0
```

for the initial public implementation.

A `1.0.0` release should only happen once:

* public interfaces are stable
* safety model is mature
* benchmarks exist
* documentation is complete
* extension mechanisms are proven

---

# 116. Documentation Structure

Minimum documentation:

```text
README
Architecture
Installation
Tool Development
Model Adapters
Safety
Testing
Benchmarking
Fine-Tuning
Contributing
```

The architecture document should contain the same conceptual model as this specification.

---

# 117. Fine-Tuned Model Support

The runtime must allow:

```python
NeedleActionModel(weights="custom.cact")
```

without any graph modification.

Needle's current ecosystem supports custom model weights and LoRA/fine-tuning workflows.

The project should document custom-weight usage separately from the normal runtime.

---

# 118. Needle Telemetry

Needle currently documents anonymous telemetry and provides an opt-out mechanism.

The framework documentation should mention that users should consult Needle's configuration if they require fully disabled telemetry in privacy-sensitive deployments.

The framework itself must not add telemetry unless explicitly enabled by users.

---

# 119. Open-Source Licensing

Choose a permissive license compatible with the project's dependencies.

Apache-2.0 is a sensible candidate for the framework because Needle 2 itself is distributed under Apache-2.0.

The final repository should clearly distinguish:

```text
framework license
third-party dependency licenses
model licenses
```

especially for redistributed model files.

---

# 120. Implementation Order

The coding agent should implement in this order.

## Phase 1 — Core contracts

Create:

```text
Tool
ToolCall
ToolResult
NeedleResult
AgentState
AgentConfig
ReasoningModel
ActionModel
```

## Phase 2 — Tool registry

Implement:

```text
ToolRegistry
schema generation
tool descriptions
```

## Phase 3 — Built-in tools

Implement:

```text
read_file
read_directory
search_files
write_file
calculator
get_time
ask_user
```

## Phase 4 — Safety

Implement:

```text
resolve_safe_path
workspace policy
resource limits
confirmation mechanism
```

## Phase 5 — Protocol

Implement parser and tests.

## Phase 6 — Action model

Implement:

```text
NeedleActionModel
sanitizer
validator
confidence routing
```

## Phase 7 — Context

Implement bounded context manager.

## Phase 8 — Graph

Implement the complete LangGraph workflow.

## Phase 9 — Mock integration

Run all graph tests without real models.

## Phase 10 — Real models

Integrate:

```text
real reasoning model
real Needle
```

## Phase 11 — E2E

Run real project scenarios.

## Phase 12 — Benchmarks

Implement the baseline comparison.

## Phase 13 — Fine-tuning

Only after the baseline runtime and benchmark suite are stable.

---

# 121. Definition of Done

The first public release is complete when all of the following are true:

### Architecture

* reasoning/action/runtime separation works
* action model is replaceable
* reasoning model is replaceable
* tools are independently extensible

### Runtime

* LangGraph loop works
* one action per reasoning turn
* max-step protection works
* low-confidence routing works
* tool errors recover correctly

### Tools

* all seven V0 tools work
* filesystem tools are workspace restricted
* calculator cannot execute code
* output limits work

### Protocol

* parser handles valid and malformed responses
* explicit final responses work
* invalid tool blocks do not execute

### Testing

* unit tests pass
* integration tests pass
* filesystem security tests pass
* mock-model graph tests pass

### Documentation

* installation documented
* architecture documented
* custom tools documented
* custom models documented
* safety documented
* benchmark methodology documented

### Packaging

* package builds cleanly
* dependencies are bounded, not unnecessarily exact-pinned
* lockfile provides reproducibility
* CI passes

---

# 122. Final Architectural Contract

The most important contract in the entire system is:

```text
REASONING
    ↓
natural-language action
    ↓
ACTION MODEL
    ↓
structured action
    ↓
SANITIZATION
    ↓
SCHEMA VALIDATION
    ↓
CONFIDENCE
    ↓
SAFETY
    ↓
EXECUTION
    ↓
OBSERVATION
    ↓
REASONING
```

No layer should bypass the layer immediately below it.

In particular:

```text
Reasoning model ──X──> Tool handler
Needle ───────────X──> Tool handler
Tool ─────────────X──> Reasoning model
```

The runtime remains the authority over execution.

The reasoning model remains the authority over interpretation and planning.

The action model remains a replaceable translator between the two.

That separation is the defining architectural property of the project.

---

# 123. Final Product Vision

The first release should not attempt to be the most feature-rich agent framework.

Its purpose is to establish a clean experimental architecture:

```text
small reasoning model
        +
small specialized action model
        +
deterministic runtime
        +
explicit safety boundary
        +
LangGraph orchestration
```

The central research/product question is then measurable:

> Can a small reasoning model become a more effective agent when structured action generation is delegated to a dedicated tiny action model?

Everything else in the implementation should support answering that question cleanly, reproducibly, and safely.
