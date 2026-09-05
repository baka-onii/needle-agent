"""Bounded, restricted AST arithmetic and stdlib timezone lookup."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_runtime.config import (
    MAX_AST_NODES,
    MAX_EXPONENT,
    MAX_EXPRESSION_CHARS,
    MAX_INTEGER_BITS,
    AgentConfig,
)
from agent_runtime.tools.base import Tool, ToolError

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _bounded(value: int | float) -> int | float:
    if type(value) is int and value.bit_length() <= MAX_INTEGER_BITS:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ToolError("Result is non-real, non-finite, or exceeds the calculator's size limit.")


def _calculate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _bounded(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _bounded(_UNARY_OPS[type(node.op)](_calculate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolError("Exponent exceeds the calculator's size limit.")
            if type(left) is int and right > 0 and left.bit_length() * right > MAX_INTEGER_BITS:
                raise ToolError("Power would exceed the calculator's size limit.")
        return _bounded(_BIN_OPS[type(node.op)](left, right))
    raise ToolError(f"Unsupported expression: {type(node).__name__}.")


def make_calculator_tool() -> Tool:
    def calculator(expression: str) -> str:
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ToolError("Expression exceeds the calculator's length limit.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
                raise ToolError("Expression is too complex.")
            return str(_calculate(tree))
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero.") from exc
        except (SyntaxError, ValueError, OverflowError, RecursionError) as exc:
            raise ToolError("Invalid or excessively large arithmetic expression.") from exc

    return Tool(
        name="calculator",
        description="Evaluate arithmetic with + - * / ** % and parentheses. No code execution.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_EXPRESSION_CHARS,
                    "description": "Arithmetic expression, e.g. '2 * (15 + 3)'.",
                }
            },
            "required": ["expression"],
        },
        handler=calculator,
    )


def make_get_time_tool(default_timezone: str | None = None) -> Tool:
    def get_time(timezone: str | None = None) -> str:
        name = timezone if timezone is not None else default_timezone
        try:
            tz = ZoneInfo(name) if name is not None else None
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"Unknown timezone: {name!r}") from exc
        return (datetime.now(tz) if tz else datetime.now().astimezone()).isoformat()

    return Tool(
        name="get_time",
        description="Return the current date and time as an ISO-8601 string.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "description": "IANA timezone, e.g. 'UTC'. Configured/local time if omitted.",
                }
            },
            "required": [],
        },
        handler=get_time,
    )


def utility_tools(config: AgentConfig | None = None) -> list[Tool]:
    return [make_calculator_tool(), make_get_time_tool((config or AgentConfig()).default_timezone)]
