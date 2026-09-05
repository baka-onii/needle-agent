"""Utility tools: calculator (restricted AST), get_time (stdlib datetime)."""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ToolError(f"Unsupported expression: {ast.dump(node)[:80]}")


def make_calculator_tool() -> Tool:
    def calculator(expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"Invalid expression: {exc}") from exc
        try:
            return str(_eval(tree))
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero.") from exc

    return Tool(
        name="calculator",
        description="Evaluate a simple arithmetic expression with + - * / ** % and parentheses.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 * (15 + 3)'.",
                }
            },
            "required": ["expression"],
        },
        handler=calculator,
    )


def make_get_time_tool() -> Tool:
    def get_time(timezone: str | None = None) -> str:
        try:
            tz = ZoneInfo(timezone) if timezone else None
        except ZoneInfoNotFoundError as exc:
            raise ToolError(f"Unknown timezone: {timezone!r}") from exc
        now = datetime.now(tz).astimezone() if tz is None else datetime.now(tz)
        return now.isoformat()

    return Tool(
        name="get_time",
        description="Return the current date and time as an ISO-8601 string.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'UTC'. Local time if omitted.",
                }
            },
            "required": [],
        },
        handler=get_time,
    )


def utility_tools() -> list[Tool]:
    return [make_calculator_tool(), make_get_time_tool()]
