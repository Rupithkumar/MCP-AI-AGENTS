"""Utils MCP server: calculator, sandboxed file reader, clock, text stats.

Security notes (worth mentioning in a write-up):
  * `calculate` parses the expression into an AST and only allows arithmetic nodes.
    It never calls eval().
  * `read_text_file` is confined to a sandbox directory (SANDBOX_DIR) and rejects
    path traversal such as ../../etc/passwd.
"""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SANDBOX_DIR = Path(os.environ.get("SANDBOX_DIR", Path(__file__).resolve().parent.parent / "workspace")).resolve()
MAX_READ_BYTES = 50_000

mcp = MCPServer("utils", instructions="Small safe utilities: math, time, text files in a sandbox.")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12.5 * 4) / 3 + 2**8'."""
    try:
        result = _eval_node(ast.parse(expression, mode="eval"))
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except SyntaxError:
        raise ToolError(f"could not parse: {expression!r}") from None
    return str(result)


@mcp.tool()
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@mcp.tool()
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file from the sandbox directory (path is relative to it)."""
    target = (SANDBOX_DIR / path).resolve()
    if not target.is_relative_to(SANDBOX_DIR):
        raise ToolError("access outside the sandbox is not allowed")
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    data = target.read_bytes()[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


@mcp.tool()
def word_count(text: str) -> str:
    """Count words, characters and lines in a piece of text."""
    return f"words={len(text.split())}; chars={len(text)}; lines={len(text.splitlines()) or 1}"


if __name__ == "__main__":
    mcp.run("stdio")
