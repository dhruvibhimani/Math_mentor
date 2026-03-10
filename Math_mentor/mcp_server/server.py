"""
MCP Server — Exposes SymPy math tools over the Model Context Protocol (stdio transport).

Run directly:
    python -m mcp_server.server

The solver agent launches this as a subprocess and communicates via stdin/stdout.
"""

import json
import sys
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import our pure math functions
sys.path.insert(0, ".")
from mcp_server.math_tools import (
    compute_derivative,
    solve_equation,
    simplify_expression,
    compute_probability,
    evaluate_numerically,
)


app = Server("math-tools-server")


# ── Tool Definitions ────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    Tool(
        name="derivative_tool",
        description="Compute the derivative of a mathematical expression. Input: expression (str), variable (str, default 'x'), order (int, default 1).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. 'x**2*sin(x)'"},
                "variable": {"type": "string", "default": "x"},
                "order": {"type": "integer", "default": 1},
            },
            "required": ["expression"],
        },
    ),
    Tool(
        name="equation_solver_tool",
        description="Solve an algebraic equation (expression = 0). Input: equation (str), variable (str, default 'x').",
        inputSchema={
            "type": "object",
            "properties": {
                "equation": {"type": "string", "description": "Equation expression set to 0, e.g. 'x**2 - 5*x + 6'"},
                "variable": {"type": "string", "default": "x"},
            },
            "required": ["equation"],
        },
    ),
    Tool(
        name="simplification_tool",
        description="Simplify or factor an algebraic expression. Input: expression (str).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Expression to simplify, e.g. 'x**2 + 2*x + 1'"},
            },
            "required": ["expression"],
        },
    ),
    Tool(
        name="probability_tool",
        description="Compute combinations, permutations, or factorials. Input: type ('combination'|'permutation'|'factorial'), n (int), k (int, optional).",
        inputSchema={
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["combination", "permutation", "factorial"]},
                "n": {"type": "integer"},
                "k": {"type": "integer", "default": 0},
            },
            "required": ["type", "n"],
        },
    ),
    Tool(
        name="numerical_eval_tool",
        description="Evaluate an expression numerically with given variable values. Input: expression (str), variables (object with variable names as keys and numeric values).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "variables": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Variable values, e.g. {'x': 2}",
                },
            },
            "required": ["expression"],
        },
    ),
]


# ── Handlers ────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOL_DEFINITIONS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to the appropriate math function."""
    try:
        if name == "derivative_tool":
            result = compute_derivative(
                arguments["expression"],
                arguments.get("variable", "x"),
                arguments.get("order", 1),
            )
        elif name == "equation_solver_tool":
            result = solve_equation(
                arguments["equation"],
                arguments.get("variable", "x"),
            )
        elif name == "simplification_tool":
            result = simplify_expression(arguments["expression"])
        elif name == "probability_tool":
            result = compute_probability(
                arguments["type"],
                arguments["n"],
                arguments.get("k", 0),
            )
        elif name == "numerical_eval_tool":
            variables = arguments.get("variables", {})
            result = evaluate_numerically(arguments["expression"], **variables)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ── Entry Point ─────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
