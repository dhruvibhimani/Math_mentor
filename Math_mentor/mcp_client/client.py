"""Compatibility wrapper for math tool calls.

The project imports ``mcp_client.client.call_mcp_tool_sync`` but does not ship
the external MCP client package. This module provides the expected function and
dispatches directly to the local SymPy-backed math tools.
"""

from __future__ import annotations

from mcp_server.math_tools import (
    compute_derivative,
    compute_probability,
    evaluate_numerically,
    simplify_expression,
    solve_equation,
)


def call_mcp_tool_sync(tool_name: str, arguments: dict) -> dict:
    """Synchronously dispatch a supported math tool call."""
    if tool_name == "derivative_tool":
        return compute_derivative(
            arguments["expression"],
            arguments.get("variable", "x"),
            arguments.get("order", 1),
        )
    if tool_name == "equation_solver_tool":
        return solve_equation(
            arguments["equation"],
            arguments.get("variable", "x"),
        )
    if tool_name == "simplification_tool":
        return simplify_expression(arguments["expression"])
    if tool_name == "probability_tool":
        return compute_probability(
            arguments["type"],
            arguments["n"],
            arguments.get("k", 0),
        )
    if tool_name == "numerical_eval_tool":
        return evaluate_numerically(
            arguments["expression"],
            **arguments.get("variables", {}),
        )
    raise ValueError(f"Unknown tool: {tool_name}")
