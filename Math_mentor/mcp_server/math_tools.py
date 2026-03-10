"""
Pure SymPy math computation functions.
These are the core tools used by the MCP server — no MCP dependency here.
"""

from sympy import (
    symbols, sympify, diff, solve, simplify, factor,
    binomial, factorial, N, Symbol, oo, zoo, nan, S
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor
)
from typing import Any


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _parse(expr_str: str):
    """Safely parse a string into a SymPy expression."""
    return parse_expr(expr_str, transformations=TRANSFORMATIONS)


# ── Tool 1: Derivative ──────────────────────────────────────────────────────

def compute_derivative(expression: str, variable: str = "x", order: int = 1) -> dict:
    """Compute the derivative of an expression with respect to a variable.

    Args:
        expression: Mathematical expression as string, e.g. "x**2*sin(x)"
        variable: Variable to differentiate with respect to (default "x")
        order: Order of the derivative (default 1)

    Returns:
        {"result": "<symbolic result>", "latex": "<latex form>"}
    """
    var = symbols(variable)
    expr = _parse(expression)
    result = diff(expr, var, order)
    return {
        "result": str(result),
        "latex": str(result),
        "simplified": str(simplify(result)),
    }


# ── Tool 2: Equation Solver ─────────────────────────────────────────────────

def solve_equation(equation: str, variable: str = "x") -> dict:
    """Solve an algebraic equation (set equal to zero).

    Args:
        equation: Expression string, e.g. "x**2 - 5*x + 6"
        variable: Variable to solve for (default "x")

    Returns:
        {"solutions": [list of solutions]}
    """
    var = symbols(variable)
    expr = _parse(equation)
    solutions = solve(expr, var)
    return {
        "solutions": [str(s) for s in solutions],
        "count": len(solutions),
    }


# ── Tool 3: Simplification ──────────────────────────────────────────────────

def simplify_expression(expression: str) -> dict:
    """Simplify an algebraic expression.

    Args:
        expression: Expression string, e.g. "x**2 + 2*x + 1"

    Returns:
        {"result": "<simplified expression>", "factored": "<factored form>"}
    """
    expr = _parse(expression)
    simplified = simplify(expr)
    factored = factor(expr)
    return {
        "result": str(simplified),
        "factored": str(factored),
    }


# ── Tool 4: Probability ─────────────────────────────────────────────────────

def compute_probability(prob_type: str, n: int, k: int = 0) -> dict:
    """Compute combinatorial / probability values.

    Args:
        prob_type: One of "combination", "permutation", "factorial"
        n: Total number of items
        k: Number of items to choose (for combination/permutation)

    Returns:
        {"result": <int>}
    """
    if prob_type == "combination":
        result = int(binomial(n, k))
    elif prob_type == "permutation":
        result = int(factorial(n) / factorial(n - k))
    elif prob_type == "factorial":
        result = int(factorial(n))
    else:
        return {"error": f"Unknown probability type: {prob_type}"}
    return {"result": result}


# ── Tool 5: Numerical Evaluation ────────────────────────────────────────────

def evaluate_numerically(expression: str, **variable_values) -> dict:
    """Evaluate an expression numerically with given variable values.

    Args:
        expression: Expression string, e.g. "2*x*sin(x) + x**2*cos(x)"
        **variable_values: Variable assignments, e.g. x=2

    Returns:
        {"result": <float>}
    """
    expr = _parse(expression)
    subs = {symbols(k): v for k, v in variable_values.items()}
    result = float(N(expr.subs(subs)))
    return {"result": result}
