"""
Verifier Agent - Validates mathematical solutions before explanation.
"""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import groq_client_kwargs
from mcp_client.client import call_mcp_tool_sync


VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the verifier agent in a reliable multimodal math mentor.

Numerical or symbolic checks:
{numerical_check}

Rules:
- Be strict about correctness.
- Check domain constraints and edge cases when possible.
- Do not invent evidence.

Return valid JSON only with exactly these fields:
{{
  "verified": <true or false>,
  "confidence": <number between 0 and 1>,
  "checks_performed": [{{"check": "<name>", "passed": <true or false>, "detail": "<detail>"}}],
  "issues_found": ["<issue>"],
  "feedback": "<what the solver should fix if needed>"
}}""",
    ),
    (
        "human",
        "Problem: {problem_text}\nExpression: {expression}\nIntent: {intent}\n\nSolution Steps:\n{solution_steps}\nFinal Answer: {final_answer}\nFinal Expression: {final_expression}",
    ),
])


def _build_check_context(parsed: dict, solution: dict, routing: dict) -> str:
    expression = parsed.get("expression", "")
    final_expression = solution.get("final_expression", "")
    intent = routing.get("intent", "")

    if not expression or not final_expression:
        return "No numerical check available."

    try:
        if intent == "derivative":
            base = call_mcp_tool_sync("numerical_eval_tool", {"expression": expression, "variables": {"x": 1.0}})
            nudged = call_mcp_tool_sync("numerical_eval_tool", {"expression": expression, "variables": {"x": 1.001}})
            derived = call_mcp_tool_sync("numerical_eval_tool", {"expression": final_expression, "variables": {"x": 1.0}})
            if all("result" in item for item in [base, nudged, derived]):
                approx = (nudged["result"] - base["result"]) / 0.001
                diff = abs(approx - derived["result"])
                return (
                    "Finite-difference derivative check at x=1.0: "
                    f"approx={approx:.6f}, claimed={derived['result']:.6f}, diff={diff:.6f}"
                )
        if intent in {"solve", "solve_equation"}:
            return "Equation check available if the final expression lists explicit roots."
    except Exception as exc:
        return f"Verification tool error: {exc}"

    return "No numerical check available."


def _extract_json(content: str) -> dict | None:
    content = content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def verify_solution(state: dict) -> dict:
    parsed = state.get("parsed_problem", {})
    solution = state.get("solution", {})
    routing = state.get("routing", {})
    retry_count = state.get("retry_count", 0)

    steps_text = "\n".join(
        f"Step {step.get('step', '?')}: {step.get('description', '')}\n{step.get('work', '')}"
        for step in solution.get("solution_steps", [])
    )
    numerical_check = _build_check_context(parsed, solution, routing)

    llm = ChatGroq(**groq_client_kwargs("VERIFIER_MODEL", "llama-3.3-70b-versatile", 0))
    response = (VERIFIER_PROMPT | llm).invoke(
        {
            "numerical_check": numerical_check,
            "problem_text": parsed.get("problem_text", state.get("raw_problem", "")),
            "expression": parsed.get("expression", ""),
            "intent": routing.get("intent", "general"),
            "solution_steps": steps_text,
            "final_answer": solution.get("final_answer", ""),
            "final_expression": solution.get("final_expression", ""),
        }
    )

    verification = _extract_json(response.content)
    if not verification:
        verification = {
            "verified": False,
            "confidence": 0.25,
            "checks_performed": [],
            "issues_found": ["Verifier could not parse its own output reliably."],
            "feedback": "Re-check the derivation and produce a simpler, grounded solution.",
        }

    verified = bool(verification.get("verified", False))
    new_retry_count = retry_count + 1 if not verified else retry_count

    return {
        **state,
        "verification": verification,
        "is_correct": verified,
        "verifier_feedback": verification.get("feedback", ""),
        "retry_count": new_retry_count,
    }
