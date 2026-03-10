"""
Solver Agent - Computes answers for math problems, grounded by retrieval and tools.
"""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import groq_client_kwargs
from mcp_client.client import call_mcp_tool_sync


SOLVER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the solver agent in a reliable multimodal math mentor.

Use the retrieved context and tool outputs when provided.

Retrieved context:
{knowledge_context}

Tool outputs:
{tool_results}

Retry context:
{retry_context}

Rules:
- Solve only the user's math problem.
- Use citations only from the retrieved context.
- Do not invent formulas, steps, tool results, or citations.
- If the retrieved context does not support the answer, say "No reliable source found in knowledge base."
- If the tools fail or information is insufficient, state the limitation plainly.

Return valid JSON only with exactly these fields:
{{
  "solution_steps": [
    {{"step": 1, "description": "<what you did>", "work": "<math work>", "citation": "<citation id or empty>"}}
  ],
  "final_answer": "<final answer>",
  "final_expression": "<final symbolic expression or empty string>",
  "confidence": <number between 0 and 1>,
  "method_used": "<method>",
  "citations_used": ["<citation ids actually used>"],
  "source_status": "<grounded|no_reliable_source_found>"
}}""",
    ),
    (
        "human",
        "Problem: {problem_text}\nExpression: {expression}\nTopic: {topic}\nIntent: {intent}\nStrategy: {strategy}",
    ),
])


def _call_tools(routing: dict, parsed: dict) -> str:
    if not routing.get("requires_tools", False):
        return "No tools were called."

    expression = parsed.get("expression", "")
    tool_results: list[str] = []

    for tool_name in routing.get("tools", []):
        try:
            if tool_name == "derivative_tool" and expression:
                result = call_mcp_tool_sync("derivative_tool", {"expression": expression})
            elif tool_name == "equation_solver_tool" and expression:
                result = call_mcp_tool_sync("equation_solver_tool", {"equation": expression})
            elif tool_name in {"simplify_expression_tool", "simplification_tool"} and expression:
                result = call_mcp_tool_sync("simplification_tool", {"expression": expression})
            elif tool_name == "numeric_evaluation_tool" and expression:
                result = call_mcp_tool_sync("numerical_eval_tool", {"expression": expression, "variables": {"x": 1.0}})
            else:
                continue
            tool_results.append(f"{tool_name}: {json.dumps(result, ensure_ascii=True)}")
        except Exception as exc:
            tool_results.append(f"{tool_name}: error={exc}")

    return "\n".join(tool_results) if tool_results else "No tools were called."


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


def solve_problem(state: dict) -> dict:
    parsed = state.get("parsed_problem", {})
    routing = state.get("routing", {})
    retry_count = state.get("retry_count", 0)
    verifier_feedback = state.get("verifier_feedback", "")

    knowledge = state.get("retrieved_docs", "")
    citations = state.get("citations", [])
    tool_results = _call_tools(routing, parsed)

    retry_context = ""
    if retry_count > 0 and verifier_feedback:
        retry_context = (
            f"Retry attempt {retry_count}. The verifier reported: {verifier_feedback}. "
            "Correct the mathematical issues and keep citations grounded."
        )

    llm = ChatGroq(**groq_client_kwargs("SOLVER_MODEL", "llama-3.1-70b-versatile", 0))
    response = (SOLVER_PROMPT | llm).invoke(
        {
            "knowledge_context": knowledge,
            "tool_results": tool_results,
            "retry_context": retry_context or "None.",
            "problem_text": parsed.get("problem_text", state.get("raw_problem", "")),
            "expression": parsed.get("expression", ""),
            "topic": routing.get("topic", parsed.get("topic", "general")),
            "intent": routing.get("intent", "general"),
            "strategy": routing.get("strategy", ""),
        }
    )

    solution = _extract_json(response.content)
    if not solution:
        fallback_answer = response.content.strip()
        if not citations:
            fallback_answer = "No reliable source found in knowledge base."
        solution = {
            "solution_steps": [
                {
                    "step": 1,
                    "description": "Best-effort reasoning",
                    "work": fallback_answer,
                    "citation": citations[0]["id"] if citations else "",
                }
            ],
            "final_answer": fallback_answer,
            "final_expression": "",
            "confidence": 0.35 if citations else 0.1,
            "method_used": "llm_reasoning",
            "citations_used": [c["id"] for c in citations[:1]],
            "source_status": "grounded" if citations else "no_reliable_source_found",
        }

    if not citations:
        solution["source_status"] = "no_reliable_source_found"
        solution["citations_used"] = []
        if not solution.get("final_answer"):
            solution["final_answer"] = "No reliable source found in knowledge base."

    return {**state, "solution": solution}
