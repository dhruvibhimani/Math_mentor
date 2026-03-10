"""
Explainer Agent - Produces grounded explanations for math and concept queries, and direct replies for conversation.
"""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import groq_client_kwargs

EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the explainer agent in a reliable multimodal math mentor.

Retrieved context:
{knowledge_context}

Rules:
- Explain clearly for a student.
- Use only citations that appear in the retrieved context.
- Never invent citations or formulas.
- If no retrieved document supports the answer, explicitly say "No reliable source found in knowledge base."
- For normal conversation, reply naturally and do not force math structure.

Return valid JSON only with exactly these fields:
{{
  "final_answer": "<concise final answer or reply>",
  "explanation": "<clear explanation in markdown>",
  "sources": ["<source names or citation ids>"],
  "confidence": <number between 0 and 1>,
  "feedback_prompt": "<short prompt inviting user feedback>"
}}""",
    ),
    (
        "human",
        "Query type: {query_type}\nProblem: {problem_text}\nTopic: {topic}\n\nSolution summary:\n{solution_summary}\n\nVerification summary:\n{verification_summary}",
    ),
])


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


def explain_solution(state: dict) -> dict:
    parsed = state.get("parsed_problem", {})
    routing = state.get("routing", {})
    solution = state.get("solution", {})
    verification = state.get("verification", {})
    citations = state.get("citations", [])
    query_type = routing.get("query_type", "concept_question")

    solution_steps = solution.get("solution_steps", [])
    solution_summary = "\n".join(
        f"Step {step.get('step', '?')}: {step.get('description', '')}\n{step.get('work', '')}"
        for step in solution_steps
    )
    if not solution_summary and query_type == "normal_conversation":
        solution_summary = parsed.get("problem_text", state.get("raw_problem", ""))

    verification_summary = json.dumps(verification, ensure_ascii=True, indent=2) if verification else "Not applicable."
    knowledge_context = state.get("retrieved_docs", "No reliable source found in knowledge base.")

    llm = ChatGroq(**groq_client_kwargs("EXPLAINER_MODEL", "llama-3.1-8b-instant", 0.2))
    response = (EXPLAINER_PROMPT | llm).invoke(
        {
            "knowledge_context": knowledge_context,
            "query_type": query_type,
            "problem_text": parsed.get("problem_text", state.get("raw_problem", "")),
            "topic": routing.get("topic", parsed.get("topic", "general")),
            "solution_summary": solution_summary or "No solution summary.",
            "verification_summary": verification_summary,
        }
    )

    explanation = _extract_json(response.content)
    if not explanation:
        fallback_answer = solution.get("final_answer") or parsed.get("problem_text", "")
        fallback_sources = [c.get("source", "") for c in citations if isinstance(c, dict)]
        explanation = {
            "final_answer": fallback_answer or "No reliable source found in knowledge base.",
            "explanation": response.content.strip() or fallback_answer,
            "sources": fallback_sources,
            "confidence": verification.get("confidence", solution.get("confidence", 0.5)),
            "feedback_prompt": "Was this answer correct?",
        }

    if not citations and query_type != "normal_conversation":
        explanation["sources"] = []
        if "No reliable source found in knowledge base." not in explanation.get("explanation", ""):
            explanation["explanation"] = (
                f"{explanation.get('explanation', '').strip()}\n\nNo reliable source found in knowledge base."
            ).strip()

    return {**state, "explanation": explanation}
