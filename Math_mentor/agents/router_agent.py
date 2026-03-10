"""
Router Agent - Chooses between math problem solving, concept explanation, and normal conversation.
"""

import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the conversation router for a reliable multimodal math mentor.

Classify the parsed input into one of:
- math_problem
- concept_question
- normal_conversation

Return valid JSON only with exactly these fields:
{{
  "query_type": "<math_problem|concept_question|normal_conversation>",
  "requires_tools": <true or false>,
  "topic": "<topic label>",
  "intent": "<short intent label>",
  "solver_type": "<symbolic_solver|numerical_solver|reasoning_solver|none>",
  "tools": ["<tool names>"],
  "strategy": "<brief routing strategy>"
}}

Rules:
- Use math_problem only when the user needs computation, symbolic manipulation, or numeric solving.
- Use concept_question when the user is asking for an explanation, definition, intuition, or comparison.
- Use normal_conversation for greetings, thanks, small talk, or non-math conversation.
- Tools are allowed only for math_problem.
- Prefer no tools unless computation is genuinely needed.
- Output JSON only.""",
    ),
    ("human", "Route this parsed input:\n\n{parsed_problem}"),
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


def route_problem(state: dict) -> dict:
    parsed = state.get("parsed_problem", {})

    llm = ChatGroq(model=os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant"), temperature=0)
    response = (ROUTER_PROMPT | llm).invoke({"parsed_problem": json.dumps(parsed, ensure_ascii=True, indent=2)})
    routing = _extract_json(response.content)

    if not routing:
        topic = parsed.get("topic", "general")
        routing = {
            "query_type": "normal_conversation" if topic == "conversation" else "concept_question",
            "requires_tools": False,
            "topic": topic,
            "intent": "conversation" if topic == "conversation" else "general_explanation",
            "solver_type": "none" if topic == "conversation" else "reasoning_solver",
            "tools": [],
            "strategy": "Respond conservatively and avoid unsupported claims.",
        }

    routing.setdefault("query_type", "concept_question")
    routing.setdefault("requires_tools", False)
    routing.setdefault("topic", parsed.get("topic", "general"))
    routing.setdefault("intent", "general")
    routing.setdefault("solver_type", "reasoning_solver")
    routing.setdefault("tools", [])
    routing.setdefault("strategy", "Respond conservatively and avoid unsupported claims.")

    if routing["query_type"] != "math_problem":
        routing["requires_tools"] = False
        routing["tools"] = []
        if routing["query_type"] == "normal_conversation":
            routing["solver_type"] = "none"

    return {**state, "routing": routing}
