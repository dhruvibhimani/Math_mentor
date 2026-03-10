"""
Parser Agent - Normalizes multimodal user input into a compact structured form.
"""

import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


PARSER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the parser for a reliable multimodal math mentor.

Convert raw user input into a clean JSON object with exactly these fields:
{{
  "problem_text": "<cleaned user input>",
  "topic": "<calculus|algebra|probability|linear_algebra|geometry|trigonometry|general|conversation>",
  "expression": "<main mathematical expression or empty string>",
  "variables": ["<variable names>"],
  "constraints": ["<constraints or domain notes>"],
  "needs_clarification": <true or false>,
  "clarification_reason": "<reason or empty string>"
}}

Rules:
- Fix obvious OCR/ASR noise.
- Preserve the user's meaning.
- For normal conversation, use topic "conversation" and leave expression empty.
- Do not invent missing math structure.
- If the query is too incomplete to answer reliably, set needs_clarification=true.
- Output JSON only.""",
    ),
    ("human", "Parse this input:\n\n{raw_problem}"),
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


def parse_problem(state: dict) -> dict:
    raw_problem = state.get("raw_problem", "").strip()
    if not raw_problem:
        parsed = {
            "problem_text": "",
            "topic": "general",
            "expression": "",
            "variables": [],
            "constraints": [],
            "needs_clarification": True,
            "clarification_reason": "No problem text was provided.",
        }
        return {**state, "parsed_problem": parsed, "needs_clarification": True}

    llm = ChatGroq(model=os.getenv("PARSER_MODEL", "llama-3.1-8b-instant"), temperature=0)
    response = (PARSER_PROMPT | llm).invoke({"raw_problem": raw_problem})
    parsed = _extract_json(response.content)

    if not parsed:
        parsed = {
            "problem_text": raw_problem,
            "topic": "general",
            "expression": "",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
            "clarification_reason": "",
        }

    parsed.setdefault("problem_text", raw_problem)
    parsed.setdefault("topic", "general")
    parsed.setdefault("expression", "")
    parsed.setdefault("variables", [])
    parsed.setdefault("constraints", [])
    parsed.setdefault("needs_clarification", False)
    parsed.setdefault("clarification_reason", "")

    return {
        **state,
        "parsed_problem": parsed,
        "needs_clarification": bool(parsed.get("needs_clarification", False)),
    }
