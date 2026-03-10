"""
LangGraph workflow for the Reliable Multimodal Math Mentor.
"""

import os
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.explainer_agent import explain_solution
from agents.parser_agent import parse_problem
from agents.retriever_agent import retrieve_context
from agents.router_agent import route_problem
from agents.solver_agent import solve_problem
from agents.verifier_agent import verify_solution


MAX_RETRIES = 3


class MathMentorState(TypedDict, total=False):
    raw_problem: str
    input_mode: str
    parsed_problem: dict
    needs_clarification: bool
    routing: dict
    retrieved_docs: str
    citations: list
    has_retrieval_support: bool
    solution: dict
    verification: dict
    explanation: dict
    retry_count: int
    is_correct: bool
    verifier_feedback: str


def should_clarify(state: MathMentorState) -> str:
    return "needs_clarification" if state.get("needs_clarification", False) else "proceed"


def route_branch(state: MathMentorState) -> str:
    query_type = state.get("routing", {}).get("query_type", "concept_question")
    if query_type == "math_problem":
        return "math_problem"
    if query_type == "concept_question":
        return "concept_question"
    return "normal_conversation"


def after_retrieval_branch(state: MathMentorState) -> str:
    query_type = state.get("routing", {}).get("query_type", "concept_question")
    return "solver" if query_type == "math_problem" else "explainer"


def verification_decision(state: MathMentorState) -> str:
    if state.get("is_correct", False):
        return "correct"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "max_retries"


def build_graph() -> StateGraph:
    graph = StateGraph(MathMentorState)

    graph.add_node("parser", parse_problem)
    graph.add_node("router", route_problem)
    graph.add_node("retriever", retrieve_context)
    graph.add_node("solver", solve_problem)
    graph.add_node("verifier", verify_solution)
    graph.add_node("explainer", explain_solution)

    graph.set_entry_point("parser")

    graph.add_conditional_edges(
        "parser",
        should_clarify,
        {"needs_clarification": END, "proceed": "router"},
    )

    graph.add_conditional_edges(
        "router",
        route_branch,
        {
            "math_problem": "retriever",
            "concept_question": "retriever",
            "normal_conversation": "explainer",
        },
    )

    graph.add_conditional_edges(
        "retriever",
        after_retrieval_branch,
        {"solver": "solver", "explainer": "explainer"},
    )

    graph.add_edge("solver", "verifier")

    graph.add_conditional_edges(
        "verifier",
        verification_decision,
        {"correct": "explainer", "retry": "solver", "max_retries": "explainer"},
    )

    graph.add_edge("explainer", END)
    return graph.compile()


def run_pipeline(raw_problem: str, input_mode: str = "text") -> dict:
    graph = build_graph()
    initial_state: MathMentorState = {
        "raw_problem": raw_problem,
        "input_mode": input_mode,
        "retry_count": 0,
    }
    return graph.invoke(initial_state)
