"""
Retriever Agent - Fetches grounded knowledge and citations from the vector store.
"""

from rag.retriever import retrieve_with_citations


def retrieve_context(state: dict) -> dict:
    parsed = state.get("parsed_problem", {})
    routing = state.get("routing", {})

    query_text = parsed.get("problem_text", state.get("raw_problem", ""))
    topic = routing.get("topic") or parsed.get("topic")

    context, citations = retrieve_with_citations(query_text, topic)
    has_support = bool(citations)

    return {
        **state,
        "retrieved_docs": context,
        "citations": citations,
        "has_retrieval_support": has_support,
    }
