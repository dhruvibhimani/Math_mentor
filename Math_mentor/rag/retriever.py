"""
Retriever — Agentic knowledge retrieval with LLM re-ranking and citations.

Performs:
1. Initial vector search to get candidate chunks
2. LLM-based relevance scoring and re-ranking
3. If top results are low-relevance, widens search to other topics
4. Returns structured citations for each used formula/chunk
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rag.vector_store import similarity_search_with_scores, get_retriever


# ── LLM Re-ranker Prompt ────────────────────────────────────────────────────

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a math knowledge relevance judge.
Given a math problem and a list of retrieved knowledge chunks, score each chunk's relevance to solving the problem.

For EACH chunk, output a JSON object with:
- "index": the chunk index (integer)
- "relevant": true/false
- "reason": brief reason why it is or isn't relevant

Output a JSON array of these objects. Only output the JSON array, nothing else."""),
    ("human", """Math Problem: {problem}

Retrieved Chunks:
{chunks_text}

Score each chunk's relevance."""),
])


def _format_chunk_for_rerank(idx: int, doc, score: float) -> str:
    """Format a chunk for the re-ranker prompt."""
    source = doc.metadata.get("source", "unknown")
    formula = doc.metadata.get("formula_name", "Unknown")
    topic = doc.metadata.get("topic", "general")
    content_preview = doc.page_content[:400]
    return (
        f"[Chunk {idx}] Source: {source} | Topic: {topic} | Formula: {formula} "
        f"| VectorScore: {score:.3f}\n{content_preview}\n"
    )


def agentic_retrieve(
    query: str,
    topic: str | None = None,
    k_initial: int = 8,
    k_final: int = 4,
) -> list[dict]:
    """Agentic retrieval: vector search + LLM re-ranking.

    Args:
        query: The math problem text
        topic: Optional topic hint from the router
        k_initial: Number of initial candidates to retrieve
        k_final: Number of final chunks to keep after re-ranking

    Returns:
        List of citation dicts: {id, content, source, topic, formula_name, relevance_score, reason}
    """
    # 1. Initial vector search (with topic filter if provided)
    results = similarity_search_with_scores(query, k=k_initial, topic_filter=topic)

    # If topic-filtered search returns too few, also search without filter
    if len(results) < 3 and topic:
        broad_results = similarity_search_with_scores(query, k=k_initial, topic_filter=None)
        seen_contents = {doc.page_content for doc, _ in results}
        for doc, score in broad_results:
            if doc.page_content not in seen_contents:
                results.append((doc, score))
                seen_contents.add(doc.page_content)

    if not results:
        return []

    # 2. Format chunks for LLM re-ranking
    chunks_text = "\n".join(
        _format_chunk_for_rerank(i, doc, score)
        for i, (doc, score) in enumerate(results)
    )

    # 3. LLM re-ranking
    try:
        llm = ChatGroq(
            model=os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
        )
        chain = RERANK_PROMPT | llm
        response = chain.invoke({"problem": query, "chunks_text": chunks_text})
        content = response.content.strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        rankings = json.loads(content)
    except Exception:
        # Fallback: keep all, sorted by vector score
        rankings = [
            {"index": i, "relevant": True, "reason": "vector match"}
            for i in range(len(results))
        ]

    # 4. Filter relevant chunks and build citations
    relevant_indices = [
        r["index"] for r in rankings
        if r.get("relevant", False) and r["index"] < len(results)
    ]

    # If LLM filtered everything, fall back to top vector matches
    if not relevant_indices:
        relevant_indices = list(range(min(k_final, len(results))))

    # Build citation list
    citations = []
    reason_map = {r["index"]: r.get("reason", "") for r in rankings}

    for rank, idx in enumerate(relevant_indices[:k_final]):
        doc, score = results[idx]
        citations.append({
            "id": f"[{rank + 1}]",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "topic": doc.metadata.get("topic", "general"),
            "formula_name": doc.metadata.get("formula_name", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "relevance_score": round(score, 3),
            "reason": reason_map.get(idx, ""),
        })

    return citations


def retrieve_with_citations(
    query: str,
    topic: str | None = None,
    k_initial: int = 8,
    k_final: int = 4,
) -> tuple[str, list[dict]]:
    """Agentic retrieval returning both formatted context and citation list.

    Args:
        query: The math problem text
        topic: Optional topic hint
        k_initial: Number of initial candidates
        k_final: Number of final chunks

    Returns:
        Tuple of (context_string, citations_list)
    """
    citations = agentic_retrieve(query, topic, k_initial, k_final)

    if not citations:
        return "No relevant knowledge found in the knowledge base.", []

    context_parts = ["## Retrieved Knowledge (Agentic RAG — LLM-ranked)\n"]
    for c in citations:
        context_parts.append(
            f"### {c['id']} {c['formula_name']}\n"
            f"Source: {c['source']} (topic: {c['topic']}, page: {c['page']})\n"
            f"{c['content']}\n"
        )

    context_string = "\n".join(context_parts)
    return context_string, citations


# Legacy compatibility
def retrieve_as_context(query: str, topic: str | None = None, k: int = 4) -> str:
    """Legacy wrapper — returns only context string."""
    context, _ = retrieve_with_citations(query, topic, k_final=k)
    return context
