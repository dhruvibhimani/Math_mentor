"""
Vector Store — Build and manage ChromaDB vector store from knowledge documents.

Loads PDF knowledge files, chunks them, embeds with HuggingFace embeddings
(all-MiniLM-L6-v2), and stores in a persistent ChromaDB collection.
"""

import os
import re
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Paths
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


def _get_topic_from_filename(filename: str) -> str:
    """Extract topic metadata from knowledge doc filename."""
    name = os.path.basename(filename).lower()
    if "calculus" in name:
        return "calculus"
    elif "algebra" in name and "linear" not in name:
        return "algebra"
    elif "probability" in name:
        return "probability"
    elif "linear" in name:
        return "linear_algebra"
    elif "trig" in name or "sincos" in name:
        return "trigonometry"
    elif "scalar" in name or "vector" in name:
        return "vectors"
    elif "quotient" in name:
        return "calculus"
    elif "mistake" in name or "common" in name:
        return "common_mistakes"
    elif "jee" in name or "formula" in name:
        return "general"
    return "general"


def _extract_formula_label(text: str) -> str:
    """Try to extract a formula/section name from a text chunk."""
    patterns = [
        r"(?:^|\n)\s*(?:#+)\s*(.+)",
        r"(?:^|\n)\s*(\d+\.\d*\s+.+)",
        r"((?:Theorem|Lemma|Formula|Rule|Identity|Definition|Property)\s*[:\-].+)",
    ]
    for pat in patterns:
        match = re.search(pat, text[:300])
        if match:
            return match.group(1).strip()[:80]
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 5:
            return line[:80]
    return "Unknown"


def _clean_section_label(label: str) -> str:
    """Normalize extracted section labels for use in chunk headers."""
    label = re.sub(r"\s+", " ", label or "").strip()
    return label[:120] if label else "General"


def _build_contextual_chunk_header(chunk) -> str:
    """Build a lightweight contextual header for embedding short math chunks."""
    topic = chunk.metadata.get("topic", "general")
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", "?")
    section = _clean_section_label(chunk.metadata.get("formula_name", "General"))

    return (
        f"Topic: {topic}\n"
        f"Section: {section}\n"
        f"Source: {source}\n"
        f"Page: {page}\n"
    )


def _get_embeddings():
    """Get the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """Build or load the ChromaDB vector store from knowledge documents.

    Args:
        force_rebuild: If True, rebuild even if the store already exists

    Returns:
        Chroma vector store instance
    """
    embeddings = _get_embeddings()

    # Check if already built
    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="math_knowledge",
        )

    # Load all knowledge documents (PDFs)
    all_docs = []
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf"))

    for filepath in pdf_files:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        topic = _get_topic_from_filename(filepath)
        source_name = os.path.basename(filepath)
        for doc in docs:
            doc.metadata["topic"] = topic
            doc.metadata["source"] = source_name
        all_docs.extend(docs)

    if not all_docs:
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="math_knowledge",
        )

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(all_docs)

    # Enrich metadata with formula labels
    for chunk in chunks:
        chunk.metadata["formula_name"] = _extract_formula_label(chunk.page_content)
        chunk.metadata["raw_content"] = chunk.page_content
        contextual_header = _build_contextual_chunk_header(chunk)
        chunk.page_content = f"{contextual_header}\n{chunk.page_content.strip()}"

    # Build vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="math_knowledge",
    )

    return vectorstore


def get_retriever(k: int = 6, topic_filter: str | None = None):
    """Get a retriever from the vector store.

    Args:
        k: Number of documents to retrieve
        topic_filter: Optional topic to filter by

    Returns:
        LangChain retriever
    """
    vectorstore = build_vector_store()

    search_kwargs = {"k": k}
    if topic_filter:
        search_kwargs["filter"] = {"topic": topic_filter}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def similarity_search_with_scores(query: str, k: int = 8, topic_filter: str | None = None):
    """Search the vector store and return documents with relevance scores.

    Args:
        query: Search query
        k: Number of results
        topic_filter: Optional topic filter

    Returns:
        List of (Document, score) tuples
    """
    vectorstore = build_vector_store()
    kwargs = {}
    if topic_filter:
        kwargs["filter"] = {"topic": topic_filter}
    return vectorstore.similarity_search_with_relevance_scores(query, k=k, **kwargs)
