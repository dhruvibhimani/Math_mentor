"""
Memory Store — SQLite-backed persistent memory for solved problems.

Stores:
- Input (raw and parsed)
- Retrieved documents
- Solver result
- Verifier result
- Explanation
- User feedback

Used for:
- Retrieving similar solved problems
- Reusing solving strategies
- Learning OCR corrections
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "math_mentor_memory.db")


def _get_connection() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            raw_input TEXT,
            parsed_problem TEXT,
            topic TEXT,
            intent TEXT,
            retrieved_docs TEXT,
            solution TEXT,
            verification TEXT,
            explanation TEXT,
            user_feedback TEXT,
            feedback_rating INTEGER DEFAULT 0,
            input_mode TEXT DEFAULT 'text'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ocr_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT,
            corrected_text TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def save_interaction(
    raw_input: str,
    parsed_problem: dict,
    retrieved_docs: str,
    solution: dict,
    verification: dict,
    explanation: dict,
    input_mode: str = "text",
    user_feedback: str = "",
    feedback_rating: int = 0,
) -> int:
    """Save a completed interaction to memory.

    Returns:
        The ID of the saved interaction.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """INSERT INTO interactions 
        (timestamp, raw_input, parsed_problem, topic, intent,
         retrieved_docs, solution, verification, explanation,
         user_feedback, feedback_rating, input_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            raw_input,
            json.dumps(parsed_problem),
            parsed_problem.get("topic", ""),
            "",  # intent populated separately if available
            retrieved_docs,
            json.dumps(solution),
            json.dumps(verification),
            json.dumps(explanation),
            user_feedback,
            feedback_rating,
            input_mode,
        ),
    )
    conn.commit()
    interaction_id = cursor.lastrowid
    conn.close()
    return interaction_id


def save_feedback(interaction_id: int, feedback: str, rating: int = 0):
    """Update an interaction with user feedback."""
    conn = _get_connection()
    conn.execute(
        "UPDATE interactions SET user_feedback = ?, feedback_rating = ? WHERE id = ?",
        (feedback, rating, interaction_id),
    )
    conn.commit()
    conn.close()


def save_ocr_correction(original: str, corrected: str):
    """Save an OCR text correction for learning."""
    conn = _get_connection()
    conn.execute(
        "INSERT INTO ocr_corrections (original_text, corrected_text, timestamp) VALUES (?, ?, ?)",
        (original, corrected, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def search_similar(query: str, limit: int = 5) -> list[dict]:
    """Search for similar solved problems using keyword matching.

    Args:
        query: Search query text
        limit: Maximum results to return

    Returns:
        List of similar past interactions
    """
    conn = _get_connection()
    # Simple keyword search — split query into words and search
    words = query.lower().split()
    if not words:
        conn.close()
        return []

    # Build WHERE clause for keyword matching
    conditions = []
    params = []
    for word in words[:5]:  # Limit to first 5 words
        conditions.append("(LOWER(raw_input) LIKE ? OR LOWER(parsed_problem) LIKE ?)")
        params.extend([f"%{word}%", f"%{word}%"])

    where_clause = " OR ".join(conditions)
    query_sql = f"""
        SELECT id, timestamp, raw_input, parsed_problem, topic, solution, explanation, 
               user_feedback, feedback_rating 
        FROM interactions 
        WHERE {where_clause}
        ORDER BY timestamp DESC 
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query_sql, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "raw_input": row["raw_input"],
            "parsed_problem": json.loads(row["parsed_problem"]) if row["parsed_problem"] else {},
            "topic": row["topic"],
            "solution": json.loads(row["solution"]) if row["solution"] else {},
            "explanation": json.loads(row["explanation"]) if row["explanation"] else {},
            "feedback_rating": row["feedback_rating"],
        })

    return results


def get_history(limit: int = 20) -> list[dict]:
    """Get recent interaction history.

    Args:
        limit: Maximum results

    Returns:
        List of recent interactions (newest first)
    """
    conn = _get_connection()
    rows = conn.execute(
        """SELECT id, timestamp, raw_input, topic, feedback_rating 
        FROM interactions ORDER BY timestamp DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "raw_input": row["raw_input"][:100] + "..." if len(row["raw_input"] or "") > 100 else row["raw_input"],
            "topic": row["topic"],
            "feedback_rating": row["feedback_rating"],
        }
        for row in rows
    ]


def get_interaction(interaction_id: int) -> Optional[dict]:
    """Retrieve a specific interaction by ID."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "raw_input": row["raw_input"],
        "parsed_problem": json.loads(row["parsed_problem"]) if row["parsed_problem"] else {},
        "solution": json.loads(row["solution"]) if row["solution"] else {},
        "verification": json.loads(row["verification"]) if row["verification"] else {},
        "explanation": json.loads(row["explanation"]) if row["explanation"] else {},
        "user_feedback": row["user_feedback"],
        "feedback_rating": row["feedback_rating"],
    }


def delete_interaction(interaction_id: int):
    """Delete a specific interaction from memory."""
    conn = _get_connection()
    conn.execute("DELETE FROM interactions WHERE id = ?", (interaction_id,))
    conn.commit()
    conn.close()


def clear_history():
    """Delete all saved interactions from memory."""
    conn = _get_connection()
    conn.execute("DELETE FROM interactions")
    conn.commit()
    conn.close()
