"""
Streamlit UI for the Math Mentor pipeline.

Run:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import streamlit as st
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in cloud deploys
    def load_dotenv(*_args, **_kwargs):
        return False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from graph.langgraph_workflow import run_pipeline
from input.paddle_ocr import extract_text_from_image
from input.whisper_asr import transcribe_audio
from memory.memory_store import (
    clear_history,
    delete_interaction,
    get_history,
    get_interaction,
    save_feedback,
    save_interaction,
    search_similar,
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        h1 {
            font-size: 2rem !important;
            line-height: 1.2 !important;
        }
        h2, h3 {
            font-size: 1.05rem !important;
        }
        p, li, label, .stMarkdown, .stCaption, .stChatMessage, .stTextInput, .stTextArea {
            font-size: 0.92rem !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            font-size: 0.88rem !important;
        }
        [data-testid="stChatInput"] textarea {
            font-size: 0.95rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def stream_response(text: str):
    for word in text.split():
        yield f"{word} "
        time.sleep(0.02)


def _normalize_value(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            return value
    return value


def _stringify(value) -> str:
    value = _normalize_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def build_response_text(result: dict) -> str:
    explanation = _normalize_value(result.get("explanation", {}))
    if not isinstance(explanation, dict):
        explanation = {"final_answer": _stringify(explanation), "explanation": _stringify(explanation)}

    sections: list[str] = []

    final_answer = explanation.get("final_answer") or result.get("solution", {}).get("final_answer", "N/A")
    sections.append("## Final Answer")
    sections.append(_stringify(final_answer))

    body = explanation.get("explanation") or result.get("solution", {}).get("solution_steps", "")
    if body:
        sections.append("## Explanation")
        if isinstance(body, list):
            body = "\n".join(_stringify(item) for item in body)
        sections.append(_stringify(body))

    sources = explanation.get("sources", [])
    if not sources:
        citations = result.get("citations", []) or result.get("solution", {}).get("citations_used", [])
        for citation in citations:
            if isinstance(citation, dict):
                label = citation.get("source") or citation.get("formula_name") or citation.get("id", "")
            else:
                label = _stringify(citation)
            if label:
                sources.append(label)

    sections.append("## Sources")
    if sources:
        sections.append("\n".join(f"- {_stringify(source)}" for source in sources))
    else:
        sections.append("No reliable source found in knowledge base.")

    confidence = explanation.get("confidence", result.get("verification", {}).get("confidence", result.get("solution", {}).get("confidence", "")))
    if confidence != "":
        sections.append("## Confidence")
        sections.append(f"Confidence Score: {_stringify(confidence)}")

    sections.append("## Feedback")
    sections.append("[ Correct ]   [ Incorrect ]")

    return "\n\n".join(section for section in sections if section).strip()


def run_text_pipeline(raw_problem: str, input_mode: str) -> tuple[str, dict, int]:
    result = run_pipeline(raw_problem=raw_problem, input_mode=input_mode)
    response_text = build_response_text(result)

    interaction_id = save_interaction(
        raw_input=raw_problem,
        parsed_problem=result.get("parsed_problem", {}),
        retrieved_docs=result.get("retrieved_docs", ""),
        solution=result.get("solution", {}),
        verification=result.get("verification", {}),
        explanation=result.get("explanation", {}),
        input_mode=input_mode,
    )
    return response_text, result, interaction_id


def load_interaction_into_chat(interaction_id: int):
    interaction = get_interaction(interaction_id)
    if not interaction:
        st.warning("Could not load that past conversation.")
        return

    response_text = build_response_text(
        {
            "solution": interaction.get("solution", {}),
            "explanation": interaction.get("explanation", {}),
            "citations": interaction.get("solution", {}).get("citations", []),
        }
    )

    st.session_state.messages = [
        {"role": "user", "content": interaction.get("raw_input", "")},
        {"role": "assistant", "content": response_text, "interaction_id": interaction_id},
    ]
    st.session_state.input_mode = "text"


def render_feedback_controls(message: dict):
    interaction_id = message.get("interaction_id")
    if not interaction_id or message.get("role") != "assistant":
        return

    feedback_key = f"feedback_choice_{interaction_id}"
    comment_key = f"feedback_comment_{interaction_id}"

    choice = st.radio(
        "Was this answer correct?",
        options=["Correct", "Incorrect"],
        key=feedback_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    comment = ""
    if choice == "Incorrect":
        comment = st.text_area("Comment", key=comment_key, placeholder="What was wrong?")

    if st.button("Save Feedback", key=f"save_feedback_{interaction_id}"):
        rating = 1 if choice == "Correct" else -1
        save_feedback(interaction_id, comment or choice, rating=rating)
        st.success("Feedback saved.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"


with st.sidebar:
    st.title("🧮 Math Mentor")

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("History")
    search_query = st.text_input("Search past problems")

    if search_query:
        results = search_similar(search_query)
        if results:
            for item in results:
                label = item.get("raw_input", "Untitled problem")[:80]
                with st.expander(label):
                    st.write(f"**Topic:** {item.get('topic', 'general')}")
                    st.write(f"**Answer:** {_stringify(item.get('explanation', {}).get('final_answer', 'N/A'))}")
                    open_col, delete_col = st.columns(2)
                    if open_col.button("Open", key=f"open_search_{item['id']}", use_container_width=True):
                        load_interaction_into_chat(item["id"])
                        st.rerun()
                    if delete_col.button("Delete", key=f"delete_search_{item['id']}", use_container_width=True):
                        delete_interaction(item["id"])
                        st.rerun()
        else:
            st.info("No matching problems found.")

    st.subheader("Recent Problems")
    if st.button("Clear All History", use_container_width=True):
        clear_history()
        st.session_state.messages = []
        st.rerun()

    history = get_history(10)
    if history:
        for item in history:
            st.markdown(f"**{item.get('topic', 'general').title()}**")
            st.caption(item.get("raw_input", ""))
            open_col, delete_col = st.columns(2)
            if open_col.button("Open", key=f"open_history_{item['id']}", use_container_width=True):
                load_interaction_into_chat(item["id"])
                st.rerun()
            if delete_col.button("Delete", key=f"delete_history_{item['id']}", use_container_width=True):
                delete_interaction(item["id"])
                st.rerun()
            st.divider()
    else:
        st.info("No problems solved yet.")

    st.subheader("Input Mode")
    selected_mode = st.radio(
        "Choose input mode",
        options=["Text", "Image", "Audio"],
        index=["text", "image", "audio"].index(st.session_state.input_mode),
        label_visibility="collapsed",
    )
    st.session_state.input_mode = selected_mode.lower()


st.title("🧮 Math Mentor")
st.caption("Ask math questions with text, image OCR, or audio transcription.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_feedback_controls(message)


raw_problem = ""
actual_mode = st.session_state.input_mode

if actual_mode == "image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded image", width=320)
        if st.button("Extract and Solve"):
            with st.spinner("Running OCR..."):
                ocr_result = extract_text_from_image(image_bytes)
            if ocr_result.get("error"):
                st.error(ocr_result["error"])
            else:
                raw_problem = st.text_area(
                    "Review extracted text",
                    value=ocr_result.get("raw_problem", ""),
                    key="image_text_review",
                ).strip()

elif actual_mode == "audio":
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    recorded_audio = st.audio_input("Or record audio") if hasattr(st, "audio_input") else None
    chosen_audio = recorded_audio if recorded_audio is not None else audio_file

    if chosen_audio is not None:
        audio_bytes = chosen_audio.getvalue()
        st.audio(audio_bytes)
        if st.button("Transcribe and Solve"):
            with st.spinner("Transcribing audio..."):
                asr_result = transcribe_audio(audio_bytes)
            if asr_result.get("error"):
                st.error(asr_result["error"])
            else:
                raw_problem = st.text_area(
                    "Review transcript",
                    value=asr_result.get("raw_problem", ""),
                    key="audio_text_review",
                ).strip()

if actual_mode == "text":
    raw_problem = st.chat_input("Type your question and press Enter...")

if raw_problem:
    st.session_state.messages.append({"role": "user", "content": raw_problem})
    with st.chat_message("user"):
        st.markdown(raw_problem)

    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            try:
                response_text, _, interaction_id = run_text_pipeline(raw_problem, actual_mode)
                st.write_stream(stream_response(response_text))
                feedback_message = {"role": "assistant", "content": response_text, "interaction_id": interaction_id}
                st.session_state.messages.append(feedback_message)
                render_feedback_controls(feedback_message)
            except Exception as exc:
                error_text = f"Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
