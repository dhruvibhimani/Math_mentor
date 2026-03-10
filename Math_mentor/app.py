"""
CLI entrypoint for the Math Mentor pipeline.

Examples:
    python app.py --text "Find derivative of x^2 sin(x)"
    python app.py --image .\\sample.png
    python app.py --audio .\\question.wav
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

from dotenv import load_dotenv

from graph.langgraph_workflow import run_pipeline
from input.paddle_ocr import extract_text_from_image
from input.whisper_asr import transcribe_audio


def _read_problem_from_args(args: argparse.Namespace) -> Tuple[str, str]:
    if args.text:
        return args.text.strip(), "text"

    if args.image:
        result = extract_text_from_image(args.image)
        if result.get("error"):
            raise RuntimeError(f"OCR failed: {result['error']}")
        return result.get("raw_problem", "").strip(), "image"

    if args.audio:
        result = transcribe_audio(args.audio)
        if result.get("error"):
            raise RuntimeError(f"Transcription failed: {result['error']}")
        return result.get("raw_problem", "").strip(), "audio"

    # Fallback interactive input
    prompt = input("Enter math problem: ").strip()
    return prompt, "text"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Math Mentor pipeline from terminal.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Math problem as plain text.")
    group.add_argument("--image", type=str, help="Path to image for OCR.")
    group.add_argument("--audio", type=str, help="Path to audio for ASR.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full pipeline output as JSON (default prints explanation only).",
    )
    args = parser.parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    raw_problem, input_mode = _read_problem_from_args(args)
    if not raw_problem:
        raise ValueError("No problem text was provided or extracted.")

    result = run_pipeline(raw_problem=raw_problem, input_mode=input_mode)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    else:
        explanation = result.get("explanation", {})
        print("\nTitle:", explanation.get("title", "Solution"))
        print("Final answer:", explanation.get("final_answer", "N/A"))
        summary = explanation.get("summary", "")
        if summary:
            print("Summary:", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
