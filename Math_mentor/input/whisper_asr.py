"""
Whisper ASR wrapper — Convert audio to text for math problem input.

Accepts audio files and returns structured output:
    {"raw_problem": "<transcribed text>", "language": "en"}
"""

import os
import tempfile
from functools import lru_cache
from typing import Union


@lru_cache(maxsize=3)
def _get_whisper_model(model_size: str = "base"):
    """Lazy-load Whisper model."""
    try:
        import whisper
        return whisper.load_model(model_size)
    except ImportError:
        return None


def transcribe_audio(audio_input: Union[str, bytes], model_size: str = "base") -> dict:
    """Transcribe an audio file to text using OpenAI Whisper.

    Args:
        audio_input: File path (str) or raw audio bytes
        model_size: Whisper model size — "tiny", "base", "small", "medium", "large"

    Returns:
        {
            "raw_problem": "<transcribed text>",
            "language": "<detected language>",
            "segments": [{"start": 0.0, "end": 2.5, "text": "..."}]
        }
    """
    model = _get_whisper_model(model_size)
    if model is None:
        return {
            "raw_problem": "",
            "language": "",
            "error": "Whisper is not installed. Install with: pip install openai-whisper",
        }

    # Handle bytes input by writing to temp file
    if isinstance(audio_input, bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_input)
            audio_path = f.name
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            return {"raw_problem": "", "language": "", "error": f"File not found: {audio_input}"}
        audio_path = audio_input
    else:
        return {"raw_problem": "", "language": "", "error": "Unsupported audio input type"}

    try:
        # Force FP32 on CPU to avoid noisy FP16 warnings.
        result = model.transcribe(audio_path, fp16=False, verbose=False)

        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        return {
            "raw_problem": result["text"].strip(),
            "language": result.get("language", "en"),
            "segments": segments,
        }

    finally:
        # Clean up temp file if we created one
        if isinstance(audio_input, bytes) and os.path.exists(audio_path):
            os.unlink(audio_path)
