"""
PaddleOCR wrapper - Extract text from math problem images.
"""

import io
import os
import tempfile
import time
from typing import Any, Dict, List, Union

from PIL import Image


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_CACHE_ROOT = os.path.join(_PROJECT_ROOT, ".cache")
_LOCAL_PADDLE_HOME = os.path.join(_LOCAL_CACHE_ROOT, "paddle")
_OCR_ENGINE = None
_OCR_INIT_ERROR = None


def _ensure_paddle_env():
    os.makedirs(_LOCAL_PADDLE_HOME, exist_ok=True)
    # Force conservative runtime flags to avoid recent Paddle PIR/oneDNN issues.
    os.environ["PADDLE_HOME"] = _LOCAL_PADDLE_HOME
    os.environ["XDG_CACHE_HOME"] = _LOCAL_CACHE_ROOT
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _build_ocr_engine(use_legacy_fallback: bool = False):
    from paddleocr import PaddleOCR

    common_kwargs = {
        "lang": "en",
        "ocr_version": "PP-OCRv3",
    }
    if use_legacy_fallback:
        common_kwargs.update(
            {
                "use_angle_cls": False,
            }
        )
    else:
        common_kwargs["use_angle_cls"] = False
    return PaddleOCR(**common_kwargs)


def _get_ocr_engine():
    """Initialize PaddleOCR once and reuse it."""
    global _OCR_ENGINE
    global _OCR_INIT_ERROR

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        _ensure_paddle_env()
        _OCR_ENGINE = _build_ocr_engine()
        _OCR_INIT_ERROR = None
        return _OCR_ENGINE
    except ImportError as exc:
        _OCR_INIT_ERROR = f"PaddleOCR import failed: {exc}"
        return None
    except Exception as exc:
        _OCR_INIT_ERROR = f"PaddleOCR initialization failed: {exc}"
        try:
            _ensure_paddle_env()
            _OCR_ENGINE = _build_ocr_engine(use_legacy_fallback=True)
            _OCR_INIT_ERROR = None
            return _OCR_ENGINE
        except Exception as retry_exc:
            _OCR_INIT_ERROR = (
                f"PaddleOCR initialization failed: {exc}. "
                f"Legacy fallback also failed: {retry_exc}"
            )
            return None


def _is_pir_runtime_error(exc: Exception) -> bool:
    text = str(exc)
    return "ConvertPirAttribute2RuntimeAttribute" in text or "onednn_instruction.cc" in text


def _run_ocr(ocr, file_path: str):
    """Prefer the classic OCR API over the newer predict pipeline."""
    if hasattr(ocr, "ocr"):
        return ocr.ocr(file_path, cls=False), "legacy"
    return ocr.predict(file_path), "predict"


def run_paddle_ocr(file_path: str) -> Dict[str, Any]:
    """Run PaddleOCR on a document path and return page/block details."""
    ocr = _get_ocr_engine()
    if ocr is None:
        return {
            "engine": "paddleocr",
            "pages": [],
            "overall_confidence": 0.0,
            "doc_language": "en",
            "duration_ms": 0,
            "error": _OCR_INIT_ERROR or (
                "PaddleOCR could not be initialized. "
                "Verify paddleocr and paddlepaddle are installed and compatible with the deployment runtime."
            ),
        }

    start = time.time()
    try:
        result, mode = _run_ocr(ocr, file_path)
    except Exception as exc:
        if _is_pir_runtime_error(exc):
            global _OCR_ENGINE
            _OCR_ENGINE = None
            try:
                _ensure_paddle_env()
                ocr = _build_ocr_engine(use_legacy_fallback=True)
                _OCR_ENGINE = ocr
                result, mode = _run_ocr(ocr, file_path)
            except Exception as retry_exc:
                return {
                    "engine": "paddleocr",
                    "pages": [],
                    "overall_confidence": 0.0,
                    "doc_language": "en",
                    "duration_ms": int((time.time() - start) * 1000),
                    "error": (
                        "PaddleOCR runtime error after retry with conservative settings: "
                        f"{retry_exc}. If this persists, reinstall a non-PIR-compatible stack "
                        "or pin paddlepaddle/paddleocr to a known-compatible pair."
                    ),
                }
        else:
            return {
                "engine": "paddleocr",
                "pages": [],
                "overall_confidence": 0.0,
                "doc_language": "en",
                "duration_ms": int((time.time() - start) * 1000),
                "error": f"PaddleOCR runtime error: {exc}",
            }

    pages: List[Dict[str, Any]] = []
    all_scores: List[float] = []

    if mode == "legacy":
        for page_idx, page in enumerate(result, start=1):
            page_blocks = []
            page_scores = []
            for line in page or []:
                if not line or len(line) < 2:
                    continue
                text = str(line[1][0])
                score_f = float(line[1][1])
                page_scores.append(score_f)
                all_scores.append(score_f)
                page_blocks.append({"text": text, "confidence": round(score_f, 4)})

            page_confidence = round(sum(page_scores) / len(page_scores), 4) if page_scores else 0.0
            pages.append(
                {
                    "page": page_idx,
                    "text": "\n".join(block["text"] for block in page_blocks),
                    "confidence": page_confidence,
                    "blocks": page_blocks,
                }
            )
    else:
        for page_idx, page in enumerate(result, start=1):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])

            page_blocks = []
            page_scores = []

            for text, score in zip(texts, scores):
                score_f = float(score)
                page_scores.append(score_f)
                all_scores.append(score_f)
                page_blocks.append({"text": text, "confidence": round(score_f, 4)})

            page_confidence = round(sum(page_scores) / len(page_scores), 4) if page_scores else 0.0
            pages.append(
                {
                    "page": page_idx,
                    "text": "\n".join(texts),
                    "confidence": page_confidence,
                    "blocks": page_blocks,
                }
            )

    overall_confidence = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    duration_ms = int((time.time() - start) * 1000)

    return {
        "engine": "paddleocr",
        "pages": pages,
        "overall_confidence": overall_confidence,
        "doc_language": "en",
        "duration_ms": duration_ms,
    }


def _to_temp_image_path(image_input: Union[str, bytes, "Image.Image"]) -> tuple[str, str | None]:
    """Convert supported image inputs to a filesystem path for PaddleOCR."""
    if isinstance(image_input, str):
        return image_input, None

    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif hasattr(image_input, "convert"):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        image.save(tmp.name, format="PNG")
    finally:
        tmp.close()
    return tmp.name, tmp.name


def extract_text_from_image(image_input: Union[str, bytes, "Image.Image"]) -> dict:
    """Extract text from an image using the PaddleOCR predict pipeline."""
    temp_path = None

    try:
        file_path, temp_path = _to_temp_image_path(image_input)
        if not os.path.exists(file_path):
            return {"raw_problem": "", "confidence": 0.0, "error": f"File not found: {file_path}"}

        result = run_paddle_ocr(file_path)
        if result.get("error"):
            return {"raw_problem": "", "confidence": 0.0, "error": result["error"]}

        lines = []
        for page in result.get("pages", []):
            for block in page.get("blocks", []):
                lines.append(
                    {
                        "text": block.get("text", ""),
                        "confidence": round(float(block.get("confidence", 0.0)), 4),
                    }
                )

        full_text = " ".join(line["text"] for line in lines).strip()
        return {
            "raw_problem": full_text,
            "confidence": round(float(result.get("overall_confidence", 0.0)), 4),
            "line_count": len(lines),
            "lines": lines,
            "pages": result.get("pages", []),
            "duration_ms": result.get("duration_ms", 0),
            "engine": result.get("engine", "paddleocr"),
        }
    except Exception as exc:
        return {"raw_problem": "", "confidence": 0.0, "error": str(exc)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
