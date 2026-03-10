"""
Runtime configuration helpers for local and deployed environments.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in cloud deploys
    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "PARSER_MODEL": "llama-3.1-8b-instant",
    "ROUTER_MODEL": "llama-3.1-8b-instant",
    "SOLVER_MODEL": "llama-3.1-70b-versatile",
    "VERIFIER_MODEL": "llama-3.3-70b-versatile",
    "EXPLAINER_MODEL": "llama-3.1-8b-instant",
}


def load_runtime_env() -> None:
    """Load local dotenv values when present and normalize model defaults."""
    load_dotenv(PROJECT_ROOT / ".env")

    for key, default in DEFAULT_MODELS.items():
        if not os.getenv(key):
            os.environ[key] = default


def require_groq_api_key() -> str:
    """Return the Groq API key or raise a clear startup error."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in the environment or in the project .env file."
        )
    if not api_key.startswith("gsk_"):
        raise RuntimeError(
            "Invalid GROQ_API_KEY format. Groq API keys should start with 'gsk_'."
        )
    return api_key


def bootstrap_runtime() -> str:
    """Load environment configuration and validate required secrets."""
    load_runtime_env()
    return require_groq_api_key()


def groq_client_kwargs(model_env_var: str, default_model: str, temperature: float) -> dict:
    """Build a normalized ChatGroq configuration."""
    return {
        "model": os.getenv(model_env_var, default_model),
        "temperature": temperature,
        "api_key": require_groq_api_key(),
    }
