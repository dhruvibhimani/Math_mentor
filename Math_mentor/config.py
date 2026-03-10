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
    "SOLVER_MODEL": "llama-3.3-70b-versatile",
    "VERIFIER_MODEL": "llama-3.3-70b-versatile",
    "EXPLAINER_MODEL": "llama-3.1-8b-instant",
}


def load_runtime_env() -> None:
    """Load local dotenv values when present."""
    load_dotenv(PROJECT_ROOT / ".env")

def get_config(key: str, default: str | None = None) -> str | None:
    """Read configuration from env, then the provided default."""
    value = os.getenv(key)
    if value:
        return value

    return default


def require_groq_api_key() -> str:
    """Return the Groq API key or raise a clear startup error."""
    api_key = (get_config("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in environment variables or the project .env file."
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
        "model": get_config(model_env_var, default_model),
        "temperature": temperature,
        "api_key": require_groq_api_key(),
    }
