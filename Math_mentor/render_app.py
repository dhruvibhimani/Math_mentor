"""Render entrypoint for launching the Streamlit UI."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    port = os.environ.get("PORT", "8501")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
