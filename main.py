"""Top-level launcher for the Crop Fusion AI browser app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> None:
    """Launch the browser-first app by default, with an optional desktop fallback."""
    parser = argparse.ArgumentParser(description="Launch the Crop Fusion AI user interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local web server to.")
    parser.add_argument("--port", default=8000, type=int, help="Port for the local web server.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode for development.")
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Launch the legacy Tkinter window instead of the browser app.",
    )
    args = parser.parse_args(argv)

    if args.desktop:
        try:
            from crop_fusion_ai.gui.app import main as gui_main
        except ModuleNotFoundError as exc:
            message = str(exc)
            if "tkinter" in message.lower():
                raise SystemExit(
                    "tkinter is not available in this Python environment. Install Tk support, then run main.py again."
                ) from exc
            raise

        gui_main()
        return

    from crop_fusion_ai.web.app import main as web_main

    web_main(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
