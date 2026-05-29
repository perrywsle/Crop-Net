"""Top-level launcher for the Crop Fusion AI desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    """Launch the Tkinter GUI."""
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


if __name__ == "__main__":
    main()
