from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_PATH = ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
