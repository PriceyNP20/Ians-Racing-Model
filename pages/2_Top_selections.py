from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PAGE_PATH = ROOT / "app" / "pages" / "2_Top_selections.py"
runpy.run_path(str(PAGE_PATH), run_name="__main__")
