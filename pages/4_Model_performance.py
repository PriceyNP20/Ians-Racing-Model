from __future__ import annotations

from pathlib import Path
import runpy

PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Model_performance.py"
runpy.run_path(str(PAGE_PATH), run_name="__main__")
