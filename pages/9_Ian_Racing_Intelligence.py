from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(ROOT / "app" / "pages" / "9_Ian_Racing_Intelligence.py"), run_name="__main__")
