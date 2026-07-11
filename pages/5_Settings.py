from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PAGE_PATH = ROOT / "app" / "pages" / "5_Settings.py"
spec = importlib.util.spec_from_file_location("ian_racing_model_page_settings", PAGE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
