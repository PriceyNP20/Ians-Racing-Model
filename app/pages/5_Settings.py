from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import IAN_FORMULA_V3_1_WEIGHTS, Settings

st.title("Settings")
settings = Settings()
st.write("Provider", settings.provider)
st.write("Database", settings.database_url)
st.subheader("Ian Formula V3.1 weights")
st.dataframe([{"component": name, "weight": weight} for name, weight in IAN_FORMULA_V3_1_WEIGHTS.items()], use_container_width=True, hide_index=True)
st.caption("Weights are code/config controlled in this MVP and total 100.")
