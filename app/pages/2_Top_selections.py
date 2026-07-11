from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import default_date, scores_to_dataframe


st.title("Top Selections")
result = get_scored_card_result(default_date(), "Ascot", Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
df = scores_to_dataframe(scores)
if not df.empty:
    df = df[df["recommendation"].isin(["WIN", "EACH_WAY", "PLACE"])].head(10)
st.dataframe(df, use_container_width=True, hide_index=True)
