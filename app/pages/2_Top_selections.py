from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import (
    default_date,
    race_selection_screener_dataframe,
    scores_to_dataframe,
    value_screener_dataframe,
)


st.title("Top Selections")
selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
st.subheader("Race-by-Race Picks")
race_picks = race_selection_screener_dataframe(scores)
if race_picks.empty:
    st.info("No race-level picks available.")
else:
    st.dataframe(race_picks, width="stretch", hide_index=True)
    st.caption("Winner and Best EW value are selected by separate scoring logic.")

st.subheader("Best Value")
value_df = value_screener_dataframe(scores, limit=12)
if value_df.empty:
    st.info("No positive model-versus-market value edges are available from the current odds.")
else:
    st.dataframe(value_df, width="stretch", hide_index=True)

st.subheader("Top Model Scores")
df = scores_to_dataframe(scores)
if not df.empty:
    df = df[df["recommendation"].isin(["WIN", "EACH_WAY", "PLACE"])].head(10)
st.dataframe(df, width="stretch", hide_index=True)
