from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model import ui as ui_helpers
from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import (
    default_date,
    picks_tracker_dataframe,
    scores_to_dataframe,
    value_screener_dataframe,
)


def _model_signal_dataframe(scores, limit=20):
    helper = getattr(ui_helpers, "model_signal_dataframe", None)
    if helper is not None:
        return helper(scores, limit=limit)
    return scores_to_dataframe([])


def _race_selection_screener_dataframe(scores):
    helper = getattr(ui_helpers, "race_selection_screener_dataframe", None)
    if helper is not None:
        return helper(scores)
    picks = picks_tracker_dataframe(scores)
    if picks.empty:
        return picks
    return picks.rename(
        columns={
            "pick_type": "pick",
            "score": "model_score",
            "win_value_edge": "win_edge",
            "place_value_edge": "place_edge",
            "selection_reason": "reason",
        }
    )[
        [
            "course",
            "off_time",
            "race",
            "pick",
            "horse",
            "selection_score",
            "model_score",
            "confidence",
            "odds",
            "fair_win_odds",
            "fair_place_odds",
            "win_edge",
            "place_edge",
            "reason",
        ]
    ]


st.title("Top Selections")
selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
st.subheader("Race-by-Race Picks")
race_picks = _race_selection_screener_dataframe(scores)
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

st.subheader("Model Signals")
signal_df = _model_signal_dataframe(scores, limit=20)
if signal_df.empty:
    st.info("No setup or market-move signals are available in the imported fields yet.")
else:
    st.dataframe(signal_df, width="stretch", hide_index=True)

st.subheader("Top Model Scores")
df = scores_to_dataframe(scores)
if not df.empty:
    df = df[df["recommendation"].isin(["WIN", "EACH_WAY", "PLACE"])].head(10)
st.dataframe(df, width="stretch", hide_index=True)
