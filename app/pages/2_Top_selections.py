from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model import ui as ui_helpers
from ian_racing_model.config import Settings
from ian_racing_model.edge_quality import edge_quality_dataframe
from ian_racing_model.edge_lab import enhanced_undervalued_edge_dataframe, negative_value_dataframe
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.table_styles import research_table_style
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


def _undervalued_edge_dataframe(scores, limit=12):
    helper = getattr(ui_helpers, "undervalued_edge_dataframe", None)
    if helper is not None:
        base = helper(scores, limit=max(limit * 2, limit))
        if not base.empty and "edge_score" in base.columns:
            return enhanced_undervalued_edge_dataframe(scores, limit=limit)
        return base
    return enhanced_undervalued_edge_dataframe(scores, limit=limit)


def _negative_value_dataframe(scores, limit=12):
    return negative_value_dataframe(scores, limit=limit)


st.title("Top Selections")
selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
st.subheader("Edge Quality Screener")
quality_df = edge_quality_dataframe(scores, limit=30)
if quality_df.empty:
    st.info("No edge-quality shortlist is available from the current runners.")
else:
    st.caption("Separates win quality from place/EW quality and shows evidence, confidence and market confirmation.")
    st.dataframe(research_table_style(quality_df), width="stretch", hide_index=True)

st.subheader("Race-by-Race Picks")
race_picks = _race_selection_screener_dataframe(scores)
if race_picks.empty:
    st.info("No race-level picks available.")
else:
    st.dataframe(research_table_style(race_picks), width="stretch", hide_index=True)
    st.caption("Winner and Best EW value are selected by separate scoring logic.")

st.subheader("Undervalued Edge")
edge_df = _undervalued_edge_dataframe(scores, limit=12)
if edge_df.empty:
    st.info("No clear undervalued edge is available from the current model, odds and evidence.")
else:
    st.dataframe(research_table_style(edge_df), width="stretch", hide_index=True)
    st.caption("Shortlist of horses where the model price is bigger than the market expects and the case is supported by racing evidence.")

st.subheader("Negative Value Watchlist")
negative_df = _negative_value_dataframe(scores, limit=12)
if negative_df.empty:
    st.info("No obvious overbet or weak-value runners are flagged from the current prices.")
else:
    st.dataframe(research_table_style(negative_df), width="stretch", hide_index=True)

st.subheader("Best Value")
value_df = value_screener_dataframe(scores, limit=12)
if value_df.empty:
    st.info("No positive model-versus-market value edges are available from the current odds.")
else:
    st.dataframe(research_table_style(value_df), width="stretch", hide_index=True)

st.subheader("Model Signals")
signal_df = _model_signal_dataframe(scores, limit=20)
if signal_df.empty:
    st.info("No setup or market-move signals are available in the imported fields yet.")
else:
    st.dataframe(research_table_style(signal_df), width="stretch", hide_index=True)

st.subheader("Top Model Scores")
df = scores_to_dataframe(scores)
if not df.empty:
    df = df[df["recommendation"].isin(["WIN", "EACH_WAY", "PLACE"])].head(10)
st.dataframe(df, width="stretch", hide_index=True)
