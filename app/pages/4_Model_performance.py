from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import (
    available_courses,
    default_date,
    model_upgrade_notes,
    outsider_last_time_dataframe,
    picks_tracker_dataframe,
    picks_tracker_style,
    picks_tracker_summary,
)


st.title("Model Performance")
selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
if result.results_imported:
    st.success("Verified results have been matched to today's picks.")
else:
    st.info("Results are not matched yet. The tracker will settle picks when verified results are available.")

scores = result.scores
course_options = ["All UK courses"] + available_courses(scores)
selected_course = st.selectbox("Course", course_options)
if selected_course != "All UK courses":
    scores = [score for score in scores if score.runner.course == selected_course]

picks_df = picks_tracker_dataframe(scores)
if picks_df.empty:
    st.info("No selections are available for this date.")
else:
    summary = picks_tracker_summary(picks_df)
    winner_metric, ew_metric = st.columns(2)
    winner_metric.metric("Winner pick win rate", summary["winner_win_rate"])
    ew_metric.metric("Best EW place rate", summary["ew_place_rate"])
    st.dataframe(picks_tracker_style(picks_df), width="stretch", hide_index=True)
    st.caption("Green means won or placed, red means lost, and blue means just missed. Unsettled rows wait for verified results.")

st.subheader("Outsider Last-Time Signals")
outsider_df = outsider_last_time_dataframe(scores)
if outsider_df.empty:
    st.info("No verified last-time rank-outsider win/place signals are available in the imported fields.")
else:
    st.dataframe(outsider_df, width="stretch", hide_index=True)

with st.expander("Model Edge Upgrade Notes", expanded=False):
    for note in model_upgrade_notes():
        st.markdown(f"- {note}")
