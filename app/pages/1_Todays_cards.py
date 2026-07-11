from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import available_courses, default_date, scores_to_dataframe


st.title("Today's Cards")
selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
course_options = ["All UK courses"] + available_courses(scores)
selected_course = st.selectbox("Course", course_options)
if selected_course != "All UK courses":
    scores = [score for score in scores if score.runner.course == selected_course]
st.dataframe(scores_to_dataframe(scores), width="stretch", hide_index=True)
