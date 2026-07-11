from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card
from ian_racing_model.ui import default_date, scores_to_dataframe

st.title("Today's Cards")
selected_date = st.date_input("Date", value=default_date())
course = st.text_input("Course", value="Ascot")
scores = get_scored_card(selected_date, course, Settings())
st.dataframe(scores_to_dataframe(scores), use_container_width=True, hide_index=True)
