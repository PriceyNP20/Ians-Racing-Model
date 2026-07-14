from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.table_styles import research_table_style
from ian_racing_model.ui import available_courses, default_date
from racing_intelligence.scoring import intelligence_dataframe


st.set_page_config(page_title="Ian Racing Intelligence", layout="wide")
st.title("Ian Racing Intelligence Platform")
st.caption(
    "Read-only UK and Irish race research. Separate Win Index and Place Index. "
    "No bet placement, no automated wagering, no invented data."
)

settings = Settings()
selected_date = st.date_input("Meeting date", value=default_date())
result = get_scored_card_result(selected_date, None, settings)
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")

scores = result.scores
course_options = ["All UK and Irish courses"] + available_courses(scores)
selected_course = st.selectbox("Course", course_options)
if selected_course != "All UK and Irish courses":
    scores = [score for score in scores if score.runner.course == selected_course]

df = intelligence_dataframe(scores)

st.subheader("Platform Status")
cols = st.columns(4)
cols[0].metric("Runners analysed", len(df))
cols[1].metric("Win value signals", 0 if df.empty else int(df["recommendation"].eq("WIN_VALUE").sum()))
cols[2].metric("Place value signals", 0 if df.empty else int(df["recommendation"].eq("PLACE_VALUE").sum()))
cols[3].metric("Data source", result.provider)

with st.expander("Plugin architecture", expanded=False):
    st.markdown(
        """
        Replaceable interfaces are now defined for racecards, results, markets, weather,
        going, ratings, trainer, jockey, pace, draw bias, course profile, win models,
        place models, fair odds, recommendations, calibration and exports.
        """
    )

if df.empty:
    st.info("No runners are available for the selected date/course.")
else:
    st.subheader("Value Intelligence")
    value_df = df[df["recommendation"].isin(["WIN_VALUE", "PLACE_VALUE", "PLACE_PROFILE"])].copy()
    if value_df.empty:
        st.info("No runner currently clears the value/profile gates.")
    else:
        st.dataframe(research_table_style(value_df), width="stretch", hide_index=True)

    st.subheader("All Runner Intelligence")
    st.dataframe(research_table_style(df), width="stretch", hide_index=True)
    st.download_button(
        "Download intelligence CSV",
        data=df.drop(columns=[column for column in df.columns if column.startswith("_")], errors="ignore").to_csv(index=False),
        file_name=f"ian-racing-intelligence-{selected_date.isoformat()}.csv",
        mime="text/csv",
    )

with st.expander("Research guardrails", expanded=False):
    st.markdown(
        """
        - Win probability and place probability are separate outputs.
        - Place value is not treated as a small adjustment to win value.
        - Missing fields lower confidence rather than being filled with invented data.
        - This platform must never connect to bet-placement endpoints.
        """
    )
