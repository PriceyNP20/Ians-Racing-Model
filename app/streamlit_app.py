from __future__ import annotations

from html import escape
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import (
    available_courses,
    default_date,
    picks_tracker_dataframe,
    picks_tracker_style,
    picks_tracker_summary,
    scores_to_dataframe,
    screener_dataframe,
)


st.set_page_config(page_title="Ian Racing Model", layout="wide")
st.markdown(
    """
    <style>
    .screener-card {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        background: #ffffff;
        min-height: 156px;
    }
    .screener-label {
        color: #2457a6;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .screener-horse {
        color: #202633;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.35rem 0 0.15rem;
    }
    .screener-meta {
        color: #4b5565;
        font-size: 0.9rem;
        line-height: 1.35;
    }
    .screener-numbers {
        color: #202633;
        font-size: 0.92rem;
        font-weight: 650;
        margin-top: 0.5rem;
    }
    .screener-warning {
        color: #8a5a00;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Ian Racing Model")
st.caption("Read-only UK horse-racing research and tracking dashboard.")

settings = Settings()
selected_date = st.date_input("Meeting date", value=default_date())

result = get_scored_card_result(selected_date, None, settings)
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
display_scores = scores
if selected_course != "All UK courses":
    display_scores = [score for score in display_scores if score.runner.course == selected_course]

df = scores_to_dataframe(display_scores)

st.subheader("Screener")
screener_df = screener_dataframe(display_scores, limit=8)
if screener_df.empty:
    st.info("No eligible runners available for the screener.")
else:
    top_cards = screener_df.head(3).to_dict("records")
    columns = st.columns(len(top_cards))
    for column, item in zip(columns, top_cards):
        warnings = escape(item["warnings"] or "No major model warning")
        with column:
            st.markdown(
                f"""
                <div class="screener-card">
                    <div class="screener-label">#{item["rank"]} {escape(str(item["screen"]))}</div>
                    <div class="screener-horse">{escape(str(item["horse"]))}</div>
                    <div class="screener-meta">{escape(str(item["off_time"]))} - {escape(str(item["race"]))}</div>
                    <div class="screener-numbers">
                        Score {item["score"]} | Confidence {item["confidence"]} | Odds {escape(str(item["odds"]))}
                    </div>
                    <div class="screener-meta">Value signal: {escape(str(item["value_edge_pct"]))}</div>
                    <div class="screener-warning">{warnings}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with st.expander("Open full screener list", expanded=True):
        st.dataframe(screener_df, width="stretch", hide_index=True)
    st.caption("Screener signals are for research only. They do not place or automate bets.")

race_options = ["All races"] + sorted(df["race"].dropna().unique().tolist()) if not df.empty else ["All races"]
race = st.selectbox("Race", race_options)
if race != "All races":
    df = df[df["race"] == race]
    display_scores = [score for score in display_scores if score.runner.race_name == race]

st.subheader("Selections Tracker")
picks_df = picks_tracker_dataframe(display_scores)
if picks_df.empty:
    st.info("No race selections available to track.")
else:
    summary = picks_tracker_summary(picks_df)
    winner_metric, ew_metric = st.columns(2)
    winner_metric.metric("Winner pick win rate", summary["winner_win_rate"])
    ew_metric.metric("Best EW place rate", summary["ew_place_rate"])
    st.dataframe(picks_tracker_style(picks_df), width="stretch", hide_index=True)
    st.caption("Green means won or placed, red means lost, and blue means just missed. Unsettled rows wait for verified results.")

st.subheader("Ranked runners")
st.dataframe(df, width="stretch", hide_index=True)
st.download_button(
    "Download CSV",
    data=df.to_csv(index=False),
    file_name=f"ian-racing-model-{selected_date.isoformat()}.csv",
    mime="text/csv",
)

with st.expander("Component explanations", expanded=False):
    for score in display_scores:
        st.markdown(f"**{score.runner.horse}** - {score.recommendation}")
        st.write(
            {
                component.name: {
                    "score": component.score,
                    "confidence": component.confidence,
                    "data_quality": component.data_quality,
                    "explanation": component.explanation,
                }
                for component in score.components
            }
        )
