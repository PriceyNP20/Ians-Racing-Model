from __future__ import annotations

from html import escape
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model import ui as ui_helpers
from ian_racing_model.config import Settings
from ian_racing_model.services import get_refresh_statuses, get_scored_card_result
from ian_racing_model.ui import (
    available_courses,
    default_date,
    model_upgrade_notes,
    outsider_last_time_dataframe,
    picks_tracker_breakdown,
    picks_tracker_dataframe,
    picks_tracker_style,
    picks_tracker_summary,
    scores_to_dataframe,
    screener_dataframe,
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


def _refresh_health_dataframe(statuses):
    helper = getattr(ui_helpers, "refresh_health_dataframe", None)
    if helper is not None:
        return helper(statuses)
    return scores_to_dataframe([])


def _refresh_health_summary(statuses, provider, warning=None):
    helper = getattr(ui_helpers, "refresh_health_summary", None)
    if helper is not None:
        return helper(statuses, provider, warning)
    if warning or provider == "mock":
        return {
            "label": "Using sample data",
            "detail": "Live API data is not currently powering this view.",
            "state": "warning",
        }
    return {
        "label": "Live API active",
        "detail": "Refresh details are loading.",
        "state": "success",
    }


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
refresh_statuses = get_refresh_statuses(settings, limit=12)
health = _refresh_health_summary(refresh_statuses, result.provider, result.warning)
if health["state"] == "success":
    st.success(f"{health['label']}: {health['detail']}")
elif health["state"] == "error":
    st.error(f"{health['label']}: {health['detail']}")
elif health["state"] == "warning":
    st.warning(f"{health['label']}: {health['detail']}")
else:
    st.info(f"{health['label']}: {health['detail']}")
with st.expander("API refresh health", expanded=False):
    health_df = _refresh_health_dataframe(refresh_statuses)
    if health_df.empty:
        st.info("No API refresh records are available yet.")
    else:
        st.dataframe(health_df, width="stretch", hide_index=True)
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

st.subheader("Best Value")
value_df = value_screener_dataframe(display_scores, limit=10)
if value_df.empty:
    st.info("No positive model-versus-market value edges are available from the current odds.")
else:
    st.dataframe(value_df, width="stretch", hide_index=True)
    st.caption("Value edge compares model probability against available odds. It is for research only.")

st.subheader("Model Signals")
signal_df = _model_signal_dataframe(display_scores, limit=20)
if signal_df.empty:
    st.info("No setup or market-move signals are available in the imported fields yet.")
else:
    st.dataframe(signal_df, width="stretch", hide_index=True)

st.subheader("Race Picks")
race_pick_df = _race_selection_screener_dataframe(display_scores)
if race_pick_df.empty:
    st.info("No race-level picks available.")
else:
    st.dataframe(race_pick_df, width="stretch", hide_index=True)
    st.caption("Winner and EW/value picks use separate scoring logic, so the EW pick is biased toward place chance and price value.")

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
    breakdown_df = picks_tracker_breakdown(picks_df)
    if not breakdown_df.empty:
        st.dataframe(breakdown_df, width="stretch", hide_index=True)
    st.dataframe(picks_tracker_style(picks_df), width="stretch", hide_index=True)
    st.caption("Green means won or placed, red means lost, and blue means just missed. Unsettled rows wait for verified results.")

st.subheader("Outsider Last-Time Signals")
outsider_df = outsider_last_time_dataframe(display_scores)
if outsider_df.empty:
    st.info("No verified last-time rank-outsider win/place signals are available in the imported fields.")
else:
    st.dataframe(outsider_df, width="stretch", hide_index=True)

with st.expander("Model Edge Upgrade Notes", expanded=False):
    for note in model_upgrade_notes():
        st.markdown(f"- {note}")

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
