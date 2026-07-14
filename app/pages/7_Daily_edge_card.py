from __future__ import annotations

from html import escape
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ian_racing_model import ui as ui_helpers
from ian_racing_model.acca import ew_accumulator_dataframe
from ian_racing_model.config import Settings
from ian_racing_model.edge_lab import enhanced_undervalued_edge_dataframe, negative_value_dataframe
from ian_racing_model.outsider import outsider_last_time_dataframe
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.table_styles import research_table_style
from ian_racing_model.ui import (
    available_courses,
    default_date,
    picks_tracker_dataframe,
    scores_to_dataframe,
    value_screener_dataframe,
)


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
    )


def _model_signal_dataframe(scores, limit=20):
    helper = getattr(ui_helpers, "model_signal_dataframe", None)
    if helper is not None:
        return helper(scores, limit=limit)
    return pd.DataFrame()


def _top_row(df: pd.DataFrame, match_column: str, match_text: str) -> dict | None:
    if df.empty or match_column not in df.columns:
        return None
    subset = df[df[match_column].astype(str).str.contains(match_text, case=False, na=False)]
    if subset.empty:
        return None
    sort_columns = [column for column in ("selection_score", "edge_score", "score", "model_score", "confidence") if column in subset.columns]
    if sort_columns:
        subset = subset.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return subset.iloc[0].to_dict()


def _edge_row(edge_df: pd.DataFrame) -> dict | None:
    if edge_df.empty:
        return None
    sort_columns = [column for column in ("edge_score", "confidence", "model_score", "score") if column in edge_df.columns]
    if sort_columns:
        edge_df = edge_df.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return edge_df.iloc[0].to_dict()


def _card(label: str, row: dict | None, tone: str) -> None:
    colours = {
        "positive": ("#fef3c7", "#78350f", "#f59e0b"),
        "watch": ("#dbeafe", "#1e3a8a", "#3b82f6"),
        "negative": ("#fee2e2", "#7f1d1d", "#ef4444"),
    }
    background, text_colour, border = colours[tone]
    if row is None:
        st.markdown(
            f"""
            <div style="border:1px solid #d8dee8;border-radius:8px;padding:14px;min-height:150px;background:#ffffff;">
              <div style="font-size:13px;font-weight:700;text-transform:uppercase;color:#6b7280;">{escape(label)}</div>
              <div style="font-size:20px;font-weight:700;margin-top:12px;color:#374151;">No signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    horse = escape(str(row.get("horse", "Unknown")))
    race = escape(str(row.get("race", "")))
    off_time = escape(str(row.get("off_time", "")))
    odds = escape(str(row.get("odds", row.get("today_odds", "Unavailable"))))
    score_value = row.get("selection_score", row.get("edge_score", row.get("score", row.get("model_score", ""))))
    score = escape(str(score_value))
    reason = escape(str(row.get("reason", row.get("evidence", row.get("avoid_reason", row.get("signal", ""))))))
    st.markdown(
        f"""
        <div style="border:1px solid {border};border-radius:8px;padding:14px;min-height:170px;background:{background};">
          <div style="font-size:13px;font-weight:750;text-transform:uppercase;color:{text_colour};">{escape(label)}</div>
          <div style="font-size:22px;font-weight:750;margin-top:8px;color:#202633;">{horse}</div>
          <div style="font-size:14px;line-height:1.35;color:#374151;">{off_time} - {race}</div>
          <div style="font-size:14px;font-weight:700;margin-top:8px;color:#202633;">Odds {odds} | Signal {score}</div>
          <div style="font-size:13px;line-height:1.35;margin-top:8px;color:#4b5563;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _acca_card(row: dict) -> None:
    rank = escape(str(row.get("acca_rank", "")))
    horse = escape(str(row.get("horse", "Unknown")))
    course = escape(str(row.get("course", "")))
    race = escape(str(row.get("race", "")))
    off_time = escape(str(row.get("off_time", "")))
    odds = escape(str(row.get("odds", "Unavailable")))
    place_probability = escape(str(row.get("place_probability", "Unavailable")))
    place_edge = escape(str(row.get("place_edge", "Unavailable")))
    score = escape(str(row.get("acca_score", "")))
    pillars = escape(str(row.get("evidence_pillars", "")))
    warnings = escape(str(row.get("warnings", "")))
    st.markdown(
        f"""
        <div style="border:1px solid #f59e0b;border-radius:8px;padding:14px;min-height:210px;background:#fef3c7;">
          <div style="font-size:13px;font-weight:750;text-transform:uppercase;color:#78350f;">EW Acca #{rank}</div>
          <div style="font-size:23px;font-weight:800;margin-top:8px;color:#202633;">{horse}</div>
          <div style="font-size:14px;line-height:1.35;color:#374151;">{off_time} - {course}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">{race}</div>
          <div style="font-size:14px;font-weight:750;margin-top:8px;color:#202633;">Place {place_probability} | Odds {odds}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">Edge {place_edge} | Acca score {score}</div>
          <div style="font-size:13px;line-height:1.35;margin-top:8px;color:#4b5563;">{pillars}</div>
          <div style="font-size:12px;line-height:1.35;margin-top:6px;color:#6b7280;">{warnings}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Daily Edge Card", layout="wide")
st.title("Daily Edge Card")
st.caption("Race-day research shortlist. No bet placement or automation.")

selected_date = st.date_input("Date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
if result.results_imported:
    st.success("Verified results have been matched to today's picks.")
else:
    st.info("Results are not matched yet. This card is pre-race research until verified results arrive.")

scores = result.scores
course_options = ["All UK courses"] + available_courses(scores)
selected_course = st.selectbox("Course", course_options)
if selected_course != "All UK courses":
    scores = [score for score in scores if score.runner.course == selected_course]

runner_df = scores_to_dataframe(scores)
race_picks = _race_selection_screener_dataframe(scores)
acca_df = ew_accumulator_dataframe(scores, limit=6)
edge_df = enhanced_undervalued_edge_dataframe(scores, limit=20)
value_df = value_screener_dataframe(scores, limit=20)
negative_df = negative_value_dataframe(scores, limit=20)
outsider_df = outsider_last_time_dataframe(scores)
signal_df = _model_signal_dataframe(scores, limit=30)

winner_row = _top_row(race_picks, "pick", "Winner")
ew_row = _top_row(race_picks, "pick", "EW|value")
edge_row = _edge_row(edge_df if not edge_df.empty else value_df)
outsider_row = outsider_df.iloc[0].to_dict() if not outsider_df.empty else None
avoid_row = negative_df.iloc[0].to_dict() if not negative_df.empty else None

metric_cols = st.columns(4)
metric_cols[0].metric("Races", f"{runner_df[['course', 'off_time', 'race']].drop_duplicates().shape[0] if not runner_df.empty else 0}")
metric_cols[1].metric("Runners", f"{len(runner_df)}")
metric_cols[2].metric("EW acca picks", f"{len(acca_df)}/6")
metric_cols[3].metric("Avoid flags", f"{len(negative_df)}")

st.subheader("EW Accumulator 6")
st.caption("Six strongest place profiles across the selected cards. Research shortlist only; no staking or bet placement.")
if acca_df.empty:
    st.info("No six-runner EW accumulator shortlist is available from the current place evidence.")
else:
    acca_rows = acca_df.to_dict("records")
    for start in range(0, len(acca_rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, acca_rows[start : start + 3]):
            with col:
                _acca_card(row)
    with st.expander("Open EW accumulator table", expanded=True):
        st.dataframe(research_table_style(acca_df), width="stretch", hide_index=True)

st.subheader("Today's Edge Card")
card_cols = st.columns(5)
with card_cols[0]:
    _card("Best Winner", winner_row, "positive")
with card_cols[1]:
    _card("Best EW Value", ew_row, "positive")
with card_cols[2]:
    _card("Undervalued", edge_row, "positive")
with card_cols[3]:
    _card("Outsider Signal", outsider_row, "watch")
with card_cols[4]:
    _card("Avoid", avoid_row, "negative")

st.subheader("Race-by-Race Picks")
if race_picks.empty:
    st.info("No race-level picks are available.")
else:
    st.dataframe(research_table_style(race_picks), width="stretch", hide_index=True)

st.subheader("Undervalued Edge Shortlist")
if edge_df.empty:
    st.info("No clear undervalued edge is available from the current model, odds and evidence.")
else:
    st.dataframe(research_table_style(edge_df), width="stretch", hide_index=True)

st.subheader("Best Value")
if value_df.empty:
    st.info("No positive model-versus-market value edges are available from the current odds.")
else:
    st.dataframe(research_table_style(value_df), width="stretch", hide_index=True)

st.subheader("Outsider Last-Time Signals")
if outsider_df.empty:
    st.info("No last-time 30/1+ placed runners with a similar setup are available in the imported fields.")
else:
    st.dataframe(research_table_style(outsider_df), width="stretch", hide_index=True)

st.subheader("Negative Value / Avoid")
if negative_df.empty:
    st.info("No obvious overbet or weak-value runners are flagged from the current prices.")
else:
    st.dataframe(research_table_style(negative_df), width="stretch", hide_index=True)

st.subheader("Model Signals")
if signal_df.empty:
    st.info("No setup or market-move signals are available in the imported fields yet.")
else:
    st.dataframe(research_table_style(signal_df), width="stretch", hide_index=True)
