from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_model_snapshots


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


st.title("Backtesting")
st.caption("Pre-race model snapshots. These rows are saved before results are attached, so they are safe for later backtesting.")

snapshots = get_model_snapshots(Settings(), limit=5000)
df = pd.DataFrame(snapshots)
if df.empty:
    st.info("No model snapshots have been saved yet. Open Today's cards or the main dashboard to trigger the first capture.")
else:
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce").dt.date
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["win_value_edge"] = pd.to_numeric(df["win_value_edge"], errors="coerce")
    df["place_value_edge"] = pd.to_numeric(df["place_value_edge"], errors="coerce")

    min_date = df["meeting_date"].min()
    max_date = df["meeting_date"].max()
    date_range = st.date_input("Date range", value=(min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["meeting_date"] >= start_date) & (df["meeting_date"] <= end_date)]

    course_options = ["All UK courses"] + sorted(df["course"].dropna().unique().tolist())
    selected_course = st.selectbox("Course", course_options)
    if selected_course != "All UK courses":
        df = df[df["course"].eq(selected_course)]

    recommendation_options = ["All recommendations"] + sorted(df["recommendation"].dropna().unique().tolist())
    selected_recommendation = st.selectbox("Recommendation", recommendation_options)
    if selected_recommendation != "All recommendations":
        df = df[df["recommendation"].eq(selected_recommendation)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshots", f"{len(df):,}")
    c2.metric("Races", f"{df[['meeting_date', 'course', 'off_time', 'race']].drop_duplicates().shape[0]:,}")
    c3.metric("Avg score", _format_number(df["total_score"].mean()))
    c4.metric("Avg confidence", _format_number(df["confidence"].mean()))

    st.subheader("Snapshot Mix")
    summary = (
        df.groupby("recommendation", dropna=False)
        .agg(
            runners=("horse", "count"),
            avg_score=("total_score", "mean"),
            avg_confidence=("confidence", "mean"),
            avg_win_edge=("win_value_edge", "mean"),
            avg_place_edge=("place_value_edge", "mean"),
        )
        .reset_index()
    )
    for column in ("avg_score", "avg_confidence", "avg_win_edge", "avg_place_edge"):
        summary[column] = summary[column].round(3)
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Captured Runner Snapshots")
    visible_columns = [
        "snapshot_at",
        "meeting_date",
        "course",
        "off_time",
        "race",
        "horse",
        "race_type",
        "race_class",
        "going",
        "distance",
        "odds",
        "total_score",
        "confidence",
        "recommendation",
        "win_probability",
        "place_probability",
        "fair_win_odds",
        "fair_place_odds",
        "win_value_edge",
        "place_value_edge",
        "red_flags",
        "warnings",
    ]
    st.dataframe(df[visible_columns], width="stretch", hide_index=True)
    st.download_button(
        "Download backtest snapshots CSV",
        data=df[visible_columns].to_csv(index=False),
        file_name="ian-racing-model-backtest-snapshots.csv",
        mime="text/csv",
    )
