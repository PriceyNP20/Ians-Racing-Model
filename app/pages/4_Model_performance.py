from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.edge_lab import closing_value_dataframe, edge_calibration_dataframe, edge_filter_recommendations
from ian_racing_model.outsider import outsider_last_time_dataframe
from ian_racing_model.services import get_model_snapshots, get_scored_card_result
from ian_racing_model.table_styles import picks_tracker_style, research_table_style
from ian_racing_model import ui as ui_helpers
from ian_racing_model.ui import (
    available_courses,
    default_date,
    model_upgrade_notes,
    performance_by_odds_band,
    picks_tracker_breakdown,
    picks_tracker_dataframe,
    picks_tracker_summary,
)


st.title("Model Performance")
selected_date = st.date_input("Date", value=default_date())
settings = Settings()
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
if selected_course != "All UK courses":
    scores = [score for score in scores if score.runner.course == selected_course]

st.subheader("All-Time Model Accuracy")
snapshot_dates = sorted(
    {
        snapshot["meeting_date"]
        for snapshot in get_model_snapshots(settings, limit=5000)
        if snapshot.get("provider") != "mock" and snapshot.get("meeting_date")
    }
)
if not snapshot_dates:
    snapshot_dates = [selected_date.isoformat()]

all_picks = []
for meeting_date_iso in snapshot_dates:
    day_result = get_scored_card_result(date.fromisoformat(meeting_date_iso), None, settings)
    day_scores = day_result.scores
    if selected_course != "All UK courses":
        day_scores = [score for score in day_scores if score.runner.course == selected_course]
    day_picks = picks_tracker_dataframe(day_scores)
    if day_picks.empty:
        continue
    day_picks.insert(0, "meeting_date", meeting_date_iso)
    all_picks.append(day_picks)

all_picks_df = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
if all_picks_df.empty:
    st.info("No combined selections are available yet.")
else:
    all_settled = all_picks_df[~all_picks_df["outcome"].eq("Awaiting result")]
    all_summary = picks_tracker_summary(all_picks_df)
    all_breakdown = picks_tracker_breakdown(all_picks_df)
    top_cols = st.columns(5)
    top_cols[0].metric("Days tracked", f"{all_picks_df['meeting_date'].nunique()}")
    top_cols[1].metric("Settled picks", f"{len(all_settled)}")
    top_cols[2].metric("Winner win rate", all_summary["winner_win_rate"])
    top_cols[3].metric("EW place rate", all_summary["ew_place_rate"])
    top_cols[4].metric("Just missed", f"{int(all_settled['outcome'].isin(['JUST LOST', 'JUST MISSED']).sum())}")
    if not all_breakdown.empty:
        st.dataframe(all_breakdown, width="stretch", hide_index=True)

    daily_rows = []
    for day, day_rows in all_picks_df.groupby("meeting_date", sort=True):
        settled_day = day_rows[~day_rows["outcome"].eq("Awaiting result")]
        summary_day = picks_tracker_summary(day_rows)
        daily_rows.append(
            {
                "meeting_date": day,
                "settled": len(settled_day),
                "winner_win_rate": summary_day["winner_win_rate"],
                "ew_place_rate": summary_day["ew_place_rate"],
                "wins": int(settled_day["outcome"].eq("WIN").sum()),
                "places": int(settled_day["outcome"].isin(["WIN", "PLACED"]).sum()),
                "just_missed": int(settled_day["outcome"].isin(["JUST LOST", "JUST MISSED"]).sum()),
                "losses": int(settled_day["outcome"].eq("LOSE").sum()),
            }
        )
    daily_df = pd.DataFrame(daily_rows).sort_values("meeting_date", ascending=False)
    with st.expander("Daily Accuracy Breakdown", expanded=True):
        st.dataframe(daily_df, width="stretch", hide_index=True)

st.subheader("Selected Day Detail")
picks_df = picks_tracker_dataframe(scores)
if picks_df.empty:
    st.info("No selections are available for this date.")
else:
    summary = picks_tracker_summary(picks_df)
    winner_metric, ew_metric = st.columns(2)
    winner_metric.metric("Winner pick win rate", summary["winner_win_rate"])
    ew_metric.metric("Best EW place rate", summary["ew_place_rate"])
    breakdown_df = picks_tracker_breakdown(picks_df)
    if not breakdown_df.empty:
        st.dataframe(breakdown_df, width="stretch", hide_index=True)
    odds_band_df = performance_by_odds_band(picks_df)
    if not odds_band_df.empty:
        st.subheader("Performance by Odds Band")
        st.dataframe(odds_band_df, width="stretch", hide_index=True)
    st.subheader("Edge Calibration Lab")
    for recommendation in edge_filter_recommendations(picks_df):
        st.markdown(f"- {recommendation}")
    calibration_dimensions = {
        "Odds band": "odds_band",
        "Score band": "score_band",
        "Confidence band": "confidence_band",
        "Pick type": "pick_type",
    }
    calibration_tabs = st.tabs(list(calibration_dimensions))
    for tab, (label, dimension) in zip(calibration_tabs, calibration_dimensions.items()):
        with tab:
            calibration_df = edge_calibration_dataframe(picks_df, dimension)
            if calibration_df.empty:
                st.info(f"No settled calibration data yet for {label.lower()}.")
            else:
                st.dataframe(calibration_df, width="stretch", hide_index=True)
    st.subheader("Closing Price Value")
    clv_df = closing_value_dataframe(scores)
    if clv_df.empty:
        st.info("No starting-price or closing-price sample is available yet from verified results.")
    else:
        st.caption("Positive closing value means the model found a bigger price than the market returned at the off.")
        st.dataframe(research_table_style(clv_df), width="stretch", hide_index=True)
    st.subheader("Performance Lab")
    lab_dimensions = {
        "Race type": "race_type",
        "Class": "race_class",
        "Field size": "field_size_band",
        "Going": "going",
        "Surface": "surface",
        "Odds band": "odds_band",
        "Selection reason": "selection_reason",
    }
    lab_tabs = st.tabs(list(lab_dimensions))
    for tab, (label, dimension) in zip(lab_tabs, lab_dimensions.items()):
        with tab:
            if dimension == "odds_band":
                lab_df = odds_band_df
            elif hasattr(ui_helpers, "performance_lab_dataframe"):
                lab_df = ui_helpers.performance_lab_dataframe(picks_df, dimension)
            else:
                lab_df = None
                st.info("Performance Lab is still finishing its rebuild. Refresh again in a moment.")
                continue
            if lab_df.empty:
                st.info(f"No settled data yet for {label.lower()}.")
            else:
                st.dataframe(lab_df, width="stretch", hide_index=True)
    st.dataframe(picks_tracker_style(picks_df), width="stretch", hide_index=True)
    st.caption("Yellow means won or placed, red means lost, and blue means just missed. Unsettled rows wait for verified results.")

st.subheader("Outsider Last-Time Signals")
outsider_df = outsider_last_time_dataframe(scores)
if outsider_df.empty:
    st.info("No verified last-time rank-outsider win/place signals are available in the imported fields.")
else:
    st.dataframe(research_table_style(outsider_df), width="stretch", hide_index=True)

with st.expander("Model Edge Upgrade Notes", expanded=False):
    for note in model_upgrade_notes():
        st.markdown(f"- {note}")
