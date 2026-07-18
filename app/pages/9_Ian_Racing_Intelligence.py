from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ian_racing_model.config import Settings
from ian_racing_model.results_summary import (
    daily_winning_placing_summary_dataframe,
    selection_hit,
    winning_placing_selections_dataframe,
)
from ian_racing_model.services import get_model_snapshots, get_scored_card_result
from ian_racing_model.table_styles import picks_tracker_style, research_table_style
from ian_racing_model.ui import (
    available_courses,
    default_date,
    picks_tracker_breakdown,
    picks_tracker_dataframe,
    picks_tracker_summary,
)
from racing_intelligence.scoring import intelligence_dataframe
from racing_intelligence.scoring.v5 import V5_ENGINE_WEIGHTS, V5_PLACE_WEIGHTS, V5_WIN_WEIGHTS


def _all_tracked_picks(settings: Settings, selected_course: str) -> pd.DataFrame:
    snapshot_dates = sorted(
        {
            snapshot["meeting_date"]
            for snapshot in get_model_snapshots(settings, limit=5000)
            if snapshot.get("provider") != "mock" and snapshot.get("meeting_date")
        }
    )
    if not snapshot_dates:
        return pd.DataFrame()

    all_picks = []
    for meeting_date_iso in snapshot_dates:
        day_result = get_scored_card_result(date.fromisoformat(meeting_date_iso), None, settings)
        day_scores = day_result.scores
        if selected_course != "All UK and Irish courses":
            day_scores = [score for score in day_scores if score.runner.course == selected_course]
        day_picks = picks_tracker_dataframe(day_scores)
        if day_picks.empty:
            continue
        day_picks.insert(0, "meeting_date", meeting_date_iso)
        all_picks.append(day_picks)
    return pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()


def _intelligence_selection_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        if bool(row.get("_selection_hit", False)):
            return ["background-color: #dcfce7; color: #14532d;"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(row_style, axis=1)
    if "_selection_hit" in df.columns:
        styler = styler.hide(axis="columns", subset=["_selection_hit"])
    return styler


def _add_selection_outcomes(intelligence_df: pd.DataFrame, picks_df: pd.DataFrame) -> pd.DataFrame:
    if intelligence_df.empty or picks_df.empty:
        return intelligence_df

    key_columns = ["course", "off_time", "race", "horse"]
    if any(column not in intelligence_df.columns for column in key_columns):
        return intelligence_df

    pick_columns = key_columns + ["pick_type", "result", "outcome"]
    picks = picks_df[[column for column in pick_columns if column in picks_df.columns]].copy()
    if not set(key_columns).issubset(picks.columns):
        return intelligence_df

    picks["_selection_hit"] = picks.apply(selection_hit, axis=1)
    merged = intelligence_df.merge(picks, on=key_columns, how="left")
    merged["pick_type"] = merged["pick_type"].fillna("")
    merged["result"] = merged["result"].fillna("")
    merged["outcome"] = merged["outcome"].fillna("")
    merged["_selection_hit"] = merged["_selection_hit"].fillna(False).astype(bool)
    return merged


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
picks_df = picks_tracker_dataframe(scores)
if not picks_df.empty:
    picks_df.insert(0, "meeting_date", selected_date.isoformat())
tracked_df = _all_tracked_picks(settings, selected_course)
if tracked_df.empty and not picks_df.empty:
    tracked_df = picks_df.copy()
df = _add_selection_outcomes(df, picks_df)

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

with st.expander("V5 engine principles", expanded=False):
    st.markdown(
        """
        Version 5 scores runners through eight explainable engines. Each engine returns a score,
        confidence, data-quality status and explanation. Missing evidence lowers confidence; it is
        not filled in with made-up ratings.
        """
    )
    weight_rows = [
        {"engine": name.replace("_", " ").title(), "engine_weight": weight}
        for name, weight in V5_ENGINE_WEIGHTS.items()
    ]
    st.dataframe(pd.DataFrame(weight_rows), width="stretch", hide_index=True)
    st.caption(
        "Win Index leans more towards Ability. Place Index leans more towards Suitability, "
        "Race Shape and repeatable reliability."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "engine": [name.replace("_", " ").title() for name in V5_WIN_WEIGHTS],
                "win_weight": list(V5_WIN_WEIGHTS.values()),
                "place_weight": [V5_PLACE_WEIGHTS[name] for name in V5_WIN_WEIGHTS],
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("Selection Results Tracker")
if picks_df.empty:
    st.info("No model selections are available to track for this date/course.")
else:
    summary = picks_tracker_summary(picks_df)
    settled = picks_df[~picks_df["outcome"].eq("Awaiting result")]
    hits = winning_placing_selections_dataframe(picks_df)
    tracker_cols = st.columns(4)
    tracker_cols[0].metric("Settled today", f"{len(settled)}")
    tracker_cols[1].metric("Winning / placing today", f"{len(hits)}")
    tracker_cols[2].metric("Winner win rate", summary["winner_win_rate"])
    tracker_cols[3].metric("EW place rate", summary["ew_place_rate"])
    breakdown_df = picks_tracker_breakdown(picks_df)
    if not breakdown_df.empty:
        st.dataframe(breakdown_df, width="stretch", hide_index=True)
    if hits.empty:
        st.info("No winning or placing selections have been matched for this date yet.")
    else:
        st.dataframe(picks_tracker_style(hits), width="stretch", hide_index=True)

if not tracked_df.empty:
    all_hits = winning_placing_selections_dataframe(tracked_df)
    all_settled = tracked_df[~tracked_df["outcome"].eq("Awaiting result")]
    daily_hits = daily_winning_placing_summary_dataframe(tracked_df)
    cumulative_cols = st.columns(4)
    cumulative_cols[0].metric("Days tracked", f"{tracked_df['meeting_date'].nunique()}")
    cumulative_cols[1].metric("Settled all-time", f"{len(all_settled)}")
    cumulative_cols[2].metric("Winning / placing all-time", f"{len(all_hits)}")
    cumulative_cols[3].metric(
        "All-time hit rate",
        "No settled picks" if len(all_settled) == 0 else f"{(len(all_hits) / len(all_settled)) * 100:.1f}%",
    )
    with st.expander("Daily winning / placing summary", expanded=False):
        if daily_hits.empty:
            st.info("No daily winning or placing selections are available yet.")
        else:
            st.dataframe(picks_tracker_style(daily_hits), width="stretch", hide_index=True)
    with st.expander("Cumulative winning / placing selections", expanded=False):
        if all_hits.empty:
            st.info("No cumulative winning or placing selections have been matched yet.")
        else:
            st.dataframe(picks_tracker_style(all_hits), width="stretch", hide_index=True)

if df.empty:
    st.info("No runners are available for the selected date/course.")
else:
    st.subheader("V5 Place Index")
    v5_columns = [
        "rank",
        "horse",
        "course",
        "off_time",
        "race",
        "odds",
        "field_size",
        "v5_place_index",
        "v5_win_index",
        "v5_recommendation",
        "v5_confidence",
        "v5_data_quality",
        "ability_engine",
        "suitability_engine",
        "race_shape_engine",
        "trainer_intent_engine",
        "current_wellbeing_engine",
        "improvement_engine",
        "market_value_engine",
        "historical_performance_engine",
        "v5_explanation",
        "_selection_hit",
    ]
    v5_df = df[[column for column in v5_columns if column in df.columns]].copy()
    if not v5_df.empty:
        v5_df = v5_df.sort_values(["v5_place_index", "v5_confidence"], ascending=[False, False])
        v5_df["rank"] = range(1, len(v5_df) + 1)
        st.dataframe(_intelligence_selection_style(v5_df.head(25)), width="stretch", hide_index=True)

    st.subheader("Value Intelligence")
    value_df = df[df["recommendation"].isin(["WIN_VALUE", "PLACE_VALUE", "PLACE_PROFILE"])].copy()
    if value_df.empty:
        st.info("No runner currently clears the value/profile gates.")
    else:
        st.dataframe(_intelligence_selection_style(value_df), width="stretch", hide_index=True)

    st.subheader("All Runner Intelligence")
    st.dataframe(_intelligence_selection_style(df), width="stretch", hide_index=True)
    st.download_button(
        "Download intelligence CSV",
        data=df.drop(columns=[column for column in df.columns if column.startswith("_")], errors="ignore").to_csv(index=False),
        file_name=f"ian-racing-intelligence-{selected_date.isoformat()}.csv",
        mime="text/csv",
    )
    st.caption("Green highlights selections that won, or EW/place selections that finished in the available places.")

with st.expander("Research guardrails", expanded=False):
    st.markdown(
        """
        - Win probability and place probability are separate outputs.
        - Place value is not treated as a small adjustment to win value.
        - Missing fields lower confidence rather than being filled with invented data.
        - This platform must never connect to bet-placement endpoints.
        """
    )
