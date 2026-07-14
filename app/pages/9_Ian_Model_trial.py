from __future__ import annotations

from html import escape
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ian_racing_model.config import Settings
from ian_racing_model.ian_index import ian_index_place_dataframe, ian_index_weights_dataframe
try:
    from ian_racing_model.ian_index import ian_index_acca_dataframe
except ImportError:
    def ian_index_acca_dataframe(scores, limit=6):
        trial = ian_index_place_dataframe(scores)
        if trial.empty:
            return trial
        rows = []
        used_races = set()
        for row in trial.to_dict("records"):
            key = (str(row.get("course", "")).lower(), str(row.get("off_time", "")).lower())
            if key in used_races:
                continue
            row["acca_rank"] = len(rows) + 1
            row["pick_type"] = "Ian Trial EW pick"
            rows.append(row)
            used_races.add(key)
            if len(rows) == limit:
                break
        return pd.DataFrame(rows)
from ian_racing_model.results_summary import winning_placing_selections_dataframe
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.table_styles import research_table_style
from ian_racing_model.ui import available_courses, default_date


def _trial_card(label: str, row: dict | None) -> None:
    if row is None:
        st.markdown(
            f"""
            <div style="border:1px solid #d8dee8;border-radius:8px;padding:14px;min-height:150px;background:#ffffff;">
              <div style="font-size:13px;font-weight:750;text-transform:uppercase;color:#6b7280;">{escape(label)}</div>
              <div style="font-size:20px;font-weight:750;margin-top:12px;color:#374151;">No signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    horse = escape(str(row.get("horse", "Unknown")))
    course = escape(str(row.get("course", "")))
    off_time = escape(str(row.get("off_time", "")))
    race = escape(str(row.get("race", "")))
    rating = escape(str(row.get("place_rating", "")))
    probability = escape(str(row.get("place_probability", "Unavailable")))
    edge = escape(str(row.get("place_value_edge", "Needs odds")))
    odds = escape(str(row.get("odds", "Unavailable")))
    evidence = escape(str(row.get("evidence_summary", "")))
    explanation = escape(str(row.get("explanation", "")))
    outcome = str(row.get("outcome", "")).upper()
    pick_type = str(row.get("pick_type", "")).lower()
    is_hit = outcome == "WIN" or (("ew" in pick_type or "place" in pick_type) and outcome == "PLACED")
    background = "#dcfce7" if is_hit else "#ffffff"
    border = "#86efac" if is_hit else "#d8dee8"
    st.markdown(
        f"""
        <div style="border:1px solid {border};border-radius:8px;padding:14px;min-height:210px;background:{background};">
          <div style="font-size:13px;font-weight:750;text-transform:uppercase;color:#374151;">{escape(label)}</div>
          <div style="font-size:23px;font-weight:800;margin-top:8px;color:#202633;">{horse}</div>
          <div style="font-size:14px;line-height:1.35;color:#374151;">{off_time} - {course}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">{race}</div>
          <div style="font-size:14px;font-weight:750;margin-top:8px;color:#202633;">Place rating {rating} | Odds {odds}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">Place {probability} | Edge {edge}</div>
          <div style="font-size:12px;line-height:1.35;margin-top:6px;color:#374151;">{evidence}</div>
          <div style="font-size:12px;line-height:1.35;margin-top:8px;color:#6b7280;">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _acca_card(row: dict) -> None:
    rank = escape(str(row.get("acca_rank", "")))
    horse = escape(str(row.get("horse", "Unknown")))
    course = escape(str(row.get("course", "")))
    off_time = escape(str(row.get("off_time", "")))
    race = escape(str(row.get("race", "")))
    rating = escape(str(row.get("place_rating", "")))
    probability = escape(str(row.get("place_probability", "Unavailable")))
    odds = escape(str(row.get("odds", "Unavailable")))
    edge = escape(str(row.get("place_value_edge", "Needs odds")))
    evidence = escape(str(row.get("evidence_summary", "")))
    result = escape(str(row.get("result", "Awaiting result")))
    outcome = str(row.get("outcome", "")).upper()
    is_hit = outcome in {"WIN", "PLACED"}
    background = "#dcfce7" if is_hit else "#ffffff"
    border = "#86efac" if is_hit else "#d8dee8"
    st.markdown(
        f"""
        <div style="border:1px solid {border};border-radius:8px;padding:14px;min-height:215px;background:{background};">
          <div style="font-size:13px;font-weight:750;text-transform:uppercase;color:#374151;">Ian Trial Acca #{rank}</div>
          <div style="font-size:23px;font-weight:800;margin-top:8px;color:#202633;">{horse}</div>
          <div style="font-size:14px;line-height:1.35;color:#374151;">{off_time} - {course}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">{race}</div>
          <div style="font-size:14px;font-weight:750;margin-top:8px;color:#202633;">Place rating {rating} | Place {probability}</div>
          <div style="font-size:13px;line-height:1.35;color:#4b5563;">Odds {odds} | Edge {edge}</div>
          <div style="font-size:12px;line-height:1.35;margin-top:6px;color:#374151;">{evidence}</div>
          <div style="font-size:12px;line-height:1.35;margin-top:8px;color:#6b7280;">Result {result} | {escape(outcome.title() if outcome else 'Awaiting result')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Ian Model Trial", layout="wide")
st.title("Ian Model Trial")
st.caption("Ian Index V4: a place-rating trial. This ranks who is most likely to place, not who wins. Research only.")

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

trial_df = ian_index_place_dataframe(scores)
acca_df = ian_index_acca_dataframe(scores)

if trial_df.empty:
    st.info("No eligible runners are available for the Ian Index trial.")
else:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Runners scored", f"{len(trial_df)}")
    metric_cols[1].metric("Top place rating", f"{trial_df.iloc[0]['place_rating']}")
    metric_cols[2].metric("Positive place value", f"{trial_df['place_value_edge'].astype(str).str.startswith('+').sum()}")
    metric_cols[3].metric("Weak data rows", f"{trial_df['data_quality'].eq('weak').sum()}")

    st.subheader("Evidence Quality")
    evidence_cols = st.columns(3)
    evidence_cols[0].metric("Imported signals", f"{int(trial_df['imported_signals'].sum())}")
    evidence_cols[1].metric("Proxy signals", f"{int(trial_df['proxy_signals'].sum())}")
    evidence_cols[2].metric("Missing signals", f"{int(trial_df['missing_signals'].sum())}")
    with st.expander("Open evidence-quality guide", expanded=False):
        st.markdown(
            """
            - **Imported** means the API supplied a direct or specialist field for that principle.
            - **Proxy** means the model used available nearby evidence, such as official rating, recent form, draw, odds, history, or trainer/jockey names.
            - **Missing** means the signal is not present and confidence is reduced.
            """
        )

    top = trial_df.iloc[0].to_dict()
    value_rows = trial_df[trial_df["place_value_edge"].astype(str).str.startswith("+")]
    value = value_rows.iloc[0].to_dict() if not value_rows.empty else None
    clean_rows = trial_df[trial_df["red_flags"].eq("None")]
    clean = clean_rows.iloc[0].to_dict() if not clean_rows.empty else None

    st.subheader("Trial Signals")
    cols = st.columns(3)
    with cols[0]:
        _trial_card("Top Place Rating", top)
    with cols[1]:
        _trial_card("Best Place Value", value)
    with cols[2]:
        _trial_card("Cleanest Place Profile", clean)

    st.subheader("Ian Trial EW Accumulator 6")
    st.caption("Six strongest Ian Index place profiles across the selected cards. One runner per race; fields under 8 runners are excluded.")
    if acca_df.empty:
        st.info("No six-runner Ian Trial acca shortlist is available from the current place evidence.")
    else:
        rows = acca_df.to_dict("records")
        for start in range(0, len(rows), 3):
            acca_cols = st.columns(3)
            for col, row in zip(acca_cols, rows[start : start + 3]):
                with col:
                    _acca_card(row)
        hits_df = winning_placing_selections_dataframe(acca_df)
        st.subheader("Ian Trial Winning / Placing Acca Picks")
        if hits_df.empty:
            st.info("No Ian Trial acca picks have won or placed yet for this date.")
        else:
            st.dataframe(research_table_style(hits_df), width="stretch", hide_index=True)
        with st.expander("Open Ian Trial acca table", expanded=True):
            st.dataframe(research_table_style(acca_df), width="stretch", hide_index=True)

    st.subheader("Ian Index Place Ratings")
    st.dataframe(research_table_style(trial_df), width="stretch", hide_index=True)

    with st.expander("Open principle evidence by runner", expanded=False):
        evidence_columns = [
            "rank",
            "horse",
            "course",
            "off_time",
            "race",
            "evidence_summary",
            "ability_evidence",
            "speed_evidence",
            "class_evidence",
            "pace_evidence",
            "value_evidence",
            "trainer_evidence",
            "jockey_evidence",
            "course_going_evidence",
        ]
        st.dataframe(
            trial_df[[column for column in evidence_columns if column in trial_df.columns]],
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "Download Ian Model trial CSV",
        data=trial_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ian_model_trial_{selected_date.isoformat()}.csv",
        mime="text/csv",
    )

with st.expander("Ian Index V4 weights", expanded=False):
    st.dataframe(ian_index_weights_dataframe(), width="stretch", hide_index=True)
    st.caption(
        "The score is built as a 0-100 PLACE RATING. Missing imported fields reduce confidence and are explained in the row notes."
    )
