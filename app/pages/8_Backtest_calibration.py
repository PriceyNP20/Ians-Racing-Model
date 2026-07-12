from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ian_racing_model.config import Settings
from ian_racing_model.services import get_model_snapshots, get_scored_card_result
from ian_racing_model.table_styles import research_table_style
from ian_racing_model.ui import available_courses, default_date, picks_tracker_dataframe


def _decimal_odds(value: Any) -> float | None:
    if value in (None, "", "Unavailable"):
        return None
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0:
                return None
            return float(num) / denominator + 1.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _odds_band(odds: float | None) -> str:
    if odds is None:
        return "No odds"
    if odds < 3:
        return "Under 3.0"
    if odds < 6:
        return "3.0 to 5.99"
    if odds < 10:
        return "6.0 to 9.99"
    if odds < 20:
        return "10.0 to 19.99"
    return "20.0+"


def _score_band(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Unknown"
    if score < 50:
        return "Under 50"
    if score < 60:
        return "50 to 59.9"
    if score < 70:
        return "60 to 69.9"
    return "70+"


def _confidence_band(confidence: float | None) -> str:
    if confidence is None or pd.isna(confidence):
        return "Unknown"
    if confidence < 0.5:
        return "Under 0.50"
    if confidence < 0.6:
        return "0.50 to 0.59"
    if confidence < 0.7:
        return "0.60 to 0.69"
    return "0.70+"


def _field_size_band(field_size: Any) -> str:
    try:
        size = int(field_size)
    except (TypeError, ValueError):
        return "Unknown"
    if size < 5:
        return "1 to 4"
    if size < 8:
        return "5 to 7"
    if size < 12:
        return "8 to 11"
    if size < 16:
        return "12 to 15"
    return "16+"


def _ratio_text(successes: int, total: int) -> str:
    if total == 0:
        return "No settled picks"
    return f"{(successes / total) * 100:.1f}% ({successes}/{total})"


def _prepare_picks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    prepared["_decimal_odds"] = prepared["odds"].map(_decimal_odds)
    prepared["odds_band"] = prepared["_decimal_odds"].map(_odds_band)
    prepared["_score"] = pd.to_numeric(prepared.get("score"), errors="coerce")
    prepared["_selection_score"] = pd.to_numeric(prepared.get("selection_score"), errors="coerce")
    prepared["_confidence"] = pd.to_numeric(prepared.get("confidence"), errors="coerce")
    prepared["score_band"] = prepared["_score"].map(_score_band)
    prepared["confidence_band"] = prepared["_confidence"].map(_confidence_band)
    prepared["field_size_band"] = prepared.get("field_size", pd.Series(dtype=object)).map(_field_size_band)
    prepared["pick_type"] = prepared["pick_type"].astype(str)
    prepared["outcome"] = prepared["outcome"].astype(str)
    return prepared


def _settled_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "outcome" not in df.columns:
        return pd.DataFrame()
    return df[~df["outcome"].eq("Awaiting result")].copy()


def _winner_profit(row: pd.Series) -> float:
    if str(row.get("pick_type")) != "Winner pick":
        return 0.0
    odds = row.get("_decimal_odds")
    if pd.isna(odds) or odds is None:
        return 0.0
    return float(odds) - 1.0 if row.get("outcome") == "WIN" else -1.0


def _edge_read(total: int, wins: int, places: int, just_missed: int, roi: float | None) -> str:
    if total < 3:
        return "Small sample"
    win_rate = wins / total
    place_rate = places / total
    miss_rate = just_missed / total
    if roi is not None and roi >= 0.10:
        return "Profitable-looking pocket"
    if place_rate >= 0.30 or win_rate >= 0.18:
        return "Positive pocket"
    if miss_rate >= 0.20:
        return "Near-miss pocket"
    if place_rate < 0.12 or (roi is not None and roi <= -0.30):
        return "Bad pocket"
    return "Monitor"


def _calibration_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    settled = _settled_rows(df)
    if settled.empty or dimension not in settled.columns:
        return pd.DataFrame()
    rows = []
    for (pick_type, bucket), bucket_rows in settled.groupby(["pick_type", dimension], dropna=False):
        total = len(bucket_rows)
        wins = int(bucket_rows["outcome"].eq("WIN").sum())
        places = int(bucket_rows["outcome"].isin(["WIN", "PLACED"]).sum())
        just_missed = int(bucket_rows["outcome"].isin(["JUST LOST", "JUST MISSED"]).sum())
        losses = int(bucket_rows["outcome"].eq("LOSE").sum())
        winner_profit = bucket_rows.apply(_winner_profit, axis=1).sum()
        winner_stakes = int(bucket_rows["pick_type"].eq("Winner pick").sum())
        roi = winner_profit / winner_stakes if winner_stakes else None
        rows.append(
            {
                "pick_type": pick_type,
                dimension: bucket or "Unknown",
                "settled": total,
                "wins": wins,
                "places": places,
                "just_missed": just_missed,
                "losses": losses,
                "win_rate": _ratio_text(wins, total),
                "place_rate": _ratio_text(places, total),
                "avg_odds": _mean(bucket_rows["_decimal_odds"]),
                "avg_score": _mean(bucket_rows["_score"]),
                "avg_confidence": _mean(bucket_rows["_confidence"]),
                "winner_roi": _format_percent(roi),
                "edge_read": _edge_read(total, wins, places, just_missed, roi),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pick_type", "settled", "places", "wins"],
        ascending=[True, False, False, False],
    )


def _mean(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.1f}%"


def _pockets(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    if label == "good":
        mask = df["edge_read"].isin(["Profitable-looking pocket", "Positive pocket"])
    else:
        mask = df["edge_read"].eq("Bad pocket")
    return df[mask].copy()


def _rule_suggestions(calibration_tables: dict[str, pd.DataFrame]) -> list[str]:
    suggestions: list[str] = []
    for dimension, table in calibration_tables.items():
        if table.empty:
            continue
        good = _pockets(table, "good")
        bad = _pockets(table, "bad")
        for _, row in good.head(3).iterrows():
            suggestions.append(
                f"Lean into {row['pick_type']} when {dimension} is {row[dimension]}: {row['edge_read']} over {row['settled']} settled picks."
            )
        for _, row in bad.head(3).iterrows():
            suggestions.append(
                f"Tighten {row['pick_type']} when {dimension} is {row[dimension]}: {row['edge_read']} over {row['settled']} settled picks."
            )
    if not suggestions:
        suggestions.append("Keep collecting settled results before changing rules aggressively. Current sample is still thin.")
    return suggestions[:10]


def _snapshot_readiness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    prepared["meeting_date"] = pd.to_datetime(prepared["meeting_date"], errors="coerce").dt.date
    prepared["total_score"] = pd.to_numeric(prepared["total_score"], errors="coerce")
    prepared["confidence"] = pd.to_numeric(prepared["confidence"], errors="coerce")
    prepared["_decimal_odds"] = prepared["odds"].map(_decimal_odds)
    prepared["odds_band"] = prepared["_decimal_odds"].map(_odds_band)
    prepared["confidence_band"] = prepared["confidence"].map(_confidence_band)
    return (
        prepared.groupby(["recommendation", "odds_band", "confidence_band"], dropna=False)
        .agg(
            snapshots=("horse", "count"),
            races=("race", "nunique"),
            avg_score=("total_score", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .round({"avg_score": 2, "avg_confidence": 2})
        .sort_values(["snapshots", "recommendation"], ascending=[False, True])
    )


st.set_page_config(page_title="Backtest Calibration", layout="wide")
st.title("Backtest Calibration V1")
st.caption("Turns settled selections and saved pre-race snapshots into model-improvement rules.")

selected_date = st.date_input("Settled results date", value=default_date())
result = get_scored_card_result(selected_date, None, Settings())
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")

scores = result.scores
course_options = ["All UK courses"] + available_courses(scores)
selected_course = st.selectbox("Course", course_options)
if selected_course != "All UK courses":
    scores = [score for score in scores if score.runner.course == selected_course]

picks_df = _prepare_picks(picks_tracker_dataframe(scores))
settled = _settled_rows(picks_df)

metric_cols = st.columns(4)
metric_cols[0].metric("Settled picks", f"{len(settled)}")
metric_cols[1].metric("Winner wins", f"{int(settled['outcome'].eq('WIN').sum()) if not settled.empty else 0}")
metric_cols[2].metric("Places", f"{int(settled['outcome'].isin(['WIN', 'PLACED']).sum()) if not settled.empty else 0}")
metric_cols[3].metric("Just missed", f"{int(settled['outcome'].isin(['JUST LOST', 'JUST MISSED']).sum()) if not settled.empty else 0}")

if settled.empty:
    st.info("No settled picks are available for this date yet. This page will populate once verified results are matched.")
else:
    dimensions = {
        "Odds band": "odds_band",
        "Race class": "race_class",
        "Race type": "race_type",
        "Field size": "field_size_band",
        "Going": "going",
        "Confidence band": "confidence_band",
        "Score band": "score_band",
        "Recommendation": "recommendation",
    }
    calibration_tables = {label: _calibration_table(picks_df, dimension) for label, dimension in dimensions.items()}

    st.subheader("Suggested Rule Changes")
    for suggestion in _rule_suggestions(calibration_tables):
        st.markdown(f"- {suggestion}")

    good_pockets = pd.concat([_pockets(table, "good").assign(dimension=label) for label, table in calibration_tables.items() if not table.empty], ignore_index=True)
    bad_pockets = pd.concat([_pockets(table, "bad").assign(dimension=label) for label, table in calibration_tables.items() if not table.empty], ignore_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Good Pockets")
        if good_pockets.empty:
            st.info("No positive pockets yet. More settled races needed.")
        else:
            st.dataframe(research_table_style(good_pockets), width="stretch", hide_index=True)
    with c2:
        st.subheader("Bad Pockets")
        if bad_pockets.empty:
            st.info("No bad pockets detected yet.")
        else:
            st.dataframe(research_table_style(bad_pockets), width="stretch", hide_index=True)

    st.subheader("Calibration Tables")
    tabs = st.tabs(list(dimensions))
    for tab, (label, table) in zip(tabs, calibration_tables.items()):
        with tab:
            if table.empty:
                st.info(f"No settled calibration data yet for {label.lower()}.")
            else:
                st.dataframe(research_table_style(table), width="stretch", hide_index=True)

    st.download_button(
        "Download settled calibration CSV",
        data=settled.to_csv(index=False),
        file_name=f"ian-racing-model-settled-calibration-{selected_date.isoformat()}.csv",
        mime="text/csv",
    )

st.subheader("Pre-Race Snapshot Coverage")
snapshots = pd.DataFrame(get_model_snapshots(Settings(), limit=10000))
snapshot_summary = _snapshot_readiness(snapshots)
if snapshot_summary.empty:
    st.info("No pre-race snapshots have been captured yet. Open Today's cards or Daily Edge Card to start collecting them.")
else:
    st.dataframe(research_table_style(snapshot_summary), width="stretch", hide_index=True)
    st.caption("Snapshot coverage shows where the model is collecting enough pre-race examples for future probability calibration.")
