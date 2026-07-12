from __future__ import annotations

from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore
from ian_racing_model.edge import undervalued_edge_dataframe


def enhanced_undervalued_edge_dataframe(scores: list[RunnerScore], limit: int = 12) -> pd.DataFrame:
    df = undervalued_edge_dataframe(scores, limit=max(limit * 2, limit))
    if df.empty:
        return df
    adjusted = df.copy()
    adjusted["_edge_score"] = pd.to_numeric(adjusted["edge_score"], errors="coerce").fillna(0.0)
    adjusted["_odds"] = adjusted["odds"].map(_decimal_odds)
    adjusted["_evidence"] = adjusted["evidence"].astype(str)
    adjusted["_edge_score"] = adjusted.apply(
        lambda row: row["_edge_score"] - _longshot_drag(row["_odds"], row["_evidence"]),
        axis=1,
    )
    adjusted["edge_score"] = adjusted["_edge_score"].round(2)
    adjusted = adjusted.sort_values(["_edge_score", "confidence"], ascending=[False, False]).head(limit)
    adjusted = adjusted.drop(columns=["rank", "_edge_score", "_odds", "_evidence"], errors="ignore")
    adjusted.insert(0, "rank", range(1, len(adjusted) + 1))
    return adjusted


def negative_value_dataframe(scores: list[RunnerScore], limit: int = 12) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        odds = _decimal_odds(runner.current_odds)
        if odds is None:
            continue
        win_edge = item.win_value_edge if item.win_value_edge is not None else 0.0
        place_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
        reasons = _negative_value_reasons(item, odds, win_edge, place_edge)
        if not reasons:
            continue
        danger_score = (
            len(reasons) * 18.0
            + max(0.0, -win_edge) * 90.0
            + max(0.0, -place_edge) * 55.0
            + len(item.red_flags) * 5.0
            + max(0.0, 0.6 - item.confidence) * 20.0
        )
        rows.append(
            {
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "odds": runner.current_odds or "Unavailable",
                "model_score": item.total_score,
                "confidence": item.confidence,
                "win_edge": _format_edge(item.win_value_edge),
                "place_edge": _format_edge(item.place_value_edge),
                "avoid_reason": "; ".join(reasons[:4]),
                "red_flags": "; ".join(item.red_flags[:3]) if item.red_flags else "None",
                "danger_score": round(danger_score, 2),
                "_danger_score": danger_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["_danger_score", "model_score"], ascending=[False, False]).head(limit)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["_danger_score"])


def edge_calibration_dataframe(picks_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if picks_df.empty or dimension not in {"odds_band", "score_band", "confidence_band", "pick_type"}:
        return pd.DataFrame()
    settled = picks_df[~picks_df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["_decimal_odds"] = settled["odds"].map(_decimal_odds)
    settled["_score"] = pd.to_numeric(settled.get("score"), errors="coerce")
    settled["_confidence"] = pd.to_numeric(settled.get("confidence"), errors="coerce")
    settled["odds_band"] = settled["_decimal_odds"].map(_odds_band)
    settled["score_band"] = settled["_score"].map(_score_band)
    settled["confidence_band"] = settled["_confidence"].map(_confidence_band)
    settled["pick_type"] = settled["pick_type"].astype(str)

    rows = []
    for bucket, bucket_rows in settled.groupby(dimension, dropna=False):
        total = len(bucket_rows)
        wins = int(bucket_rows["outcome"].eq("WIN").sum())
        places = int(bucket_rows["outcome"].isin(["WIN", "PLACED"]).sum())
        just_missed = int(bucket_rows["outcome"].isin(["JUST LOST", "JUST MISSED"]).sum())
        rows.append(
            {
                dimension: bucket or "Unknown",
                "settled": total,
                "wins": wins,
                "places": places,
                "just_missed": just_missed,
                "win_rate": _ratio_text(wins, total),
                "place_rate": _ratio_text(places, total),
                "avg_odds": _mean_value(bucket_rows["_decimal_odds"]),
                "edge_read": _edge_read(total, wins, places, just_missed),
            }
        )
    return pd.DataFrame(rows).sort_values(["settled", "places", "wins"], ascending=[False, False, False])


def edge_filter_recommendations(picks_df: pd.DataFrame) -> list[str]:
    if picks_df.empty:
        return ["No settled picks yet. Let results accumulate before changing the model."]
    settled = picks_df[~picks_df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return ["No settled picks yet. Let results accumulate before changing the model."]
    settled["_decimal_odds"] = settled["odds"].map(_decimal_odds)
    settled["odds_band"] = settled["_decimal_odds"].map(_odds_band)
    recs: list[str] = []

    ew_rows = settled[settled["pick_type"].astype(str).eq("Best EW pick")]
    if not ew_rows.empty:
        by_band = {band: rows for band, rows in ew_rows.groupby("odds_band")}
        mid = by_band.get("10.0 to 19.99")
        huge = by_band.get("20.0+")
        if mid is not None and len(mid) >= 3:
            places = int(mid["outcome"].isin(["WIN", "PLACED"]).sum())
            recs.append(f"EW watch: 10.0 to 19.99 has placed {_ratio_text(places, len(mid))}; keep this as a priority value band.")
        if huge is not None and len(huge) >= 3:
            places = int(huge["outcome"].isin(["WIN", "PLACED"]).sum())
            if places / len(huge) < 0.12:
                recs.append("Tighten 20.0+ outsiders unless they have proven setup, speed or trainer/jockey evidence.")

    winner_rows = settled[settled["pick_type"].astype(str).eq("Winner pick")]
    if not winner_rows.empty:
        wins = int(winner_rows["outcome"].eq("WIN").sum())
        if wins / len(winner_rows) < 0.15:
            recs.append("Winner picks need sharper filters: require either positive win edge, high confidence, or low red-flag count.")

    if not recs:
        recs.append("No strong correction yet. Keep collecting results before changing weights aggressively.")
    return recs


def _longshot_drag(odds: float | None, evidence: str) -> float:
    if odds is None or odds < 20:
        return 0.0
    evidence_text = evidence.lower()
    strong_evidence = any(token in evidence_text for token in ("speed", "trainer", "jockey", "proven setup", "supported"))
    if strong_evidence:
        return 3.0
    if odds >= 34:
        return 18.0
    return 10.0


def _negative_value_reasons(item: RunnerScore, odds: float, win_edge: float, place_edge: float) -> list[str]:
    reasons = []
    if odds < 3.0 and win_edge < 0.02:
        reasons.append("short price without a positive win edge")
    if win_edge < -0.05 and place_edge < 0.02:
        reasons.append("market shorter than model fair price")
    if item.total_score >= 60 and item.red_flags:
        reasons.append("good headline score but red flags remain")
    if item.confidence < 0.5 and odds < 6.0:
        reasons.append("low confidence at a short or mid price")
    market = _market_signal(item.runner.source_payload, item.runner.current_odds)
    if market.startswith("Drifting") and odds < 8.0:
        reasons.append("drifting in the market at a short or mid price")
    return reasons


def _market_signal(payload: dict[str, Any], current_odds: str | None) -> str:
    current = _decimal_odds(current_odds)
    opening = _opening_odds(payload)
    if current is None or opening is None or opening <= 1:
        return "No market move data"
    move = (current - opening) / opening
    if move <= -0.12:
        return f"Supported: {opening:.2f} to {current:.2f}"
    if move >= 0.2:
        return f"Drifting: {opening:.2f} to {current:.2f}"
    return f"Stable: {opening:.2f} to {current:.2f}"


def _opening_odds(payload: dict[str, Any]) -> float | None:
    sources = [payload]
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        sources.append(source_runner)
    for source in sources:
        for key in ("opening_odds", "open_odds", "early_odds", "first_odds", "opening_price", "open_price"):
            odds = _decimal_odds(str(source.get(key))) if source.get(key) not in (None, "") else None
            if odds is not None:
                return odds
        odds_rows = source.get("odds") or source.get("odds_history") or source.get("price_history")
        if isinstance(odds_rows, list):
            for row in odds_rows:
                if isinstance(row, dict):
                    odds = _decimal_odds(str(row.get("opening") or row.get("open") or row.get("first") or ""))
                    if odds is not None:
                        return odds
    return None


def _decimal_odds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return float(num) / float(den) + 1.0
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
    if score < 80:
        return "70 to 79.9"
    return "80+"


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


def _ratio_text(successes: int, total: int) -> str:
    if total == 0:
        return "No settled picks"
    return f"{(successes / total) * 100:.1f}% ({successes}/{total})"


def _mean_value(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _edge_read(total: int, wins: int, places: int, just_missed: int) -> str:
    if total < 5:
        return "Small sample"
    place_rate = places / total
    win_rate = wins / total
    if place_rate >= 0.3 or win_rate >= 0.18:
        return "Positive pocket"
    if just_missed / total >= 0.2:
        return "Near miss pocket"
    if place_rate < 0.12:
        return "Weak pocket"
    return "Monitor"


def _format_edge(edge: float | None) -> str:
    if edge is None:
        return "Unavailable"
    return f"{edge * 100:+.1f} pts"
