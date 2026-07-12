from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore


RECOMMENDATION_PRIORITY = {
    "WIN": 5,
    "EACH_WAY": 4,
    "PLACE": 3,
    "WATCH": 2,
    "PASS": 1,
}

RESULT_POSITION_KEYS = (
    "result_position",
    "finish_position",
    "finishing_position",
    "position",
    "pos",
    "place",
)

REFRESH_SOURCE_LABELS = {
    "racecard": "Racecards",
    "results": "Results",
    "horse_history": "Horse history",
}


def scores_to_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        row = {
            "course": runner.course,
            "off_time": runner.off_time,
            "race": runner.race_name,
            "horse": runner.horse,
            "total_score": item.total_score,
            "confidence": item.confidence,
            "odds": runner.current_odds,
            "win_probability": _format_probability(item.win_probability),
            "place_probability": _format_probability(item.place_probability),
            "fair_win_odds": _format_decimal(item.fair_win_odds),
            "fair_place_odds": _format_decimal(item.fair_place_odds),
            "win_value_edge": _format_edge(item.win_value_edge),
            "place_value_edge": _format_edge(item.place_value_edge),
            "recommendation": item.recommendation,
            "warnings": "; ".join(item.data_quality_warnings),
        }
        for component in item.components:
            row[component.name] = component.score
        rows.append(row)
    return pd.DataFrame(rows)


def screener_dataframe(scores: list[RunnerScore], limit: int = 8) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        odds = _decimal_odds(runner.current_odds)
        value_edge = item.win_value_edge
        place_edge = item.place_value_edge
        label = _screener_label(item, odds, value_edge)
        warnings = item.red_flags or item.data_quality_warnings
        rows.append(
            {
                "screen": label,
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "score": item.total_score,
                "confidence": item.confidence,
                "odds": runner.current_odds or "Unavailable",
                "fair_win_odds": _format_decimal(item.fair_win_odds),
                "fair_place_odds": _format_decimal(item.fair_place_odds),
                "win_probability": _format_probability(item.win_probability),
                "place_probability": _format_probability(item.place_probability),
                "value_edge_pct": _format_edge(value_edge),
                "place_value_edge_pct": _format_edge(place_edge),
                "recommendation": item.recommendation,
                "warnings": "; ".join(warnings[:3]),
                "_priority": RECOMMENDATION_PRIORITY.get(item.recommendation, 0),
                "_edge_sort": value_edge if value_edge is not None else -1.0,
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["_priority", "_edge_sort", "score", "confidence"],
        ascending=[False, False, False, False],
    ).head(limit)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["_priority", "_edge_sort"])


def value_screener_dataframe(scores: list[RunnerScore], limit: int = 10) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        if item.win_value_edge is None and item.place_value_edge is None:
            continue
        best_edge = max(item.win_value_edge or -1.0, item.place_value_edge or -1.0)
        if best_edge <= 0:
            continue
        signal = "Best EW value" if (item.place_value_edge or -1) >= (item.win_value_edge or -1) else "Best win value"
        rows.append(
            {
                "signal": signal,
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "odds": runner.current_odds or "Unavailable",
                "fair_win_odds": _format_decimal(item.fair_win_odds),
                "fair_place_odds": _format_decimal(item.fair_place_odds),
                "win_probability": _format_probability(item.win_probability),
                "place_probability": _format_probability(item.place_probability),
                "win_value_edge": _format_edge(item.win_value_edge),
                "place_value_edge": _format_edge(item.place_value_edge),
                "value_confidence": _value_confidence_label(item, best_edge),
                "score": item.total_score,
                "confidence": item.confidence,
                "warning": "; ".join((item.red_flags or item.data_quality_warnings)[:2]),
                "_edge_sort": best_edge,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        by=["_edge_sort", "score", "confidence"],
        ascending=[False, False, False],
    ).head(limit)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["_edge_sort"])


def model_signal_dataframe(scores: list[RunnerScore], limit: int = 20) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        history = _history_candidates(runner.source_payload)
        setup_hits = _setup_hits(runner, history)
        market_signal = _market_signal(runner.source_payload, runner.current_odds)
        if not setup_hits and market_signal == "No market move data":
            continue
        rows.append(
            {
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "setup_evidence": ", ".join(setup_hits) if setup_hits else "No proven setup in imported history",
                "market_signal": market_signal,
                "score": item.total_score,
                "confidence": item.confidence,
                "red_flags": "; ".join(item.red_flags) if item.red_flags else "None",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["confidence", "score"], ascending=[False, False]).head(limit)


def race_selection_screener_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for (_, course, off_time, race), race_scores in _race_groups(scores).items():
        sorted_scores = sorted(race_scores, key=_winner_selection_score, reverse=True)
        winner_pick = sorted_scores[0]
        ew_pick = _best_each_way_pick(race_scores, winner_pick)
        rows.append(_selection_screener_row(winner_pick, "Winner", _winner_selection_score(winner_pick)))
        if ew_pick is not None:
            rows.append(_selection_screener_row(ew_pick, "Best EW value", _each_way_selection_score(ew_pick)))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["course", "off_time", "race", "pick"], ascending=[True, True, True, False])


def refresh_health_dataframe(statuses: list[dict]) -> pd.DataFrame:
    if not statuses:
        return pd.DataFrame()
    rows = []
    seen: set[tuple[str, str]] = set()
    for item in statuses:
        source = str(item.get("source") or "unknown")
        course = str(item.get("course") or "All UK courses")
        key = (source, course)
        if key in seen:
            continue
        seen.add(key)
        status = str(item.get("status") or "unknown")
        rows.append(
            {
                "data": REFRESH_SOURCE_LABELS.get(source, source.replace("_", " ").title()),
                "course": course,
                "status": status.title(),
                "last refreshed": _friendly_refresh_time(item.get("refreshed_at")),
                "message": item.get("message") or "",
            }
        )
    return pd.DataFrame(rows)


def refresh_health_summary(statuses: list[dict], provider: str, warning: str | None = None) -> dict[str, str]:
    if warning or provider == "mock":
        return {"label": "Using sample data", "detail": "Live API data is not currently powering this view.", "state": "warning"}
    if not statuses:
        return {"label": "Waiting for first refresh", "detail": "Open a card to trigger the first API refresh record.", "state": "info"}
    latest = statuses[0]
    if str(latest.get("status")).lower() == "error":
        return {"label": "API needs attention", "detail": latest.get("message") or "The latest refresh recorded an error.", "state": "error"}
    return {"label": "Live API active", "detail": f"Latest {latest.get('source')} refresh: {_friendly_refresh_time(latest.get('refreshed_at'))}.", "state": "success"}


def picks_tracker_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for (_, _, _, _), race_scores in _race_groups(scores).items():
        sorted_scores = sorted(race_scores, key=_winner_selection_score, reverse=True)
        winner_pick = sorted_scores[0]
        ew_pick = _best_each_way_pick(race_scores, winner_pick)
        rows.append(_pick_row(winner_pick, "Winner pick", _winner_selection_score(winner_pick)))
        if ew_pick is not None:
            rows.append(_pick_row(ew_pick, "Best EW pick", _each_way_selection_score(ew_pick)))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by=["course", "off_time", "race", "pick_type"], ascending=[True, True, True, False])


def picks_tracker_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        status = str(row.get("outcome", ""))
        if status in {"WIN", "PLACED"}:
            colour = "background-color: #dcfce7; color: #14532d;"
        elif status in {"LOSE"}:
            colour = "background-color: #fee2e2; color: #7f1d1d;"
        elif status in {"JUST LOST", "JUST MISSED"}:
            colour = "background-color: #dbeafe; color: #1e3a8a;"
        else:
            colour = ""
        return [colour] * len(row)

    return df.style.apply(row_style, axis=1)


def picks_tracker_summary(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {"winner_win_rate": "No settled picks", "ew_place_rate": "No settled picks"}

    settled = df[~df["outcome"].eq("Awaiting result")]
    pick_type = settled["pick_type"].astype(str).str.strip()
    winner_rows = settled[pick_type.eq("Winner pick")]
    ew_rows = settled[pick_type.eq("Best EW pick")]
    return {
        "winner_win_rate": _ratio_text(winner_rows["outcome"].eq("WIN").sum(), len(winner_rows)),
        "ew_place_rate": _ratio_text(ew_rows["outcome"].isin(["WIN", "PLACED"]).sum(), len(ew_rows)),
    }


def picks_tracker_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    settled["pick_type"] = settled["pick_type"].astype(str).str.strip()
    for pick_type in ("Winner pick", "Best EW pick"):
        pick_rows = settled[settled["pick_type"].eq(pick_type)]
        total = len(pick_rows)
        positions = pd.to_numeric(pick_rows["result"].map(_parse_position), errors="coerce")
        place_cutoffs = pd.to_numeric(pick_rows["place_cutoff"], errors="coerce").fillna(0)
        place_hits = int(((positions.notna()) & (positions <= place_cutoffs)).sum())
        wins = pick_rows["outcome"].eq("WIN").sum()
        rows.append(
            {
                "pick_type": pick_type,
                "settled": total,
                "wins": int(wins),
                "places": int(place_hits),
                "just_missed": int(pick_rows["outcome"].isin(["JUST LOST", "JUST MISSED"]).sum()),
                "losses": int(pick_rows["outcome"].eq("LOSE").sum()),
                "win_rate": _ratio_text(int(wins), total),
                "place_rate": _ratio_text(int(place_hits), total),
            }
        )
    return pd.DataFrame(rows)


def performance_by_odds_band(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["_decimal_odds"] = settled["odds"].map(_decimal_odds)
    settled["odds_band"] = settled["_decimal_odds"].map(_odds_band)
    rows = []
    for (pick_type, odds_band), band_rows in settled.groupby(["pick_type", "odds_band"]):
        total = len(band_rows)
        wins = int(band_rows["outcome"].eq("WIN").sum())
        places = int(band_rows["outcome"].isin(["WIN", "PLACED"]).sum())
        rows.append({"pick_type": pick_type, "odds_band": odds_band, "settled": total, "wins": wins, "places": places, "win_rate": _ratio_text(wins, total), "place_rate": _ratio_text(places, total)})
    return pd.DataFrame(rows).sort_values(["pick_type", "odds_band"])


def performance_lab_dataframe(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if df.empty or dimension not in df.columns:
        return pd.DataFrame()

    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()

    settled["_dimension"] = settled[dimension].fillna("Unknown").astype(str).str.strip()
    settled.loc[settled["_dimension"].eq(""), "_dimension"] = "Unknown"
    settled["_score"] = pd.to_numeric(settled.get("score"), errors="coerce")
    settled["_selection_score"] = pd.to_numeric(settled.get("selection_score"), errors="coerce")
    settled["_confidence"] = pd.to_numeric(settled.get("confidence"), errors="coerce")
    settled["_decimal_odds"] = settled["odds"].map(_decimal_odds)

    rows = []
    for (pick_type, bucket), bucket_rows in settled.groupby(["pick_type", "_dimension"], dropna=False):
        total = len(bucket_rows)
        wins = int(bucket_rows["outcome"].eq("WIN").sum())
        places = int(bucket_rows["outcome"].isin(["WIN", "PLACED"]).sum())
        just_missed = int(bucket_rows["outcome"].isin(["JUST LOST", "JUST MISSED"]).sum())
        rows.append(
            {
                "pick_type": pick_type,
                dimension: bucket,
                "settled": total,
                "wins": wins,
                "places": places,
                "just_missed": just_missed,
                "win_rate": _ratio_text(wins, total),
                "place_rate": _ratio_text(places, total),
                "avg_score": _mean_value(bucket_rows["_score"]),
                "avg_selection_score": _mean_value(bucket_rows["_selection_score"]),
                "avg_confidence": _mean_value(bucket_rows["_confidence"]),
                "avg_odds": _mean_value(bucket_rows["_decimal_odds"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["pick_type", "settled", "places", "wins"], ascending=[True, False, False, False])


def outsider_last_time_dataframe(scores: list[RunnerScore], min_decimal_odds: float = 15.0) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        signal = _last_time_outsider_signal(runner.source_payload, min_decimal_odds)
        if signal is None:
            continue
        rows.append(
            {
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "horse": runner.horse,
                "last_result": signal["last_result"],
                "last_odds": signal["last_odds"],
                "signal": signal["signal"],
                "today_odds": runner.current_odds or "Unavailable",
                "fair_win_odds": _format_decimal(item.fair_win_odds),
                "win_value_edge": _format_edge(item.win_value_edge),
                "score": item.total_score,
                "recommendation": item.recommendation,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["score", "course", "off_time"], ascending=[False, True, True])


def model_upgrade_notes() -> list[str]:
    return [
        "Calibrate fair odds from actual results, not the current placeholder. Convert model scores into a probability curve by race type, field size and price band.",
        "Separate win probability from place probability. EW picks should be judged on place value and place terms, not the same score used for win picks.",
        "Use Pro horse history for course, distance, going and class performance. Replace neutral baselines with proven or unproven evidence.",
        "Add market movement: compare opening odds, current odds and SP. Strong runners drifting late and weak runners shortening should be treated differently.",
        "Use trainer and jockey analysis endpoints for recent strike rate, course record, A/E and 1 unit profit/loss, then score suitability rather than name presence.",
        "Model outsider resilience: horses that recently won or placed at big odds should get a controlled positive for hidden ability, but only if today's setup is similar.",
    ]


def available_courses(scores: list[RunnerScore]) -> list[str]:
    return sorted({score.runner.course for score in scores})


def available_races(scores: list[RunnerScore]) -> list[str]:
    return sorted({f"{score.runner.off_time} - {score.runner.race_name}" for score in scores})


def default_date() -> date:
    return date(2026, 7, 11)


def _decimal_odds(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
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


def _value_edge(score: float, odds: float | None) -> float | None:
    if odds is None or odds <= 1:
        return None
    model_probability = score / 100.0
    market_probability = 1.0 / odds
    return model_probability - market_probability


def _format_edge(edge: float | None) -> str:
    if edge is None:
        return "Needs odds"
    return f"{edge * 100:+.1f} pts"


def _format_probability(probability: float | None) -> str:
    if probability is None:
        return "Unavailable"
    return f"{probability * 100:.1f}%"


def _format_decimal(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2f}"


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


def _field_size_band(field_size: int | None) -> str:
    if field_size is None:
        return "Unknown"
    if field_size < 5:
        return "1 to 4"
    if field_size < 8:
        return "5 to 7"
    if field_size < 12:
        return "8 to 11"
    if field_size < 16:
        return "12 to 15"
    return "16+"


def _mean_value(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _screener_label(item: RunnerScore, odds: float | None, edge: float | None) -> str:
    has_positive_edge = edge is None or edge > 0
    if item.recommendation == "WIN" and has_positive_edge:
        return "Top win"
    if item.recommendation in {"EACH_WAY", "PLACE"} and odds is not None and odds >= 5 and has_positive_edge:
        return "EW value"
    if edge is not None and edge >= 0.08 and item.total_score >= 55:
        return "Undervalued"
    if item.recommendation == "WATCH":
        return "Watchlist"
    return item.recommendation.title()


def _race_groups(scores: list[RunnerScore]) -> dict[tuple[date, str, str, str], list[RunnerScore]]:
    groups: dict[tuple[date, str, str, str], list[RunnerScore]] = {}
    for score in scores:
        runner = score.runner
        key = (runner.meeting_date, runner.course, runner.off_time, runner.race_name)
        groups.setdefault(key, []).append(score)
    return groups


def _best_each_way_pick(race_scores: list[RunnerScore], winner_pick: RunnerScore) -> RunnerScore | None:
    candidates = [score for score in race_scores if score.runner.horse != winner_pick.runner.horse]
    if not candidates:
        candidates = race_scores
    ew_candidates = [score for score in candidates if _is_each_way_candidate(score)]
    pool = ew_candidates or candidates
    return sorted(pool, key=_each_way_selection_score, reverse=True)[0]


def _pick_row(item: RunnerScore, pick_type: str, selection_score: float) -> dict[str, Any]:
    runner = item.runner
    position = _finish_position(runner.source_payload)
    place_cutoff = _place_cutoff(runner.field_size)
    outcome = _pick_outcome(pick_type, position, place_cutoff)
    return {
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "race_type": runner.race_type or "Unknown",
        "race_class": runner.race_class or "Unknown",
        "surface": runner.surface or "Unknown",
        "going": runner.going or "Unknown",
        "field_size": runner.field_size or 0,
        "field_size_band": _field_size_band(runner.field_size),
        "pick_type": pick_type,
        "horse": runner.horse,
        "selection_score": round(selection_score, 2),
        "selection_reason": _selection_reason(item, pick_type),
        "score": item.total_score,
        "confidence": item.confidence,
        "odds": runner.current_odds or "Unavailable",
        "recommendation": item.recommendation,
        "win_probability": _format_probability(item.win_probability),
        "place_probability": _format_probability(item.place_probability),
        "fair_win_odds": _format_decimal(item.fair_win_odds),
        "fair_place_odds": _format_decimal(item.fair_place_odds),
        "win_value_edge": _format_edge(item.win_value_edge),
        "place_value_edge": _format_edge(item.place_value_edge),
        "result": str(position) if position is not None else "Awaiting result",
        "outcome": outcome,
        "place_cutoff": place_cutoff,
        "warnings": "; ".join((item.red_flags or item.data_quality_warnings)[:3]),
    }


def _winner_selection_score(item: RunnerScore) -> float:
    win_probability = item.win_probability or 0.0
    win_edge = item.win_value_edge if item.win_value_edge is not None else 0.0
    red_flag_drag = min(0.2, len(item.red_flags) * 0.035)
    return win_probability * 100.0 + item.total_score * 0.55 + item.confidence * 12.0 + max(-0.08, min(0.16, win_edge)) * 100.0 - red_flag_drag * 100.0


def _each_way_selection_score(item: RunnerScore) -> float:
    place_probability = item.place_probability or 0.0
    place_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
    win_edge = item.win_value_edge if item.win_value_edge is not None else 0.0
    odds = _decimal_odds(item.runner.current_odds)
    price_bonus = _each_way_price_bonus(odds)
    red_flag_drag = min(0.18, len(item.red_flags) * 0.03)
    value_bonus = max(-0.08, min(0.26, place_edge)) * 165.0
    win_bias_penalty = max(0.0, win_edge - place_edge) * 35.0
    return place_probability * 100.0 + item.total_score * 0.2 + item.confidence * 18.0 + value_bonus + price_bonus - win_bias_penalty - red_flag_drag * 100.0


def _is_each_way_candidate(item: RunnerScore) -> bool:
    odds = _decimal_odds(item.runner.current_odds)
    if odds is not None and odds < 5.0:
        return False
    if item.place_probability >= 0.42 and (item.place_value_edge is None or item.place_value_edge >= -0.02):
        return True
    if item.place_value_edge is not None and item.place_value_edge >= 0.03:
        return True
    return item.recommendation in {"EACH_WAY", "PLACE", "WATCH"}


def _each_way_price_bonus(odds: float | None) -> float:
    if odds is None:
        return 0.0
    if 5.0 <= odds <= 14.0:
        return 8.0
    if 14.0 < odds <= 26.0:
        return 5.0
    if odds > 26.0:
        return 1.5
    return -4.0


def _selection_reason(item: RunnerScore, pick_type: str) -> str:
    if pick_type == "Winner pick":
        parts = [f"win { _format_probability(item.win_probability) }", f"edge { _format_edge(item.win_value_edge) }"]
    else:
        parts = [f"place { _format_probability(item.place_probability) }", f"place edge { _format_edge(item.place_value_edge) }"]
    if item.red_flags:
        parts.append(f"{len(item.red_flags)} red flag(s)")
    return "; ".join(parts)


def _selection_screener_row(item: RunnerScore, pick: str, selection_score: float) -> dict[str, Any]:
    runner = item.runner
    return {
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "pick": pick,
        "horse": runner.horse,
        "selection_score": round(selection_score, 2),
        "model_score": item.total_score,
        "confidence": item.confidence,
        "odds": runner.current_odds or "Unavailable",
        "fair_win_odds": _format_decimal(item.fair_win_odds),
        "fair_place_odds": _format_decimal(item.fair_place_odds),
        "win_edge": _format_edge(item.win_value_edge),
        "place_edge": _format_edge(item.place_value_edge),
        "reason": _selection_reason(item, "Winner pick" if pick == "Winner" else "Best EW pick"),
    }


def _pick_outcome(pick_type: str, position: int | None, place_cutoff: int) -> str:
    if position is None:
        return "Awaiting result"
    if pick_type == "Winner pick":
        if position == 1:
            return "WIN"
        if position == 2:
            return "JUST LOST"
        return "LOSE"
    if position <= place_cutoff:
        return "PLACED"
    if position == place_cutoff + 1:
        return "JUST MISSED"
    return "LOSE"


def _place_cutoff(field_size: int | None) -> int:
    if field_size is None:
        return 3
    if field_size >= 16:
        return 4
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return 1


def _finish_position(payload: dict[str, Any]) -> int | None:
    for key in RESULT_POSITION_KEYS:
        parsed = _parse_position(payload.get(key))
        if parsed is not None:
            return parsed
    for value in payload.values():
        if isinstance(value, dict):
            parsed = _finish_position(value)
            if parsed is not None:
                return parsed
    return None


def _parse_position(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"nr", "non-runner", "pu", "f", "ur", "bd"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _ratio_text(successes: int, total: int) -> str:
    if total == 0:
        return "No settled picks"
    return f"{(successes / total) * 100:.1f}% ({successes}/{total})"


def _last_time_outsider_signal(payload: dict[str, Any], min_decimal_odds: float) -> dict[str, str] | None:
    for candidate in _history_candidates(payload):
        if not isinstance(candidate, dict):
            continue
        position = _parse_position(candidate.get("position") or candidate.get("pos"))
        odds = _history_decimal_odds(candidate)
        if position is None or odds is None:
            continue
        if position <= 3 and odds >= min_decimal_odds:
            return {"last_result": f"{position}", "last_odds": _format_decimal_odds(odds), "signal": "Won/placed at big odds last time"}
    return None


def _history_candidates(payload: dict[str, Any]) -> list[Any]:
    history_keys = ("last_result", "last_run_result", "previous_result", "previous_run", "latest_result", "history", "horse_history", "results", "past_results", "horse_results")
    candidates: list[Any] = []
    for key in history_keys:
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value[:3])
        elif value:
            candidates.append(value)
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        for key in history_keys:
            value = source_runner.get(key)
            if isinstance(value, list):
                candidates.extend(value[:3])
            elif value:
                candidates.append(value)
    return candidates


def _history_decimal_odds(payload: dict[str, Any]) -> float | None:
    for key in ("sp_dec", "bsp", "decimal", "odds_decimal"):
        odds = _decimal_odds(str(payload.get(key))) if payload.get(key) not in (None, "") else None
        if odds is not None:
            return odds
    for key in ("sp", "odds", "starting_price"):
        odds = _decimal_odds(str(payload.get(key))) if payload.get(key) not in (None, "") else None
        if odds is not None:
            return odds
    return None


def _format_decimal_odds(odds: float) -> str:
    return f"{odds:.1f}"


def _value_confidence_label(item: RunnerScore, best_edge: float) -> str:
    if item.confidence >= 0.62 and best_edge >= 0.08:
        return "Strong value"
    if item.confidence >= 0.52 and best_edge > 0:
        return "Speculative value"
    return "Weak data"


def _setup_hits(runner, history: list[Any]) -> list[str]:
    hits = []
    if any(_history_match(item, "course", runner.course) for item in history):
        hits.append("course")
    if any(_history_match(item, "going", runner.going) or _history_match(item, "ground", runner.going) for item in history):
        hits.append("going")
    if any(_history_match(item, "surface", runner.surface) for item in history):
        hits.append("surface")
    if any(_history_match(item, "distance", runner.distance) or _history_match(item, "dist", runner.distance) for item in history):
        hits.append("distance")
    return hits


def _history_match(item: Any, key: str, expected: str | None) -> bool:
    if not isinstance(item, dict) or not expected:
        return False
    position = _parse_position(item.get("position") or item.get("pos"))
    if position is None or position > 3:
        return False
    actual = item.get(key)
    if actual in (None, ""):
        return False
    actual_text = " ".join(str(actual).lower().strip().split())
    expected_text = " ".join(str(expected).lower().strip().split())
    return bool(actual_text and expected_text and (actual_text in expected_text or expected_text in actual_text))


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
                if not isinstance(row, dict):
                    continue
                odds = _decimal_odds(str(row.get("opening") or row.get("open") or row.get("first") or row.get("fractional_open") or row.get("decimal_open")))
                if odds is not None:
                    return odds
    return None


def _friendly_refresh_time(value: Any) -> str:
    if value in (None, ""):
        return "Unknown"
    text = str(value)
    try:
        refreshed_at = datetime.fromisoformat(text)
    except ValueError:
        return text
    return refreshed_at.strftime("%Y-%m-%d %H:%M")
