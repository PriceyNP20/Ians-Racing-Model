from __future__ import annotations

from datetime import date
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
            "fair_odds": item.fair_odds_placeholder,
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
        value_edge = _value_edge(item.total_score, odds)
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
                "value_edge_pct": _format_edge(value_edge),
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


def picks_tracker_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for (_, _, _, _), race_scores in _race_groups(scores).items():
        sorted_scores = sorted(race_scores, key=lambda item: item.total_score, reverse=True)
        winner_pick = sorted_scores[0]
        ew_pick = _best_each_way_pick(sorted_scores, winner_pick)
        rows.append(_pick_row(winner_pick, "Winner pick"))
        if ew_pick is not None:
            rows.append(_pick_row(ew_pick, "Best EW pick"))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        by=["course", "off_time", "race", "pick_type"],
        ascending=[True, True, True, False],
    )


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
        return {
            "winner_win_rate": "No settled picks",
            "ew_place_rate": "No settled picks",
        }

    settled = df[~df["outcome"].eq("Awaiting result")]
    winner_rows = settled[settled["pick_type"].eq("Winner pick")]
    ew_rows = settled[settled["pick_type"].eq("Best EW pick")]
    return {
        "winner_win_rate": _ratio_text(winner_rows["outcome"].eq("WIN").sum(), len(winner_rows)),
        "ew_place_rate": _ratio_text(ew_rows["outcome"].isin(["WIN", "PLACED"]).sum(), len(ew_rows)),
    }


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


def _best_each_way_pick(
    sorted_scores: list[RunnerScore], winner_pick: RunnerScore
) -> RunnerScore | None:
    candidates = [score for score in sorted_scores if score.runner.horse != winner_pick.runner.horse]
    if not candidates:
        candidates = sorted_scores
    ew_candidates = [
        score
        for score in candidates
        if score.recommendation in {"EACH_WAY", "PLACE", "WATCH"}
        and (_decimal_odds(score.runner.current_odds) or 0) >= 5
    ]
    pool = ew_candidates or candidates
    return sorted(
        pool,
        key=lambda score: (
            _value_edge(score.total_score, _decimal_odds(score.runner.current_odds)) or -1,
            score.total_score,
            score.confidence,
        ),
        reverse=True,
    )[0]


def _pick_row(item: RunnerScore, pick_type: str) -> dict[str, Any]:
    runner = item.runner
    position = _finish_position(runner.source_payload)
    place_cutoff = _place_cutoff(runner.field_size)
    outcome = _pick_outcome(pick_type, position, place_cutoff)
    return {
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "pick_type": pick_type,
        "horse": runner.horse,
        "score": item.total_score,
        "confidence": item.confidence,
        "odds": runner.current_odds or "Unavailable",
        "recommendation": item.recommendation,
        "result": position if position is not None else "Awaiting result",
        "outcome": outcome,
        "place_cutoff": place_cutoff,
        "warnings": "; ".join((item.red_flags or item.data_quality_warnings)[:3]),
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
