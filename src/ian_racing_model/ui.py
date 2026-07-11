from __future__ import annotations

from datetime import date

import pandas as pd

from ian_racing_model.domain import RunnerScore


RECOMMENDATION_PRIORITY = {
    "WIN": 5,
    "EACH_WAY": 4,
    "PLACE": 3,
    "WATCH": 2,
    "PASS": 1,
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
