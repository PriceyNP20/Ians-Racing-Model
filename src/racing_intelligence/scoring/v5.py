from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from ian_racing_model.analysis_engines import (
    DATA_MISSING,
    DATA_OK,
    DATA_PARTIAL,
    EngineSignal,
    course_conditions_signal,
    pace_race_shape_signal,
    trainer_intent_signal,
)
from ian_racing_model.domain import Runner, RunnerScore


V5_ENGINE_WEIGHTS = {
    "ability": 20,
    "suitability": 20,
    "race_shape": 15,
    "trainer_intent": 10,
    "current_wellbeing": 10,
    "improvement_potential": 10,
    "market_value": 10,
    "historical_performance": 5,
}

V5_WIN_WEIGHTS = {
    "ability": 25,
    "suitability": 15,
    "race_shape": 15,
    "trainer_intent": 12,
    "current_wellbeing": 12,
    "improvement_potential": 10,
    "market_value": 8,
    "historical_performance": 3,
}

V5_PLACE_WEIGHTS = {
    "ability": 15,
    "suitability": 25,
    "race_shape": 18,
    "trainer_intent": 8,
    "current_wellbeing": 12,
    "improvement_potential": 10,
    "market_value": 7,
    "historical_performance": 5,
}


@dataclass(frozen=True)
class V5Analysis:
    score: RunnerScore
    engines: dict[str, EngineSignal]
    win_index: float
    place_index: float
    confidence: float
    data_quality: str
    recommendation: str


def validate_v5_weights() -> None:
    for name, weights in (
        ("engine", V5_ENGINE_WEIGHTS),
        ("win", V5_WIN_WEIGHTS),
        ("place", V5_PLACE_WEIGHTS),
    ):
        total = sum(weights.values())
        if total != 100:
            raise ValueError(f"V5 {name} weights must total 100, got {total}.")


def v5_analysis(score: RunnerScore, race_scores: list[RunnerScore] | None = None) -> V5Analysis:
    validate_v5_weights()
    engines = _engine_map(score, race_scores or [score])
    win_index = _weighted_index(engines, V5_WIN_WEIGHTS)
    place_index = _weighted_index(engines, V5_PLACE_WEIGHTS)
    confidence = _weighted_confidence(engines, V5_ENGINE_WEIGHTS)
    quality = _combined_quality(engines)
    return V5Analysis(
        score=score,
        engines=engines,
        win_index=win_index,
        place_index=place_index,
        confidence=confidence,
        data_quality=quality,
        recommendation=_recommendation(win_index, place_index, confidence, score),
    )


def v5_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_scores in _race_groups(scores).values():
        for score in race_scores:
            analysis = v5_analysis(score, race_scores)
            runner = score.runner
            rows.append(
                {
                    "course": runner.course,
                    "off_time": runner.off_time,
                    "race": runner.race_name,
                    "horse": runner.horse,
                    "odds": runner.current_odds or "Unavailable",
                    "field_size": runner.field_size,
                    "v5_win_index": analysis.win_index,
                    "v5_place_index": analysis.place_index,
                    "v5_confidence": analysis.confidence,
                    "v5_recommendation": analysis.recommendation,
                    "v5_data_quality": analysis.data_quality,
                    "ability": analysis.engines["ability"].score,
                    "suitability": analysis.engines["suitability"].score,
                    "race_shape": analysis.engines["race_shape"].score,
                    "trainer_intent": analysis.engines["trainer_intent"].score,
                    "current_wellbeing": analysis.engines["current_wellbeing"].score,
                    "improvement_potential": analysis.engines["improvement_potential"].score,
                    "market_value": analysis.engines["market_value"].score,
                    "historical_performance": analysis.engines["historical_performance"].score,
                    "v5_explanation": _headline_explanation(analysis),
                    "_v5_sort": analysis.place_index + max(0.0, analysis.confidence - 0.5) * 10,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["_v5_sort", "v5_win_index"], ascending=[False, False])
    df.insert(0, "v5_rank", range(1, len(df) + 1))
    return df.drop(columns=["_v5_sort"])


def _engine_map(score: RunnerScore, race_scores: list[RunnerScore]) -> dict[str, EngineSignal]:
    runner = score.runner
    return {
        "ability": _ability_signal(score),
        "suitability": _suitability_signal(runner),
        "race_shape": pace_race_shape_signal(runner),
        "trainer_intent": trainer_intent_signal(runner),
        "current_wellbeing": _current_wellbeing_signal(runner),
        "improvement_potential": _improvement_potential_signal(runner),
        "market_value": _market_value_signal(score),
        "historical_performance": _historical_performance_signal(runner),
    }


def _ability_signal(score: RunnerScore) -> EngineSignal:
    runner = score.runner
    payload = _payload(runner)
    imported = _first_metric(
        payload,
        (
            "timeform_rating",
            "rpr",
            "official_rating",
            "speed_figure",
            "topspeed",
            "beyer",
        ),
    )
    if imported is not None:
        return EngineSignal("ability", _rating_to_score(imported), 0.72, DATA_PARTIAL, "Ability scored from imported ratings/figures.")
    if runner.official_rating is not None:
        return EngineSignal("ability", _rating_to_score(runner.official_rating), 0.62, DATA_PARTIAL, "Ability inferred from official rating.")
    return EngineSignal("ability", score.total_score, max(0.35, score.confidence * 0.75), DATA_PARTIAL, "Ability proxied from current model score.")


def _suitability_signal(runner: Runner) -> EngineSignal:
    conditions = course_conditions_signal(runner)
    if conditions.data_quality in {DATA_OK, DATA_PARTIAL}:
        return EngineSignal("suitability", conditions.score, conditions.confidence, conditions.data_quality, conditions.explanation)
    return EngineSignal("suitability", 45, 0.35, DATA_MISSING, "Suitability evidence unavailable.")


def _current_wellbeing_signal(runner: Runner) -> EngineSignal:
    history = _history(runner)
    positions = [_position(item) for item in history[:4]]
    positions = [position for position in positions if position is not None]
    if positions:
        avg = sum(positions) / len(positions)
        placed = sum(1 for position in positions if position <= 3)
        score = max(25.0, min(88.0, 82.0 - avg * 6.0 + placed * 4.0))
        return EngineSignal("current_wellbeing", score, 0.7, DATA_OK, "Current wellbeing scored from recent finishing positions.")
    if runner.recent_form:
        digits = [int(ch) for ch in runner.recent_form if ch.isdigit()]
        if digits:
            avg = sum(digits[:4]) / min(4, len(digits))
            return EngineSignal("current_wellbeing", max(30.0, min(82.0, 84.0 - avg * 7.0)), 0.58, DATA_PARTIAL, "Current wellbeing inferred from form string.")
    return EngineSignal("current_wellbeing", 45, 0.35, DATA_MISSING, "Recent wellbeing evidence unavailable.")


def _improvement_potential_signal(runner: Runner) -> EngineSignal:
    payload = _payload(runner)
    flags: list[str] = []
    starts = _first_metric(payload, ("career_starts", "runs", "total_runs"))
    if starts is not None and starts <= 4:
        flags.append("lightly raced")
    if _truthy(payload, ("first_handicap", "handicap_debut")):
        flags.append("first handicap")
    if _truthy(payload, ("first_time_gelding", "gelded_first_time", "first_run_after_gelding")):
        flags.append("first run after gelding")
    if _truthy(payload, ("class_drop", "down_in_class")):
        flags.append("down in class")
    if _truthy(payload, ("step_up_trip", "up_in_trip", "trip_change")):
        flags.append("trip change")
    if _truthy(payload, ("seasonal_reappearance", "first_run_season")):
        flags.append("seasonal return")
    race_text = " ".join(str(value or "").lower() for value in (runner.race_name, runner.race_type))
    if "maiden" in race_text or "novice" in race_text:
        flags.append("development race type")
    if not flags:
        return EngineSignal("improvement_potential", 50, 0.42, DATA_PARTIAL, "No clear imported improvement trigger.")
    score = min(82.0, 54.0 + len(set(flags)) * 7.0)
    return EngineSignal("improvement_potential", score, 0.62, DATA_PARTIAL, "Improvement triggers: " + ", ".join(sorted(set(flags))) + ".")


def _market_value_signal(score: RunnerScore) -> EngineSignal:
    if score.place_value_edge is None and score.win_value_edge is None:
        return EngineSignal("market_value", 45, 0.35, DATA_MISSING, "Market odds/value data unavailable.")
    best_edge = max(score.place_value_edge or -1.0, score.win_value_edge or -1.0)
    payload = _payload(score.runner)
    move = _market_move(payload, score.runner.current_odds)
    base = 52.0 + max(-0.2, min(0.25, best_edge)) * 120.0
    if move is not None:
        base += -move * 20.0
    quality = DATA_OK if move is not None else DATA_PARTIAL
    explanation = "Market value uses model edge"
    explanation += " and price movement." if move is not None else "; price movement unavailable."
    return EngineSignal("market_value", max(20.0, min(88.0, base)), 0.66 if move is not None else 0.52, quality, explanation)


def _historical_performance_signal(runner: Runner) -> EngineSignal:
    history = _history(runner)
    if not history:
        return EngineSignal("historical_performance", 45, 0.35, DATA_MISSING, "Horse history unavailable.")
    positions = [_position(item) for item in history]
    positions = [position for position in positions if position is not None]
    if not positions:
        return EngineSignal("historical_performance", 42, 0.42, DATA_PARTIAL, "History present but finishing positions unavailable.")
    placed = sum(1 for position in positions if position <= 3)
    wins = sum(1 for position in positions if position == 1)
    score = 42.0 + (placed / len(positions)) * 30.0 + (wins / len(positions)) * 18.0
    return EngineSignal("historical_performance", min(88.0, score), 0.68, DATA_OK, "Historical performance scored from win/place record in imported history.")


def _weighted_index(engines: dict[str, EngineSignal], weights: dict[str, int]) -> float:
    return round(sum(engines[name].score * weight / 100.0 for name, weight in weights.items()), 2)


def _weighted_confidence(engines: dict[str, EngineSignal], weights: dict[str, int]) -> float:
    return round(sum(engines[name].confidence * weight / 100.0 for name, weight in weights.items()), 2)


def _combined_quality(engines: dict[str, EngineSignal]) -> str:
    qualities = {signal.data_quality for signal in engines.values()}
    if DATA_MISSING in qualities:
        return DATA_PARTIAL
    if qualities == {DATA_OK}:
        return DATA_OK
    return DATA_PARTIAL


def _recommendation(win_index: float, place_index: float, confidence: float, score: RunnerScore) -> str:
    odds = _decimal_odds(score.runner.current_odds)
    if confidence < 0.45:
        return "WATCH"
    if place_index >= 68 and odds is not None and 4.0 <= odds <= 21.0 and (score.runner.field_size or 0) >= 8:
        return "V5_PLACE_VALUE"
    if win_index >= 72 and odds is not None and odds <= 12.0:
        return "V5_WIN_VALUE"
    if place_index >= 64:
        return "V5_PLACE_PROFILE"
    return "PASS"


def _headline_explanation(analysis: V5Analysis) -> str:
    top = sorted(analysis.engines.values(), key=lambda signal: signal.score, reverse=True)[:3]
    return "; ".join(f"{signal.name.replace('_', ' ')} {signal.score:.0f}" for signal in top)


def _race_groups(scores: list[RunnerScore]) -> dict[tuple[str, str, str], list[RunnerScore]]:
    groups: dict[tuple[str, str, str], list[RunnerScore]] = {}
    for score in scores:
        runner = score.runner
        groups.setdefault((runner.course, runner.off_time, runner.race_name), []).append(score)
    return groups


def _payload(runner: Runner) -> dict[str, Any]:
    payload = dict(runner.source_payload or {})
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        payload = {**source_runner, **payload}
    return payload


def _history(runner: Runner) -> list[dict[str, Any]]:
    payload = _payload(runner)
    keys = ("horse_history", "history", "results", "past_results", "horse_results", "last_result", "previous_result")
    items: list[dict[str, Any]] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            items.append(value)
    return items


def _position(item: dict[str, Any]) -> int | None:
    for key in ("position", "pos", "finishing_position", "finish_position", "result_position", "place"):
        value = item.get(key)
        if value in (None, ""):
            continue
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _first_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalised = {key.lower().replace(" ", "_").replace("-", "_"): key for key in payload}
    for key in keys:
        raw_key = normalised.get(key.lower().replace(" ", "_").replace("-", "_"))
        if raw_key is None:
            continue
        value = payload.get(raw_key)
        if value in (None, ""):
            continue
        text = str(value).strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            continue
    return None


def _truthy(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    normalised = {key.lower().replace(" ", "_").replace("-", "_"): key for key in payload}
    for key in keys:
        raw_key = normalised.get(key.lower().replace(" ", "_").replace("-", "_"))
        if raw_key is None:
            continue
        value = payload.get(raw_key)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"1", "true", "yes", "y", "positive"}:
            return True
    return False


def _rating_to_score(value: float) -> float:
    if value > 100:
        return max(30.0, min(92.0, 35.0 + value * 0.32))
    return max(30.0, min(88.0, value))


def _market_move(payload: dict[str, Any], current_odds: str | None) -> float | None:
    opening = _first_metric(payload, ("opening_odds_decimal", "open_decimal", "first_decimal"))
    current = _decimal_odds(current_odds)
    if opening is None or current is None or opening <= 1:
        return None
    return (current - opening) / opening


def _decimal_odds(value: Any) -> float | None:
    if value in (None, ""):
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
