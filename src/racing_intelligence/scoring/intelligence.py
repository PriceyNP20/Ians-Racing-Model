from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from ian_racing_model.analysis_engines import engine_signal_map
from ian_racing_model.domain import RunnerScore
from racing_intelligence.domain import IntelligenceRunner, ProbabilityAssessment
from racing_intelligence.scoring.v5 import v5_analysis


def analyse_scores(scores: list[RunnerScore]) -> list[IntelligenceRunner]:
    by_race: dict[tuple[str, str, str], list[RunnerScore]] = defaultdict(list)
    for score in scores:
        runner = score.runner
        by_race[(runner.course, runner.off_time, runner.race_name)].append(score)

    rows: list[IntelligenceRunner] = []
    for race_scores in by_race.values():
        for score in race_scores:
            signals = engine_signal_map(score)
            win = _win_assessment(score, race_scores, signals)
            place = _place_assessment(score, race_scores, signals)
            win_edge = _edge(win.probability, _decimal_odds(score.runner.current_odds))
            place_edge = _edge(place.probability, _estimated_place_odds(_decimal_odds(score.runner.current_odds)))
            rows.append(
                IntelligenceRunner(
                    course=score.runner.course,
                    off_time=score.runner.off_time,
                    race=score.runner.race_name,
                    horse=score.runner.horse,
                    odds=score.runner.current_odds,
                    field_size=score.runner.field_size,
                    win=win,
                    place=place,
                    win_value_edge=win_edge,
                    place_value_edge=place_edge,
                    recommendation=_recommendation(win, place, win_edge, place_edge, score),
                    data_quality=_combined_quality(win, place, score),
                    warnings=list(score.red_flags or score.data_quality_warnings),
                )
            )
    return sorted(rows, key=lambda row: (row.recommendation != "PASS", row.place.probability, row.win.probability), reverse=True)


def intelligence_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    analysed = analyse_scores(scores)
    score_lookup = {
        (score.runner.course, score.runner.off_time, score.runner.race_name, score.runner.horse): score
        for score in scores
    }
    for item in analysed:
        source_score = score_lookup[(item.course, item.off_time, item.race, item.horse)]
        signals = engine_signal_map(source_score)
        v5 = v5_analysis(source_score)
        rows.append(
            {
                "course": item.course,
                "off_time": item.off_time,
                "race": item.race,
                "horse": item.horse,
                "odds": item.odds or "Unavailable",
                "field_size": item.field_size,
                "win_probability": _format_probability(item.win.probability),
                "place_probability": _format_probability(item.place.probability),
                "fair_win_odds": _format_odds(item.win.fair_odds),
                "fair_place_odds": _format_odds(item.place.fair_odds),
                "win_value_edge": _format_edge(item.win_value_edge),
                "place_value_edge": _format_edge(item.place_value_edge),
                "recommendation": item.recommendation,
                "v5_win_index": v5.win_index,
                "v5_place_index": v5.place_index,
                "v5_recommendation": v5.recommendation,
                "v5_confidence": v5.confidence,
                "v5_data_quality": v5.data_quality,
                "pace_shape_score": signals["pace_race_shape"].score,
                "trainer_intent_score": signals["trainer_intent"].score,
                "course_conditions_score": signals["course_conditions"].score,
                "ability_engine": v5.engines["ability"].score,
                "suitability_engine": v5.engines["suitability"].score,
                "race_shape_engine": v5.engines["race_shape"].score,
                "trainer_intent_engine": v5.engines["trainer_intent"].score,
                "current_wellbeing_engine": v5.engines["current_wellbeing"].score,
                "improvement_engine": v5.engines["improvement_potential"].score,
                "market_value_engine": v5.engines["market_value"].score,
                "historical_performance_engine": v5.engines["historical_performance"].score,
                "data_quality": item.data_quality,
                "win_explanation": item.win.explanation,
                "place_explanation": item.place.explanation,
                "pace_shape": signals["pace_race_shape"].explanation,
                "trainer_intent": signals["trainer_intent"].explanation,
                "course_conditions": signals["course_conditions"].explanation,
                "v5_explanation": "; ".join(
                    [
                        v5.engines["ability"].explanation,
                        v5.engines["suitability"].explanation,
                        v5.engines["race_shape"].explanation,
                        v5.engines["trainer_intent"].explanation,
                        v5.engines["improvement_potential"].explanation,
                    ]
                ),
                "warnings": "; ".join(item.warnings[:3]) if item.warnings else "None",
                "_win_probability": item.win.probability,
                "_place_probability": item.place.probability,
                "_place_edge": item.place_value_edge if item.place_value_edge is not None else -1.0,
                "_win_edge": item.win_value_edge if item.win_value_edge is not None else -1.0,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["_place_probability", "_place_edge", "_win_probability", "_win_edge"],
        ascending=[False, False, False, False],
    )
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["_win_probability", "_place_probability", "_place_edge", "_win_edge"])


def _win_assessment(score: RunnerScore, race_scores: list[RunnerScore], signals: dict[str, Any]) -> ProbabilityAssessment:
    probability = score.win_probability or _normalised_probability(score.total_score, race_scores)
    engine_factor = _engine_probability_factor(signals, mode="win")
    probability = max(0.001, min(0.85, probability * engine_factor))
    confidence = min(0.95, max(0.05, score.confidence))
    quality = "ok" if score.win_probability else "partial"
    return ProbabilityAssessment(
        probability=probability,
        fair_odds=_fair_odds(probability),
        confidence=confidence,
        data_quality=quality,
        explanation="Separate win probability adjusted by race shape, trainer intent and course/conditions evidence.",
    )


def _place_assessment(score: RunnerScore, race_scores: list[RunnerScore], signals: dict[str, Any]) -> ProbabilityAssessment:
    field_size = score.runner.field_size or len(race_scores)
    place_slots = _place_slots(field_size)
    base = score.place_probability or min(0.9, (score.win_probability or 0.05) * place_slots * 0.9)
    reliability = _place_reliability(score)
    engine_factor = _engine_probability_factor(signals, mode="place")
    probability = max(0.01, min(0.92, base * reliability * engine_factor))
    confidence = min(0.95, max(0.05, score.confidence * reliability * _engine_confidence_factor(signals)))
    quality = "ok" if score.place_probability else "partial"
    return ProbabilityAssessment(
        probability=probability,
        fair_odds=_fair_odds(probability),
        confidence=confidence,
        data_quality=quality,
        explanation="Independent place assessment using place slots, reliability, race shape, intent and conditions.",
    )


def _engine_probability_factor(signals: dict[str, Any], mode: str) -> float:
    weights = {
        "win": {"pace_race_shape": 0.4, "trainer_intent": 0.35, "course_conditions": 0.25},
        "place": {"pace_race_shape": 0.35, "trainer_intent": 0.25, "course_conditions": 0.4},
    }[mode]
    weighted_edge = 0.0
    for name, weight in weights.items():
        signal = signals[name]
        usable_confidence = max(0.0, min(1.0, signal.confidence))
        weighted_edge += ((signal.score - 55.0) / 100.0) * weight * usable_confidence
    return max(0.78, min(1.22, 1.0 + weighted_edge))


def _engine_confidence_factor(signals: dict[str, Any]) -> float:
    confidence = sum(signal.confidence for signal in signals.values()) / max(1, len(signals))
    return max(0.78, min(1.12, 0.82 + confidence * 0.3))


def _place_reliability(score: RunnerScore) -> float:
    runner = score.runner
    reliability = 1.0
    if runner.field_size and runner.field_size < 8:
        reliability -= 0.25
    if runner.recent_form:
        digits = [int(ch) for ch in runner.recent_form[:5] if ch.isdigit()]
        if digits:
            placed = sum(1 for value in digits if value <= 3)
            reliability += min(0.18, placed / len(digits) * 0.18)
    reliability -= min(0.32, len(score.red_flags) * 0.06)
    odds = _decimal_odds(runner.current_odds)
    if odds is not None and odds > 21:
        reliability -= 0.2
    return max(0.45, min(1.15, reliability))


def _recommendation(
    win: ProbabilityAssessment,
    place: ProbabilityAssessment,
    win_edge: float | None,
    place_edge: float | None,
    score: RunnerScore,
) -> str:
    odds = _decimal_odds(score.runner.current_odds)
    if win.probability >= 0.22 and (win_edge or -1) > 0.03 and win.confidence >= 0.55:
        return "WIN_VALUE"
    if (
        odds is not None
        and 3.0 <= odds <= 21.0
        and score.runner.field_size
        and score.runner.field_size >= 8
        and place.probability >= 0.34
        and (place_edge or -1) > 0.02
        and place.confidence >= 0.45
    ):
        return "PLACE_VALUE"
    if place.probability >= 0.45 and place.confidence >= 0.5:
        return "PLACE_PROFILE"
    return "PASS"


def _combined_quality(win: ProbabilityAssessment, place: ProbabilityAssessment, score: RunnerScore) -> str:
    if score.data_quality_warnings:
        return "partial"
    if win.data_quality == "ok" and place.data_quality == "ok":
        return "ok"
    return "partial"


def _normalised_probability(score: float, race_scores: list[RunnerScore]) -> float:
    total = sum(max(1.0, item.total_score) for item in race_scores) or 1.0
    return max(0.001, min(0.85, max(1.0, score) / total))


def _place_slots(field_size: int) -> int:
    if field_size >= 16:
        return 4
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return 1


def _edge(probability: float, odds: float | None) -> float | None:
    if odds is None or odds <= 1:
        return None
    return probability - (1.0 / odds)


def _fair_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return round(1.0 / probability, 2)


def _estimated_place_odds(win_odds: float | None) -> float | None:
    if win_odds is None or win_odds <= 1:
        return None
    return 1.0 + ((win_odds - 1.0) / 5.0)


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


def _format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_edge(value: float | None) -> str:
    if value is None:
        return "Needs odds"
    return f"{value * 100:+.1f} pts"


def _format_odds(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2f}"
