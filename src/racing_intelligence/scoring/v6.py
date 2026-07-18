from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from ian_racing_model.analysis_engines import DATA_MISSING, DATA_OK, DATA_PARTIAL, EngineSignal
from ian_racing_model.domain import Runner, RunnerScore


V6_WIN_WEIGHTS = {
    "ability": 18,
    "horse_profile": 10,
    "pace_race_shape": 15,
    "course_suitability": 8,
    "distance_suitability": 8,
    "going_suitability": 6,
    "handicap_position": 10,
    "trainer_intent": 10,
    "current_wellbeing": 8,
    "improvement_potential": 4,
    "market_value": 3,
}

V6_PLACE_WEIGHTS = {
    "ability": 14,
    "horse_profile": 16,
    "pace_race_shape": 15,
    "course_suitability": 10,
    "distance_suitability": 10,
    "going_suitability": 8,
    "handicap_position": 8,
    "trainer_intent": 6,
    "current_wellbeing": 9,
    "improvement_potential": 2,
    "market_value": 2,
}


@dataclass(frozen=True)
class RaceDifficulty:
    grade: str
    score: float
    explanation: str


@dataclass(frozen=True)
class V6Analysis:
    score: RunnerScore
    engines: dict[str, EngineSignal]
    win_index: float
    place_index: float
    confidence: float
    data_quality: str
    recommendation: str
    race_difficulty: RaceDifficulty
    explanation: str


def validate_v6_weights() -> None:
    for name, weights in (("win", V6_WIN_WEIGHTS), ("place", V6_PLACE_WEIGHTS)):
        total = sum(weights.values())
        if total != 100:
            raise ValueError(f"V6 {name} weights must total 100, got {total}.")


def v6_analysis(score: RunnerScore, race_scores: list[RunnerScore] | None = None) -> V6Analysis:
    validate_v6_weights()
    race = race_scores or [score]
    engines = _engine_map(score, race)
    win_index = _weighted_index(engines, V6_WIN_WEIGHTS)
    place_index = _weighted_index(engines, V6_PLACE_WEIGHTS)
    confidence = _weighted_confidence(engines, V6_PLACE_WEIGHTS)
    difficulty = race_difficulty(race)
    return V6Analysis(
        score=score,
        engines=engines,
        win_index=win_index,
        place_index=place_index,
        confidence=confidence,
        data_quality=_combined_quality(engines),
        recommendation=_recommendation(win_index, place_index, confidence, difficulty, score),
        race_difficulty=difficulty,
        explanation=_explanation(score, engines, difficulty),
    )


def v6_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_scores in _race_groups(scores).values():
        difficulty = race_difficulty(race_scores)
        for score in race_scores:
            if score.runner.is_non_runner:
                continue
            analysis = v6_analysis(score, race_scores)
            runner = score.runner
            rows.append(
                {
                    "course": runner.course,
                    "off_time": runner.off_time,
                    "race": runner.race_name,
                    "horse": runner.horse,
                    "odds": runner.current_odds or "Unavailable",
                    "field_size": runner.field_size,
                    "race_difficulty": difficulty.grade,
                    "difficulty_note": difficulty.explanation,
                    "v6_win_index": analysis.win_index,
                    "v6_place_index": analysis.place_index,
                    "v6_confidence": analysis.confidence,
                    "v6_recommendation": analysis.recommendation,
                    "v6_data_quality": analysis.data_quality,
                    "ability": analysis.engines["ability"].score,
                    "horse_profile": analysis.engines["horse_profile"].score,
                    "pace_race_shape": analysis.engines["pace_race_shape"].score,
                    "course_suitability": analysis.engines["course_suitability"].score,
                    "distance_suitability": analysis.engines["distance_suitability"].score,
                    "going_suitability": analysis.engines["going_suitability"].score,
                    "handicap_position": analysis.engines["handicap_position"].score,
                    "trainer_intent": analysis.engines["trainer_intent"].score,
                    "current_wellbeing": analysis.engines["current_wellbeing"].score,
                    "improvement_potential": analysis.engines["improvement_potential"].score,
                    "market_value": analysis.engines["market_value"].score,
                    "v6_explanation": analysis.explanation,
                    "_v6_sort": analysis.place_index + analysis.confidence * 8 - difficulty.score * 0.08,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["_v6_sort", "v6_win_index"], ascending=[False, False])
    df.insert(0, "v6_rank", range(1, len(df) + 1))
    return df.drop(columns=["_v6_sort"])


def race_difficulty_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for race_scores in _race_groups(scores).values():
        if not race_scores:
            continue
        runner = race_scores[0].runner
        difficulty = race_difficulty(race_scores)
        rows.append(
            {
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "field_size": runner.field_size,
                "race_type": runner.race_type or "Unknown",
                "race_class": runner.race_class or "Unknown",
                "difficulty": difficulty.grade,
                "difficulty_score": difficulty.score,
                "explanation": difficulty.explanation,
            }
        )
    return pd.DataFrame(rows).sort_values(["difficulty", "course", "off_time"]) if rows else pd.DataFrame()


def race_difficulty(race_scores: list[RunnerScore]) -> RaceDifficulty:
    active = [score for score in race_scores if not score.runner.is_non_runner]
    if not active:
        return RaceDifficulty("D", 90, "No active runners available.")
    place_indexes = [_base_place_strength(score) for score in active]
    top = max(place_indexes)
    second = sorted(place_indexes, reverse=True)[1] if len(place_indexes) > 1 else 0
    field_size = active[0].runner.field_size or len(active)
    race_text = _race_text(active[0].runner)
    uncertainty = 30.0
    if field_size >= 14:
        uncertainty += 18
    elif field_size <= 6:
        uncertainty += 8
    if any(token in race_text for token in ("maiden", "novice", "juvenile", "2yo", "two year")):
        uncertainty += 18
    if any(token in race_text for token in ("handicap", "nursery")):
        uncertainty += 10
    if top - second >= 10:
        uncertainty -= 16
    if top - second <= 4:
        uncertainty += 12
    if sum(1 for score in active if score.confidence < 0.5) > len(active) * 0.35:
        uncertainty += 12
    uncertainty = _clamp(uncertainty, 0, 100)
    if uncertainty <= 30:
        return RaceDifficulty("A", uncertainty, "Clearer race profile with a standout contender.")
    if uncertainty <= 50:
        return RaceDifficulty("B", uncertainty, "Two or three credible contenders; selective opportunities.")
    if uncertainty <= 70:
        return RaceDifficulty("C", uncertainty, "Competitive race with limited separation between runners.")
    return RaceDifficulty("D", uncertainty, "Unpredictable race; require exceptional value or strong evidence.")


def _engine_map(score: RunnerScore, race_scores: list[RunnerScore]) -> dict[str, EngineSignal]:
    runner = score.runner
    return {
        "ability": _ability_engine(score),
        "horse_profile": _horse_profile_engine(runner),
        "pace_race_shape": _pace_engine(score, race_scores),
        "course_suitability": _profile_match_engine(runner, "course", runner.course, "course"),
        "distance_suitability": _distance_engine(runner),
        "going_suitability": _profile_match_engine(runner, "going", runner.going, "going"),
        "handicap_position": _handicap_engine(score, race_scores),
        "trainer_intent": _trainer_intent_engine(runner),
        "current_wellbeing": _wellbeing_engine(score),
        "improvement_potential": _improvement_engine(runner),
        "market_value": _market_engine(score),
    }


def _ability_engine(score: RunnerScore) -> EngineSignal:
    runner = score.runner
    payload = _payload(runner)
    rating_sources = {
        "official rating": _metric(payload, ("ofr", "official_rating", "or")) or runner.official_rating,
        "RPR": _metric(payload, ("rpr", "racing_post_rating")),
        "Topspeed": _metric(payload, ("ts", "topspeed", "speed_rating")),
        "Timeform": _metric(payload, ("timeform", "timeform_rating")),
        "performance": _metric(payload, ("performance_rating",)),
    }
    values = [(name, _rating_score(value)) for name, value in rating_sources.items() if value is not None]
    if values:
        score_value = sum(value for _, value in values) / len(values)
        labels = ", ".join(name for name, _ in values)
        return EngineSignal("ability", score_value, min(0.82, 0.45 + len(values) * 0.08), DATA_OK, f"Blended ability from {labels}.")
    return EngineSignal("ability", score.total_score, max(0.35, score.confidence * 0.7), DATA_PARTIAL, "Ability proxied from current model score.")


def _horse_profile_engine(runner: Runner) -> EngineSignal:
    history = _history(runner)
    if not history:
        return EngineSignal("horse_profile", 43, 0.32, DATA_MISSING, "Full horse profile unavailable.")
    runs = len(history)
    wins = sum(1 for item in history if _position(item) == 1)
    places = sum(1 for item in history if (_position(item) or 99) <= 3)
    course_places = _match_places(history, "course", runner.course)
    distance_places = _distance_places(history, runner.distance)
    going_places = _match_places(history, "going", runner.going, alternate="ground")
    score = 42 + (wins / runs) * 18 + (places / runs) * 30
    score += min(12, course_places * 3 + distance_places * 3 + going_places * 2)
    explanation = (
        f"Career profile: {runs} runs, {wins} wins, {places} places; "
        f"matched positives today - course {course_places}, distance {distance_places}, going {going_places}."
    )
    return EngineSignal("horse_profile", _clamp(score), 0.72, DATA_OK, explanation)


def _pace_engine(score: RunnerScore, race_scores: list[RunnerScore]) -> EngineSignal:
    runner = score.runner
    style = _running_style(runner)
    pressure = sum(1 for item in race_scores if _running_style(item.runner) in {"Front runner", "Prominent"})
    if style == "Unknown" and runner.draw is None:
        return EngineSignal("pace_race_shape", 43, 0.34, DATA_MISSING, "No running style, pace map or draw evidence available.")
    field_size = runner.field_size or len(race_scores)
    score_value = 55.0
    notes = [f"{style.lower()} profile"]
    if style == "Front runner":
        if pressure <= 2:
            score_value += 14
            notes.append("possible uncontested lead")
        else:
            score_value -= 8
            notes.append("pace pressure likely")
    elif style == "Hold-up":
        if pressure >= max(4, field_size * 0.35):
            score_value += 12
            notes.append("pace collapse could help closers")
        else:
            score_value -= 5
            notes.append("tactical pace could blunt closer")
    elif style == "Prominent":
        score_value += 6 if pressure <= 4 else -3
        notes.append("should hold position near pace")
    if runner.draw is not None and field_size:
        if runner.draw <= max(2, field_size * 0.25):
            score_value += 5
            notes.append("low draw position")
        elif runner.draw >= max(8, field_size * 0.75):
            score_value -= 6
            notes.append("wide/high draw risk")
    return EngineSignal("pace_race_shape", _clamp(score_value), 0.62 if style != "Unknown" else 0.48, DATA_PARTIAL, "; ".join(notes) + ".")


def _profile_match_engine(runner: Runner, name: str, expected: str | None, label: str) -> EngineSignal:
    history = _history(runner)
    if not history or not expected:
        return EngineSignal(f"{name}_suitability", 45, 0.35, DATA_MISSING, f"{label.title()} performance profile unavailable.")
    runs = _match_runs(history, name, expected, alternate="ground" if name == "going" else None)
    places = _match_places(history, name, expected, alternate="ground" if name == "going" else None)
    wins = _match_wins(history, name, expected, alternate="ground" if name == "going" else None)
    if runs == 0:
        return EngineSignal(f"{name}_suitability", 45, 0.55, DATA_PARTIAL, f"No previous comparable {label} run found.")
    score = 42 + (places / runs) * 32 + (wins / runs) * 16
    return EngineSignal(f"{name}_suitability", _clamp(score), 0.74, DATA_OK, f"{label.title()} record: {wins} wins and {places} places from {runs} comparable runs.")


def _distance_engine(runner: Runner) -> EngineSignal:
    history = _history(runner)
    if not history or not runner.distance:
        return EngineSignal("distance_suitability", 45, 0.35, DATA_MISSING, "Distance performance profile unavailable.")
    runs = sum(1 for item in history if _distance_match(runner.distance, _text(item, "distance", "dist")))
    places = _distance_places(history, runner.distance)
    wins = sum(1 for item in history if _distance_match(runner.distance, _text(item, "distance", "dist")) and _position(item) == 1)
    if runs == 0:
        return EngineSignal("distance_suitability", 44, 0.55, DATA_PARTIAL, "No proven run over today's trip or close equivalent.")
    score = 42 + (places / runs) * 34 + (wins / runs) * 16
    return EngineSignal("distance_suitability", _clamp(score), 0.74, DATA_OK, f"Distance record: {wins} wins and {places} places from {runs} comparable trips.")


def _handicap_engine(score: RunnerScore, race_scores: list[RunnerScore]) -> EngineSignal:
    runner = score.runner
    if runner.official_rating is None:
        return EngineSignal("handicap_position", 48, 0.38, DATA_MISSING, "Official rating/handicap position unavailable.")
    ratings = [item.runner.official_rating for item in race_scores if item.runner.official_rating is not None]
    if not ratings:
        return EngineSignal("handicap_position", 48, 0.38, DATA_MISSING, "Race official ratings unavailable.")
    rank = sorted(ratings, reverse=True).index(runner.official_rating) + 1
    score_value = 68 - (rank - 1) * 3.2
    payload = _payload(runner)
    last_win_mark = _metric(payload, ("last_winning_or", "last_winning_mark", "winning_mark"))
    if last_win_mark is not None:
        delta = runner.official_rating - last_win_mark
        score_value += max(-12, min(8, -delta * 1.5))
        note = f"rated {delta:+.0f} lb versus last winning mark"
    else:
        note = f"rated {rank} of {len(ratings)} on official ratings"
    return EngineSignal("handicap_position", _clamp(score_value), 0.66, DATA_PARTIAL, f"Handicap position: {note}.")


def _trainer_intent_engine(runner: Runner) -> EngineSignal:
    payload = _payload(runner)
    flags = []
    score = 52.0
    for keys, label, boost in (
        (("target_race", "trainer_target", "declared_target"), "target flag", 12),
        (("trainer_jockey_ae", "jockey_trainer_ae"), "trainer/jockey combo", 8),
        (("first_handicap", "handicap_debut"), "handicap debut", 6),
        (("class_drop", "down_in_class"), "class drop", 6),
        (("first_time_gelding", "first_run_after_gelding"), "gelded angle", 5),
    ):
        metric = _metric(payload, keys)
        if metric is not None:
            score += (metric - 1.0) * boost if "ae" in keys[0] else boost
            flags.append(label)
        elif _truthy(payload, keys):
            score += boost
            flags.append(label)
    recent = _metric(payload, ("trainer_rtf", "trainer_win_pct", "trainer_14_day_win_pct"))
    if recent is not None:
        score += max(-8, min(10, (recent - 10) * 0.55))
        flags.append("recent trainer form")
    if flags:
        return EngineSignal("trainer_intent", _clamp(score), 0.68, DATA_PARTIAL, "Trainer intent: " + ", ".join(flags) + ".")
    return EngineSignal("trainer_intent", 50, 0.42, DATA_PARTIAL, "Trainer named but no clear target-placement evidence.")


def _wellbeing_engine(score: RunnerScore) -> EngineSignal:
    runner = score.runner
    payload = _payload(runner)
    recent = _metric(payload, ("trainer_rtf", "trainer_14_day_win_pct"))
    history = _history(runner)
    positions = [_position(item) for item in history[:3]]
    positions = [position for position in positions if position is not None]
    score_value = 50.0
    notes = []
    if positions:
        placed = sum(1 for position in positions if position <= 3)
        score_value += placed * 7 - sum(max(0, position - 4) for position in positions) * 1.5
        notes.append(f"last three runs include {placed} places")
    elif runner.recent_form:
        digits = [int(ch) for ch in runner.recent_form if ch.isdigit()]
        if digits:
            placed = sum(1 for digit in digits[:3] if digit <= 3)
            score_value += placed * 6
            notes.append("recent form string used")
    if recent is not None:
        score_value += max(-8, min(10, (recent - 10) * 0.5))
        notes.append("trainer current form")
    if _truthy(payload, ("wind_surgery", "wind_surgery_run")):
        score_value += 3
        notes.append("wind operation noted")
    if runner.current_odds is None:
        notes.append("market absent")
    if notes:
        return EngineSignal("current_wellbeing", _clamp(score_value), 0.62, DATA_PARTIAL, "Current wellbeing: " + ", ".join(notes) + ".")
    return EngineSignal("current_wellbeing", 45, 0.35, DATA_MISSING, "Current wellbeing evidence unavailable.")


def _improvement_engine(runner: Runner) -> EngineSignal:
    payload = _payload(runner)
    history = _history(runner)
    flags = []
    starts = _metric(payload, ("career_starts", "runs", "total_runs")) or len(history)
    if starts and starts <= 4:
        flags.append("lightly raced")
    race_text = _race_text(runner)
    if "maiden" in race_text or "novice" in race_text:
        flags.append("development race")
    for keys, label in (
        (("first_handicap", "handicap_debut"), "handicap debut"),
        (("first_time_headgear", "headgear_run"), "headgear angle"),
        (("first_time_gelding", "first_run_after_gelding"), "gelded angle"),
        (("step_up_trip", "up_in_trip"), "step up in trip"),
        (("class_drop", "down_in_class"), "down in class"),
    ):
        if _truthy(payload, keys):
            flags.append(label)
    if not flags:
        return EngineSignal("improvement_potential", 48, 0.42, DATA_PARTIAL, "No clear improvement trigger found.")
    return EngineSignal("improvement_potential", min(82, 52 + len(set(flags)) * 6), 0.58, DATA_PARTIAL, "Improvement case: " + ", ".join(sorted(set(flags))) + ".")


def _market_engine(score: RunnerScore) -> EngineSignal:
    odds = _decimal_odds(score.runner.current_odds)
    if odds is None:
        return EngineSignal("market_value", 42, 0.32, DATA_MISSING, "Current odds unavailable.")
    place_edge = score.place_value_edge if score.place_value_edge is not None else 0.0
    win_edge = score.win_value_edge if score.win_value_edge is not None else 0.0
    opening = _metric(_payload(score.runner), ("opening_odds_decimal", "open_decimal"))
    value = 50 + max(place_edge, win_edge) * 120
    note = "model overlay"
    if opening:
        move = (odds - opening) / opening
        value += max(-8, min(8, -move * 25))
        note += f"; opened {opening:.2f}, now {odds:.2f}"
    return EngineSignal("market_value", _clamp(value), 0.58 if opening else 0.48, DATA_PARTIAL, "Market intelligence: " + note + ".")


def _recommendation(win_index: float, place_index: float, confidence: float, difficulty: RaceDifficulty, score: RunnerScore) -> str:
    odds = _decimal_odds(score.runner.current_odds)
    if confidence < 0.45 or difficulty.grade == "D":
        return "WATCH"
    if place_index >= 70 and odds is not None and 3.5 <= odds <= 21 and (score.runner.field_size or 0) >= 8:
        return "V6_PLACE_EDGE"
    if win_index >= 74 and odds is not None and odds <= 14:
        return "V6_WIN_EDGE"
    if place_index >= 66 and difficulty.grade in {"A", "B"}:
        return "V6_PLACE_PROFILE"
    return "PASS"


def _explanation(score: RunnerScore, engines: dict[str, EngineSignal], difficulty: RaceDifficulty) -> str:
    strengths = [signal.explanation for signal in sorted(engines.values(), key=lambda signal: signal.score, reverse=True)[:4]]
    concerns = [signal.explanation for signal in engines.values() if signal.score < 48][:2]
    parts = strengths + concerns + [f"Race difficulty {difficulty.grade}: {difficulty.explanation}"]
    return " ".join(parts)


def _base_place_strength(score: RunnerScore) -> float:
    return (score.place_probability or 0) * 100 + score.total_score * 0.35 + score.confidence * 10


def _weighted_index(engines: dict[str, EngineSignal], weights: dict[str, int]) -> float:
    return round(sum(engines[name].score * weight / 100 for name, weight in weights.items()), 2)


def _weighted_confidence(engines: dict[str, EngineSignal], weights: dict[str, int]) -> float:
    return round(sum(engines[name].confidence * weight / 100 for name, weight in weights.items()), 2)


def _combined_quality(engines: dict[str, EngineSignal]) -> str:
    qualities = {signal.data_quality for signal in engines.values()}
    if DATA_MISSING in qualities:
        return DATA_PARTIAL
    if qualities == {DATA_OK}:
        return DATA_OK
    return DATA_PARTIAL


def _race_groups(scores: list[RunnerScore]) -> dict[tuple[str, str, str], list[RunnerScore]]:
    grouped: dict[tuple[str, str, str], list[RunnerScore]] = defaultdict(list)
    for score in scores:
        runner = score.runner
        grouped[(runner.course, runner.off_time, runner.race_name)].append(score)
    return grouped


def _payload(runner: Runner) -> dict[str, Any]:
    payload = dict(runner.source_payload or {})
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        payload = {**source_runner, **payload}
    return payload


def _history(runner: Runner) -> list[dict[str, Any]]:
    payload = _payload(runner)
    items = []
    for key in ("horse_history", "history", "results", "past_results", "horse_results"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            items.append(value)
    return items


def _metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalised = {key.lower().replace(" ", "_").replace("-", "_").replace("/", "_"): key for key in payload}
    for key in keys:
        raw = normalised.get(key.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))
        if raw is None:
            continue
        value = payload.get(raw)
        if value in (None, ""):
            continue
        try:
            return float(str(value).strip().replace("%", ""))
        except ValueError:
            continue
    return None


def _truthy(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    normalised = {key.lower().replace(" ", "_").replace("-", "_").replace("/", "_"): key for key in payload}
    for key in keys:
        raw = normalised.get(key.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))
        if raw is None:
            continue
        value = payload.get(raw)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"1", "true", "yes", "y", "first", "positive", "target"}:
            return True
    return False


def _position(item: dict[str, Any]) -> int | None:
    for key in ("position", "pos", "finishing_position", "finish_position", "result_position", "place"):
        value = item.get(key)
        if value in (None, ""):
            continue
        digits = "".join(ch for ch in str(value).split("/")[0] if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _match_runs(history: list[dict[str, Any]], key: str, expected: str | None, alternate: str | None = None) -> int:
    return sum(1 for item in history if _text_match(expected, _text(item, key, *(tuple([alternate]) if alternate else tuple()))))


def _match_places(history: list[dict[str, Any]], key: str, expected: str | None, alternate: str | None = None) -> int:
    return sum(1 for item in history if (_position(item) or 99) <= 3 and _text_match(expected, _text(item, key, *(tuple([alternate]) if alternate else tuple()))))


def _match_wins(history: list[dict[str, Any]], key: str, expected: str | None, alternate: str | None = None) -> int:
    return sum(1 for item in history if _position(item) == 1 and _text_match(expected, _text(item, key, *(tuple([alternate]) if alternate else tuple()))))


def _distance_places(history: list[dict[str, Any]], expected: str | None) -> int:
    return sum(1 for item in history if (_position(item) or 99) <= 3 and _distance_match(expected, _text(item, "distance", "dist")))


def _text(item: dict[str, Any], *keys: str | None) -> str:
    for key in keys:
        if not key:
            continue
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _text_match(expected: str | None, actual: str | None) -> bool:
    expected_text = _normalise(expected)
    actual_text = _normalise(actual)
    return bool(expected_text and actual_text and (expected_text in actual_text or actual_text in expected_text))


def _normalise(value: str | None) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _distance_match(expected: str | None, actual: str | None) -> bool:
    expected_trip = _distance_furlongs(expected)
    actual_trip = _distance_furlongs(actual)
    return expected_trip is not None and actual_trip is not None and abs(expected_trip - actual_trip) <= 0.3


def _distance_furlongs(value: str | None) -> float | None:
    if not value:
        return None
    text = _normalise(value).replace(" ", "")
    miles = furlongs = yards = 0.0
    mile_match = re.search(r"(\d+(?:\.\d+)?)m", text)
    furlong_match = re.search(r"(\d+(?:\.\d+)?)f", text)
    yard_match = re.search(r"(\d+(?:\.\d+)?)y", text)
    if mile_match:
        miles = float(mile_match.group(1))
    if furlong_match:
        furlongs = float(furlong_match.group(1))
    if yard_match:
        yards = float(yard_match.group(1))
    if miles or furlongs or yards:
        return miles * 8 + furlongs + yards / 220
    return None


def _running_style(runner: Runner) -> str:
    text = " ".join(
        str(_payload(runner).get(key) or "")
        for key in ("run_style", "pace_style", "early_pace", "comment", "spotlight", "comments")
    ).lower()
    if any(token in text for token in ("front", "leader", "made all", "led")):
        return "Front runner"
    if any(token in text for token in ("prominent", "chased", "tracked")):
        return "Prominent"
    if any(token in text for token in ("midfield", "mid-division")):
        return "Midfield"
    if any(token in text for token in ("held up", "rear", "dwelt", "slowly away")):
        return "Hold-up"
    return "Unknown"


def _race_text(runner: Runner) -> str:
    return " ".join(str(value or "").lower() for value in (runner.race_name, runner.race_type, runner.race_class))


def _rating_score(value: float) -> float:
    if value > 110:
        return _clamp(34 + value * 0.33, 25, 94)
    return _clamp(value, 25, 92)


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
            return float(num) / denominator + 1
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, round(value, 2)))
