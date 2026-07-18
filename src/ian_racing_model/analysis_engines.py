from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ian_racing_model.domain import Runner, RunnerScore


DATA_OK = "ok"
DATA_PARTIAL = "partial"
DATA_MISSING = "missing"


@dataclass(frozen=True)
class EngineSignal:
    name: str
    score: float
    confidence: float
    data_quality: str
    explanation: str


def pace_race_shape_signal(runner: Runner) -> EngineSignal:
    payload = _combined_payload(runner)
    imported = _metric_value(payload, ("pace_rating", "pace_score", "early_pace_rating", "run_style_score"))
    draw_bias = _metric_value(payload, ("draw_bias", "draw_bias_score", "draw_advantage"))
    if imported is not None:
        score = _clamp(imported + ((draw_bias - 50.0) * 0.25 if draw_bias is not None else 0.0))
        return EngineSignal(
            "pace_race_shape",
            score,
            0.78 if draw_bias is not None else 0.68,
            DATA_OK if draw_bias is not None else DATA_PARTIAL,
            "Imported pace/race-shape signal used with draw context."
            if draw_bias is not None
            else "Imported pace signal used; draw-bias feed unavailable.",
        )

    if runner.draw is None or runner.field_size is None:
        return EngineSignal(
            "pace_race_shape",
            45,
            0.35,
            DATA_MISSING,
            "Pace map and draw/field-size data unavailable.",
        )

    score = 55.0
    explanation = "Race shape inferred from draw, field size and available run-style hints."
    if runner.draw <= max(2, runner.field_size * 0.25):
        score += 12
    elif runner.draw > max(8, runner.field_size * 0.75):
        score -= 16

    run_style = _run_style_text(payload)
    if any(token in run_style for token in ("front", "leader", "led", "prominent", "made all")):
        score += 8
        explanation = "Likely forward run style may help control or hold position."
    elif any(token in run_style for token in ("held up", "slowly", "rear", "dwelt")):
        score -= 6
        explanation = "Hold-up or slow-start hints make race shape less reliable."

    return EngineSignal("pace_race_shape", _clamp(score), 0.58, DATA_PARTIAL, explanation)


def trainer_intent_signal(runner: Runner) -> EngineSignal:
    payload = _combined_payload(runner)
    trainer_ae = _metric_value(payload, ("trainer_ae", "trainer_a/e", "trainer_actual_expected"))
    trainer_sr = _metric_value(
        payload,
        (
            "trainer_strike_rate",
            "trainer_win_pct",
            "trainer_win_percentage",
            "trainer_14_day_win_pct",
            "trainer_30_day_win_pct",
            "trainer_rtf",
        ),
    )
    trainer_recent = _trainer_recent_percent(payload)
    if trainer_sr is None:
        trainer_sr = trainer_recent
    course_sr = _metric_value(payload, ("trainer_course_win_pct", "trainer_course_strike_rate"))
    target_flag = _truthy(payload, ("target_race", "trainer_target", "intent_flag", "declared_target"))

    if any(value is not None for value in (trainer_ae, trainer_sr, course_sr)) or target_flag:
        score = 55.0
        notes: list[str] = []
        if trainer_ae is not None:
            score += (trainer_ae - 1.0) * 35.0
            notes.append("trainer A/E")
        if trainer_sr is not None:
            score += min(14.0, max(-10.0, (trainer_sr - 10.0) * 0.7))
            notes.append("recent strike rate")
        if course_sr is not None:
            score += min(10.0, max(-8.0, (course_sr - 10.0) * 0.55))
            notes.append("course record")
        if target_flag:
            score += 8.0
            notes.append("target flag")
        return EngineSignal(
            "trainer_intent",
            _clamp(score),
            0.74,
            DATA_OK,
            "Trainer intent scored from imported " + ", ".join(notes) + ".",
        )

    if not runner.trainer:
        return EngineSignal("trainer_intent", 42, 0.32, DATA_MISSING, "Trainer unavailable.")

    form = runner.recent_form or ""
    history = _history_items(runner)
    if _last_position(form, history) in {1, 2, 3}:
        return EngineSignal(
            "trainer_intent",
            66,
            0.54,
            DATA_PARTIAL,
            "Intent inferred from recent competitiveness; trainer stats unavailable.",
        )
    if "-" in form:
        return EngineSignal(
            "trainer_intent",
            44,
            0.5,
            DATA_PARTIAL,
            "Break in form string may indicate prep/fitness uncertainty.",
        )
    return EngineSignal("trainer_intent", 55, 0.45, DATA_PARTIAL, "Trainer present, but intent evidence is limited.")


def course_conditions_signal(runner: Runner) -> EngineSignal:
    payload = _combined_payload(runner)
    imported_values = [
        _metric_value(payload, keys)
        for keys in (
            ("course_place_pct", "course_win_pct", "course_strike_rate"),
            ("going_place_pct", "going_win_pct", "ground_place_pct"),
            ("distance_place_pct", "distance_win_pct", "trip_place_pct"),
            ("course_distance_place_pct", "cd_place_pct", "course_distance_win_pct"),
        )
    ]
    imported_values = [value for value in imported_values if value is not None]
    if imported_values:
        avg = sum(imported_values) / len(imported_values)
        score = _clamp(48.0 + avg * 0.85)
        return EngineSignal(
            "course_conditions",
            score,
            0.76,
            DATA_OK,
            "Course, distance or going suitability scored from imported performance rates.",
        )

    history = _history_items(runner)
    if history:
        course_hits = _placed_matches(history, "course", runner.course)
        going_hits = _placed_matches(history, "going", runner.going, alternate_key="ground")
        distance_hits = _placed_distance_matches(history, runner.distance)
        surface_hits = _placed_matches(history, "surface", runner.surface)
        score = 48 + course_hits * 8 + going_hits * 6 + distance_hits * 7 + surface_hits * 4
        if any((course_hits, going_hits, distance_hits, surface_hits)):
            return EngineSignal(
                "course_conditions",
                _clamp(score),
                0.68,
                DATA_PARTIAL,
                "Course/conditions profile inferred from placed runs in horse history.",
            )
        return EngineSignal(
            "course_conditions",
            44,
            0.62,
            DATA_PARTIAL,
            "Horse history available, but no clear course, distance or going proof found.",
        )

    if not any((runner.course, runner.distance, runner.going, runner.surface)):
        return EngineSignal(
            "course_conditions",
            40,
            0.3,
            DATA_MISSING,
            "Course, distance, going and surface data unavailable.",
        )
    return EngineSignal(
        "course_conditions",
        55,
        0.42,
        DATA_PARTIAL,
        "Race setup fields are present, but proven suitability data is unavailable.",
    )


def engine_signal_map(score: RunnerScore) -> dict[str, EngineSignal]:
    runner = score.runner
    return {
        "pace_race_shape": pace_race_shape_signal(runner),
        "trainer_intent": trainer_intent_signal(runner),
        "course_conditions": course_conditions_signal(runner),
    }


def _combined_payload(runner: Runner) -> dict[str, Any]:
    payload = dict(runner.source_payload or {})
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        payload = {**source_runner, **payload}
    return payload


def _metric_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalised_keys = {key.lower().replace(" ", "_").replace("-", "_"): key for key in payload}
    for key in keys:
        raw_key = normalised_keys.get(key.lower().replace(" ", "_").replace("-", "_"))
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


def _trainer_recent_percent(payload: dict[str, Any]) -> float | None:
    value = payload.get("trainer_14_days")
    if isinstance(value, dict):
        percent = value.get("percent")
        if percent not in (None, ""):
            try:
                return float(str(percent).strip().replace("%", ""))
            except ValueError:
                return None
    return None


def _truthy(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    normalised_keys = {key.lower().replace(" ", "_").replace("-", "_"): key for key in payload}
    for key in keys:
        raw_key = normalised_keys.get(key.lower().replace(" ", "_").replace("-", "_"))
        if raw_key is None:
            continue
        value = payload.get(raw_key)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"1", "true", "yes", "y", "target", "positive"}:
            return True
    return False


def _history_items(runner: Runner) -> list[dict[str, Any]]:
    payload = runner.source_payload or {}
    keys = ("horse_history", "history", "results", "past_results", "horse_results", "last_result", "previous_result")
    items: list[dict[str, Any]] = []
    for source in (payload, payload.get("source_runner") if isinstance(payload.get("source_runner"), dict) else {}):
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                items.append(value)
    return items


def _history_position(item: dict[str, Any]) -> int | None:
    for key in ("position", "pos", "finishing_position", "finish_position", "result_position", "place"):
        value = item.get(key)
        if value in (None, ""):
            continue
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _history_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _normalise_text(value: str | None) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _text_match(expected: str | None, actual: str | None) -> bool:
    expected_text = _normalise_text(expected)
    actual_text = _normalise_text(actual)
    return bool(expected_text and actual_text and (expected_text in actual_text or actual_text in expected_text))


def _placed_matches(history: list[dict[str, Any]], key: str, expected: str | None, alternate_key: str | None = None) -> int:
    return sum(
        1
        for item in history
        if (_history_position(item) or 99) <= 3
        and _text_match(expected, _history_text(item, key, *(tuple([alternate_key]) if alternate_key else tuple())))
    )


def _placed_distance_matches(history: list[dict[str, Any]], expected: str | None) -> int:
    return sum(
        1
        for item in history
        if (_history_position(item) or 99) <= 3
        and _distance_match(expected, _history_text(item, "distance", "dist"))
    )


def _distance_match(expected: str | None, actual: str | None) -> bool:
    expected_trip = _distance_furlongs(expected)
    actual_trip = _distance_furlongs(actual)
    return expected_trip is not None and actual_trip is not None and abs(expected_trip - actual_trip) <= 0.25


def _distance_furlongs(value: str | None) -> float | None:
    if not value:
        return None
    text = _normalise_text(value).replace(" ", "")
    miles = 0.0
    furlongs = 0.0
    yards = 0.0
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
        return miles * 8.0 + furlongs + yards / 220.0
    return None


def _last_position(form: str, history: list[dict[str, Any]]) -> int | None:
    for item in history[:1]:
        position = _history_position(item)
        if position is not None:
            return position
    for char in form:
        if char.isdigit():
            return int(char)
    return None


def _run_style_text(payload: dict[str, Any]) -> str:
    values = [
        str(payload.get(key) or "")
        for key in ("run_style", "pace_style", "early_pace", "comments", "analysis")
    ]
    return " ".join(values).lower()


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, round(value, 2)))
