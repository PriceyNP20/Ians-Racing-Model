from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore


def outsider_last_time_dataframe(scores: list[RunnerScore], min_decimal_odds: float = 31.0) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        signal = _last_time_outsider_signal(item, min_decimal_odds)
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
                "last_class": signal["last_class"],
                "last_going": signal["last_going"],
                "last_distance": signal["last_distance"],
                "similarity": signal["similarity"],
                "signal": signal["signal"],
                "today_odds": runner.current_odds or "Unavailable",
                "score": item.total_score,
                "recommendation": item.recommendation,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["similarity", "score", "course", "off_time"],
        ascending=[False, False, True, True],
    )


def _last_time_outsider_signal(item: RunnerScore, min_decimal_odds: float) -> dict[str, Any] | None:
    runner = item.runner
    for candidate in _history_candidates(runner.source_payload):
        if not isinstance(candidate, dict):
            continue
        position = _parse_position(candidate.get("position") or candidate.get("pos"))
        odds = _history_decimal_odds(candidate)
        if position is None or position > 3 or odds is None or odds < min_decimal_odds:
            continue
        setup = _setup_similarity(item, candidate)
        if not setup["qualified"]:
            continue
        return {
            "last_result": f"{position}",
            "last_odds": f"{odds:.1f}",
            "last_class": _history_text(candidate, "race_class", "class", "race_grade", "grade") or "Unknown",
            "last_going": _history_text(candidate, "going", "ground") or "Unknown",
            "last_distance": _history_text(candidate, "distance", "dist", "race_distance") or "Unknown",
            "similarity": setup["score"],
            "signal": f"Placed at >30/1 last time; similar {', '.join(setup['matches'])}",
        }
    return None


def _setup_similarity(item: RunnerScore, candidate: dict[str, Any]) -> dict[str, Any]:
    runner = item.runner
    matches: list[str] = []
    if _class_similarity(runner.race_class, _history_text(candidate, "race_class", "class", "race_grade", "grade")):
        matches.append("class/grade")
    if _text_similarity(runner.going, _history_text(candidate, "going", "ground")):
        matches.append("going")
    if _distance_similarity(runner.distance, _history_text(candidate, "distance", "dist", "race_distance")):
        matches.append("distance")
    if _text_similarity(runner.surface, _history_text(candidate, "surface")):
        matches.append("surface")

    if "class/grade" not in matches and _component_score(item, "class_strength") >= 6.4:
        matches.append("class/grade")
    if "distance" not in matches and _component_score(item, "distance_suitability") >= 6.4:
        matches.append("distance")

    has_core_match = any(match in matches for match in ("class/grade", "distance"))
    return {"matches": matches, "score": len(matches), "qualified": len(matches) >= 2 and has_core_match}


def _history_candidates(payload: dict[str, Any]) -> list[Any]:
    keys = (
        "last_result",
        "last_run_result",
        "previous_result",
        "previous_run",
        "latest_result",
        "history",
        "horse_history",
        "results",
        "past_results",
        "horse_results",
    )
    candidates: list[Any] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value[:3])
        elif value:
            candidates.append(value)
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        for key in keys:
            value = source_runner.get(key)
            if isinstance(value, list):
                candidates.extend(value[:3])
            elif value:
                candidates.append(value)
    return candidates


def _history_decimal_odds(payload: dict[str, Any]) -> float | None:
    for key in ("sp_dec", "bsp", "decimal", "odds_decimal", "sp", "odds", "starting_price"):
        odds = _decimal_odds(payload.get(key))
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


def _parse_position(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"nr", "non-runner", "pu", "f", "ur", "bd"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _history_text(payload: dict[str, Any], *keys: str) -> str:
    normalised_keys = {key.lower().replace(" ", "_").replace("-", "_"): key for key in payload}
    for key in keys:
        raw_key = normalised_keys.get(key.lower().replace(" ", "_").replace("-", "_"))
        if raw_key is None:
            continue
        value = payload.get(raw_key)
        if value not in (None, ""):
            return str(value)
    return ""


def _text_similarity(expected: str | None, actual: str | None) -> bool:
    expected_text = _normalise_text(expected)
    actual_text = _normalise_text(actual)
    return bool(expected_text and actual_text and (expected_text in actual_text or actual_text in expected_text))


def _class_similarity(expected: str | None, actual: str | None) -> bool:
    expected_class = _class_number(expected)
    actual_class = _class_number(actual)
    if expected_class is not None and actual_class is not None:
        return abs(expected_class - actual_class) <= 1
    return _text_similarity(expected, actual)


def _class_number(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b([1-7])\b", str(value))
    if not match:
        return None
    return int(match.group(1))


def _distance_similarity(expected: str | None, actual: str | None) -> bool:
    expected_text = _normalise_text(expected).replace(" ", "")
    actual_text = _normalise_text(actual).replace(" ", "")
    if expected_text and actual_text and expected_text == actual_text:
        return True
    expected_trip = _distance_furlongs(expected)
    actual_trip = _distance_furlongs(actual)
    return bool(expected_trip is not None and actual_trip is not None and abs(expected_trip - actual_trip) <= 0.5)


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


def _normalise_text(value: str | None) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _component_score(item: RunnerScore, name: str) -> float:
    for component in item.components:
        if component.name == name:
            return component.score
    return 0.0
