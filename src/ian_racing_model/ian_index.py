from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore


IAN_INDEX_V4_WEIGHTS = {
    "ability_timeform": 20,
    "speed_beyer_topspeed": 15,
    "class_rpr": 15,
    "pace_race_shape": 15,
    "value_hugh_taylor": 15,
    "trainer_intent": 10,
    "jockey": 5,
    "course_going": 5,
}


@dataclass(frozen=True)
class IanIndexComponent:
    score: float
    confidence: float
    data_quality: str
    explanation: str


def validate_ian_index_weights() -> None:
    total = sum(IAN_INDEX_V4_WEIGHTS.values())
    if total != 100:
        raise ValueError(f"Ian Index V4 weights must total 100, got {total}.")


def ian_index_place_dataframe(scores: list[RunnerScore], limit: int | None = None) -> pd.DataFrame:
    validate_ian_index_weights()
    rows = []
    for item in scores:
        if item.runner.is_non_runner:
            continue
        components = _components(item)
        weighted_score = sum(
            components[name].score * weight / 100.0
            for name, weight in IAN_INDEX_V4_WEIGHTS.items()
        )
        red_flag_drag = min(8.0, len(item.red_flags) * 1.8)
        confidence = sum(
            components[name].confidence * weight / 100.0
            for name, weight in IAN_INDEX_V4_WEIGHTS.items()
        )
        place_rating = _clip(weighted_score - red_flag_drag)
        rows.append(
            {
                "rank": 0,
                "pick_type": "Ian Trial place profile",
                "horse": item.runner.horse,
                "course": item.runner.course,
                "off_time": item.runner.off_time,
                "race": item.runner.race_name,
                "place_rating": round(place_rating, 2),
                "place_probability": _format_probability(item.place_probability),
                "place_value_edge": _format_edge(item.place_value_edge),
                "odds": item.runner.current_odds or "Unavailable",
                "confidence": round(confidence, 2),
                "data_quality": _data_quality(components),
                "ability_timeform": round(components["ability_timeform"].score, 1),
                "speed_beyer_topspeed": round(components["speed_beyer_topspeed"].score, 1),
                "class_rpr": round(components["class_rpr"].score, 1),
                "pace_race_shape": round(components["pace_race_shape"].score, 1),
                "value_hugh_taylor": round(components["value_hugh_taylor"].score, 1),
                "trainer_intent": round(components["trainer_intent"].score, 1),
                "jockey": round(components["jockey"].score, 1),
                "course_going": round(components["course_going"].score, 1),
                "red_flags": "; ".join(item.red_flags[:3]) if item.red_flags else "None",
                "result": _result_text(item),
                "outcome": _place_outcome(item),
                "place_cutoff": _place_cutoff(item.runner.field_size),
                "explanation": _explanation(components, item),
                "_sort_rating": place_rating,
                "_sort_probability": item.place_probability or 0.0,
                "_sort_edge": item.place_value_edge if item.place_value_edge is not None else -1.0,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["_sort_rating", "_sort_probability", "_sort_edge", "confidence"],
        ascending=[False, False, False, False],
    )
    if limit is not None:
        df = df.head(limit)
    df = df.copy()
    df["rank"] = range(1, len(df) + 1)
    return df.drop(columns=["_sort_rating", "_sort_probability", "_sort_edge"])


def ian_index_acca_dataframe(scores: list[RunnerScore], limit: int = 6) -> pd.DataFrame:
    trial = ian_index_place_dataframe(scores)
    if trial.empty:
        return pd.DataFrame()

    trial = trial.copy()
    trial["_field_size"] = trial.apply(lambda row: _field_size_for_row(row, scores), axis=1)
    trial = trial[trial["_field_size"] >= 8].copy()
    if trial.empty:
        return pd.DataFrame()

    selected = []
    used_races = set()
    for row in trial.to_dict("records"):
        race_key = _trial_race_key(row)
        if race_key in used_races:
            continue
        row["acca_rank"] = len(selected) + 1
        row["pick_type"] = "Ian Trial EW pick"
        row["field_size"] = int(row.pop("_field_size", 0))
        selected.append(row)
        used_races.add(race_key)
        if len(selected) == limit:
            break

    if not selected:
        return pd.DataFrame()
    df = pd.DataFrame(selected)
    display_columns = [
        "acca_rank",
        "course",
        "off_time",
        "race",
        "pick_type",
        "horse",
        "place_rating",
        "place_probability",
        "place_value_edge",
        "odds",
        "confidence",
        "field_size",
        "result",
        "outcome",
        "red_flags",
        "explanation",
    ]
    return df[[column for column in display_columns if column in df.columns]]


def ian_index_weights_dataframe() -> pd.DataFrame:
    labels = {
        "ability_timeform": "Ability (Timeform)",
        "speed_beyer_topspeed": "Speed (Beyer + Topspeed)",
        "class_rpr": "Class (RPR)",
        "pace_race_shape": "Pace / Race shape",
        "value_hugh_taylor": "Value",
        "trainer_intent": "Trainer intent",
        "jockey": "Jockey",
        "course_going": "Course / Going",
    }
    return pd.DataFrame(
        [{"principle": labels[key], "weight": f"{weight}%"} for key, weight in IAN_INDEX_V4_WEIGHTS.items()]
    )


def _components(item: RunnerScore) -> dict[str, IanIndexComponent]:
    return {
        "ability_timeform": _ability_timeform(item),
        "speed_beyer_topspeed": _speed_beyer_topspeed(item),
        "class_rpr": _class_rpr(item),
        "pace_race_shape": _pace_race_shape(item),
        "value_hugh_taylor": _value_hugh_taylor(item),
        "trainer_intent": _trainer_intent(item),
        "jockey": _jockey(item),
        "course_going": _course_going(item),
    }


def _ability_timeform(item: RunnerScore) -> IanIndexComponent:
    value = _metric_value(item.runner.source_payload, ("timeform_rating", "timeform", "timeform_score", "tf_rating"))
    if value is not None:
        return IanIndexComponent(_rating_to_score(value), 0.9, "ok", "Timeform-style rating imported")
    if item.runner.official_rating is not None:
        return IanIndexComponent(_rating_to_score(item.runner.official_rating), 0.55, "partial", "No Timeform; official rating proxy")
    return IanIndexComponent(max(35.0, min(72.0, item.total_score)), 0.4, "partial", "No Timeform; model form proxy")


def _speed_beyer_topspeed(item: RunnerScore) -> IanIndexComponent:
    values = [
        _metric_value(item.runner.source_payload, keys)
        for keys in (
            ("beyer", "beyer_speed", "beyer_speed_figure"),
            ("topspeed", "top_speed", "top_speed_rating"),
            ("speed_figure", "speed_rating", "last_speed"),
        )
    ]
    values = [value for value in values if value is not None]
    if values:
        return IanIndexComponent(sum(_rating_to_score(value) for value in values) / len(values), 0.85, "ok", "Imported speed figure")
    recent_strength = _recent_form_strength(item.runner.recent_form)
    return IanIndexComponent(42.0 + recent_strength * 28.0, 0.35, "partial", "No Beyer/Topspeed; recent form proxy")


def _class_rpr(item: RunnerScore) -> IanIndexComponent:
    value = _metric_value(item.runner.source_payload, ("rpr", "racing_post_rating", "official_rpr", "last_rpr"))
    if value is not None:
        return IanIndexComponent(_rating_to_score(value), 0.88, "ok", "RPR imported")
    if item.runner.official_rating is not None:
        return IanIndexComponent(_rating_to_score(item.runner.official_rating), 0.55, "partial", "No RPR; official rating proxy")
    race_class = _race_class_number(item.runner.race_class)
    if race_class is not None:
        return IanIndexComponent(72.0 - race_class * 3.0, 0.35, "partial", "No RPR; race grade proxy")
    return IanIndexComponent(50.0, 0.2, "missing", "No class rating available")


def _pace_race_shape(item: RunnerScore) -> IanIndexComponent:
    value = _metric_value(item.runner.source_payload, ("pace_rating", "early_speed", "run_style_score", "pace_score"))
    draw = item.runner.draw
    field_size = item.runner.field_size
    parts = []
    confidence = 0.35
    explanation = "Draw/field-size race-shape proxy"
    if value is not None:
        parts.append(_rating_to_score(value))
        confidence = 0.75
        explanation = "Imported pace/race-shape signal"
    if draw is not None and field_size:
        draw_ratio = draw / max(field_size, 1)
        if draw_ratio <= 0.33:
            parts.append(66.0)
        elif draw_ratio <= 0.72:
            parts.append(60.0)
        else:
            parts.append(47.0)
    if not parts:
        return IanIndexComponent(50.0, 0.25, "missing", "No pace or draw signal available")
    return IanIndexComponent(sum(parts) / len(parts), confidence, "ok" if value is not None else "partial", explanation)


def _value_hugh_taylor(item: RunnerScore) -> IanIndexComponent:
    edge = item.place_value_edge
    probability = item.place_probability or 0.0
    odds = _decimal_odds(item.runner.current_odds)
    if edge is None and odds is None:
        return IanIndexComponent(46.0 + probability * 35.0, 0.35, "partial", "No odds; place chance only")
    edge_part = max(-0.12, min(0.22, edge or 0.0)) * 155.0
    odds_part = 4.0 if odds is not None and 5.0 <= odds <= 16.0 else -4.0 if odds is not None and odds >= 25.0 else 0.0
    score = 50.0 + probability * 24.0 + edge_part + odds_part
    quality = "ok" if edge is not None else "partial"
    return IanIndexComponent(_clip(score), 0.82 if edge is not None else 0.45, quality, "Place value versus available odds")


def _trainer_intent(item: RunnerScore) -> IanIndexComponent:
    payload = item.runner.source_payload
    ae = _metric_value(payload, ("trainer_ae", "trainer_a/e", "trainer_actual_expected"))
    strike_rate = _metric_value(payload, ("trainer_strike_rate", "trainer_win_pct", "trainer_win_percentage"))
    course_sr = _metric_value(payload, ("trainer_course_sr", "trainer_course_strike_rate"))
    values = []
    if ae is not None:
        values.append(50.0 + max(-0.35, min(0.45, ae - 1.0)) * 70.0)
    if strike_rate is not None:
        values.append(_percentage_score(strike_rate, 6.0, 24.0))
    if course_sr is not None:
        values.append(_percentage_score(course_sr, 5.0, 22.0))
    if values:
        return IanIndexComponent(_clip(sum(values) / len(values)), 0.78, "ok", "Trainer intent/profile imported")
    if item.runner.trainer:
        return IanIndexComponent(52.0, 0.3, "partial", "Trainer known; no intent stats imported")
    return IanIndexComponent(45.0, 0.2, "missing", "Trainer signal unavailable")


def _jockey(item: RunnerScore) -> IanIndexComponent:
    payload = item.runner.source_payload
    ae = _metric_value(payload, ("jockey_ae", "jockey_a/e", "jockey_actual_expected"))
    strike_rate = _metric_value(payload, ("jockey_strike_rate", "jockey_win_pct", "jockey_win_percentage"))
    values = []
    if ae is not None:
        values.append(50.0 + max(-0.35, min(0.45, ae - 1.0)) * 70.0)
    if strike_rate is not None:
        values.append(_percentage_score(strike_rate, 5.0, 22.0))
    if item.runner.jockey_claim:
        values.append(56.0)
    if values:
        return IanIndexComponent(_clip(sum(values) / len(values)), 0.7, "ok", "Jockey profile imported")
    if item.runner.jockey:
        return IanIndexComponent(51.0, 0.3, "partial", "Jockey known; no suitability stats imported")
    return IanIndexComponent(45.0, 0.2, "missing", "Jockey signal unavailable")


def _course_going(item: RunnerScore) -> IanIndexComponent:
    payload = item.runner.source_payload
    course_metric = _metric_value(payload, ("course_win_pct", "course_place_pct", "course_strike_rate"))
    going_metric = _metric_value(payload, ("going_win_pct", "going_place_pct", "going_strike_rate"))
    values = []
    if course_metric is not None:
        values.append(_percentage_score(course_metric, 5.0, 30.0))
    if going_metric is not None:
        values.append(_percentage_score(going_metric, 5.0, 30.0))
    flags = " ".join(item.red_flags).lower()
    if "going concern" in flags:
        values.append(34.0)
    if values:
        return IanIndexComponent(_clip(sum(values) / len(values)), 0.7, "ok", "Course/going suitability imported")
    component = _component_score(item, "course_suitability", 10)
    if component is not None:
        return IanIndexComponent(component, 0.45, "partial", "Course suitability model proxy")
    return IanIndexComponent(50.0, 0.25, "missing", "No course/going history signal available")


def _data_quality(components: dict[str, IanIndexComponent]) -> str:
    qualities = [component.data_quality for component in components.values()]
    if all(quality == "ok" for quality in qualities):
        return "ok"
    if qualities.count("missing") >= 3:
        return "weak"
    return "partial"


def _explanation(components: dict[str, IanIndexComponent], item: RunnerScore) -> str:
    strongest = sorted(components.items(), key=lambda pair: pair[1].score, reverse=True)[:3]
    notes = [f"{name.replace('_', ' ')}: {component.explanation}" for name, component in strongest]
    if item.red_flags:
        notes.append("cautions: " + "; ".join(item.red_flags[:2]))
    return " | ".join(notes)


def _component_score(item: RunnerScore, name: str, model_weight: float) -> float | None:
    for component in item.components:
        if component.name == name:
            return _clip(component.score / model_weight * 100.0)
    return None


def _field_size_for_row(row: pd.Series, scores: list[RunnerScore]) -> int:
    for item in scores:
        runner = item.runner
        if (
            runner.horse == row.get("horse")
            and runner.course == row.get("course")
            and runner.off_time == row.get("off_time")
            and runner.race_name == row.get("race")
        ):
            return int(runner.field_size or 0)
    return 0


def _trial_race_key(row: dict[str, Any]) -> tuple[str, str]:
    return (_normalise_identity(row.get("course")), _normalise_off_time(row.get("off_time")))


def _result_text(item: RunnerScore) -> str:
    position = _finish_position(item.runner.source_payload)
    return str(position) if position is not None else "Awaiting result"


def _place_outcome(item: RunnerScore) -> str:
    position = _finish_position(item.runner.source_payload)
    if position is None:
        return "Awaiting result"
    cutoff = _place_cutoff(item.runner.field_size)
    if position == 1:
        return "WIN"
    if position <= cutoff:
        return "PLACED"
    if position == cutoff + 1:
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
    for key in ("result_position", "finish_position", "finishing_position", "position", "pos", "place"):
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


def _normalise_identity(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    return " ".join(text.lower().replace("'", "").split())


def _normalise_off_time(value: Any) -> str:
    text = str(value or "").strip().lower().replace(".", ":")
    parts = text.split(":")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        hour = int(parts[0])
        minute = int(parts[1])
        if hour > 12:
            hour -= 12
        return f"{hour:02d}:{minute:02d}"
    return text


def _metric_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    key_lookup = {key.lower().replace(" ", "_") for key in keys}
    for key, value in _walk(payload):
        normalised = key.lower().replace(" ", "_")
        if normalised in key_lookup:
            number = _number(value)
            if number is not None:
                return number
    return None


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            rows.extend(_walk(item, str(key)))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_walk(item, prefix))
        return rows
    return [(prefix, value)]


def _number(value: Any) -> float | None:
    if value in (None, "", "Unavailable"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _rating_to_score(value: float) -> float:
    if value <= 1.0:
        return _clip(value * 100.0)
    if value <= 10.0:
        return _clip(value * 10.0)
    if value <= 100.0:
        return _clip(value)
    return _clip((value - 45.0) / 85.0 * 100.0)


def _percentage_score(value: float, low: float, high: float) -> float:
    if value <= 1.0:
        value *= 100.0
    return _clip((value - low) / (high - low) * 100.0)


def _recent_form_strength(form: str | None) -> float:
    if not form:
        return 0.0
    digits = [int(ch) for ch in form[:5] if ch.isdigit()]
    if not digits:
        return 0.0
    placed = sum(1 for position in digits if position <= 3)
    wins = sum(1 for position in digits if position == 1)
    return min(1.0, placed / len(digits) * 0.75 + wins / len(digits) * 0.25)


def _race_class_number(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _decimal_odds(value: str | None) -> float | None:
    if not value:
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


def _format_probability(probability: float | None) -> str:
    if probability is None:
        return "Unavailable"
    return f"{probability * 100:.1f}%"


def _format_edge(edge: float | None) -> str:
    if edge is None:
        return "Needs odds"
    return f"{edge * 100:+.1f} pts"


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))
