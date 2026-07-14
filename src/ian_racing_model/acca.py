from __future__ import annotations

import pandas as pd

from ian_racing_model.domain import RunnerScore
from ian_racing_model.edge_lab import _decimal_odds, _format_edge, _format_percent

try:
    from ian_racing_model.calibration_rules import each_way_gate, evidence_profile
except ImportError:
    each_way_gate = None
    evidence_profile = None


def ew_accumulator_dataframe(scores: list[RunnerScore], limit: int = 6) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        if runner.is_non_runner:
            continue
        if runner.field_size is not None and runner.field_size < 8:
            continue
        odds = _decimal_odds(runner.current_odds)
        if odds is not None and odds < 2.5:
            continue
        gate = _each_way_gate(item)
        if not gate["qualified"]:
            continue
        profile = _evidence_profile(item)
        place_probability = item.place_probability or 0.0
        place_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
        red_flag_drag = min(18.0, len(item.red_flags) * 4.5)
        price_drag = _accumulator_price_drag(odds)
        accumulator_score = (
            place_probability * 100.0
            + item.confidence * 24.0
            + max(-0.04, min(0.18, place_edge)) * 120.0
            + min(18.0, profile["count"] * 4.0)
            - red_flag_drag
            - price_drag
        )
        if place_probability < 0.34 and profile["count"] < 3:
            continue
        rows.append(
            {
                "acca_rank": 0,
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "field_size": runner.field_size or 0,
                "odds": runner.current_odds or "Unavailable",
                "place_probability": _format_percent(place_probability),
                "place_edge": _format_edge(item.place_value_edge),
                "confidence": item.confidence,
                "acca_score": round(accumulator_score, 2),
                "evidence_count": profile["count"],
                "evidence_pillars": ", ".join(profile["pillars"]) if profile["pillars"] else "None",
                "gate_reason": gate["reason"],
                "warnings": "; ".join(item.red_flags[:2]) if item.red_flags else "None",
                "_score": accumulator_score,
                "_place_probability": place_probability,
                "_confidence": item.confidence,
                "_race_key": (runner.meeting_date, runner.course, runner.off_time, runner.race_name),
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows).sort_values(
        ["_score", "_place_probability", "_confidence"],
        ascending=[False, False, False],
    )
    selected = []
    used_races = set()
    for row in ranked.to_dict("records"):
        race_key = row["_race_key"]
        if race_key in used_races:
            continue
        selected.append(row)
        used_races.add(race_key)
        if len(selected) == limit:
            break
    df = pd.DataFrame(selected)
    df["acca_rank"] = range(1, len(df) + 1)
    return df.drop(columns=["_score", "_place_probability", "_confidence", "_race_key"])


def _accumulator_price_drag(odds: float | None) -> float:
    if odds is None:
        return 4.0
    if odds < 6.0:
        return 0.0
    if odds < 10.0:
        return 2.0
    if odds < 20.0:
        return 7.0
    return 16.0


def _each_way_gate(item: RunnerScore) -> dict:
    if each_way_gate is not None:
        return each_way_gate(item)
    odds = _decimal_odds(item.runner.current_odds)
    place_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
    place_probability = item.place_probability or 0.0
    profile = _evidence_profile(item)
    if odds is not None and odds < 5.0:
        return {"qualified": False, "reason": "too short for EW/value gate"}
    if odds is not None and odds >= 20.0:
        qualified = profile["count"] >= 3 and (place_edge >= 0.05 or place_probability >= 0.45)
        return {"qualified": qualified, "reason": "fallback 20.0+ evidence gate"}
    qualified = place_edge >= 0 or place_probability >= 0.38 or profile["count"] >= 2
    return {"qualified": qualified, "reason": "fallback EW evidence gate"}


def _evidence_profile(item: RunnerScore) -> dict:
    if evidence_profile is not None:
        return evidence_profile(item)
    pillars = []
    payload = item.runner.source_payload
    if item.confidence >= 0.6:
        pillars.append("confidence")
    if item.place_probability and item.place_probability >= 0.4:
        pillars.append("place profile")
    if item.place_value_edge is not None and item.place_value_edge >= 0.05:
        pillars.append("place edge")
    payload_text = " ".join(str(value).lower() for value in _flatten(payload))
    if "speed" in payload_text or "rpr" in payload_text:
        pillars.append("speed")
    if "trainer" in payload_text or "jockey" in payload_text:
        pillars.append("trainer/jockey")
    if "opening" in payload_text or "open_odds" in payload_text:
        pillars.append("market")
    return {"count": len(set(pillars)), "pillars": sorted(set(pillars)), "market": "unknown"}


def _flatten(value) -> list:
    if isinstance(value, dict):
        rows = []
        for item in value.values():
            rows.extend(_flatten(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten(item))
        return rows
    return [value]
