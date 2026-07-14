from __future__ import annotations

import pandas as pd

from ian_racing_model.calibration_rules import each_way_gate, evidence_profile
from ian_racing_model.domain import RunnerScore
from ian_racing_model.edge_lab import _decimal_odds, _format_edge, _format_percent


def ew_accumulator_dataframe(scores: list[RunnerScore], limit: int = 6) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        if runner.is_non_runner:
            continue
        odds = _decimal_odds(runner.current_odds)
        if odds is not None and odds < 2.5:
            continue
        gate = each_way_gate(item)
        if not gate["qualified"]:
            continue
        profile = evidence_profile(item)
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
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["_score", "_place_probability", "_confidence"],
        ascending=[False, False, False],
    ).head(limit)
    df["acca_rank"] = range(1, len(df) + 1)
    return df.drop(columns=["_score", "_place_probability", "_confidence"])


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
