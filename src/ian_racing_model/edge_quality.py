from __future__ import annotations

from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore
from ian_racing_model.ui import (
    _decimal_odds,
    _each_way_selection_score,
    _edge_evidence,
    _format_edge,
    _format_probability,
    _is_each_way_candidate,
    _market_signal,
    _ratio_text,
    _winner_selection_score,
)


def edge_quality_dataframe(scores: list[RunnerScore], limit: int = 30) -> pd.DataFrame:
    rows = []
    for item in scores:
        if item.runner.is_non_runner:
            continue
        win_score = _winner_selection_score(item)
        place_score = _each_way_selection_score(item)
        win_quality = _edge_quality_score(item, "win", win_score)
        place_quality = _edge_quality_score(item, "place", place_score)
        best_type = "Place/EW" if place_quality >= win_quality else "Win"
        best_score = max(win_quality, place_quality)
        rows.append(_edge_quality_row(item, win_quality, place_quality, best_score, best_type))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        by=["edge_quality_score", "evidence_count", "confidence"],
        ascending=[False, False, False],
    ).head(limit)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def probability_calibration_dataframe(picks_df: pd.DataFrame) -> pd.DataFrame:
    if picks_df.empty:
        return pd.DataFrame()
    required = {"pick_type", "outcome", "win_probability", "place_probability"}
    if not required.issubset(set(picks_df.columns)):
        return pd.DataFrame()

    settled = picks_df[~picks_df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()

    settled["_target_probability"] = settled.apply(_target_probability, axis=1)
    settled = settled[settled["_target_probability"].notna()].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["probability_band"] = settled["_target_probability"].map(_probability_band)
    settled["_hit"] = settled.apply(_target_hit, axis=1)

    rows = []
    for (pick_type, band), bucket in settled.groupby(["pick_type", "probability_band"], dropna=False):
        total = len(bucket)
        hits = int(bucket["_hit"].sum())
        expected = float(bucket["_target_probability"].mean())
        actual = hits / total if total else 0.0
        rows.append(
            {
                "pick_type": pick_type,
                "probability_band": band,
                "settled": total,
                "expected_rate": _format_probability(expected),
                "actual_rate": _ratio_text(hits, total),
                "calibration_gap": f"{(actual - expected) * 100:+.1f} pts",
                "edge_read": _calibration_read(actual, expected, total),
            }
        )
    return pd.DataFrame(rows).sort_values(["pick_type", "probability_band"])


def _edge_quality_row(
    item: RunnerScore,
    win_quality: float,
    place_quality: float,
    best_score: float,
    best_type: str,
) -> dict[str, Any]:
    runner = item.runner
    profile = _safe_evidence_profile(item)
    gate = _safe_each_way_gate(item)
    return {
        "horse": runner.horse,
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "best_edge_type": best_type,
        "edge_label": _edge_quality_label(item, best_score, profile, gate, best_type),
        "edge_quality_score": round(best_score, 2),
        "win_quality_score": round(win_quality, 2),
        "place_quality_score": round(place_quality, 2),
        "odds": runner.current_odds or "Unavailable",
        "win_probability": _format_probability(item.win_probability),
        "place_probability": _format_probability(item.place_probability),
        "win_edge": _format_edge(item.win_value_edge),
        "place_edge": _format_edge(item.place_value_edge),
        "confidence": item.confidence,
        "confidence_gate": _confidence_gate(item, profile),
        "evidence_count": profile["count"],
        "evidence_pillars": ", ".join(profile["pillars"]) if profile["pillars"] else "None",
        "market_state": str(profile["market"]).title(),
        "closing_value_signal": _closing_value_signal(runner.source_payload, runner.current_odds),
        "red_flags": "; ".join(item.red_flags[:3]) if item.red_flags else "None",
    }


def _edge_quality_score(item: RunnerScore, mode: str, selection_score: float) -> float:
    profile = _safe_evidence_profile(item)
    evidence_bonus = min(18.0, profile["count"] * 3.5)
    red_flag_drag = min(22.0, len(item.red_flags) * 4.0)
    confidence_drag = max(0.0, 0.56 - item.confidence) * 45.0
    market_bonus = 5.0 if profile["market"] == "supported" else -5.0 if profile["market"] == "drifting" else 0.0

    if mode == "place":
        value_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
        probability = item.place_probability or 0.0
        gate_drag = 0.0 if _safe_each_way_gate(item)["qualified"] else 24.0
    else:
        value_edge = item.win_value_edge if item.win_value_edge is not None else 0.0
        probability = item.win_probability or 0.0
        gate_drag = 10.0 if value_edge < 0 and item.confidence < 0.66 else 0.0

    return (
        selection_score * 0.45
        + probability * 55.0
        + max(-0.08, min(0.18, value_edge)) * 95.0
        + evidence_bonus
        + item.confidence * 18.0
        + market_bonus
        - red_flag_drag
        - confidence_drag
        - gate_drag
    )


def _edge_quality_label(item: RunnerScore, score: float, profile: dict[str, Any], gate: dict[str, Any], best_type: str) -> str:
    if item.confidence < 0.5 or profile["count"] == 0:
        return "Speculative only"
    if best_type == "Place/EW" and not gate["qualified"]:
        return "Blocked by EW gate"
    if score >= 86 and profile["count"] >= 3:
        return "High-confidence edge"
    if score >= 74 and profile["count"] >= 2:
        return "Qualified edge"
    if score >= 64:
        return "Watchlist edge"
    return "Avoid/weak edge"


def _confidence_gate(item: RunnerScore, profile: dict[str, Any]) -> str:
    if item.confidence >= 0.66 and profile["count"] >= 2:
        return "Strong"
    if item.confidence >= 0.56 and profile["count"] >= 1:
        return "Usable"
    if item.confidence >= 0.5:
        return "Thin evidence"
    return "Low confidence"


def _safe_evidence_profile(item: RunnerScore) -> dict[str, Any]:
    try:
        from ian_racing_model.calibration_rules import evidence_profile

        profile = evidence_profile(item)
        return {
            "count": int(profile.get("count", 0)),
            "pillars": list(profile.get("pillars", [])),
            "market": str(profile.get("market", "unknown")),
        }
    except Exception:
        evidence = _edge_evidence(item)
        market = _market_signal(item.runner.source_payload, item.runner.current_odds)
        return {
            "count": len(evidence),
            "pillars": evidence,
            "market": "supported" if market.startswith("Supported") else "drifting" if market.startswith("Drifting") else "stable",
        }


def _safe_each_way_gate(item: RunnerScore) -> dict[str, Any]:
    try:
        from ian_racing_model.calibration_rules import each_way_gate

        gate = each_way_gate(item)
        return {"qualified": bool(gate.get("qualified")), "reason": str(gate.get("reason", ""))}
    except Exception:
        qualified = _is_each_way_candidate(item)
        return {"qualified": qualified, "reason": "EW base gate passed" if qualified else "EW base gate failed"}


def _target_probability(row: pd.Series) -> float | None:
    if str(row.get("pick_type")) == "Best EW pick":
        return _parse_probability_text(row.get("place_probability"))
    return _parse_probability_text(row.get("win_probability"))


def _target_hit(row: pd.Series) -> bool:
    outcome = str(row.get("outcome", "")).upper()
    if str(row.get("pick_type")) == "Best EW pick":
        return outcome in {"WIN", "PLACED"}
    return outcome == "WIN"


def _parse_probability_text(value: Any) -> float | None:
    if value in (None, "", "Unavailable"):
        return None
    try:
        return float(str(value).strip().replace("%", "")) / 100.0
    except ValueError:
        return None


def _probability_band(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    if value < 0.2:
        return "Under 20%"
    if value < 0.3:
        return "20% to 29%"
    if value < 0.4:
        return "30% to 39%"
    if value < 0.5:
        return "40% to 49%"
    return "50%+"


def _calibration_read(actual: float, expected: float, total: int) -> str:
    if total < 5:
        return "Small sample"
    gap = actual - expected
    if gap >= 0.08:
        return "Outperforming"
    if gap <= -0.08:
        return "Underperforming"
    return "Well calibrated"


def _closing_value_signal(payload: dict[str, Any], current_odds: str | None) -> str:
    current = _decimal_odds(current_odds)
    closing = _closing_odds(payload)
    if current is None or closing is None or closing <= 1:
        return "No closing price"
    clv = (current / closing) - 1.0
    if clv >= 0.08:
        return f"Beat close {_format_edge(clv)}"
    if clv <= -0.08:
        return f"Lost value {_format_edge(clv)}"
    return f"Held price {_format_edge(clv)}"


def _closing_odds(payload: dict[str, Any]) -> float | None:
    for source in _closing_sources(payload):
        odds = _closing_odds_from_tree(source)
        if odds is not None:
            return odds
    return None


def _closing_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [payload]
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        sources.append(source_runner)
    for key in ("result_payload", "result_runner", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            sources.append(value)
            runner_value = value.get("result_runner") or value.get("runner")
            if isinstance(runner_value, dict):
                sources.append(runner_value)
    return sources


def _closing_odds_from_tree(payload: dict[str, Any], depth: int = 0) -> float | None:
    if depth > 3:
        return None
    direct_keys = (
        "sp_dec",
        "bsp",
        "starting_price_decimal",
        "starting_price_dec",
        "sp_decimal",
        "closing_odds",
        "closing_price",
        "closing_price_decimal",
        "result_sp",
        "sp",
        "starting_price",
        "starting_price_fractional",
    )
    for key in direct_keys:
        odds = _decimal_odds(str(payload.get(key))) if payload.get(key) not in (None, "") else None
        if odds is not None:
            return odds
    for value in payload.values():
        if isinstance(value, dict):
            odds = _closing_odds_from_tree(value, depth + 1)
            if odds is not None:
                return odds
    return None
