from __future__ import annotations

from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore


def undervalued_edge_dataframe(scores: list[RunnerScore], limit: int = 12) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        odds = _decimal_odds(runner.current_odds)
        if odds is None:
            continue
        win_edge = item.win_value_edge if item.win_value_edge is not None else -1.0
        place_edge = item.place_value_edge if item.place_value_edge is not None else -1.0
        best_edge = max(win_edge, place_edge)
        if best_edge <= 0:
            continue

        evidence = _edge_evidence(item)
        edge_score = (
            best_edge * 170.0
            + item.confidence * 22.0
            + item.total_score * 0.28
            + len(evidence) * 4.0
            + _speed_signal_bonus(runner.source_payload)
            - min(24.0, len(item.red_flags) * 4.0)
        )
        if item.confidence < 0.45:
            edge_score -= 8.0
        if not evidence:
            edge_score -= 10.0

        rows.append(
            {
                "horse": runner.horse,
                "course": runner.course,
                "off_time": runner.off_time,
                "race": runner.race_name,
                "edge_type": "EW/place" if place_edge >= win_edge else "Win",
                "odds": runner.current_odds or "Unavailable",
                "fair_win_odds": _format_decimal(item.fair_win_odds),
                "fair_place_odds": _format_decimal(item.fair_place_odds),
                "win_edge": _format_edge(item.win_value_edge),
                "place_edge": _format_edge(item.place_value_edge),
                "edge_score": round(edge_score, 2),
                "confidence": item.confidence,
                "model_score": item.total_score,
                "evidence": "; ".join(evidence[:5]) if evidence else "Price edge only; needs more proof",
                "cautions": "; ".join(item.red_flags[:3]) if item.red_flags else "None",
                "_edge_score": edge_score,
                "_best_edge": best_edge,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        by=["_edge_score", "_best_edge", "confidence"],
        ascending=[False, False, False],
    ).head(limit)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["_edge_score", "_best_edge"])


def _edge_evidence(item: RunnerScore) -> list[str]:
    runner = item.runner
    payload = runner.source_payload
    evidence: list[str] = []
    setup = _setup_hits(runner, _history_candidates(payload))
    if setup:
        evidence.append("Proven setup: " + ", ".join(setup))
    if item.place_probability >= 0.42:
        evidence.append(f"Place profile {_format_probability(item.place_probability)}")
    if item.win_probability >= 0.18 and (item.win_value_edge or 0.0) > 0:
        evidence.append(f"Win chance {_format_probability(item.win_probability)}")
    if _component_score(item, "current_performance") >= 6.4 or _recent_form_strength(runner.recent_form) >= 0.55:
        evidence.append("Recent form strength")
    if _component_score(item, "course_suitability") >= 6.2:
        evidence.append("Course or setup suitability")
    if _component_score(item, "distance_suitability") >= 6.2:
        evidence.append("Trip suitability")
    speed = _speed_signal(payload)
    if speed:
        evidence.append(speed)
    trainer_jockey = _trainer_jockey_signal(payload)
    if trainer_jockey:
        evidence.append(trainer_jockey)
    market = _market_signal(payload, runner.current_odds)
    if market.startswith("Supported"):
        evidence.append(market)
    return evidence


def _component_score(item: RunnerScore, name: str) -> float:
    for component in item.components:
        if component.name == name:
            return component.score
    return 0.0


def _recent_form_strength(form: str | None) -> float:
    if not form:
        return 0.0
    digits = [int(ch) for ch in form[:5] if ch.isdigit()]
    if not digits:
        return 0.0
    in_frame = sum(1 for position in digits if position <= 3)
    wins = sum(1 for position in digits if position == 1)
    return min(1.0, in_frame / len(digits) * 0.75 + wins / len(digits) * 0.25)


def _setup_hits(runner: Any, history: list[Any]) -> list[str]:
    hits = []
    if any(_history_match(item, "course", runner.course) for item in history):
        hits.append("course")
    if any(_history_match(item, "going", runner.going) or _history_match(item, "ground", runner.going) for item in history):
        hits.append("going")
    if any(_history_match(item, "surface", runner.surface) for item in history):
        hits.append("surface")
    if any(_history_match(item, "distance", runner.distance) or _history_match(item, "dist", runner.distance) for item in history):
        hits.append("distance")
    return hits


def _history_candidates(payload: dict[str, Any]) -> list[Any]:
    history_keys = (
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
    for key in history_keys:
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value[:3])
        elif value:
            candidates.append(value)
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        for key in history_keys:
            value = source_runner.get(key)
            if isinstance(value, list):
                candidates.extend(value[:3])
            elif value:
                candidates.append(value)
    return candidates


def _history_match(item: Any, key: str, expected: str | None) -> bool:
    if not isinstance(item, dict) or not expected:
        return False
    position = _parse_position(item.get("position") or item.get("pos"))
    if position is None or position > 3:
        return False
    actual = item.get(key)
    if actual in (None, ""):
        return False
    actual_text = " ".join(str(actual).lower().strip().split())
    expected_text = " ".join(str(expected).lower().strip().split())
    return bool(actual_text and expected_text and (actual_text in expected_text or expected_text in actual_text))


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


def _trainer_jockey_signal(payload: dict[str, Any]) -> str | None:
    trainer_ae = _metric_value(payload, ("trainer_ae", "trainer_a/e", "trainer_actual_expected"))
    trainer_sr = _metric_value(payload, ("trainer_strike_rate", "trainer_win_pct", "trainer_win_percentage"))
    jockey_ae = _metric_value(payload, ("jockey_ae", "jockey_a/e", "jockey_actual_expected"))
    jockey_sr = _metric_value(payload, ("jockey_strike_rate", "jockey_win_pct", "jockey_win_percentage"))
    signals = []
    if trainer_ae is not None and trainer_ae >= 1.05:
        signals.append("trainer A/E")
    elif trainer_sr is not None and trainer_sr >= 14:
        signals.append("trainer strike-rate")
    if jockey_ae is not None and jockey_ae >= 1.05:
        signals.append("jockey A/E")
    elif jockey_sr is not None and jockey_sr >= 14:
        signals.append("jockey strike-rate")
    if not signals:
        return None
    return "Positive " + " and ".join(signals)


def _speed_signal(payload: dict[str, Any]) -> str | None:
    speed = _speed_value(payload)
    if speed is None:
        return None
    if speed >= 90:
        return f"Strong speed figure {speed:g}"
    if speed >= 75:
        return f"Useful speed figure {speed:g}"
    return None


def _speed_signal_bonus(payload: dict[str, Any]) -> float:
    speed = _speed_value(payload)
    if speed is None:
        return 0.0
    if speed >= 90:
        return 7.0
    if speed >= 75:
        return 4.0
    return 0.0


def _speed_value(payload: dict[str, Any]) -> float | None:
    return _metric_value(
        payload,
        (
            "speed_figure",
            "speed_rating",
            "topspeed",
            "top_speed",
            "rpr",
            "official_speed",
            "timeform_rating",
        ),
    )


def _metric_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for source in _payload_sources(payload):
        value = _metric_from_source(source, keys)
        if value is not None:
            return value
    return None


def _metric_from_source(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalised_keys = {key.lower().replace(" ", "_").replace("-", "_"): key for key in source}
    for key in keys:
        lookup = key.lower().replace(" ", "_").replace("-", "_")
        raw_key = normalised_keys.get(lookup)
        if raw_key is not None:
            parsed = _parse_float(source.get(raw_key))
            if parsed is not None:
                return parsed
    for value in source.values():
        if isinstance(value, dict):
            parsed = _metric_from_source(value, keys)
            if parsed is not None:
                return parsed
    return None


def _payload_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [payload]
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        sources.append(source_runner)
    return sources


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _market_signal(payload: dict[str, Any], current_odds: str | None) -> str:
    current = _decimal_odds(current_odds)
    opening = _opening_odds(payload)
    if current is None or opening is None or opening <= 1:
        return "No market move data"
    move = (current - opening) / opening
    if move <= -0.12:
        return f"Supported: {opening:.2f} to {current:.2f}"
    if move >= 0.2:
        return f"Drifting: {opening:.2f} to {current:.2f}"
    return f"Stable: {opening:.2f} to {current:.2f}"


def _opening_odds(payload: dict[str, Any]) -> float | None:
    for source in _payload_sources(payload):
        for key in ("opening_odds", "open_odds", "early_odds", "first_odds", "opening_price", "open_price"):
            odds = _decimal_odds(str(source.get(key))) if source.get(key) not in (None, "") else None
            if odds is not None:
                return odds
        odds_rows = source.get("odds") or source.get("odds_history") or source.get("price_history")
        if isinstance(odds_rows, list):
            for row in odds_rows:
                if not isinstance(row, dict):
                    continue
                odds = _decimal_odds(
                    str(
                        row.get("opening")
                        or row.get("open")
                        or row.get("first")
                        or row.get("fractional_open")
                        or row.get("decimal_open")
                    )
                )
                if odds is not None:
                    return odds
    return None


def _decimal_odds(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return float(num) / float(den) + 1.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_edge(edge: float | None) -> str:
    if edge is None:
        return "Unavailable"
    return f"{edge * 100:+.1f} pts"


def _format_probability(probability: float | None) -> str:
    if probability is None:
        return "Unavailable"
    return f"{probability * 100:.1f}%"


def _format_decimal(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2f}"
