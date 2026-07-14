from __future__ import annotations

from typing import Any

from ian_racing_model.domain import RunnerScore


def install_calibration_rules(ui_module: Any) -> None:
    if getattr(ui_module, "_ian_calibration_rules_installed", False):
        return

    original_winner_score = ui_module._winner_selection_score
    original_each_way_score = ui_module._each_way_selection_score
    original_is_each_way_candidate = ui_module._is_each_way_candidate
    original_pick_row = ui_module._pick_row
    original_selection_screener_row = ui_module._selection_screener_row

    def winner_selection_score(item: RunnerScore) -> float:
        return original_winner_score(item) + calibration_adjustment(item, "Winner pick")[0] + winner_gate_adjustment(item)[0]

    def each_way_selection_score(item: RunnerScore) -> float:
        return original_each_way_score(item) + calibration_adjustment(item, "Best EW pick")[0] + ew_gate_adjustment(item)[0]

    def is_each_way_candidate(item: RunnerScore) -> bool:
        if not original_is_each_way_candidate(item):
            return False
        return each_way_gate(item)["qualified"]

    def pick_row(item: RunnerScore, pick_type: str, selection_score: float) -> dict[str, Any]:
        row = original_pick_row(item, pick_type, selection_score)
        adjustment, reason = full_calibration_adjustment(item, pick_type)
        row.update(_calibration_columns(item, pick_type, adjustment, reason))
        return row

    def selection_screener_row(item: RunnerScore, pick: str, selection_score: float) -> dict[str, Any]:
        row = original_selection_screener_row(item, pick, selection_score)
        pick_type = "Winner pick" if pick == "Winner" else "Best EW pick"
        adjustment, reason = full_calibration_adjustment(item, pick_type)
        row.update(_calibration_columns(item, pick_type, adjustment, reason))
        return row

    ui_module._winner_selection_score = winner_selection_score
    ui_module._each_way_selection_score = each_way_selection_score
    ui_module._is_each_way_candidate = is_each_way_candidate
    ui_module._pick_row = pick_row
    ui_module._selection_screener_row = selection_screener_row
    ui_module._ian_calibration_rules_installed = True


def full_calibration_adjustment(item: RunnerScore, pick_type: str) -> tuple[float, str]:
    base_adjustment, base_reason = calibration_adjustment(item, pick_type)
    if pick_type == "Winner pick":
        gate_adjustment, gate_reason = winner_gate_adjustment(item)
    else:
        gate_adjustment, gate_reason = ew_gate_adjustment(item)

    reasons = [reason for reason in (base_reason, gate_reason) if reason and reason != "No calibration adjustment"]
    if not reasons:
        return 0.0, "No calibration adjustment"
    return base_adjustment + gate_adjustment, "; ".join(reasons)


def calibration_adjustment(item: RunnerScore, pick_type: str) -> tuple[float, str]:
    odds = _decimal_odds(item.runner.current_odds)
    race_class = _normalise(item.runner.race_class)
    strong_evidence = _has_strong_evidence(item)
    reasons: list[str] = []
    adjustment = 0.0

    if pick_type == "Winner pick":
        if odds is not None and 6.0 <= odds < 10.0:
            adjustment += 6.0
            reasons.append("winner sweet spot 6.0 to 9.99")
        elif odds is not None and 10.0 <= odds < 20.0:
            penalty = -8.0 if strong_evidence else -14.0
            adjustment += penalty
            reasons.append("tightened winner 10.0 to 19.99")
        elif odds is not None and odds >= 20.0:
            penalty = -14.0 if strong_evidence else -24.0
            adjustment += penalty
            reasons.append("tightened winner 20.0+")
    elif pick_type == "Best EW pick":
        if odds is not None and 6.0 <= odds < 10.0:
            adjustment += 7.0
            reasons.append("EW positive pocket 6.0 to 9.99")
        elif odds is not None and 10.0 <= odds < 20.0:
            adjustment += 9.0
            reasons.append("EW positive pocket 10.0 to 19.99")

        if race_class == "class 2":
            adjustment += 5.0
            reasons.append("EW Class 2 pocket")
        elif race_class == "class 5":
            adjustment += 4.0
            reasons.append("EW Class 5 pocket")
        elif race_class in {"class 1", "class 6"}:
            penalty = -5.0 if strong_evidence else -12.0
            adjustment += penalty
            reasons.append(f"tightened EW {race_class.title()}")

    if not reasons:
        return 0.0, "No calibration adjustment"
    if strong_evidence and adjustment < 0:
        reasons.append("penalty softened by evidence")
    return adjustment, "; ".join(reasons)


def winner_gate_adjustment(item: RunnerScore) -> tuple[float, str]:
    odds = _decimal_odds(item.runner.current_odds)
    win_edge = item.win_value_edge if item.win_value_edge is not None else 0.0
    evidence = evidence_profile(item)
    adjustment = 0.0
    reasons: list[str] = []

    if win_edge < 0 and item.confidence < 0.66:
        adjustment -= 10.0
        reasons.append("winner needs stronger value proof")
    if odds is not None and odds < 3.0 and win_edge < 0.03:
        adjustment -= 8.0
        reasons.append("short winner price needs clear edge")
    if odds is not None and odds >= 10.0 and evidence["count"] < 2:
        adjustment -= 10.0
        reasons.append("winner price over 10.0 needs at least two evidence pillars")
    if item.red_flags:
        adjustment -= min(10.0, len(item.red_flags) * 2.5)
    if evidence["market"] == "supported" and win_edge >= 0:
        adjustment += 4.0
        reasons.append("market support confirms win case")

    if not reasons:
        return 0.0, ""
    return adjustment, "; ".join(reasons)


def ew_gate_adjustment(item: RunnerScore) -> tuple[float, str]:
    gate = each_way_gate(item)
    odds = _decimal_odds(item.runner.current_odds)
    adjustment = 0.0
    reasons: list[str] = []

    if not gate["qualified"]:
        adjustment -= 35.0
        reasons.append(gate["reason"])
    elif odds is not None and 6.0 <= odds < 20.0:
        adjustment += min(8.0, gate["evidence_count"] * 2.0)
        reasons.append("EW value has enough evidence support")
    elif odds is not None and odds >= 20.0:
        adjustment += min(6.0, gate["evidence_count"] * 1.5)
        reasons.append("big-price EW allowed by evidence gate")

    if gate["market"] == "supported":
        adjustment += 4.0
        reasons.append("market support")
    elif gate["market"] == "drifting" and not gate["qualified"]:
        adjustment -= 5.0
        reasons.append("drifting without enough evidence")

    if not reasons:
        return 0.0, ""
    return adjustment, "; ".join(reasons)


def each_way_gate(item: RunnerScore) -> dict[str, Any]:
    odds = _decimal_odds(item.runner.current_odds)
    place_edge = item.place_value_edge if item.place_value_edge is not None else 0.0
    place_probability = item.place_probability or 0.0
    race_class = _normalise(item.runner.race_class)
    evidence = evidence_profile(item)
    evidence_count = int(evidence["count"])

    if odds is not None and odds < 5.0:
        return _gate(False, "too short for EW/value gate", evidence)
    if odds is not None and odds >= 20.0:
        qualified = evidence_count >= 3 and (place_edge >= 0.05 or place_probability >= 0.45)
        reason = "20.0+ EW needs 3 evidence pillars plus place edge" if not qualified else "20.0+ EW evidence gate passed"
        return _gate(qualified, reason, evidence)
    if race_class in {"class 1", "class 6"}:
        qualified = evidence_count >= 2 and place_edge >= 0.03
        reason = f"{race_class.title()} EW needs stronger evidence" if not qualified else f"{race_class.title()} EW evidence gate passed"
        return _gate(qualified, reason, evidence)
    if odds is not None and 10.0 <= odds < 20.0:
        qualified = evidence_count >= 1 and (place_edge > 0 or place_probability >= 0.40)
        reason = "10.0 to 19.99 EW needs at least one evidence pillar" if not qualified else "10.0 to 19.99 EW gate passed"
        return _gate(qualified, reason, evidence)
    qualified = place_edge >= 0 or place_probability >= 0.38 or evidence_count >= 2
    reason = "EW candidate needs place edge, place chance, or two evidence pillars" if not qualified else "EW evidence gate passed"
    return _gate(qualified, reason, evidence)


def evidence_profile(item: RunnerScore) -> dict[str, Any]:
    pillars: list[str] = []
    if _component_score(item, "course_suitability") >= 6.2:
        pillars.append("course")
    if _component_score(item, "distance_suitability") >= 6.2:
        pillars.append("distance")
    if _component_score(item, "current_performance") >= 6.4 or _recent_form_strength(item.runner.recent_form) >= 0.55:
        pillars.append("recent form")
    if _setup_history_hits(item) >= 2:
        pillars.append("proven setup")
    if _speed_signal(item.runner.source_payload):
        pillars.append("speed")
    if _trainer_jockey_signal(item.runner.source_payload):
        pillars.append("trainer/jockey")
    market = _market_state(item.runner.source_payload, item.runner.current_odds)
    if market == "supported":
        pillars.append("market support")
    if _last_time_outsider_signal(item):
        pillars.append("last-time outsider placed")
    return {
        "count": len(set(pillars)),
        "pillars": sorted(set(pillars)),
        "market": market,
    }


def _calibration_columns(item: RunnerScore, pick_type: str, adjustment: float, reason: str) -> dict[str, Any]:
    profile = evidence_profile(item)
    gate = each_way_gate(item) if pick_type == "Best EW pick" else _gate(True, "winner gate uses score penalties", profile)
    return {
        "calibration_adjustment": round(adjustment, 2),
        "calibration_reason": reason,
        "evidence_gate": "Passed" if gate["qualified"] else "Blocked",
        "evidence_count": profile["count"],
        "evidence_pillars": ", ".join(profile["pillars"]) if profile["pillars"] else "None",
        "market_state": profile["market"].title(),
    }


def _gate(qualified: bool, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "qualified": qualified,
        "reason": reason,
        "evidence_count": evidence["count"],
        "evidence_pillars": evidence["pillars"],
        "market": evidence["market"],
    }


def _has_strong_evidence(item: RunnerScore) -> bool:
    if evidence_profile(item)["count"] >= 2:
        return True
    if item.confidence >= 0.66:
        return True
    if item.win_value_edge is not None and item.win_value_edge >= 0.05:
        return True
    if item.place_value_edge is not None and item.place_value_edge >= 0.07:
        return True
    if item.red_flags and len(item.red_flags) >= 3:
        return False
    payload = item.runner.source_payload
    text = " ".join(str(value).lower() for value in _flatten_values(payload))
    return any(token in text for token in ("speed", "supported", "trainer_ae", "jockey_ae", "strike_rate"))


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


def _setup_history_hits(item: RunnerScore) -> int:
    runner = item.runner
    hits = 0
    for candidate in _history_candidates(runner.source_payload):
        if not isinstance(candidate, dict):
            continue
        if _history_match(candidate, ("course",), runner.course):
            hits += 1
        if _history_match(candidate, ("going", "ground"), runner.going):
            hits += 1
        if _history_match(candidate, ("distance", "dist", "race_distance"), runner.distance):
            hits += 1
        if _class_similarity(runner.race_class, _history_text(candidate, "race_class", "class", "race_grade", "grade")):
            hits += 1
        if hits >= 2:
            return hits
    return hits


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
    rows: list[Any] = []
    for source in _payload_sources(payload):
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                rows.extend(value[:5])
            elif value:
                rows.append(value)
    return rows


def _last_time_outsider_signal(item: RunnerScore) -> bool:
    runner = item.runner
    for candidate in _history_candidates(runner.source_payload):
        if not isinstance(candidate, dict):
            continue
        position = _parse_position(candidate.get("position") or candidate.get("pos"))
        odds = _history_decimal_odds(candidate)
        if position is None or odds is None or position > 3 or odds < 31.0:
            continue
        matches = 0
        if _class_similarity(runner.race_class, _history_text(candidate, "race_class", "class", "race_grade", "grade")):
            matches += 1
        if _history_match(candidate, ("going", "ground"), runner.going):
            matches += 1
        if _distance_similarity(runner.distance, _history_text(candidate, "distance", "dist", "race_distance")):
            matches += 1
        if matches >= 2:
            return True
    return False


def _history_decimal_odds(payload: dict[str, Any]) -> float | None:
    for key in ("sp_dec", "bsp", "decimal", "odds_decimal", "sp", "odds", "starting_price"):
        value = payload.get(key)
        odds = _decimal_odds(str(value)) if value not in (None, "") else None
        if odds is not None:
            return odds
    return None


def _history_match(payload: dict[str, Any], keys: tuple[str, ...], expected: str | None) -> bool:
    actual = _history_text(payload, *keys)
    return _text_similarity(expected, actual)


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
    expected_text = _normalise(expected)
    actual_text = _normalise(actual)
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
    for token in str(value).replace("-", " ").split():
        if token.isdigit() and 1 <= int(token) <= 7:
            return int(token)
    return None


def _distance_similarity(expected: str | None, actual: str | None) -> bool:
    expected_trip = _distance_furlongs(expected)
    actual_trip = _distance_furlongs(actual)
    if expected_trip is not None and actual_trip is not None:
        return abs(expected_trip - actual_trip) <= 0.5
    return _text_similarity(expected, actual)


def _distance_furlongs(value: str | None) -> float | None:
    if not value:
        return None
    text = _normalise(value).replace(" ", "")
    miles = _number_before(text, "m")
    furlongs = _number_before(text, "f")
    yards = _number_before(text, "y")
    if miles is None and furlongs is None and yards is None:
        return None
    return (miles or 0.0) * 8.0 + (furlongs or 0.0) + (yards or 0.0) / 220.0


def _number_before(text: str, marker: str) -> float | None:
    marker_index = text.find(marker)
    if marker_index <= 0:
        return None
    start = marker_index - 1
    while start >= 0 and (text[start].isdigit() or text[start] == "."):
        start -= 1
    number_text = text[start + 1 : marker_index]
    try:
        return float(number_text)
    except ValueError:
        return None


def _speed_signal(payload: dict[str, Any]) -> bool:
    speed = _metric_value(
        payload,
        ("speed_figure", "speed_rating", "topspeed", "top_speed", "rpr", "official_speed", "timeform_rating"),
    )
    return bool(speed is not None and speed >= 75)


def _trainer_jockey_signal(payload: dict[str, Any]) -> bool:
    trainer_ae = _metric_value(payload, ("trainer_ae", "trainer_a/e", "trainer_actual_expected"))
    trainer_sr = _metric_value(payload, ("trainer_strike_rate", "trainer_win_pct", "trainer_win_percentage"))
    jockey_ae = _metric_value(payload, ("jockey_ae", "jockey_a/e", "jockey_actual_expected"))
    jockey_sr = _metric_value(payload, ("jockey_strike_rate", "jockey_win_pct", "jockey_win_percentage"))
    return bool(
        (trainer_ae is not None and trainer_ae >= 1.05)
        or (jockey_ae is not None and jockey_ae >= 1.05)
        or (trainer_sr is not None and trainer_sr >= 14)
        or (jockey_sr is not None and jockey_sr >= 14)
    )


def _metric_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for source in _payload_sources(payload):
        parsed = _metric_from_source(source, keys)
        if parsed is not None:
            return parsed
    return None


def _metric_from_source(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalised_keys = {key.lower().replace(" ", "_").replace("-", "_"): key for key in source}
    for key in keys:
        raw_key = normalised_keys.get(key.lower().replace(" ", "_").replace("-", "_"))
        if raw_key is None:
            continue
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
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def _market_state(payload: dict[str, Any], current_odds: str | None) -> str:
    current = _decimal_odds(current_odds)
    opening = _opening_odds(payload)
    if current is None or opening is None or opening <= 1:
        return "unknown"
    move = (current - opening) / opening
    if move <= -0.12:
        return "supported"
    if move >= 0.2:
        return "drifting"
    return "stable"


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


def _parse_position(value: Any) -> int | None:
    if value in (None, ""):
        return None
    digits = "".join(ch for ch in str(value).lower() if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        rows: list[Any] = []
        for item in value.values():
            rows.extend(_flatten_values(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten_values(item))
        return rows
    return [value]


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


def _normalise(value: str | None) -> str:
    return " ".join(str(value or "").lower().strip().split())
