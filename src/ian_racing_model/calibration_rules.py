from __future__ import annotations

from typing import Any

from ian_racing_model.domain import RunnerScore


def install_calibration_rules(ui_module: Any) -> None:
    if getattr(ui_module, "_ian_calibration_rules_installed", False):
        return

    original_winner_score = ui_module._winner_selection_score
    original_each_way_score = ui_module._each_way_selection_score
    original_pick_row = ui_module._pick_row
    original_selection_screener_row = ui_module._selection_screener_row

    def winner_selection_score(item: RunnerScore) -> float:
        return original_winner_score(item) + calibration_adjustment(item, "Winner pick")[0]

    def each_way_selection_score(item: RunnerScore) -> float:
        return original_each_way_score(item) + calibration_adjustment(item, "Best EW pick")[0]

    def pick_row(item: RunnerScore, pick_type: str, selection_score: float) -> dict[str, Any]:
        row = original_pick_row(item, pick_type, selection_score)
        adjustment, reason = calibration_adjustment(item, pick_type)
        row["calibration_adjustment"] = round(adjustment, 2)
        row["calibration_reason"] = reason
        return row

    def selection_screener_row(item: RunnerScore, pick: str, selection_score: float) -> dict[str, Any]:
        row = original_selection_screener_row(item, pick, selection_score)
        pick_type = "Winner pick" if pick == "Winner" else "Best EW pick"
        adjustment, reason = calibration_adjustment(item, pick_type)
        row["calibration_adjustment"] = round(adjustment, 2)
        row["calibration_reason"] = reason
        return row

    ui_module._winner_selection_score = winner_selection_score
    ui_module._each_way_selection_score = each_way_selection_score
    ui_module._pick_row = pick_row
    ui_module._selection_screener_row = selection_screener_row
    ui_module._ian_calibration_rules_installed = True


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


def _has_strong_evidence(item: RunnerScore) -> bool:
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
