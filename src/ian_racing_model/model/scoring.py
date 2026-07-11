from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp
import re

from ian_racing_model.config import IAN_FORMULA_V3_1_WEIGHTS, validate_weights
from ian_racing_model.domain import ComponentScore, Runner, RunnerScore


DATA_OK = "ok"
DATA_PARTIAL = "partial"
DATA_MISSING = "missing"


@dataclass(frozen=True)
class FeatureValue:
    score: float
    confidence: float
    quality: str
    explanation: str


class IanFormulaV31:
    def __init__(self, weights: dict[str, int] | None = None) -> None:
        self.weights = weights or IAN_FORMULA_V3_1_WEIGHTS
        validate_weights(self.weights)

    def score_runner(self, runner: Runner) -> RunnerScore:
        components = [
            self._component("handicap_position", runner, self._handicap_position),
            self._component("target_race_intent", runner, self._target_race_intent),
            self._component("pace_and_draw", runner, self._pace_and_draw),
            self._component("course_suitability", runner, self._course_suitability),
            self._component("distance_suitability", runner, self._distance_suitability),
            self._component("class_strength", runner, self._class_strength),
            self._component("current_performance", runner, self._current_performance),
            self._component("trainer_profile", runner, self._trainer_profile),
            self._component("jockey_suitability", runner, self._jockey_suitability),
            self._component("market_value", runner, self._market_value),
        ]
        base_score = sum(c.score for c in components)
        red_flags, deduction = self._red_flags(runner)
        total = max(0.0, min(100.0, round(base_score - deduction, 2)))
        confidence = round(sum(c.confidence for c in components) / len(components), 2)
        warnings = [
            f"{c.name}: {c.explanation}"
            for c in components
            if c.data_quality in {DATA_PARTIAL, DATA_MISSING}
        ]
        warnings.extend(red_flags)
        return RunnerScore(
            runner=runner,
            total_score=total,
            confidence=confidence,
            recommendation=self._recommendation(total, confidence, runner.current_odds),
            fair_odds_placeholder="TBD after results calibration",
            win_probability=0.0,
            place_probability=0.0,
            fair_win_odds=None,
            fair_place_odds=None,
            win_value_edge=None,
            place_value_edge=None,
            components=components,
            red_flags=red_flags,
            data_quality_warnings=warnings,
        )

    def score_runners(self, runners: list[Runner]) -> list[RunnerScore]:
        eligible = [runner for runner in runners if not runner.is_non_runner]
        scored = [self.score_runner(runner) for runner in eligible]
        scored = self._calibrate_probabilities(scored)
        return sorted(scored, key=lambda score: score.total_score, reverse=True)

    def _component(self, name: str, runner: Runner, fn) -> ComponentScore:
        feature = fn(runner)
        weighted = round((feature.score / 100.0) * self.weights[name], 2)
        return ComponentScore(name, weighted, feature.confidence, feature.quality, feature.explanation)

    def _handicap_position(self, runner: Runner) -> FeatureValue:
        if runner.official_rating is None or not runner.weight:
            return FeatureValue(45, 0.45, DATA_MISSING, "Official rating or weight unavailable.")
        if runner.official_rating >= 96:
            return FeatureValue(38, 0.8, DATA_OK, "High handicap mark creates limited margin.")
        if runner.official_rating >= 86:
            return FeatureValue(58, 0.8, DATA_OK, "Competitive but not obviously well treated.")
        return FeatureValue(72, 0.75, DATA_OK, "Handicap mark looks manageable from available data.")

    def _target_race_intent(self, runner: Runner) -> FeatureValue:
        if not runner.recent_form:
            return FeatureValue(45, 0.4, DATA_MISSING, "Recent form unavailable; intent cannot be inferred.")
        if re.search(r"[12]", runner.recent_form):
            return FeatureValue(72, 0.65, DATA_PARTIAL, "Recent placing suggests active campaign.")
        if "-" in runner.recent_form:
            return FeatureValue(42, 0.55, DATA_PARTIAL, "Break in form string may indicate absence.")
        return FeatureValue(55, 0.55, DATA_PARTIAL, "Intent inferred only from limited form string.")

    def _pace_and_draw(self, runner: Runner) -> FeatureValue:
        if runner.draw is None or runner.field_size is None:
            return FeatureValue(45, 0.4, DATA_MISSING, "Draw or field size unavailable.")
        if runner.draw > max(8, runner.field_size * 0.75):
            return FeatureValue(35, 0.7, DATA_OK, "Wide draw may make the race shape harder.")
        if runner.draw <= max(2, runner.field_size * 0.25):
            return FeatureValue(68, 0.65, DATA_OK, "Low draw is a possible tactical positive.")
        return FeatureValue(58, 0.6, DATA_OK, "Draw looks broadly neutral.")

    def _course_suitability(self, runner: Runner) -> FeatureValue:
        if not runner.course:
            return FeatureValue(40, 0.35, DATA_MISSING, "Course unavailable.")
        history = _history_items(runner)
        if history:
            course_runs = [
                item for item in history if _text_match(runner.course, _history_text(item, "course"))
            ]
            if course_runs:
                placed = sum(1 for item in course_runs if (_history_position(item) or 99) <= 3)
                wins = sum(1 for item in course_runs if _history_position(item) == 1)
                score = min(82, 58 + wins * 12 + placed * 6)
                return FeatureValue(
                    score,
                    0.75,
                    DATA_OK,
                    "Course suitability scored from recent horse history.",
                )
            return FeatureValue(52, 0.65, DATA_PARTIAL, "No recent course evidence in horse history.")
        return FeatureValue(58, 0.45, DATA_PARTIAL, "No course-history feed available; using neutral baseline.")

    def _distance_suitability(self, runner: Runner) -> FeatureValue:
        if not runner.distance:
            return FeatureValue(40, 0.35, DATA_MISSING, "Distance unavailable.")
        history = _history_items(runner)
        if history:
            distance_runs = [
                item
                for item in history
                if _distance_match(runner.distance, _history_text(item, "distance", "dist"))
            ]
            if distance_runs:
                placed = sum(1 for item in distance_runs if (_history_position(item) or 99) <= 3)
                wins = sum(1 for item in distance_runs if _history_position(item) == 1)
                score = min(82, 58 + wins * 12 + placed * 6)
                return FeatureValue(
                    score,
                    0.75,
                    DATA_OK,
                    "Trip suitability scored from recent horse history.",
                )
            return FeatureValue(50, 0.65, DATA_PARTIAL, "No proven recent run at this trip in horse history.")
        if any(token in runner.distance.lower() for token in ["3m", "2m7f", "2m6f"]):
            return FeatureValue(50, 0.5, DATA_PARTIAL, "Stamina trip noted without full trip history.")
        return FeatureValue(58, 0.45, DATA_PARTIAL, "Distance present but prior suitability unavailable.")

    def _class_strength(self, runner: Runner) -> FeatureValue:
        if not runner.race_class:
            return FeatureValue(45, 0.35, DATA_MISSING, "Race class unavailable.")
        today_class = _class_number(runner.race_class)
        history = _history_items(runner)
        if history and today_class is not None:
            class_runs = [
                (_class_number(_history_text(item, "race_class", "class")), _history_position(item))
                for item in history[:5]
            ]
            class_runs = [(race_class, position) for race_class, position in class_runs if race_class]
            if class_runs:
                same_or_stronger_places = sum(
                    1
                    for race_class, position in class_runs
                    if race_class <= today_class and position is not None and position <= 3
                )
                last_class = class_runs[0][0]
                if same_or_stronger_places:
                    return FeatureValue(70, 0.72, DATA_OK, "Placed at this class level or stronger in horse history.")
                if last_class and last_class > today_class:
                    return FeatureValue(46, 0.68, DATA_PARTIAL, "Class rise detected from recent horse history.")
                return FeatureValue(56, 0.65, DATA_PARTIAL, "Class evidence available but no strong positive.")
        if "2" in runner.race_class or "1" in runner.race_class:
            return FeatureValue(52, 0.55, DATA_PARTIAL, "Stronger class band requires caution.")
        return FeatureValue(62, 0.55, DATA_PARTIAL, "Class band appears ordinary from available label.")

    def _current_performance(self, runner: Runner) -> FeatureValue:
        history_positions = [
            position for position in (_history_position(item) for item in _history_items(runner)[:4]) if position
        ]
        if history_positions:
            avg = sum(history_positions) / len(history_positions)
            in_frame = sum(1 for position in history_positions if position <= 3)
            score = max(30, min(84, 84 - avg * 7 + in_frame * 4))
            return FeatureValue(score, 0.75, DATA_OK, "Scored from recent horse history finishing positions.")
        if not runner.recent_form:
            return FeatureValue(42, 0.35, DATA_MISSING, "Recent form unavailable.")
        digits = [int(ch) for ch in runner.recent_form if ch.isdigit()]
        if not digits:
            return FeatureValue(44, 0.45, DATA_PARTIAL, "Form string has no finishing positions.")
        avg = sum(digits[:4]) / min(len(digits), 4)
        score = max(30, min(80, 86 - avg * 8))
        return FeatureValue(score, 0.65, DATA_PARTIAL, "Scored from recent finishing positions only.")

    def _trainer_profile(self, runner: Runner) -> FeatureValue:
        if not runner.trainer:
            return FeatureValue(45, 0.35, DATA_MISSING, "Trainer unavailable.")
        return FeatureValue(56, 0.4, DATA_PARTIAL, "Trainer strike-rate data unavailable; neutral baseline used.")

    def _jockey_suitability(self, runner: Runner) -> FeatureValue:
        if not runner.jockey:
            return FeatureValue(42, 0.35, DATA_MISSING, "Jockey unavailable.")
        claim_bonus = 5 if runner.jockey_claim and runner.jockey_claim > 0 else 0
        return FeatureValue(55 + claim_bonus, 0.45, DATA_PARTIAL, "Jockey fit inferred only from booking and claim.")

    def _market_value(self, runner: Runner) -> FeatureValue:
        odds = _decimal_odds(runner.current_odds)
        if odds is None:
            return FeatureValue(42, 0.35, DATA_MISSING, "Current odds unavailable.")
        if odds < 2.5:
            return FeatureValue(40, 0.55, DATA_PARTIAL, "Short price needs value proof not yet available.")
        if odds >= 8:
            return FeatureValue(68, 0.5, DATA_PARTIAL, "Bigger price may offer value, pending calibration.")
        return FeatureValue(58, 0.5, DATA_PARTIAL, "Market price is within a workable range.")

    def _red_flags(self, runner: Runner) -> tuple[list[str], float]:
        flags: list[str] = []
        if runner.distance and any(token in runner.distance.lower() for token in ["3m", "2m7f"]):
            flags.append("wrong or unproven trip")
        if runner.race_class and any(token in runner.race_class.lower() for token in ["class 1", "class 2"]):
            flags.append("class rise")
        if runner.draw and runner.field_size and runner.draw > max(8, runner.field_size * 0.75):
            flags.append("poor draw")
        if runner.official_rating and runner.official_rating >= 96:
            flags.append("high handicap mark")
        if runner.going and any(token in runner.going.lower() for token in ["heavy", "firm"]):
            flags.append("going concern")
        if runner.recent_form and "-" in runner.recent_form:
            flags.append("long absence")
        if runner.recent_form and runner.recent_form.upper().count("F") + runner.recent_form.upper().count("U") >= 2:
            flags.append("repeated jumping errors")
        odds = _decimal_odds(runner.current_odds)
        if odds is not None and odds < 2.5:
            flags.append("short price without value")
        return flags, min(20.0, len(flags) * 3.0)

    def _recommendation(self, score: float, confidence: float, odds: str | None) -> str:
        decimal_odds = _decimal_odds(odds)
        if score >= 78 and confidence >= 0.65:
            return "WIN"
        if score >= 68 and confidence >= 0.55 and (decimal_odds is None or decimal_odds >= 5):
            return "EACH_WAY"
        if score >= 62 and confidence >= 0.5:
            return "PLACE"
        if score >= 52:
            return "WATCH"
        return "PASS"

    def _calibrate_probabilities(self, scores: list[RunnerScore]) -> list[RunnerScore]:
        calibrated: list[RunnerScore] = []
        groups: dict[tuple[str, str, str], list[RunnerScore]] = {}
        for score in scores:
            runner = score.runner
            key = (runner.course, runner.off_time, runner.race_name)
            groups.setdefault(key, []).append(score)

        for race_scores in groups.values():
            if not race_scores:
                continue
            mean_score = sum(item.total_score for item in race_scores) / len(race_scores)
            win_strengths = [
                exp((item.total_score - mean_score) / 14.0) * max(0.35, item.confidence)
                for item in race_scores
            ]
            win_total = sum(win_strengths) or 1.0
            place_slots = _place_cutoff(race_scores[0].runner.field_size or len(race_scores))
            place_strengths = [
                exp((item.total_score - mean_score) / 18.0)
                * max(0.45, item.confidence)
                * _place_reliability(item)
                for item in race_scores
            ]
            place_total = sum(place_strengths) or 1.0
            for item, win_strength, place_strength in zip(race_scores, win_strengths, place_strengths):
                win_probability = max(0.001, min(0.85, win_strength / win_total))
                place_probability = max(
                    win_probability,
                    min(0.92, (place_strength / place_total) * place_slots),
                )
                fair_win_odds = _fair_odds(win_probability)
                fair_place_odds = _fair_odds(place_probability)
                market_odds = _decimal_odds(item.runner.current_odds)
                place_market_odds = _estimated_place_odds(market_odds)
                calibrated.append(
                    replace(
                        item,
                        win_probability=round(win_probability, 4),
                        place_probability=round(place_probability, 4),
                        fair_win_odds=fair_win_odds,
                        fair_place_odds=fair_place_odds,
                        win_value_edge=_value_edge(win_probability, market_odds),
                        place_value_edge=_value_edge(place_probability, place_market_odds),
                        fair_odds_placeholder=(
                            f"Win {fair_win_odds:.2f} | Place {fair_place_odds:.2f}"
                            if fair_win_odds and fair_place_odds
                            else "Needs probability calibration"
                        ),
                    )
                )
        return calibrated


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


def _history_items(runner: Runner) -> list[dict]:
    payload = runner.source_payload
    keys = (
        "horse_history",
        "history",
        "results",
        "past_results",
        "horse_results",
        "last_result",
        "previous_result",
    )
    items: list[dict] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            items.append(value)
    source_runner = payload.get("source_runner")
    if isinstance(source_runner, dict):
        for key in keys:
            value = source_runner.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                items.append(value)
    return items


def _history_position(item: dict) -> int | None:
    for key in ("position", "pos", "finishing_position", "finish_position", "result_position", "place"):
        value = item.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip().lower()
        if text in {"nr", "pu", "f", "ur", "bd"}:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        try:
            return int(digits)
        except ValueError:
            continue
    return None


def _history_text(item: dict, *keys: str) -> str:
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


def _distance_match(expected: str | None, actual: str | None) -> bool:
    expected_text = _normalise_text(expected).replace(" ", "")
    actual_text = _normalise_text(actual).replace(" ", "")
    return bool(expected_text and actual_text and expected_text == actual_text)


def _class_number(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b([1-7])\b", str(value))
    if not match:
        return None
    return int(match.group(1))


def _fair_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return round(1.0 / probability, 2)


def _value_edge(probability: float, market_odds: float | None) -> float | None:
    if market_odds is None or market_odds <= 1:
        return None
    return round(probability - (1.0 / market_odds), 4)


def _estimated_place_odds(win_odds: float | None) -> float | None:
    if win_odds is None or win_odds <= 1:
        return None
    return 1.0 + ((win_odds - 1.0) / 5.0)


def _place_cutoff(field_size: int) -> int:
    if field_size >= 16:
        return 4
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return 1


def _place_reliability(item: RunnerScore) -> float:
    runner = item.runner
    reliability = 1.0
    if runner.recent_form:
        recent_digits = [int(ch) for ch in runner.recent_form[:5] if ch.isdigit()]
        if recent_digits:
            in_frame = sum(1 for position in recent_digits if position <= 3)
            reliability += min(0.35, in_frame / max(1, len(recent_digits)) * 0.35)
    reliability -= min(0.35, len(item.red_flags) * 0.06)
    return max(0.65, reliability)
