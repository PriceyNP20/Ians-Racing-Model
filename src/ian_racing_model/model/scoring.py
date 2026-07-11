from __future__ import annotations

from dataclasses import dataclass
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
            self._component("course_suitability", runner, lambda r: FeatureValue(58, 0.45, DATA_PARTIAL, "No course-history feed available; using neutral baseline.")),
            self._component("distance_suitability", runner, self._distance_suitability),
            self._component("class_strength", runner, self._class_strength),
            self._component("current_performance", runner, self._current_performance),
            self._component("trainer_profile", runner, lambda r: FeatureValue(56 if r.trainer else 45, 0.4 if r.trainer else 0.35, DATA_PARTIAL if r.trainer else DATA_MISSING, "Trainer strike-rate data unavailable; neutral baseline used." if r.trainer else "Trainer unavailable.")),
            self._component("jockey_suitability", runner, self._jockey_suitability),
            self._component("market_value", runner, self._market_value),
        ]
        red_flags, deduction = self._red_flags(runner)
        total = max(0.0, min(100.0, round(sum(c.score for c in components) - deduction, 2)))
        confidence = round(sum(c.confidence for c in components) / len(components), 2)
        warnings = [f"{c.name}: {c.explanation}" for c in components if c.data_quality in {DATA_PARTIAL, DATA_MISSING}]
        warnings.extend(red_flags)
        return RunnerScore(runner, total, confidence, self._recommendation(total, confidence, runner.current_odds), "TBD after results calibration", components, red_flags, warnings)

    def score_runners(self, runners: list[Runner]) -> list[RunnerScore]:
        return sorted([self.score_runner(r) for r in runners if not r.is_non_runner], key=lambda s: s.total_score, reverse=True)

    def _component(self, name: str, runner: Runner, fn) -> ComponentScore:
        f = fn(runner)
        return ComponentScore(name, round((f.score / 100.0) * self.weights[name], 2), f.confidence, f.quality, f.explanation)

    def _handicap_position(self, r: Runner) -> FeatureValue:
        if r.official_rating is None or not r.weight:
            return FeatureValue(45, 0.45, DATA_MISSING, "Official rating or weight unavailable.")
        if r.official_rating >= 96:
            return FeatureValue(38, 0.8, DATA_OK, "High handicap mark creates limited margin.")
        if r.official_rating >= 86:
            return FeatureValue(58, 0.8, DATA_OK, "Competitive but not obviously well treated.")
        return FeatureValue(72, 0.75, DATA_OK, "Handicap mark looks manageable from available data.")

    def _target_race_intent(self, r: Runner) -> FeatureValue:
        if not r.recent_form:
            return FeatureValue(45, 0.4, DATA_MISSING, "Recent form unavailable; intent cannot be inferred.")
        if re.search(r"[12]", r.recent_form):
            return FeatureValue(72, 0.65, DATA_PARTIAL, "Recent placing suggests active campaign.")
        if "-" in r.recent_form:
            return FeatureValue(42, 0.55, DATA_PARTIAL, "Break in form string may indicate absence.")
        return FeatureValue(55, 0.55, DATA_PARTIAL, "Intent inferred only from limited form string.")

    def _pace_and_draw(self, r: Runner) -> FeatureValue:
        if r.draw is None or r.field_size is None:
            return FeatureValue(45, 0.4, DATA_MISSING, "Draw or field size unavailable.")
        if r.draw > max(8, r.field_size * 0.75):
            return FeatureValue(35, 0.7, DATA_OK, "Wide draw may make the race shape harder.")
        if r.draw <= max(2, r.field_size * 0.25):
            return FeatureValue(68, 0.65, DATA_OK, "Low draw is a possible tactical positive.")
        return FeatureValue(58, 0.6, DATA_OK, "Draw looks broadly neutral.")

    def _distance_suitability(self, r: Runner) -> FeatureValue:
        if not r.distance:
            return FeatureValue(40, 0.35, DATA_MISSING, "Distance unavailable.")
        return FeatureValue(58, 0.45, DATA_PARTIAL, "Distance present but prior suitability unavailable.")

    def _class_strength(self, r: Runner) -> FeatureValue:
        if not r.race_class:
            return FeatureValue(45, 0.35, DATA_MISSING, "Race class unavailable.")
        if "2" in r.race_class or "1" in r.race_class:
            return FeatureValue(52, 0.55, DATA_PARTIAL, "Stronger class band requires caution.")
        return FeatureValue(62, 0.55, DATA_PARTIAL, "Class band appears ordinary from available label.")

    def _current_performance(self, r: Runner) -> FeatureValue:
        if not r.recent_form:
            return FeatureValue(42, 0.35, DATA_MISSING, "Recent form unavailable.")
        digits = [int(ch) for ch in r.recent_form if ch.isdigit()]
        if not digits:
            return FeatureValue(44, 0.45, DATA_PARTIAL, "Form string has no finishing positions.")
        avg = sum(digits[:4]) / min(len(digits), 4)
        return FeatureValue(max(30, min(80, 86 - avg * 8)), 0.65, DATA_PARTIAL, "Scored from recent finishing positions only.")

    def _jockey_suitability(self, r: Runner) -> FeatureValue:
        if not r.jockey:
            return FeatureValue(42, 0.35, DATA_MISSING, "Jockey unavailable.")
        return FeatureValue(55 + (5 if r.jockey_claim else 0), 0.45, DATA_PARTIAL, "Jockey fit inferred only from booking and claim.")

    def _market_value(self, r: Runner) -> FeatureValue:
        odds = _decimal_odds(r.current_odds)
        if odds is None:
            return FeatureValue(42, 0.35, DATA_MISSING, "Current odds unavailable.")
        if odds < 2.5:
            return FeatureValue(40, 0.55, DATA_PARTIAL, "Short price needs value proof not yet available.")
        if odds >= 8:
            return FeatureValue(68, 0.5, DATA_PARTIAL, "Bigger price may offer value, pending calibration.")
        return FeatureValue(58, 0.5, DATA_PARTIAL, "Market price is within a workable range.")

    def _red_flags(self, r: Runner) -> tuple[list[str], float]:
        flags: list[str] = []
        if r.race_class and any(token in r.race_class.lower() for token in ["class 1", "class 2"]):
            flags.append("class rise")
        if r.draw and r.field_size and r.draw > max(8, r.field_size * 0.75):
            flags.append("poor draw")
        if r.official_rating and r.official_rating >= 96:
            flags.append("high handicap mark")
        if r.going and any(token in r.going.lower() for token in ["heavy", "firm"]):
            flags.append("going concern")
        if r.recent_form and "-" in r.recent_form:
            flags.append("long absence")
        if r.recent_form and r.recent_form.upper().count("F") + r.recent_form.upper().count("U") >= 2:
            flags.append("repeated jumping errors")
        odds = _decimal_odds(r.current_odds)
        if odds is not None and odds < 2.5:
            flags.append("short price without value")
        return flags, min(20.0, len(flags) * 3.0)

    def _recommendation(self, score: float, confidence: float, odds: str | None) -> str:
        decimal = _decimal_odds(odds)
        if score >= 78 and confidence >= 0.65:
            return "WIN"
        if score >= 68 and confidence >= 0.55 and (decimal is None or decimal >= 5):
            return "EACH_WAY"
        if score >= 62 and confidence >= 0.5:
            return "PLACE"
        if score >= 52:
            return "WATCH"
        return "PASS"


def _decimal_odds(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        try:
            num, den = value.split("/", 1)
            return float(num) / float(den) + 1.0
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None
