from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import pandas as pd

from ian_racing_model.domain import Runner, RunnerScore
from racing_intelligence.domain import IntelligenceRunner, ProbabilityAssessment


class RacecardProvider(Protocol):
    """Fetch declared runners without binding callers to a provider response shape."""

    def fetch_racecards(self, meeting_date: date, region: str | None = None) -> list[Runner]:
        ...


class ResultsProvider(Protocol):
    """Fetch verified results only; never infer or fabricate results."""

    def fetch_results(self, meeting_date: date) -> dict[str, Any]:
        ...


class MarketProvider(Protocol):
    """Fetch current and historical market prices."""

    def prices_for(self, runner: Runner) -> dict[str, Any]:
        ...


class WeatherProvider(Protocol):
    def forecast_for(self, course: str, meeting_date: date) -> dict[str, Any]:
        ...


class GoingProvider(Protocol):
    def conditions_for(self, course: str, meeting_date: date) -> dict[str, Any]:
        ...


class HorseRatingsProvider(Protocol):
    def ratings_for(self, runner: Runner) -> dict[str, Any]:
        ...


class TrainerRatingsProvider(Protocol):
    def ratings_for(self, trainer: str | None) -> dict[str, Any]:
        ...


class JockeyRatingsProvider(Protocol):
    def ratings_for(self, jockey: str | None) -> dict[str, Any]:
        ...


class PaceMapEngine(Protocol):
    def score(self, runner: Runner, race_runners: list[Runner]) -> ProbabilityAssessment:
        ...


class DrawBiasEngine(Protocol):
    def score(self, runner: Runner, race_runners: list[Runner]) -> ProbabilityAssessment:
        ...


class CourseProfileEngine(Protocol):
    def score(self, runner: Runner) -> ProbabilityAssessment:
        ...


class WinProbabilityModel(Protocol):
    def assess(self, score: RunnerScore, race_scores: list[RunnerScore]) -> ProbabilityAssessment:
        ...


class PlaceProbabilityModel(Protocol):
    def assess(self, score: RunnerScore, race_scores: list[RunnerScore]) -> ProbabilityAssessment:
        ...


class FairOddsCalculator(Protocol):
    def fair_odds(self, probability: float) -> float | None:
        ...


class RecommendationEngine(Protocol):
    def recommend(self, runner: IntelligenceRunner) -> str:
        ...


class ModelCalibrationEngine(Protocol):
    def calibrate(self, rows: pd.DataFrame) -> pd.DataFrame:
        ...


class ReportingModule(Protocol):
    def export(self, rows: pd.DataFrame) -> bytes:
        ...
