from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityAssessment:
    probability: float
    fair_odds: float | None
    confidence: float
    data_quality: str
    explanation: str


@dataclass(frozen=True)
class IntelligenceRunner:
    course: str
    off_time: str
    race: str
    horse: str
    odds: str | None
    field_size: int | None
    win: ProbabilityAssessment
    place: ProbabilityAssessment
    win_value_edge: float | None
    place_value_edge: float | None
    recommendation: str
    data_quality: str
    warnings: list[str]
