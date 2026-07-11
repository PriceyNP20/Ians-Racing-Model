from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Runner:
    meeting_date: date
    course: str
    off_time: str
    race_name: str
    race_class: str | None
    race_type: str | None
    surface: str | None
    distance: str | None
    going: str | None
    field_size: int | None
    horse: str
    age: int | None
    sex: str | None
    draw: int | None
    weight: str | None
    official_rating: int | None
    trainer: str | None
    jockey: str | None
    jockey_claim: int | None
    recent_form: str | None
    current_odds: str | None
    is_non_runner: bool = False
    source_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float
    confidence: float
    data_quality: str
    explanation: str


@dataclass(frozen=True)
class RunnerScore:
    runner: Runner
    total_score: float
    confidence: float
    recommendation: str
    fair_odds_placeholder: str
    win_probability: float
    place_probability: float
    fair_win_odds: float | None
    fair_place_odds: float | None
    win_value_edge: float | None
    place_value_edge: float | None
    components: list[ComponentScore]
    red_flags: list[str]
    data_quality_warnings: list[str]
