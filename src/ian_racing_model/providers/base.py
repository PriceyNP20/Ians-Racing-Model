from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from ian_racing_model.domain import Runner


class RacingDataProvider(ABC):
    """Replaceable interface for racecard data sources."""

    @abstractmethod
    def fetch_racecard(
        self, meeting_date: date, course: str | None = None
    ) -> tuple[list[Runner], dict[str, Any]]:
        """Return mapped runners and the raw provider response for audit storage."""

    def fetch_results(self, meeting_date: date) -> dict[str, Any]:
        """Return raw results for a date when the provider supports results."""
        return {}

    def fetch_horse_history(self, runner: Runner, limit: int = 6) -> dict[str, Any]:
        """Return recent horse results when the provider supports runner history."""
        return {}

    def fetch_horse_profile(self, runner: Runner) -> dict[str, Any]:
        """Return Pro horse profile fields such as breeding, sex, colour and DOB."""
        return {}

    def fetch_odds_history(self, runner: Runner) -> dict[str, Any]:
        """Return runner odds history when the provider supports market movement."""
        return {}

    def fetch_trainer_course_analysis(self, runner: Runner) -> dict[str, Any]:
        """Return trainer performance for today's course/setup where supported."""
        return {}

    def fetch_trainer_distance_analysis(self, runner: Runner) -> dict[str, Any]:
        """Return trainer performance for today's distance band where supported."""
        return {}

    def fetch_trainer_jockey_analysis(self, runner: Runner) -> dict[str, Any]:
        """Return trainer/jockey combination performance where supported."""
        return {}

    def fetch_jockey_course_analysis(self, runner: Runner) -> dict[str, Any]:
        """Return jockey performance for today's course/setup where supported."""
        return {}

    def fetch_jockey_trainer_analysis(self, runner: Runner) -> dict[str, Any]:
        """Return jockey/trainer combination performance where supported."""
        return {}
