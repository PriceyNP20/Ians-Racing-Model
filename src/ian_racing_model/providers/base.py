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
