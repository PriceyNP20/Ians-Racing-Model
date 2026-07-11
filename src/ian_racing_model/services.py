from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from ian_racing_model.config import Settings
from ian_racing_model.domain import RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from ian_racing_model.providers.factory import build_provider
from ian_racing_model.providers.mock import MockRacingDataProvider
from ian_racing_model.storage.db import make_session_factory, store_raw_response


@dataclass(frozen=True)
class ScoredCardResult:
    scores: list[RunnerScore]
    provider: str
    warning: str | None = None


def _store_raw(settings: Settings, meeting_date: date, course: str | None, payload: dict[str, Any]) -> None:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        store_raw_response(session, settings.provider, meeting_date.isoformat(), course, payload)


def get_scored_card_result(
    meeting_date: date, course: str | None, settings: Settings
) -> ScoredCardResult:
    provider = build_provider(settings)
    try:
        runners, raw = provider.fetch_racecard(meeting_date, course)
        _store_raw(settings, meeting_date, course, raw)
        return ScoredCardResult(
            scores=IanFormulaV31().score_runners(runners),
            provider=settings.provider,
        )
    except Exception as exc:
        error_payload = {
            "error": type(exc).__name__,
            "message": str(exc),
            "fallback": "mock",
        }
        _store_raw(settings, meeting_date, course, error_payload)
        if settings.provider.lower() == "mock":
            raise

        mock_provider = MockRacingDataProvider(settings.sample_racecard_path)
        runners, raw = mock_provider.fetch_racecard(meeting_date, course)
        _store_raw(settings, meeting_date, course, raw)
        return ScoredCardResult(
            scores=IanFormulaV31().score_runners(runners),
            provider="mock",
            warning=(
                "Live Racing API data could not be loaded, so this view is using "
                "sample data. Check Streamlit logs for the API status code/details."
            ),
        )


def get_scored_card(meeting_date: date, course: str | None, settings: Settings) -> list[RunnerScore]:
    return get_scored_card_result(meeting_date, course, settings).scores
