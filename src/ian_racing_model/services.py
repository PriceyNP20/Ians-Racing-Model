from __future__ import annotations

from datetime import date

from ian_racing_model.config import Settings
from ian_racing_model.domain import RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from ian_racing_model.providers.factory import build_provider
from ian_racing_model.storage.db import make_session_factory, store_raw_response


def get_scored_card(meeting_date: date, course: str | None, settings: Settings) -> list[RunnerScore]:
    provider = build_provider(settings)
    runners, raw = provider.fetch_racecard(meeting_date, course)
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        store_raw_response(session, settings.provider, meeting_date.isoformat(), course, raw)
    return IanFormulaV31().score_runners(runners)
