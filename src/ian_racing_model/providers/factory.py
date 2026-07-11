from __future__ import annotations

from ian_racing_model.config import Settings, THE_RACING_API_CONFIG
from ian_racing_model.providers.base import RacingDataProvider
from ian_racing_model.providers.mock import MockRacingDataProvider
from ian_racing_model.providers.the_racing_api import TheRacingApiProvider


def build_provider(settings: Settings) -> RacingDataProvider:
    if settings.provider.lower() == "the_racing_api":
        return TheRacingApiProvider(THE_RACING_API_CONFIG)
    return MockRacingDataProvider(settings.sample_racecard_path)
