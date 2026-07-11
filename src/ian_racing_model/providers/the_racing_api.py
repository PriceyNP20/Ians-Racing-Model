from __future__ import annotations

from datetime import date
from typing import Any

import requests

from ian_racing_model.config import THE_RACING_API_CONFIG, get_setting
from ian_racing_model.domain import Runner
from ian_racing_model.providers.base import RacingDataProvider
from ian_racing_model.providers.mapping import map_runner, reject_mismatched_runners


class TheRacingApiProvider(RacingDataProvider):
    """Adapter for The Racing API with endpoint and mapping config isolated."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or THE_RACING_API_CONFIG
        self.session = requests.Session()

    def fetch_racecard(self, meeting_date: date, course: str | None = None) -> tuple[list[Runner], dict[str, Any]]:
        username = get_setting(self.config["auth"]["username_env"])
        password = get_setting(self.config["auth"]["password_env"])
        if not username or not password:
            raise RuntimeError(
                "The Racing API credentials are missing. Set RACING_API_USERNAME "
                "and RACING_API_PASSWORD or use RACING_DATA_PROVIDER=mock."
            )
        url = self.config["base_url"].rstrip("/") + self.config["racecards_endpoint"]
        params = {"date": meeting_date.isoformat()}
        response = self.session.get(url, params=params, auth=(username, password), timeout=30)
        response.raise_for_status()
        raw = response.json()
        items = raw.get("runners") or raw.get("racecards") or raw.get("data") or []
        mapped = [
            runner
            for runner in (map_runner(item, self.config["field_map"]) for item in items)
            if runner is not None
        ]
        return reject_mismatched_runners(mapped, meeting_date, course), raw
