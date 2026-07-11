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
        items = _flatten_racecards(raw)
        mapped = [
            runner
            for runner in (map_runner(item, self.config["field_map"]) for item in items)
            if runner is not None
        ]
        return reject_mismatched_runners(mapped, meeting_date, course), raw


def _flatten_racecards(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if raw.get("runners"):
        return raw["runners"]

    racecards = raw.get("racecards") or raw.get("data") or []
    flattened: list[dict[str, Any]] = []
    for race in racecards:
        if not isinstance(race, dict):
            continue
        race_fields = {
            "date": race.get("date"),
            "course": race.get("course"),
            "off_time": race.get("off_time"),
            "race_name": race.get("race_name"),
            "race_class": race.get("race_class"),
            "race_type": race.get("type"),
            "surface": race.get("surface"),
            "distance": race.get("distance"),
            "going": race.get("going"),
            "field_size": race.get("field_size"),
        }
        for runner in race.get("runners") or []:
            if not isinstance(runner, dict):
                continue
            odds = runner.get("odds") or []
            current_odds = None
            if odds and isinstance(odds[0], dict):
                current_odds = odds[0].get("fractional") or odds[0].get("decimal")
            flattened.append(
                {
                    **race_fields,
                    "horse": runner.get("horse"),
                    "age": runner.get("age"),
                    "sex": runner.get("sex"),
                    "draw": runner.get("draw"),
                    "weight": runner.get("lbs"),
                    "official_rating": runner.get("ofr"),
                    "trainer": runner.get("trainer"),
                    "jockey": runner.get("jockey"),
                    "jockey_claim": None,
                    "recent_form": runner.get("form"),
                    "current_odds": current_odds,
                    "non_runner": runner.get("non_runner") or runner.get("is_non_runner"),
                    "source_runner": runner,
                }
            )
    return flattened
