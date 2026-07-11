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

    def fetch_racecard(
        self, meeting_date: date, course: str | None = None
    ) -> tuple[list[Runner], dict[str, Any]]:
        username = get_setting(self.config["auth"]["username_env"])
        password = get_setting(self.config["auth"]["password_env"])
        if not username or not password:
            raise RuntimeError(
                "The Racing API credentials are missing. Set RACING_API_USERNAME "
                "and RACING_API_PASSWORD or use RACING_DATA_PROVIDER=mock."
            )

        url = self.config["base_url"].rstrip("/") + self.config["racecards_endpoint"]
        params = {"date": meeting_date.isoformat()}
        region_codes = _csv_values(self.config.get("racecards_region_codes"))
        if region_codes:
            params["region_codes"] = region_codes
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

    def fetch_results(self, meeting_date: date) -> dict[str, Any]:
        username = get_setting(self.config["auth"]["username_env"])
        password = get_setting(self.config["auth"]["password_env"])
        if not username or not password:
            return {}

        url = self.config["base_url"].rstrip("/") + self.config["results_endpoint"]
        params: dict[str, Any] = {
            "start_date": meeting_date.isoformat(),
            "end_date": meeting_date.isoformat(),
        }
        region_codes = _csv_values(self.config.get("racecards_region_codes"))
        if region_codes:
            params["region"] = region_codes
        response = self.session.get(url, params=params, auth=(username, password), timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_horse_history(self, runner: Runner, limit: int = 6) -> dict[str, Any]:
        username = get_setting(self.config["auth"]["username_env"])
        password = get_setting(self.config["auth"]["password_env"])
        horse_id = _horse_id(runner)
        if not username or not password or not horse_id:
            return {}

        endpoint = str(self.config["horse_results_endpoint"]).format(horse_id=horse_id)
        url = self.config["base_url"].rstrip("/") + endpoint
        response = self.session.get(
            url,
            params={"limit": limit},
            auth=(username, password),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def _flatten_racecards(raw: dict[str, Any]) -> list[dict[str, Any]]:
    racecards = raw.get("racecards") or raw.get("data") or []
    if raw.get("runners"):
        return raw["runners"]
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
                    "horse_id": runner.get("horse_id") or runner.get("id"),
                    "source_runner": runner,
                }
            )
    return flattened


def _csv_values(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _horse_id(runner: Runner) -> str | None:
    payload = runner.source_payload
    source_runner = payload.get("source_runner")
    for source in (payload, source_runner if isinstance(source_runner, dict) else {}):
        for key in ("horse_id", "id"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return None
