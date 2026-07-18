from __future__ import annotations

from datetime import date
import time
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

    def fetch_horse_profile(self, runner: Runner) -> dict[str, Any]:
        horse_id = _horse_id(runner)
        if not horse_id:
            return {}
        return self._get_json("horse_profile_endpoint", {"horse_id": horse_id})

    def fetch_odds_history(self, runner: Runner) -> dict[str, Any]:
        horse_id = _horse_id(runner)
        race_id = _race_id(runner)
        if not horse_id or not race_id:
            return {}
        return self._get_json("odds_endpoint", {"race_id": race_id, "horse_id": horse_id})

    def fetch_trainer_course_analysis(self, runner: Runner) -> dict[str, Any]:
        trainer_id = _trainer_id(runner)
        if not trainer_id:
            return {}
        return self._get_json(
            "trainer_course_analysis_endpoint",
            {"trainer_id": trainer_id},
            params=_analysis_params(runner, include_distance=False),
        )

    def fetch_trainer_distance_analysis(self, runner: Runner) -> dict[str, Any]:
        trainer_id = _trainer_id(runner)
        if not trainer_id:
            return {}
        return self._get_json(
            "trainer_distance_analysis_endpoint",
            {"trainer_id": trainer_id},
            params=_analysis_params(runner, include_course=False),
        )

    def fetch_trainer_jockey_analysis(self, runner: Runner) -> dict[str, Any]:
        trainer_id = _trainer_id(runner)
        if not trainer_id:
            return {}
        return self._get_json(
            "trainer_jockey_analysis_endpoint",
            {"trainer_id": trainer_id},
            params=_analysis_params(runner),
        )

    def fetch_jockey_course_analysis(self, runner: Runner) -> dict[str, Any]:
        jockey_id = _jockey_id(runner)
        if not jockey_id:
            return {}
        return self._get_json(
            "jockey_course_analysis_endpoint",
            {"jockey_id": jockey_id},
            params=_analysis_params(runner, include_distance=False),
        )

    def fetch_jockey_trainer_analysis(self, runner: Runner) -> dict[str, Any]:
        jockey_id = _jockey_id(runner)
        if not jockey_id:
            return {}
        return self._get_json(
            "jockey_trainer_analysis_endpoint",
            {"jockey_id": jockey_id},
            params=_analysis_params(runner),
        )

    def _get_json(
        self,
        endpoint_key: str,
        path_values: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        username = get_setting(self.config["auth"]["username_env"])
        password = get_setting(self.config["auth"]["password_env"])
        if not username or not password:
            return {}

        endpoint = str(self.config[endpoint_key]).format(**path_values)
        url = self.config["base_url"].rstrip("/") + endpoint
        response = self.session.get(
            url,
            params={key: value for key, value in (params or {}).items() if value not in (None, "", [])},
            auth=(username, password),
            timeout=30,
        )
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
        for attempt in range(3):
            response = self.session.get(
                url,
                params={"limit": limit},
                auth=(username, password),
                timeout=30,
            )
            if response.status_code != 429 or attempt == 2:
                response.raise_for_status()
                return response.json()
            time.sleep(1.0 + attempt)
        return {}


def _flatten_racecards(raw: dict[str, Any]) -> list[dict[str, Any]]:
    racecards = raw.get("racecards") or raw.get("data") or []
    if raw.get("runners"):
        return raw["runners"]
    flattened: list[dict[str, Any]] = []
    for race in racecards:
        if not isinstance(race, dict):
            continue
        race_fields = {
            "race_id": race.get("race_id"),
            "date": race.get("date"),
            "course": race.get("course"),
            "course_id": race.get("course_id"),
            "off_time": race.get("off_time"),
            "off_dt": race.get("off_dt"),
            "race_name": race.get("race_name"),
            "race_class": race.get("race_class"),
            "race_type": race.get("type"),
            "pattern": race.get("pattern"),
            "age_band": race.get("age_band"),
            "rating_band": race.get("rating_band"),
            "prize": race.get("prize"),
            "surface": race.get("surface"),
            "distance": race.get("distance"),
            "distance_round": race.get("distance_round"),
            "distance_f": race.get("distance_f"),
            "going": race.get("going"),
            "going_detailed": race.get("going_detailed"),
            "field_size": race.get("field_size"),
            "rail_movements": race.get("rail_movements"),
            "stalls": race.get("stalls"),
            "weather": race.get("weather"),
            "jumps": race.get("jumps"),
            "big_race": race.get("big_race"),
            "race_status": race.get("race_status"),
            "betting_forecast": race.get("betting_forecast"),
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
                    "rpr": runner.get("rpr"),
                    "topspeed": runner.get("ts"),
                    "performance_rating": runner.get("performance_rating"),
                    "speed_rating": runner.get("speed_rating"),
                    "trainer": runner.get("trainer"),
                    "trainer_id": runner.get("trainer_id"),
                    "trainer_14_days": runner.get("trainer_14_days"),
                    "trainer_rtf": runner.get("trainer_rtf"),
                    "jockey": runner.get("jockey"),
                    "jockey_id": runner.get("jockey_id"),
                    "jockey_claim": runner.get("jockey_claim_lbs"),
                    "recent_form": runner.get("form"),
                    "last_run": runner.get("last_run"),
                    "headgear": runner.get("headgear"),
                    "headgear_run": runner.get("headgear_run"),
                    "wind_surgery": runner.get("wind_surgery"),
                    "wind_surgery_run": runner.get("wind_surgery_run"),
                    "past_results_flags": runner.get("past_results_flags"),
                    "spotlight": runner.get("spotlight"),
                    "comment": runner.get("comment"),
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


def _race_id(runner: Runner) -> str | None:
    return _payload_id(runner, ("race_id",))


def _trainer_id(runner: Runner) -> str | None:
    return _payload_id(runner, ("trainer_id",))


def _jockey_id(runner: Runner) -> str | None:
    return _payload_id(runner, ("jockey_id",))


def _payload_id(runner: Runner, keys: tuple[str, ...]) -> str | None:
    payload = runner.source_payload
    source_runner = payload.get("source_runner")
    for source in (payload, source_runner if isinstance(source_runner, dict) else {}):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _analysis_params(
    runner: Runner,
    *,
    include_course: bool = True,
    include_distance: bool = True,
) -> dict[str, Any]:
    payload = runner.source_payload or {}
    params: dict[str, Any] = {
        "region": _region(payload),
        "type": _race_type(payload.get("race_type") or runner.race_type),
        "going": _going_code(runner.going),
        "race_class": _race_class_code(runner.race_class),
    }
    if include_course:
        params["course"] = payload.get("course_id")
    if include_distance:
        yards = _distance_yards(payload)
        if yards is not None:
            params["min_distance_y"] = max(0, yards - 110)
            params["max_distance_y"] = yards + 110
    return params


def _region(payload: dict[str, Any]) -> list[str] | None:
    value = payload.get("region")
    if value:
        return [str(value).lower()]
    return None


def _race_type(value: Any) -> list[str] | None:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in {"flat", "chase", "hurdle", "nh_flat"}:
        return [text]
    if "hurdle" in text:
        return ["hurdle"]
    if "chase" in text:
        return ["chase"]
    if "flat" in text:
        return ["flat"]
    return None


def _going_code(value: Any) -> list[str] | None:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return [text] if text else None


def _race_class_code(value: Any) -> list[str] | None:
    text = str(value or "").strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    return [f"class_{digits}"] if digits else None


def _distance_yards(payload: dict[str, Any]) -> int | None:
    for key in ("distance_y", "dist_y"):
        value = payload.get(key)
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            pass
    value = payload.get("distance_f")
    try:
        return int(float(str(value)) * 220)
    except (TypeError, ValueError):
        return None
