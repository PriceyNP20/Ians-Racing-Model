from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

from ian_racing_model.config import THE_RACING_API_CONFIG
from ian_racing_model.domain import Runner
from ian_racing_model.providers.base import RacingDataProvider
from ian_racing_model.providers.mapping import map_runner, reject_mismatched_runners


class MockRacingDataProvider(RacingDataProvider):
    def __init__(self, sample_path: Path) -> None:
        self.sample_path = sample_path
        self.field_map = THE_RACING_API_CONFIG["field_map"]

    def fetch_racecard(self, meeting_date: date, course: str | None = None) -> tuple[list[Runner], dict[str, Any]]:
        raw = json.loads(self.sample_path.read_text(encoding="utf-8"))
        mapped = [runner for runner in (map_runner(item, self.field_map) for item in raw["runners"]) if runner is not None]
        return reject_mismatched_runners(mapped, meeting_date, course), raw
