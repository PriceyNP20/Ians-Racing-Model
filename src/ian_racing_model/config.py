from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value)
    except Exception:
        pass
    return default


def get_int_setting(name: str, default: int) -> int:
    value = get_setting(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_setting(name: str, default: float) -> float:
    value = get_setting(name, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


IAN_FORMULA_V3_1_WEIGHTS: dict[str, int] = {
    "handicap_position": 18,
    "target_race_intent": 12,
    "pace_and_draw": 14,
    "course_suitability": 10,
    "distance_suitability": 10,
    "class_strength": 10,
    "current_performance": 10,
    "trainer_profile": 7,
    "jockey_suitability": 5,
    "market_value": 4,
}


THE_RACING_API_CONFIG = {
    "base_url": get_setting("RACING_API_BASE_URL", "https://api.theracingapi.com/v1"),
    "racecards_endpoint": get_setting("RACING_API_RACECARDS_ENDPOINT", "/racecards/pro"),
    "results_endpoint": get_setting("RACING_API_RESULTS_ENDPOINT", "/results"),
    "horse_results_endpoint": get_setting(
        "RACING_API_HORSE_RESULTS_ENDPOINT", "/horses/{horse_id}/results"
    ),
    "horse_history_limit": get_int_setting("RACING_API_HORSE_HISTORY_LIMIT", 6),
    "horse_history_max_runners": get_int_setting("RACING_API_HORSE_HISTORY_MAX_RUNNERS", 40),
    "horse_history_delay_seconds": get_float_setting(
        "RACING_API_HORSE_HISTORY_DELAY_SECONDS", 0.25
    ),
    "racecard_refresh_minutes": get_int_setting("RACE_CARD_REFRESH_MINUTES", 60),
    "results_refresh_minutes": get_int_setting("RESULTS_REFRESH_MINUTES", 15),
    "horse_history_refresh_hours": get_int_setting("HORSE_HISTORY_REFRESH_HOURS", 24),
    "racecards_region_codes": get_setting("RACING_API_REGION_CODES", "gb"),
    "auth": {
        "username_env": "RACING_API_USERNAME",
        "password_env": "RACING_API_PASSWORD",
    },
    "field_map": {
        "meeting_date": "date",
        "course": "course",
        "off_time": "off_time",
        "race_name": "race_name",
        "race_class": "race_class",
        "race_type": "race_type",
        "surface": "surface",
        "distance": "distance",
        "going": "going",
        "field_size": "field_size",
        "horse": "horse",
        "age": "age",
        "sex": "sex",
        "draw": "draw",
        "weight": "weight",
        "official_rating": "official_rating",
        "trainer": "trainer",
        "jockey": "jockey",
        "jockey_claim": "jockey_claim",
        "recent_form": "recent_form",
        "current_odds": "current_odds",
        "is_non_runner": "non_runner",
    },
}


@dataclass(frozen=True)
class Settings:
    provider: str = get_setting("RACING_DATA_PROVIDER", "mock")
    database_url: str = get_setting("DATABASE_URL", "sqlite:///ian_racing_model.db")
    sample_racecard_path: Path = field(
        default_factory=lambda: SAMPLE_DATA_DIR / "mock_racecard.json"
    )


def validate_weights(weights: dict[str, int] | None = None) -> None:
    selected = weights or IAN_FORMULA_V3_1_WEIGHTS
    total = sum(selected.values())
    if total != 100:
        raise ValueError(f"Ian Formula V3.1 weights must total 100, got {total}.")
