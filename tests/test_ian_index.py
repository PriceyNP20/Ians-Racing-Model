from __future__ import annotations

from datetime import date

from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.ian_index import IAN_INDEX_V4_WEIGHTS, ian_index_place_dataframe


def _runner(**overrides) -> Runner:
    values = {
        "meeting_date": date(2026, 7, 14),
        "course": "Beverley",
        "off_time": "15:48",
        "race_name": "Trial Handicap",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 12,
        "horse": "Trial Runner",
        "age": 4,
        "sex": "gelding",
        "draw": 4,
        "weight": "9-4",
        "official_rating": 82,
        "trainer": "Trainer",
        "jockey": "Jockey",
        "jockey_claim": None,
        "recent_form": "123",
        "current_odds": "8/1",
        "is_non_runner": False,
        "source_payload": {},
    }
    values.update(overrides)
    return Runner(**values)


def _score(runner: Runner, **overrides) -> RunnerScore:
    values = {
        "total_score": 66,
        "confidence": 0.62,
        "recommendation": "PLACE",
        "fair_odds_placeholder": "",
        "win_probability": 0.12,
        "place_probability": 0.44,
        "fair_win_odds": 8.3,
        "fair_place_odds": 2.27,
        "win_value_edge": 0.01,
        "place_value_edge": 0.09,
        "components": [],
        "red_flags": [],
        "data_quality_warnings": [],
    }
    values.update(overrides)
    return RunnerScore(runner, **values)


def test_ian_index_weights_total_100() -> None:
    assert sum(IAN_INDEX_V4_WEIGHTS.values()) == 100


def test_ian_index_excludes_non_runners() -> None:
    live = _score(_runner(horse="Live Runner"))
    non_runner = _score(_runner(horse="Non Runner", is_non_runner=True))

    trial = ian_index_place_dataframe([non_runner, live])

    assert trial["horse"].tolist() == ["Live Runner"]


def test_ian_index_place_rating_stays_between_zero_and_100() -> None:
    score = _score(
        _runner(source_payload={"timeform_rating": 150, "beyer": 120, "rpr": 160, "trainer_ae": 1.5}),
        place_probability=0.8,
        place_value_edge=0.3,
    )

    trial = ian_index_place_dataframe([score])

    assert 0 <= trial.iloc[0]["place_rating"] <= 100


def test_ian_index_rewards_positive_place_value() -> None:
    value = _score(_runner(horse="Value Runner"), place_value_edge=0.12, place_probability=0.45)
    no_value = _score(_runner(horse="No Value Runner"), place_value_edge=-0.12, place_probability=0.45)

    trial = ian_index_place_dataframe([no_value, value])

    assert trial.iloc[0]["horse"] == "Value Runner"


def test_ian_index_missing_data_lowers_confidence() -> None:
    rich = _score(
        _runner(
            horse="Rich Data",
            source_payload={
                "timeform_rating": 104,
                "beyer": 86,
                "topspeed": 90,
                "rpr": 103,
                "pace_rating": 74,
                "trainer_ae": 1.12,
                "jockey_ae": 1.05,
                "course_place_pct": 28,
            },
        )
    )
    thin = _score(_runner(horse="Thin Data", source_payload={}))

    trial = ian_index_place_dataframe([rich, thin])
    confidence = dict(zip(trial["horse"], trial["confidence"], strict=True))

    assert confidence["Rich Data"] > confidence["Thin Data"]
