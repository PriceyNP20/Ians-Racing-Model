from __future__ import annotations

from datetime import date

from ian_racing_model.analysis_engines import (
    course_conditions_signal,
    pace_race_shape_signal,
    trainer_intent_signal,
)
from ian_racing_model.domain import Runner, RunnerScore
from racing_intelligence.plugins.registry import PluginRegistry
from racing_intelligence.scoring import intelligence_dataframe


def _score(**overrides) -> RunnerScore:
    runner_values = {
        "meeting_date": date(2026, 7, 14),
        "course": "Beverley",
        "off_time": "15:00",
        "race_name": "Intelligence Stakes",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 12,
        "horse": "Runner",
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
    runner_values.update(overrides.pop("runner", {}))
    values = {
        "runner": Runner(**runner_values),
        "total_score": 66,
        "confidence": 0.62,
        "recommendation": "PLACE",
        "fair_odds_placeholder": "",
        "win_probability": 0.12,
        "place_probability": 0.44,
        "fair_win_odds": 8.33,
        "fair_place_odds": 2.27,
        "win_value_edge": 0.01,
        "place_value_edge": 0.09,
        "components": [],
        "red_flags": [],
        "data_quality_warnings": [],
    }
    values.update(overrides)
    return RunnerScore(**values)


def test_intelligence_outputs_separate_win_and_place_probabilities() -> None:
    df = intelligence_dataframe([_score(runner={"horse": "Place Type"})])

    assert df.iloc[0]["win_probability"] != df.iloc[0]["place_probability"]
    assert "win_explanation" in df.columns
    assert "place_explanation" in df.columns


def test_intelligence_can_show_place_value_without_win_value() -> None:
    df = intelligence_dataframe(
        [
            _score(
                runner={"horse": "EW Type", "current_odds": "10/1"},
                win_probability=0.08,
                place_probability=0.42,
            )
        ]
    )

    row = df.iloc[0]
    assert row["recommendation"] in {"PLACE_VALUE", "PLACE_PROFILE"}
    assert row["place_value_edge"].startswith("+")


def test_plugin_registry_replaces_capabilities_by_name() -> None:
    registry = PluginRegistry()
    registry.register("racecards", object())

    assert registry.names() == ["racecards"]
    assert registry.get("racecards") is not None


def test_intelligence_dataframe_exposes_three_edge_engines() -> None:
    df = intelligence_dataframe(
        [
            _score(
                runner={
                    "horse": "Evidence Horse",
                    "source_payload": {
                        "pace_rating": 72,
                        "trainer_ae": 1.2,
                        "course_place_pct": 32,
                    },
                }
            )
        ]
    )

    assert {"pace_shape_score", "trainer_intent_score", "course_conditions_score"} <= set(df.columns)
    assert df.iloc[0]["pace_shape_score"] > 55
    assert df.iloc[0]["trainer_intent_score"] > 55
    assert df.iloc[0]["course_conditions_score"] > 55


def test_pace_race_shape_engine_does_not_invent_missing_data() -> None:
    signal = pace_race_shape_signal(_score(runner={"draw": None, "field_size": None}).runner)

    assert signal.data_quality == "missing"
    assert signal.confidence < 0.4


def test_trainer_intent_engine_uses_imported_trainer_indicators() -> None:
    runner = _score(runner={"source_payload": {"trainer_ae": 1.25, "trainer_14_day_win_pct": 22}}).runner
    signal = trainer_intent_signal(runner)

    assert signal.score > 60
    assert signal.data_quality == "ok"


def test_course_conditions_engine_uses_history_setup_evidence() -> None:
    runner = _score(
        runner={
            "course": "Beverley",
            "distance": "1m",
            "going": "Good",
            "source_payload": {
                "horse_history": [
                    {"course": "Beverley", "distance": "1m", "going": "Good", "position": "2"}
                ]
            },
        }
    ).runner
    signal = course_conditions_signal(runner)

    assert signal.score > 60
    assert signal.data_quality == "partial"
