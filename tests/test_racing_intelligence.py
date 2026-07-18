from __future__ import annotations

from datetime import date

from ian_racing_model.analysis_engines import (
    course_conditions_signal,
    pace_race_shape_signal,
    trainer_intent_signal,
)
from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from racing_intelligence.plugins.registry import PluginRegistry
from racing_intelligence.scoring import intelligence_dataframe
from racing_intelligence.scoring.v5 import (
    V5_ENGINE_WEIGHTS,
    V5_PLACE_WEIGHTS,
    V5_WIN_WEIGHTS,
    v5_analysis,
    validate_v5_weights,
)
from racing_intelligence.tracking import v5_tracker_dataframe, v5_tracker_summary


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


def test_v5_tracker_separates_win_and_place_results() -> None:
    tracker = v5_tracker_dataframe(
        [
            _score(
                runner={
                    "horse": "Placed Runner",
                    "field_size": 12,
                    "source_payload": {"result_position": 2},
                }
            )
        ],
        date(2026, 7, 14),
    )

    summary = v5_tracker_summary(tracker)

    assert set(tracker["pick_type"]) == {"V5 Win pick", "V5 Place pick"}
    assert summary["v5_win_rate"] == "0.0% (0/1)"
    assert summary["v5_place_rate"] == "100.0% (1/1)"


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


def test_engines_feed_existing_weighted_components() -> None:
    runner = _score(
        runner={
            "source_payload": {
                "pace_rating": 78,
                "trainer_ae": 1.25,
                "course_place_pct": 34,
            }
        }
    ).runner
    score = IanFormulaV31().score_runner(runner)
    components = {component.name: component for component in score.components}

    assert components["pace_and_draw"].score > 8
    assert components["target_race_intent"].score > 7
    assert components["course_suitability"].score > 6


def test_v5_weight_sets_total_100_and_separate_win_place_logic() -> None:
    validate_v5_weights()

    assert sum(V5_ENGINE_WEIGHTS.values()) == 100
    assert sum(V5_WIN_WEIGHTS.values()) == 100
    assert sum(V5_PLACE_WEIGHTS.values()) == 100
    assert V5_WIN_WEIGHTS != V5_PLACE_WEIGHTS
    assert V5_PLACE_WEIGHTS["suitability"] > V5_WIN_WEIGHTS["suitability"]
    assert V5_WIN_WEIGHTS["ability"] > V5_PLACE_WEIGHTS["ability"]


def test_v5_returns_distinct_win_and_place_indexes() -> None:
    score = _score(
        total_score=62,
        runner={
            "source_payload": {
                "official_rating": 78,
                "pace_rating": 70,
                "trainer_ae": 1.15,
                "course_place_pct": 45,
                "going_place_pct": 42,
                "distance_place_pct": 40,
                "horse_history": [
                    {"course": "Beverley", "distance": "1m", "going": "Good", "position": "2"},
                    {"course": "Beverley", "distance": "1m", "going": "Good", "position": "3"},
                ],
            }
        },
    )

    analysis = v5_analysis(score)

    assert analysis.place_index != analysis.win_index
    assert analysis.place_index > analysis.win_index
    assert analysis.engines["suitability"].score > 60


def test_v5_missing_data_lowers_confidence_without_inventing() -> None:
    analysis = v5_analysis(
        _score(
            confidence=0.4,
            win_value_edge=None,
            place_value_edge=None,
            runner={
                "draw": None,
                "field_size": None,
                "official_rating": None,
                "recent_form": None,
                "current_odds": None,
                "source_payload": {},
            },
        )
    )

    assert analysis.confidence < 0.5
    assert analysis.data_quality == "partial"
    assert any(signal.data_quality == "missing" for signal in analysis.engines.values())


def test_v5_score_outputs_remain_between_0_and_100() -> None:
    analysis = v5_analysis(
        _score(
            total_score=99,
            runner={
                "current_odds": "100/1",
                "source_payload": {
                    "timeform_rating": 180,
                    "pace_rating": 130,
                    "trainer_ae": 2.5,
                    "course_place_pct": 90,
                    "career_starts": 2,
                    "first_handicap": True,
                },
            },
        )
    )

    assert 0 <= analysis.win_index <= 100
    assert 0 <= analysis.place_index <= 100
    for signal in analysis.engines.values():
        assert 0 <= signal.score <= 100


def test_intelligence_dataframe_exposes_v5_engine_audit_columns() -> None:
    df = intelligence_dataframe([_score(runner={"horse": "V5 Audit"})])

    expected = {
        "v5_win_index",
        "v5_place_index",
        "v5_recommendation",
        "v5_confidence",
        "v5_data_quality",
        "ability_engine",
        "suitability_engine",
        "race_shape_engine",
        "trainer_intent_engine",
        "current_wellbeing_engine",
        "improvement_engine",
        "market_value_engine",
        "historical_performance_engine",
        "v5_explanation",
    }

    assert expected <= set(df.columns)
