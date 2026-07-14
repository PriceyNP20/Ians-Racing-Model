from __future__ import annotations

from datetime import date

import pandas as pd

from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.acca import ew_accumulator_dataframe
from ian_racing_model.edge_lab import (
    closing_value_dataframe,
    edge_calibration_dataframe,
    edge_filter_recommendations,
    enhanced_undervalued_edge_dataframe,
    negative_value_dataframe,
)


def _runner(**overrides) -> Runner:
    values = {
        "meeting_date": date(2026, 7, 11),
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Edge Test",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 10,
        "horse": "Runner",
        "age": 5,
        "sex": "gelding",
        "draw": 3,
        "weight": "9-4",
        "official_rating": 82,
        "trainer": "Trainer",
        "jockey": "Jockey",
        "jockey_claim": None,
        "recent_form": "123",
        "current_odds": "10/1",
        "is_non_runner": False,
        "source_payload": {},
    }
    values.update(overrides)
    return Runner(**values)


def test_undervalued_edge_requires_positive_market_edge() -> None:
    value_runner = _runner(horse="Real Edge", current_odds="10/1", source_payload={"speed_figure": 91})
    poor_value_runner = _runner(horse="Too Short", current_odds="1/2", source_payload={"speed_figure": 91})
    scores = [
        RunnerScore(value_runner, 68, 0.68, "EACH_WAY", "", 0.18, 0.48, 5.56, 2.08, 0.089, 0.15, [], [], []),
        RunnerScore(poor_value_runner, 68, 0.68, "WIN", "", 0.18, 0.48, 5.56, 2.08, -0.487, -0.22, [], [], []),
    ]

    edge = enhanced_undervalued_edge_dataframe(scores)

    assert edge["horse"].tolist() == ["Real Edge"]
    assert edge.iloc[0]["edge_type"] == "EW/place"
    assert "speed figure" in edge.iloc[0]["evidence"].lower()


def test_negative_value_flags_short_overbet_runner() -> None:
    score = RunnerScore(
        _runner(horse="Overbet", current_odds="6/4"),
        64,
        0.47,
        "WATCH",
        "",
        0.22,
        0.41,
        4.55,
        2.44,
        -0.18,
        -0.05,
        [],
        ["poor draw"],
        [],
    )

    negative = negative_value_dataframe([score])

    assert not negative.empty
    assert "short price" in negative.iloc[0]["avoid_reason"]


def test_ew_accumulator_uses_top_six_place_profiles() -> None:
    scores = []
    for index in range(8):
        runner = _runner(
            horse=f"Place Profile {index}",
            off_time=f"14:{index:02d}",
            race_name=f"Race {index}",
            current_odds="8/1",
            recent_form="123",
            source_payload={"speed_figure": 82 + index, "trainer_ae": 1.1},
        )
        scores.append(
            RunnerScore(
                runner,
                60 + index,
                0.58 + index / 100,
                "EACH_WAY",
                "",
                0.08,
                0.38 + index / 100,
                12.5,
                2.6,
                0.02,
                0.06 + index / 100,
                [],
                [],
                [],
            )
        )

    acca = ew_accumulator_dataframe(scores)

    assert len(acca) == 6
    assert acca.iloc[0]["horse"] == "Place Profile 7"
    assert acca["acca_rank"].tolist() == [1, 2, 3, 4, 5, 6]


def test_ew_accumulator_takes_one_horse_per_race_and_requires_eight_runners() -> None:
    strong_same_race = RunnerScore(
        _runner(
            horse="Same Race Strong",
            off_time="15:00",
            race_name="Shared Race",
            field_size=10,
            current_odds="8/1",
            source_payload={"speed_figure": 92, "trainer_ae": 1.2},
        ),
        70,
        0.7,
        "EACH_WAY",
        "",
        0.1,
        0.52,
        10.0,
        2.1,
        0.01,
        0.1,
        [],
        [],
        [],
    )
    weaker_same_race = RunnerScore(
        _runner(
            horse="Same Race Weaker",
            off_time="15:00",
            race_name="Shared Race",
            field_size=10,
            current_odds="8/1",
            source_payload={"speed_figure": 88, "trainer_ae": 1.1},
        ),
        66,
        0.66,
        "EACH_WAY",
        "",
        0.1,
        0.48,
        10.0,
        2.1,
        0.01,
        0.08,
        [],
        [],
        [],
    )
    small_field = RunnerScore(
        _runner(
            horse="Small Field",
            off_time="16:00",
            race_name="Tiny Field",
            field_size=7,
            current_odds="8/1",
            source_payload={"speed_figure": 94, "trainer_ae": 1.2},
        ),
        72,
        0.72,
        "EACH_WAY",
        "",
        0.1,
        0.54,
        10.0,
        2.1,
        0.01,
        0.12,
        [],
        [],
        [],
    )

    acca = ew_accumulator_dataframe([strong_same_race, weaker_same_race, small_field], limit=6)

    assert acca["horse"].tolist() == ["Same Race Strong"]
    assert acca["field_size"].min() >= 8


def test_ew_accumulator_dedupes_same_display_race_with_text_variations() -> None:
    first = RunnerScore(
        _runner(
            horse="Display Race First",
            course="Beverley",
            off_time="3:48",
            race_name="Beverley Annual Badgeholders Handicap Stakes",
            field_size=10,
            current_odds="9/2",
            source_payload={"speed_figure": 92, "trainer_ae": 1.2},
        ),
        72,
        0.7,
        "EACH_WAY",
        "",
        0.1,
        0.54,
        10.0,
        2.1,
        0.01,
        0.1,
        [],
        [],
        [],
    )
    second = RunnerScore(
        _runner(
            horse="Display Race Second",
            course=" beverley ",
            off_time="03:48",
            race_name="  Beverley Annual Badgeholders Handicap Stakes  ",
            field_size=10,
            current_odds="12/1",
            source_payload={"speed_figure": 90, "trainer_ae": 1.1},
        ),
        70,
        0.68,
        "EACH_WAY",
        "",
        0.1,
        0.5,
        10.0,
        2.1,
        0.01,
        0.09,
        [],
        [],
        [],
    )

    acca = ew_accumulator_dataframe([first, second], limit=6)

    assert acca["horse"].tolist() == ["Display Race First"]


def test_ew_accumulator_dedupes_same_course_and_time_even_if_race_name_differs() -> None:
    first = RunnerScore(
        _runner(
            horse="Beverley First",
            course="Beverley",
            off_time="3:48",
            race_name="Beverley Annual Badgeholders Handicap Stakes",
            field_size=10,
            current_odds="9/2",
            source_payload={"speed_figure": 92, "trainer_ae": 1.2},
        ),
        72,
        0.7,
        "EACH_WAY",
        "",
        0.1,
        0.54,
        10.0,
        2.1,
        0.01,
        0.1,
        [],
        [],
        [],
    )
    second = RunnerScore(
        _runner(
            horse="Beverley Second",
            course="Beverley",
            off_time="03:48",
            race_name="Racing Again On Monday Evening Handicap Stakes",
            field_size=10,
            current_odds="12/1",
            source_payload={"speed_figure": 90, "trainer_ae": 1.1},
        ),
        70,
        0.68,
        "EACH_WAY",
        "",
        0.1,
        0.5,
        10.0,
        2.1,
        0.01,
        0.09,
        [],
        [],
        [],
    )

    acca = ew_accumulator_dataframe([first, second], limit=6)

    assert acca["horse"].tolist() == ["Beverley First"]


def test_edge_calibration_groups_settled_picks() -> None:
    picks = pd.DataFrame(
        [
            {"pick_type": "Best EW pick", "outcome": "PLACED", "odds": "12/1", "score": 65, "confidence": 0.64},
            {"pick_type": "Best EW pick", "outcome": "LOSE", "odds": "20/1", "score": 58, "confidence": 0.52},
            {"pick_type": "Winner pick", "outcome": "WIN", "odds": "4/1", "score": 72, "confidence": 0.7},
        ]
    )

    calibration = edge_calibration_dataframe(picks, "odds_band")

    assert not calibration.empty
    assert "edge_read" in calibration.columns
    assert "10.0 to 19.99" in set(calibration["odds_band"])


def test_edge_filter_recommendations_flag_weak_big_outsiders() -> None:
    picks = pd.DataFrame(
        [
            {"pick_type": "Best EW pick", "outcome": "LOSE", "odds": "20/1"},
            {"pick_type": "Best EW pick", "outcome": "LOSE", "odds": "25/1"},
            {"pick_type": "Best EW pick", "outcome": "LOSE", "odds": "33/1"},
            {"pick_type": "Winner pick", "outcome": "LOSE", "odds": "4/1"},
        ]
    )

    recommendations = edge_filter_recommendations(picks)

    assert any("20.0+ outsiders" in item for item in recommendations)


def test_closing_value_dataframe_marks_runner_that_beats_the_close() -> None:
    score = RunnerScore(
        _runner(
            horse="Early Value",
            current_odds="10/1",
            source_payload={"result_payload": {"result_runner": {"sp": "6/1"}}},
        ),
        67,
        0.64,
        "EACH_WAY",
        "",
        0.18,
        0.42,
        5.56,
        2.38,
        0.09,
        0.12,
        [],
        [],
        [],
    )

    closing_value = closing_value_dataframe([score])

    assert closing_value.iloc[0]["horse"] == "Early Value"
    assert closing_value.iloc[0]["clv_signal"] == "Beat close"
    assert closing_value.iloc[0]["closing_value"] == "+57.1%"
