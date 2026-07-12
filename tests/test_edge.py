from __future__ import annotations

from datetime import date

import pandas as pd

from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.edge_lab import (
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
