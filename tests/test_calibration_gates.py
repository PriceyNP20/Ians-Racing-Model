from __future__ import annotations

from datetime import date

from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.ui import _is_each_way_candidate, picks_tracker_dataframe


def _runner(**overrides) -> Runner:
    values = {
        "meeting_date": date(2026, 7, 11),
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Big Price Gate",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 12,
        "horse": "Runner",
        "age": 4,
        "sex": "gelding",
        "draw": 3,
        "weight": "9-4",
        "official_rating": 80,
        "trainer": "Trainer",
        "jockey": "Jockey",
        "jockey_claim": None,
        "recent_form": "789",
        "current_odds": "25/1",
        "is_non_runner": False,
        "source_payload": {},
    }
    values.update(overrides)
    return Runner(**values)


def test_unsupported_big_price_each_way_candidate_is_blocked() -> None:
    score = RunnerScore(
        _runner(horse="Tempting Price"),
        61,
        0.56,
        "EACH_WAY",
        "",
        0.08,
        0.46,
        12.5,
        2.17,
        0.042,
        0.25,
        [],
        [],
        [],
    )

    assert not _is_each_way_candidate(score)


def test_big_price_each_way_candidate_can_pass_with_real_evidence() -> None:
    score = RunnerScore(
        _runner(
            horse="Evidence Price",
            recent_form="231",
            source_payload={
                "speed_figure": 88,
                "trainer_ae": 1.12,
                "opening_odds": "33/1",
                "horse_history": [
                    {
                        "position": "2",
                        "sp": "33/1",
                        "race_class": "Class 4",
                        "going": "Good",
                        "distance": "1m",
                    }
                ],
            },
        ),
        62,
        0.58,
        "EACH_WAY",
        "",
        0.08,
        0.47,
        12.5,
        2.13,
        0.042,
        0.28,
        [],
        [],
        [],
    )
    tracker = picks_tracker_dataframe([score])

    assert _is_each_way_candidate(score)
    assert tracker.iloc[0]["evidence_gate"] == "Passed"
    assert tracker.iloc[0]["evidence_count"] >= 3
