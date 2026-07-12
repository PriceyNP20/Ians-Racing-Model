from __future__ import annotations

from datetime import date

from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.edge import undervalued_edge_dataframe


def test_undervalued_edge_requires_positive_market_edge() -> None:
    runner_base = {
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
        "age": 5,
        "sex": "gelding",
        "draw": 3,
        "weight": "9-4",
        "official_rating": 82,
        "trainer": "Trainer",
        "jockey": "Jockey",
        "jockey_claim": None,
        "recent_form": "123",
        "is_non_runner": False,
        "source_payload": {"speed_figure": 91},
    }
    value_runner = Runner(**runner_base, horse="Real Edge", current_odds="10/1")
    poor_value_runner = Runner(**runner_base, horse="Too Short", current_odds="1/2")
    scores = [
        RunnerScore(value_runner, 68, 0.68, "EACH_WAY", "", 0.18, 0.48, 5.56, 2.08, 0.089, 0.15, [], [], []),
        RunnerScore(poor_value_runner, 68, 0.68, "WIN", "", 0.18, 0.48, 5.56, 2.08, -0.487, -0.22, [], [], []),
    ]

    edge = undervalued_edge_dataframe(scores)

    assert edge["horse"].tolist() == ["Real Edge"]
    assert edge.iloc[0]["edge_type"] == "EW/place"
    assert "speed figure" in edge.iloc[0]["evidence"].lower()


def test_undervalued_edge_uses_trainer_jockey_signals() -> None:
    runner = Runner(
        meeting_date=date(2026, 7, 11),
        course="Ascot",
        off_time="14:05",
        race_name="Trainer Jockey Edge",
        race_class="Class 4",
        race_type="Flat",
        surface="Turf",
        distance="1m",
        going="Good",
        field_size=10,
        horse="Hidden Signal",
        age=5,
        sex="gelding",
        draw=3,
        weight="9-4",
        official_rating=82,
        trainer="Trainer",
        jockey="Jockey",
        jockey_claim=None,
        recent_form="123",
        current_odds="12/1",
        source_payload={"source_runner": {"trainer_ae": 1.18, "jockey_strike_rate": "16"}},
    )
    score = RunnerScore(runner, 66, 0.64, "EACH_WAY", "", 0.16, 0.44, 6.25, 2.27, 0.083, 0.055, [], [], [])

    edge = undervalued_edge_dataframe([score])

    assert not edge.empty
    assert "trainer" in edge.iloc[0]["evidence"].lower()
    assert "jockey" in edge.iloc[0]["evidence"].lower()
