from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

from ian_racing_model.config import IAN_FORMULA_V3_1_WEIGHTS, SAMPLE_DATA_DIR
from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from ian_racing_model.providers.mock import MockRacingDataProvider
from ian_racing_model.services import _attach_horse_history, _attach_results
from ian_racing_model.ui import (
    outsider_last_time_dataframe,
    performance_by_odds_band,
    picks_tracker_breakdown,
    picks_tracker_dataframe,
    picks_tracker_summary,
    screener_dataframe,
    value_screener_dataframe,
)


def test_weights_total_100() -> None:
    assert sum(IAN_FORMULA_V3_1_WEIGHTS.values()) == 100


def test_wrong_date_data_is_rejected() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    assert {runner.meeting_date for runner in runners} == {date(2026, 7, 11)}
    assert "Yesterday Runner" not in {runner.horse for runner in runners}


def test_wrong_course_runners_are_rejected() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    assert {runner.course for runner in runners} == {"Ascot"}
    assert "Different Track" not in {runner.horse for runner in runners}


def test_all_courses_are_loaded_when_course_is_not_requested() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), None)
    assert {"Ascot", "York"} <= {runner.course for runner in runners}
    assert "Different Track" in {runner.horse for runner in runners}


def test_non_runners_are_excluded_from_scoring() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    assert "Declared Out" not in {score.runner.horse for score in scores}


def test_missing_data_lowers_confidence() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    complete = next(score for score in scores if score.runner.horse == "Measured Move")
    sparse = next(score for score in scores if score.runner.horse == "Quiet Baseline")
    assert sparse.confidence < complete.confidence
    assert sparse.data_quality_warnings


def test_score_remains_between_0_and_100() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    assert scores
    assert all(0 <= score.total_score <= 100 for score in scores)


def test_probabilities_and_fair_odds_are_calibrated_by_race() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    first_race = [
        score
        for score in scores
        if score.runner.race_name == "Ian Racing Model Sample Handicap"
    ]
    assert round(sum(score.win_probability for score in first_race), 2) == 1.0
    assert all(score.place_probability >= score.win_probability for score in first_race)
    assert all(score.fair_win_odds is not None for score in first_race)


def test_screener_orders_top_matches_and_value_signal() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    screener = screener_dataframe(scores)
    assert not screener.empty
    assert screener.iloc[0]["score"] >= screener.iloc[-1]["score"]
    assert "value_edge_pct" in screener.columns


def test_value_screener_uses_model_market_edge() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    value = value_screener_dataframe(scores)
    assert "fair_win_odds" in value.columns or value.empty
    assert "place_value_edge" in value.columns or value.empty
    assert "value_confidence" in value.columns or value.empty


def test_horse_history_improves_evidence_components() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    measured = next(runner for runner in runners if runner.horse == "Measured Move")
    enriched_runner = replace(
        measured,
        source_payload={
            **measured.source_payload,
            "horse_history": [
                {
                    "course": "Ascot",
                    "distance": measured.distance,
                    "race_class": measured.race_class,
                    "position": "1",
                }
            ],
        },
    )
    score = IanFormulaV31().score_runner(enriched_runner)
    components = {component.name: component for component in score.components}
    assert components["course_suitability"].confidence >= 0.7
    assert components["course_suitability"].score > 5.8
    assert "horse history" in components["current_performance"].explanation


def test_picks_tracker_selects_winner_and_each_way_per_race() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), None)
    scores = IanFormulaV31().score_runners(runners)
    tracker = picks_tracker_dataframe(scores)
    assert {"Winner pick", "Best EW pick"} <= set(tracker["pick_type"])
    assert tracker.groupby(["course", "off_time", "race"]).size().max() <= 2


def test_picks_tracker_separates_win_and_each_way_logic() -> None:
    runner_base = {
        "meeting_date": date(2026, 7, 11),
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Split Logic Stakes",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 10,
        "age": 4,
        "sex": "gelding",
        "draw": 3,
        "weight": "9-4",
        "official_rating": 80,
        "trainer": "Trainer",
        "jockey": "Jockey",
        "jockey_claim": None,
        "recent_form": "123",
        "is_non_runner": False,
        "source_payload": {},
    }
    winner = Runner(**runner_base, horse="Win Profile", current_odds="3/1")
    each_way = Runner(**runner_base, horse="EW Profile", current_odds="10/1")
    scores = [
        RunnerScore(winner, 70, 0.7, "WIN", "", 0.42, 0.48, 2.38, 2.08, 0.17, -0.02, [], [], []),
        RunnerScore(each_way, 64, 0.7, "EACH_WAY", "", 0.22, 0.62, 4.55, 1.61, 0.13, 0.44, [], [], []),
    ]
    tracker = picks_tracker_dataframe(scores)
    picks = tracker.set_index("pick_type")
    assert picks.loc["Winner pick", "horse"] == "Win Profile"
    assert picks.loc["Best EW pick", "horse"] == "EW Profile"
    assert "selection_reason" in tracker.columns


def test_picks_tracker_summary_counts_settled_results() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    settled_scores: list[RunnerScore] = []
    for index, score in enumerate(scores):
        settled_scores.append(
            replace(
                score,
                runner=replace(score.runner, source_payload={"result_position": index + 1}),
            )
        )
    tracker = picks_tracker_dataframe(settled_scores)
    summary = picks_tracker_summary(tracker)
    assert "%" in summary["winner_win_rate"]
    assert "%" in summary["ew_place_rate"]


def test_picks_tracker_summary_uses_separate_category_denominators() -> None:
    tracker = pd.DataFrame(
        [
            {"pick_type": "Winner pick", "outcome": "WIN"},
            {"pick_type": "Winner pick", "outcome": "LOSE"},
            {"pick_type": "Best EW pick", "outcome": "PLACED"},
        ]
    )
    summary = picks_tracker_summary(tracker)
    assert summary["winner_win_rate"] == "50.0% (1/2)"
    assert summary["ew_place_rate"] == "100.0% (1/1)"


def test_picks_tracker_breakdown_explains_matching_headline_rates() -> None:
    tracker = pd.DataFrame(
        [
            {"pick_type": "Winner pick", "outcome": "WIN", "result": "1", "place_cutoff": 3},
            {"pick_type": "Winner pick", "outcome": "LOSE", "result": "5", "place_cutoff": 3},
            {"pick_type": "Best EW pick", "outcome": "PLACED", "result": "3", "place_cutoff": 3},
            {"pick_type": "Best EW pick", "outcome": "LOSE", "result": "8", "place_cutoff": 3},
        ]
    )
    breakdown = picks_tracker_breakdown(tracker).set_index("pick_type")
    assert breakdown.loc["Winner pick", "wins"] == 1
    assert breakdown.loc["Winner pick", "places"] == 1
    assert breakdown.loc["Best EW pick", "wins"] == 0
    assert breakdown.loc["Best EW pick", "places"] == 1


def test_picks_tracker_breakdown_handles_string_place_cutoffs() -> None:
    tracker = pd.DataFrame(
        [
            {"pick_type": "Best EW pick", "outcome": "PLACED", "result": "2", "place_cutoff": "3"},
            {"pick_type": "Best EW pick", "outcome": "LOSE", "result": "7", "place_cutoff": "3"},
        ]
    )
    breakdown = picks_tracker_breakdown(tracker).set_index("pick_type")
    assert breakdown.loc["Best EW pick", "places"] == 1


def test_performance_by_odds_band_groups_settled_picks() -> None:
    tracker = pd.DataFrame(
        [
            {"pick_type": "Winner pick", "outcome": "WIN", "odds": "2/1"},
            {"pick_type": "Winner pick", "outcome": "LOSE", "odds": "12/1"},
            {"pick_type": "Best EW pick", "outcome": "PLACED", "odds": "8/1"},
        ]
    )
    by_band = performance_by_odds_band(tracker)
    assert not by_band.empty
    assert set(by_band["pick_type"]) == {"Winner pick", "Best EW pick"}


def test_outsider_last_time_signal_uses_verified_history_fields() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    score = next(score for score in scores if score.runner.horse == "Measured Move")
    enriched = replace(
        score,
        runner=replace(
            score.runner,
            source_payload={
                **score.runner.source_payload,
                "previous_result": {"position": "2", "sp_dec": "34.0"},
            },
        ),
    )
    signals = outsider_last_time_dataframe([enriched])
    assert not signals.empty
    assert signals.iloc[0]["horse"] == "Measured Move"


def test_outsider_last_time_signal_uses_fetched_horse_history() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    score = next(score for score in scores if score.runner.horse == "Measured Move")
    enriched = replace(
        score,
        runner=replace(
            score.runner,
            source_payload={
                **score.runner.source_payload,
                "horse_history": [{"position": "3", "sp": "20/1"}],
            },
        ),
    )
    signals = outsider_last_time_dataframe([enriched])
    assert not signals.empty
    assert signals.iloc[0]["signal"] == "Won/placed at big odds last time"


def test_horse_history_is_attached_without_breaking_card() -> None:
    class HistoryProvider:
        def fetch_horse_history(self, runner, limit: int = 6) -> dict:
            return {"results": [{"course": "Ascot", "position": "2", "sp": "33/1"}]}

    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    merged = _attach_horse_history(
        HistoryProvider(),
        settings=type(
            "SettingsStub",
            (),
            {
                "provider": "mock",
                "database_url": "sqlite:///:memory:",
            },
        )(),
        meeting_date=date(2026, 7, 11),
        course="Ascot",
        runners=runners[:1],
    )
    assert merged[0].source_payload["horse_history"][0]["position"] == "2"


def test_results_are_attached_to_matching_runners() -> None:
    class ResultsProvider:
        def fetch_results(self, meeting_date: date) -> dict:
            return {
                "results": [
                    {
                        "date": meeting_date.isoformat(),
                        "course": "Ascot",
                        "off": "14:05",
                        "race_name": "Ian Racing Model Sample Handicap",
                        "runners": [
                            {
                                "horse": "Measured Move",
                                "position": "1",
                            }
                        ],
                    }
                ]
            }

    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    merged, matched = _attach_results(
        ResultsProvider(),
        settings=type(
            "SettingsStub",
            (),
            {
                "provider": "mock",
                "database_url": "sqlite:///:memory:",
            },
        )(),
        meeting_date=date(2026, 7, 11),
        course="Ascot",
        runners=runners,
    )
    assert matched
    measured = next(runner for runner in merged if runner.horse == "Measured Move")
    assert measured.source_payload["result_position"] == "1"
