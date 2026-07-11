from __future__ import annotations

from dataclasses import replace
from datetime import date

from ian_racing_model.config import IAN_FORMULA_V3_1_WEIGHTS, SAMPLE_DATA_DIR
from ian_racing_model.domain import RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from ian_racing_model.providers.mock import MockRacingDataProvider
from ian_racing_model.ui import (
    picks_tracker_dataframe,
    picks_tracker_summary,
    screener_dataframe,
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


def test_screener_orders_top_matches_and_value_signal() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), "Ascot")
    scores = IanFormulaV31().score_runners(runners)
    screener = screener_dataframe(scores)
    assert not screener.empty
    assert screener.iloc[0]["score"] >= screener.iloc[-1]["score"]
    assert "value_edge_pct" in screener.columns


def test_picks_tracker_selects_winner_and_each_way_per_race() -> None:
    provider = MockRacingDataProvider(SAMPLE_DATA_DIR / "mock_racecard.json")
    runners, _ = provider.fetch_racecard(date(2026, 7, 11), None)
    scores = IanFormulaV31().score_runners(runners)
    tracker = picks_tracker_dataframe(scores)
    assert {"Winner pick", "Best EW pick"} <= set(tracker["pick_type"])
    assert tracker.groupby(["course", "off_time", "race"]).size().max() <= 2


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
