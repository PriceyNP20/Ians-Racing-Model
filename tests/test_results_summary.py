from __future__ import annotations

import pandas as pd

from ian_racing_model.results_summary import (
    daily_winning_placing_summary_dataframe,
    winning_placing_selections_dataframe,
)


def test_winning_placing_selection_hits_are_pick_type_aware() -> None:
    picks = pd.DataFrame(
        [
            {"meeting_date": "2026-07-14", "pick_type": "Winner pick", "horse": "Won", "outcome": "WIN", "result": "1"},
            {"meeting_date": "2026-07-14", "pick_type": "Winner pick", "horse": "Second", "outcome": "PLACED", "result": "2"},
            {"meeting_date": "2026-07-14", "pick_type": "Best EW pick", "horse": "Placed", "outcome": "PLACED", "result": "3"},
            {"meeting_date": "2026-07-14", "pick_type": "Best EW pick", "horse": "Lost", "outcome": "LOSE", "result": "8"},
        ]
    )

    hits = winning_placing_selections_dataframe(picks)

    assert set(hits["horse"]) == {"Won", "Placed"}


def test_daily_winning_placing_summary_lists_hit_horses() -> None:
    picks = pd.DataFrame(
        [
            {"meeting_date": "2026-07-14", "pick_type": "Winner pick", "horse": "Won", "outcome": "WIN", "result": "1"},
            {"meeting_date": "2026-07-14", "pick_type": "Best EW pick", "horse": "Placed", "outcome": "PLACED", "result": "3"},
            {"meeting_date": "2026-07-14", "pick_type": "Best EW pick", "horse": "Lost", "outcome": "LOSE", "result": "7"},
        ]
    )

    summary = daily_winning_placing_summary_dataframe(picks)

    assert summary.iloc[0]["winning_selections"] == 1
    assert summary.iloc[0]["ew_place_selections"] == 1
    assert "Won" in summary.iloc[0]["horses"]
    assert "Placed" in summary.iloc[0]["horses"]
