from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import pandas as pd

from ian_racing_model.domain import RunnerScore
from racing_intelligence.scoring.v5 import v5_analysis
from racing_intelligence.scoring.v6 import v6_analysis


def v5_tracker_dataframe(scores: list[RunnerScore], meeting_date: date | None = None) -> pd.DataFrame:
    rows = []
    for race_scores in _race_groups(scores).values():
        runners = [score for score in race_scores if not score.runner.is_non_runner]
        if not runners:
            continue
        win_pick = max(runners, key=lambda score: v5_analysis(score).win_index)
        place_pick = max(runners, key=lambda score: v5_analysis(score).place_index)
        rows.append(_v5_pick_row(win_pick, "V5 Win pick", meeting_date))
        rows.append(_v5_pick_row(place_pick, "V5 Place pick", meeting_date))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["course", "off_time", "race", "pick_type"],
        ascending=[True, True, True, False],
    )


def v5_tracker_summary(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "outcome" not in df.columns:
        return {
            "v5_win_rate": "No settled picks",
            "v5_place_rate": "No settled picks",
        }

    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    pick_type = settled["pick_type"].astype(str).str.strip()
    win_rows = settled[pick_type.eq("V5 Win pick")]
    place_rows = settled[pick_type.eq("V5 Place pick")]
    return {
        "v5_win_rate": _ratio_text(int(win_rows["outcome"].eq("WIN").sum()), len(win_rows)),
        "v5_place_rate": _ratio_text(int(place_rows["outcome"].isin(["WIN", "PLACED"]).sum()), len(place_rows)),
    }


def v6_tracker_dataframe(scores: list[RunnerScore], meeting_date: date | None = None) -> pd.DataFrame:
    rows = []
    for race_scores in _race_groups(scores).values():
        runners = [score for score in race_scores if not score.runner.is_non_runner]
        if not runners:
            continue
        win_pick = max(runners, key=lambda score: v6_analysis(score, runners).win_index)
        place_pick = max(runners, key=lambda score: v6_analysis(score, runners).place_index)
        rows.append(_v6_pick_row(win_pick, "V6 Win pick", meeting_date, runners))
        rows.append(_v6_pick_row(place_pick, "V6 Place pick", meeting_date, runners))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["course", "off_time", "race", "pick_type"],
        ascending=[True, True, True, False],
    )


def v6_tracker_summary(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "outcome" not in df.columns:
        return {
            "v6_win_rate": "No settled picks",
            "v6_place_rate": "No settled picks",
        }

    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    pick_type = settled["pick_type"].astype(str).str.strip()
    win_rows = settled[pick_type.eq("V6 Win pick")]
    place_rows = settled[pick_type.eq("V6 Place pick")]
    return {
        "v6_win_rate": _ratio_text(int(win_rows["outcome"].eq("WIN").sum()), len(win_rows)),
        "v6_place_rate": _ratio_text(int(place_rows["outcome"].isin(["WIN", "PLACED"]).sum()), len(place_rows)),
    }


def _v5_pick_row(item: RunnerScore, pick_type: str, meeting_date: date | None) -> dict[str, Any]:
    runner = item.runner
    v5 = v5_analysis(item)
    position = _finish_position(runner.source_payload)
    place_cutoff = _place_cutoff(runner.field_size)
    return {
        "meeting_date": meeting_date.isoformat() if meeting_date else runner.meeting_date.isoformat(),
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "race_type": runner.race_type or "Unknown",
        "race_class": runner.race_class or "Unknown",
        "surface": runner.surface or "Unknown",
        "going": runner.going or "Unknown",
        "field_size": runner.field_size,
        "place_cutoff": place_cutoff,
        "pick_type": pick_type,
        "horse": runner.horse,
        "odds": runner.current_odds or "Unavailable",
        "v5_win_index": round(v5.win_index, 2),
        "v5_place_index": round(v5.place_index, 2),
        "v5_confidence": round(v5.confidence, 3),
        "v5_recommendation": v5.recommendation,
        "v5_data_quality": v5.data_quality,
        "result": str(position) if position is not None else "Awaiting result",
        "outcome": _pick_outcome(pick_type, position, place_cutoff),
        "v5_explanation": "; ".join(
            [
                v5.engines["ability"].explanation,
                v5.engines["suitability"].explanation,
                v5.engines["race_shape"].explanation,
                v5.engines["trainer_intent"].explanation,
            ]
        ),
    }


def _v6_pick_row(
    item: RunnerScore,
    pick_type: str,
    meeting_date: date | None,
    race_scores: list[RunnerScore],
) -> dict[str, Any]:
    runner = item.runner
    v6 = v6_analysis(item, race_scores)
    position = _finish_position(runner.source_payload)
    place_cutoff = _place_cutoff(runner.field_size)
    return {
        "meeting_date": meeting_date.isoformat() if meeting_date else runner.meeting_date.isoformat(),
        "course": runner.course,
        "off_time": runner.off_time,
        "race": runner.race_name,
        "race_type": runner.race_type or "Unknown",
        "race_class": runner.race_class or "Unknown",
        "surface": runner.surface or "Unknown",
        "going": runner.going or "Unknown",
        "field_size": runner.field_size,
        "place_cutoff": place_cutoff,
        "pick_type": pick_type,
        "horse": runner.horse,
        "odds": runner.current_odds or "Unavailable",
        "race_difficulty": v6.race_difficulty.grade,
        "v6_win_index": round(v6.win_index, 2),
        "v6_place_index": round(v6.place_index, 2),
        "v6_confidence": round(v6.confidence, 3),
        "v6_recommendation": v6.recommendation,
        "v6_data_quality": v6.data_quality,
        "result": str(position) if position is not None else "Awaiting result",
        "outcome": _pick_outcome(pick_type, position, place_cutoff),
        "v6_explanation": "; ".join(
            [
                v6.engines["ability"].explanation,
                v6.engines["horse_profile"].explanation,
                v6.engines["pace_race_shape"].explanation,
                v6.engines["trainer_intent"].explanation,
            ]
        ),
    }


def _race_groups(scores: list[RunnerScore]) -> dict[tuple[str, str, str], list[RunnerScore]]:
    grouped: dict[tuple[str, str, str], list[RunnerScore]] = defaultdict(list)
    for score in scores:
        runner = score.runner
        grouped[(runner.course, runner.off_time, runner.race_name)].append(score)
    return grouped


def _pick_outcome(pick_type: str, position: int | None, place_cutoff: int) -> str:
    if position is None:
        return "Awaiting result"
    if position == 1:
        return "WIN"
    if "Place" in pick_type and position <= place_cutoff:
        return "PLACED"
    if position == place_cutoff + 1:
        return "JUST MISSED"
    return "LOSE"


def _place_cutoff(field_size: int | None) -> int:
    if field_size is None:
        return 3
    if field_size >= 16:
        return 4
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return 1


def _finish_position(payload: dict[str, Any]) -> int | None:
    for key in ("result_position", "finish_position", "finishing_position", "position", "pos", "place"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            parsed = _finish_position(value)
            if parsed is not None:
                return parsed
        try:
            return int(str(value).strip().split("/")[0])
        except ValueError:
            continue
    return None


def _ratio_text(successes: int, total: int) -> str:
    if total == 0:
        return "No settled picks"
    return f"{(successes / total) * 100:.1f}% ({successes}/{total})"
