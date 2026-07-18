from __future__ import annotations

import pandas as pd


def winning_placing_selections_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "outcome" not in df.columns:
        return pd.DataFrame()

    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()

    settled["_selection_hit"] = settled.apply(selection_hit, axis=1)
    hits = settled[settled["_selection_hit"]].copy()
    if hits.empty:
        return pd.DataFrame()

    hits["hit_type"] = hits.apply(_hit_type, axis=1)
    columns = [
        "meeting_date",
        "course",
        "off_time",
        "race",
        "pick_type",
        "horse",
        "hit_type",
        "result",
        "outcome",
        "odds",
        "selection_score",
        "score",
        "confidence",
        "win_probability",
        "place_probability",
        "place_value_edge",
    ]
    available = [column for column in columns if column in hits.columns]
    sort_columns = [column for column in ("meeting_date", "course", "off_time", "pick_type") if column in hits.columns]
    return hits[available].sort_values(sort_columns)


def daily_winning_placing_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "meeting_date" not in df.columns:
        return pd.DataFrame()

    settled = df[~df["outcome"].eq("Awaiting result")].copy()
    if settled.empty:
        return pd.DataFrame()

    settled["_selection_hit"] = settled.apply(selection_hit, axis=1)
    settled["_hit_label"] = settled.apply(
        lambda row: f"{row.get('horse')} ({row.get('pick_type')}, {row.get('result')})",
        axis=1,
    )
    rows = []
    for day, day_rows in settled.groupby("meeting_date", sort=True):
        hits = day_rows[day_rows["_selection_hit"]]
        winner_hits = hits[hits["pick_type"].astype(str).str.strip().eq("Winner pick")]
        ew_hits = hits[hits["pick_type"].astype(str).str.strip().eq("Best EW pick")]
        rows.append(
            {
                "meeting_date": day,
                "settled": len(day_rows),
                "winning_selections": len(winner_hits),
                "ew_place_selections": len(ew_hits),
                "total_hits": len(hits),
                "hit_rate": _ratio_text(len(hits), len(day_rows)),
                "horses": "; ".join(hits["_hit_label"].tolist()) if not hits.empty else "None",
            }
        )
    return pd.DataFrame(rows).sort_values("meeting_date", ascending=False)


def selection_hit(row: pd.Series) -> bool:
    outcome = str(row.get("outcome", "")).upper()
    pick_type = str(row.get("pick_type", row.get("pick", ""))).strip().lower()
    if "winner" in pick_type or "win pick" in pick_type:
        return outcome == "WIN"
    if "ew" in pick_type or "place" in pick_type:
        return outcome in {"WIN", "PLACED"}
    return outcome in {"WIN", "PLACED"}


def _hit_type(row: pd.Series) -> str:
    pick_type = str(row.get("pick_type", "")).strip()
    outcome = str(row.get("outcome", "")).upper()
    if pick_type in {"Winner pick", "V5 Win pick", "V6 Win pick"} and outcome == "WIN":
        return "Winner"
    if pick_type in {"Best EW pick", "V5 Place pick", "V6 Place pick"} and outcome == "WIN":
        return "EW win"
    if pick_type in {"Best EW pick", "V5 Place pick", "V6 Place pick"}:
        return "EW placed"
    return outcome.title()


def _ratio_text(successes: int, total: int) -> str:
    if total == 0:
        return "No settled picks"
    return f"{(successes / total) * 100:.1f}% ({successes}/{total})"
