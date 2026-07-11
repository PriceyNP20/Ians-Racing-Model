from __future__ import annotations

from datetime import date

import pandas as pd

from ian_racing_model.domain import RunnerScore


def scores_to_dataframe(scores: list[RunnerScore]) -> pd.DataFrame:
    rows = []
    for item in scores:
        runner = item.runner
        row = {
            "course": runner.course,
            "off_time": runner.off_time,
            "race": runner.race_name,
            "horse": runner.horse,
            "total_score": item.total_score,
            "confidence": item.confidence,
            "odds": runner.current_odds,
            "fair_odds": item.fair_odds_placeholder,
            "recommendation": item.recommendation,
            "warnings": "; ".join(item.data_quality_warnings),
        }
        for component in item.components:
            row[component.name] = component.score
        rows.append(row)
    return pd.DataFrame(rows)


def default_date() -> date:
    return date(2026, 7, 11)
