from __future__ import annotations

import pandas as pd

from ian_racing_model.results_summary import selection_hit


STYLE_WINNER = "background-color: #dcfce7; color: #14532d;"
STYLE_SIGNAL_COLUMNS = {
    "avoid_reason",
    "calibration_adjustment",
    "calibration_reason",
    "clv_signal",
    "danger_score",
    "edge_label",
    "edge_quality_score",
    "edge_score",
    "edge_read",
    "edge_type",
    "evidence_gate",
    "evidence_pillars",
    "gate_reason",
    "outcome",
    "pick",
    "recommendation",
    "screen",
    "signal",
    "value_confidence",
}


def picks_tracker_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        if _is_successful_selection(row):
            colour = STYLE_WINNER
        else:
            colour = ""
        return [colour] * len(row)

    return df.style.apply(row_style, axis=1)


def research_table_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        return [_research_row_style(row)] * len(row)

    return df.style.apply(row_style, axis=1)


def install_streamlit_table_styles() -> None:
    try:
        import streamlit as st
    except Exception:
        return

    if getattr(st, "_ian_table_styles_installed", False):
        return

    original_dataframe = st.dataframe
    def styled_dataframe(data=None, *args, **kwargs):
        if isinstance(data, pd.DataFrame) and _should_style_dataframe(data):
            data = research_table_style(data)
        return original_dataframe(data, *args, **kwargs)

    st.dataframe = styled_dataframe
    st._ian_table_styles_installed = True


def _should_style_dataframe(df: pd.DataFrame) -> bool:
    return bool(STYLE_SIGNAL_COLUMNS.intersection(set(df.columns)))


def _research_row_style(row: pd.Series) -> str:
    if _is_successful_selection(row):
        return STYLE_WINNER
    return ""


def _is_successful_selection(row: pd.Series) -> bool:
    return selection_hit(row)
