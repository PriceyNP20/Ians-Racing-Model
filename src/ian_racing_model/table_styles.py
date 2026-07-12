from __future__ import annotations

import pandas as pd


STYLE_POSITIVE = "background-color: #fef3c7; color: #78350f;"
STYLE_NEGATIVE = "background-color: #fee2e2; color: #7f1d1d;"
STYLE_WATCH = "background-color: #dbeafe; color: #1e3a8a;"
STYLE_MUTED = "background-color: #f3f4f6; color: #374151;"
STYLE_SIGNAL_COLUMNS = {
    "avoid_reason",
    "clv_signal",
    "danger_score",
    "edge_score",
    "edge_read",
    "edge_type",
    "outcome",
    "pick",
    "recommendation",
    "screen",
    "signal",
    "value_confidence",
}


def picks_tracker_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        status = str(row.get("outcome", "")).upper()
        if status in {"WIN", "PLACED"}:
            colour = STYLE_POSITIVE
        elif status == "LOSE":
            colour = STYLE_NEGATIVE
        elif status in {"JUST LOST", "JUST MISSED"}:
            colour = STYLE_WATCH
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
    original_caption = st.caption

    def styled_dataframe(data=None, *args, **kwargs):
        if isinstance(data, pd.DataFrame) and _should_style_dataframe(data):
            data = research_table_style(data)
        return original_dataframe(data, *args, **kwargs)

    def styled_caption(body=None, *args, **kwargs):
        if isinstance(body, str):
            body = body.replace("Green means won or placed", "Yellow means won or placed")
        return original_caption(body, *args, **kwargs)

    st.dataframe = styled_dataframe
    st.caption = styled_caption
    st._ian_table_styles_installed = True


def _should_style_dataframe(df: pd.DataFrame) -> bool:
    return bool(STYLE_SIGNAL_COLUMNS.intersection(set(df.columns)))


def _research_row_style(row: pd.Series) -> str:
    outcome = str(row.get("outcome", "")).upper()
    if outcome in {"WIN", "PLACED"}:
        return STYLE_POSITIVE
    if outcome == "LOSE":
        return STYLE_NEGATIVE
    if outcome in {"JUST LOST", "JUST MISSED"}:
        return STYLE_WATCH

    if any(name in row.index for name in ("avoid_reason", "danger_score")):
        return STYLE_NEGATIVE

    text = " | ".join(str(value).lower() for value in row.to_dict().values())
    recommendation = str(row.get("recommendation", "")).upper()
    pick = str(row.get("pick", "")).lower()
    screen = str(row.get("screen", "")).lower()
    signal = str(row.get("signal", "")).lower()
    edge_type = str(row.get("edge_type", "")).lower()
    value_confidence = str(row.get("value_confidence", "")).lower()

    if any(token in text for token in ("negative", "avoid", "danger", "lost value", "drifting", "weak value", "overbet", "bad pocket")):
        return STYLE_NEGATIVE
    if (
        recommendation in {"WIN", "EACH_WAY", "PLACE"}
        or any(token in pick for token in ("winner", "ew", "value"))
        or any(token in screen for token in ("edge", "value", "win"))
        or any(token in signal for token in ("edge", "value", "supported", "placed"))
        or any(token in edge_type for token in ("edge", "value", "win"))
        or any(token in text for token in ("positive pocket", "profitable-looking pocket"))
        or "strong value" in value_confidence
    ):
        return STYLE_POSITIVE
    if recommendation == "WATCH" or any(token in text for token in ("watch", "monitor", "held price", "stable", "speculative")):
        return STYLE_WATCH
    if recommendation == "PASS":
        return STYLE_MUTED
    return ""
