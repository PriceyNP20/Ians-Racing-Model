from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import IAN_FORMULA_V3_1_WEIGHTS, THE_RACING_API_CONFIG, Settings
from ian_racing_model.services import get_refresh_statuses


st.title("Settings")
settings = Settings()
st.write("Provider", settings.provider)
st.write("Database", settings.database_url)

st.subheader("Refresh policy")
st.dataframe(
    [
        {
            "data": "Racecards",
            "refresh window": f"{THE_RACING_API_CONFIG['racecard_refresh_minutes']} minutes",
        },
        {
            "data": "Results",
            "refresh window": f"{THE_RACING_API_CONFIG['results_refresh_minutes']} minutes",
        },
        {
            "data": "Horse history",
            "refresh window": f"{THE_RACING_API_CONFIG['horse_history_refresh_hours']} hours",
        },
        {
            "data": "Horse history runner cap",
            "refresh window": f"{THE_RACING_API_CONFIG['horse_history_max_runners']} runners",
        },
    ],
    width="stretch",
    hide_index=True,
)

st.subheader("Latest refresh status")
statuses = get_refresh_statuses(settings)
if statuses:
    st.dataframe(statuses, width="stretch", hide_index=True)
else:
    st.info("No refresh status has been recorded yet. Open a racecard page to trigger the first refresh.")

st.subheader("Ian Formula V3.1 weights")
st.dataframe(
    [{"component": name, "weight": weight} for name, weight in IAN_FORMULA_V3_1_WEIGHTS.items()],
    width="stretch",
    hide_index=True,
)
st.caption("Weights are code/config controlled in this MVP and total 100.")
