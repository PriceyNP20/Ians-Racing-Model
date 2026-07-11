from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ian_racing_model.config import Settings
from ian_racing_model.services import get_scored_card_result
from ian_racing_model.ui import default_date, scores_to_dataframe


st.set_page_config(page_title="Ian Racing Model", layout="wide")
st.title("Ian Racing Model")
st.caption("Read-only UK horse-racing research and tracking dashboard.")

settings = Settings()
selected_date = st.date_input("Meeting date", value=default_date())
course = st.text_input("Course", value="Ascot")

result = get_scored_card_result(selected_date, course, settings)
if result.warning:
    st.warning(result.warning)
st.caption(f"Data source: {result.provider}")
scores = result.scores
df = scores_to_dataframe(scores)

race_options = ["All races"] + sorted(df["race"].dropna().unique().tolist()) if not df.empty else ["All races"]
race = st.selectbox("Race", race_options)
if race != "All races":
    df = df[df["race"] == race]

st.subheader("Ranked runners")
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button(
    "Download CSV",
    data=df.to_csv(index=False),
    file_name=f"ian-racing-model-{selected_date.isoformat()}.csv",
    mime="text/csv",
)

with st.expander("Component explanations", expanded=False):
    for score in scores:
        if race != "All races" and score.runner.race_name != race:
            continue
        st.markdown(f"**{score.runner.horse}** - {score.recommendation}")
        st.write(
            {
                component.name: {
                    "score": component.score,
                    "confidence": component.confidence,
                    "data_quality": component.data_quality,
                    "explanation": component.explanation,
                }
                for component in score.components
            }
        )
