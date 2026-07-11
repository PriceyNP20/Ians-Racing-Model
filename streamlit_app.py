from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ian Racing Model", layout="wide")

WEIGHTS = {
    "handicap_position": 18,
    "target_race_intent": 12,
    "pace_and_draw": 14,
    "course_suitability": 10,
    "distance_suitability": 10,
    "class_strength": 10,
    "current_performance": 10,
    "trainer_profile": 7,
    "jockey_suitability": 5,
    "market_value": 4,
}

MOCK_RUNNERS: list[dict[str, Any]] = [
    {
        "date": "2026-07-11",
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Ian Racing Model Sample Handicap",
        "race_class": "Class 3",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m 2f",
        "going": "Good",
        "field_size": 8,
        "horse": "Measured Move",
        "age": 4,
        "sex": "Gelding",
        "draw": 2,
        "weight": "9-4",
        "official_rating": 84,
        "trainer": "A Trainer",
        "jockey": "B Jockey",
        "jockey_claim": 0,
        "recent_form": "3121",
        "current_odds": "5/1",
        "non_runner": False,
    },
    {
        "date": "2026-07-11",
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Ian Racing Model Sample Handicap",
        "race_class": "Class 3",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m 2f",
        "going": "Good",
        "field_size": 8,
        "horse": "Wide Question",
        "age": 5,
        "sex": "Mare",
        "draw": 8,
        "weight": "9-10",
        "official_rating": 98,
        "trainer": "C Trainer",
        "jockey": "D Jockey",
        "jockey_claim": 0,
        "recent_form": "78-5",
        "current_odds": "15/8",
        "non_runner": False,
    },
    {
        "date": "2026-07-11",
        "course": "Ascot",
        "off_time": "14:40",
        "race_name": "Sample Novice Stakes",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "7f",
        "going": "Good",
        "field_size": 6,
        "horse": "Quiet Baseline",
        "age": 3,
        "sex": "Colt",
        "draw": 3,
        "weight": "9-2",
        "official_rating": None,
        "trainer": "E Trainer",
        "jockey": "F Jockey",
        "jockey_claim": 3,
        "recent_form": None,
        "current_odds": None,
        "non_runner": False,
    },
    {
        "date": "2026-07-11",
        "course": "Ascot",
        "off_time": "14:40",
        "race_name": "Sample Novice Stakes",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "7f",
        "going": "Good",
        "field_size": 6,
        "horse": "Declared Out",
        "age": 3,
        "sex": "Filly",
        "draw": 4,
        "weight": "8-13",
        "official_rating": 76,
        "trainer": "G Trainer",
        "jockey": "H Jockey",
        "jockey_claim": 0,
        "recent_form": "221",
        "current_odds": "4/1",
        "non_runner": True,
    },
    {
        "date": "2026-07-10",
        "course": "Ascot",
        "off_time": "14:05",
        "race_name": "Wrong Date Race",
        "race_class": "Class 5",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "1m",
        "going": "Good",
        "field_size": 5,
        "horse": "Yesterday Runner",
        "age": 4,
        "sex": "Gelding",
        "draw": 1,
        "weight": "9-0",
        "official_rating": 70,
        "trainer": "I Trainer",
        "jockey": "J Jockey",
        "jockey_claim": 0,
        "recent_form": "111",
        "current_odds": "2/1",
        "non_runner": False,
    },
    {
        "date": "2026-07-11",
        "course": "York",
        "off_time": "15:10",
        "race_name": "Wrong Course Race",
        "race_class": "Class 4",
        "race_type": "Flat",
        "surface": "Turf",
        "distance": "6f",
        "going": "Good",
        "field_size": 9,
        "horse": "Different Track",
        "age": 4,
        "sex": "Gelding",
        "draw": 5,
        "weight": "9-1",
        "official_rating": 80,
        "trainer": "K Trainer",
        "jockey": "L Jockey",
        "jockey_claim": 0,
        "recent_form": "222",
        "current_odds": "7/2",
        "non_runner": False,
    },
]


@dataclass(frozen=True)
class Feature:
    score: float
    confidence: float
    quality: str
    explanation: str


def decimal_odds(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        try:
            num, den = value.split("/", 1)
            return float(num) / float(den) + 1.0
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def weighted(name: str, feature: Feature) -> dict[str, Any]:
    return {
        "name": name,
        "score": round(feature.score / 100 * WEIGHTS[name], 2),
        "confidence": feature.confidence,
        "quality": feature.quality,
        "explanation": feature.explanation,
    }


def score_runner(r: dict[str, Any]) -> dict[str, Any]:
    components = []
    rating = r.get("official_rating")
    odds = decimal_odds(r.get("current_odds"))
    form = r.get("recent_form") or ""

    if rating is None or not r.get("weight"):
        hp = Feature(45, 0.45, "missing", "Official rating or weight unavailable.")
    elif rating >= 96:
        hp = Feature(38, 0.8, "ok", "High handicap mark creates limited margin.")
    elif rating >= 86:
        hp = Feature(58, 0.8, "ok", "Competitive but not obviously well treated.")
    else:
        hp = Feature(72, 0.75, "ok", "Handicap mark looks manageable from available data.")
    components.append(weighted("handicap_position", hp))

    if not form:
        tri = Feature(45, 0.4, "missing", "Recent form unavailable; intent cannot be inferred.")
    elif any(ch in form for ch in "12"):
        tri = Feature(72, 0.65, "partial", "Recent placing suggests active campaign.")
    elif "-" in form:
        tri = Feature(42, 0.55, "partial", "Break in form string may indicate absence.")
    else:
        tri = Feature(55, 0.55, "partial", "Intent inferred only from limited form string.")
    components.append(weighted("target_race_intent", tri))

    draw = r.get("draw")
    field_size = r.get("field_size")
    if draw is None or field_size is None:
        pd = Feature(45, 0.4, "missing", "Draw or field size unavailable.")
    elif draw > max(8, field_size * 0.75):
        pd = Feature(35, 0.7, "ok", "Wide draw may make the race shape harder.")
    elif draw <= max(2, field_size * 0.25):
        pd = Feature(68, 0.65, "ok", "Low draw is a possible tactical positive.")
    else:
        pd = Feature(58, 0.6, "ok", "Draw looks broadly neutral.")
    components.append(weighted("pace_and_draw", pd))

    components.append(weighted("course_suitability", Feature(58, 0.45, "partial", "No course-history feed available; neutral baseline used.")))
    components.append(weighted("distance_suitability", Feature(58 if r.get("distance") else 40, 0.45 if r.get("distance") else 0.35, "partial" if r.get("distance") else "missing", "Distance present but prior suitability unavailable." if r.get("distance") else "Distance unavailable.")))
    components.append(weighted("class_strength", Feature(62 if r.get("race_class") else 45, 0.55 if r.get("race_class") else 0.35, "partial" if r.get("race_class") else "missing", "Class band inferred from imported label only." if r.get("race_class") else "Race class unavailable.")))

    digits = [int(ch) for ch in form if ch.isdigit()]
    if not form:
        current = Feature(42, 0.35, "missing", "Recent form unavailable.")
    elif not digits:
        current = Feature(44, 0.45, "partial", "Form string has no finishing positions.")
    else:
        avg = sum(digits[:4]) / min(len(digits), 4)
        current = Feature(max(30, min(80, 86 - avg * 8)), 0.65, "partial", "Scored from recent finishing positions only.")
    components.append(weighted("current_performance", current))

    components.append(weighted("trainer_profile", Feature(56 if r.get("trainer") else 45, 0.4 if r.get("trainer") else 0.35, "partial" if r.get("trainer") else "missing", "Trainer strike-rate data unavailable; neutral baseline used." if r.get("trainer") else "Trainer unavailable.")))
    components.append(weighted("jockey_suitability", Feature((55 if r.get("jockey") else 42) + (5 if r.get("jockey_claim") else 0), 0.45 if r.get("jockey") else 0.35, "partial" if r.get("jockey") else "missing", "Jockey fit inferred only from booking and claim." if r.get("jockey") else "Jockey unavailable.")))

    if odds is None:
        market = Feature(42, 0.35, "missing", "Current odds unavailable.")
    elif odds < 2.5:
        market = Feature(40, 0.55, "partial", "Short price needs value proof not yet available.")
    elif odds >= 8:
        market = Feature(68, 0.5, "partial", "Bigger price may offer value, pending calibration.")
    else:
        market = Feature(58, 0.5, "partial", "Market price is within a workable range.")
    components.append(weighted("market_value", market))

    flags = []
    if r.get("race_class") and any(x in r["race_class"].lower() for x in ["class 1", "class 2"]):
        flags.append("class rise")
    if draw and field_size and draw > max(8, field_size * 0.75):
        flags.append("poor draw")
    if rating and rating >= 96:
        flags.append("high handicap mark")
    if r.get("going") and any(x in r["going"].lower() for x in ["heavy", "firm"]):
        flags.append("going concern")
    if "-" in form:
        flags.append("long absence")
    if form.upper().count("F") + form.upper().count("U") >= 2:
        flags.append("repeated jumping errors")
    if odds is not None and odds < 2.5:
        flags.append("short price without value")

    total = max(0.0, min(100.0, round(sum(c["score"] for c in components) - min(20, len(flags) * 3), 2)))
    confidence = round(sum(c["confidence"] for c in components) / len(components), 2)
    if total >= 78 and confidence >= 0.65:
        recommendation = "WIN"
    elif total >= 68 and confidence >= 0.55 and (odds is None or odds >= 5):
        recommendation = "EACH_WAY"
    elif total >= 62 and confidence >= 0.5:
        recommendation = "PLACE"
    elif total >= 52:
        recommendation = "WATCH"
    else:
        recommendation = "PASS"

    warnings = [f"{c['name']}: {c['explanation']}" for c in components if c["quality"] != "ok"] + flags
    return {
        **r,
        "total_score": total,
        "confidence": confidence,
        "recommendation": recommendation,
        "fair_odds": "TBD after results calibration",
        "warnings": "; ".join(warnings),
        **{c["name"]: c["score"] for c in components},
    }


st.title("Ian Racing Model")
st.caption("Read-only UK horse-racing research dashboard. No bet placement functionality.")

with st.sidebar:
    page = st.radio("Page", ["Today's cards", "Top selections", "Results", "Model performance", "Settings"])

if page == "Results":
    st.info("Results import is reserved for the next milestone. No results are invented in this MVP.")
    st.stop()
if page == "Model performance":
    st.info("Performance tracking will activate after verified results have been imported.")
    st.stop()
if page == "Settings":
    st.subheader("Ian Formula V3.1 weights")
    st.dataframe(pd.DataFrame([{"component": k, "weight": v} for k, v in WEIGHTS.items()]), hide_index=True, use_container_width=True)
    st.write("Weights total", sum(WEIGHTS.values()))
    st.stop()

available_dates = sorted({r["date"] for r in MOCK_RUNNERS})
selected_date = st.date_input("Meeting date", value=pd.to_datetime("2026-07-11").date())
selected_date_text = selected_date.isoformat()
all_courses = sorted({r["course"] for r in MOCK_RUNNERS})
selected_course = st.selectbox("Course", all_courses, index=all_courses.index("Ascot") if "Ascot" in all_courses else 0)

requested = [r for r in MOCK_RUNNERS if r["date"] == selected_date_text and r["course"] == selected_course]
rejected_count = len(MOCK_RUNNERS) - len(requested)
non_runner_count = sum(1 for r in requested if r["non_runner"])
eligible = [r for r in requested if not r["non_runner"]]
scored = sorted([score_runner(r) for r in eligible], key=lambda r: r["total_score"], reverse=True)

race_options = ["All races"] + sorted({f"{r['off_time']} - {r['race_name']}" for r in scored})
selected_race = st.selectbox("Race", race_options)
if selected_race != "All races":
    scored = [r for r in scored if f"{r['off_time']} - {r['race_name']}" == selected_race]

if page == "Top selections":
    scored = [r for r in scored if r["recommendation"] in {"WIN", "EACH_WAY", "PLACE"}]

left, mid, right, far = st.columns(4)
left.metric("Eligible runners", len(scored))
mid.metric("Rejected records", rejected_count)
right.metric("Non-runners excluded", non_runner_count)
far.metric("Top score", scored[0]["total_score"] if scored else 0)

if not scored:
    st.warning("No eligible runners match the selected card.")
    st.stop()

df = pd.DataFrame(scored)
columns = [
    "off_time", "race_name", "horse", "total_score", "confidence", "current_odds", "fair_odds", "recommendation",
    "handicap_position", "target_race_intent", "pace_and_draw", "course_suitability", "distance_suitability",
    "class_strength", "current_performance", "trainer_profile", "jockey_suitability", "market_value", "warnings"
]
st.dataframe(df[columns], hide_index=True, use_container_width=True)
st.download_button("Download CSV", df[columns].to_csv(index=False), file_name="ian-racing-model.csv", mime="text/csv")

with st.expander("Audit notes"):
    st.write("Wrong-date and wrong-course records are rejected before scoring.")
    st.write("Non-runners are excluded from scoring and recommendations.")
    st.write("Unavailable evidence lowers confidence and is shown in warnings.")
