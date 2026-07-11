from __future__ import annotations

from datetime import date, datetime
from typing import Any

from ian_racing_model.domain import Runner


def parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "nr", "non-runner"}


def normalize_course(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def map_runner(payload: dict[str, Any], field_map: dict[str, str]) -> Runner | None:
    get = lambda field: payload.get(field_map[field])
    meeting_date = parse_date(get("meeting_date"))
    if meeting_date is None or not get("course") or not get("horse"):
        return None
    return Runner(
        meeting_date=meeting_date,
        course=str(get("course")).strip(),
        off_time=str(get("off_time") or "").strip(),
        race_name=str(get("race_name") or "").strip(),
        race_class=_clean(get("race_class")),
        race_type=_clean(get("race_type")),
        surface=_clean(get("surface")),
        distance=_clean(get("distance")),
        going=_clean(get("going")),
        field_size=parse_int(get("field_size")),
        horse=str(get("horse")).strip(),
        age=parse_int(get("age")),
        sex=_clean(get("sex")),
        draw=parse_int(get("draw")),
        weight=_clean(get("weight")),
        official_rating=parse_int(get("official_rating")),
        trainer=_clean(get("trainer")),
        jockey=_clean(get("jockey")),
        jockey_claim=parse_int(get("jockey_claim")),
        recent_form=_clean(get("recent_form")),
        current_odds=_clean(get("current_odds")),
        is_non_runner=parse_bool(get("is_non_runner")),
        source_payload=payload,
    )


def reject_mismatched_runners(runners: list[Runner], requested_date: date, requested_course: str | None) -> list[Runner]:
    expected_course = normalize_course(requested_course) if requested_course else None
    accepted = []
    for runner in runners:
        if runner.meeting_date != requested_date:
            continue
        if expected_course and normalize_course(runner.course) != expected_course:
            continue
        accepted.append(runner)
    return accepted


def _clean(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()
