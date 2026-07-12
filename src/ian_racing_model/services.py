from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import date
from pathlib import Path
import time
from typing import Any

from ian_racing_model.config import Settings, THE_RACING_API_CONFIG
from ian_racing_model.domain import Runner, RunnerScore
from ian_racing_model.model.scoring import IanFormulaV31
from ian_racing_model.providers.factory import build_provider
from ian_racing_model.providers.mock import MockRacingDataProvider
from ian_racing_model.storage.db import (
    list_refresh_statuses,
    make_session_factory,
    record_refresh_status,
    store_raw_response,
)

try:
    import streamlit as st
except Exception:
    st = None


@dataclass(frozen=True)
class ScoredCardResult:
    scores: list[RunnerScore]
    provider: str
    warning: str | None = None
    results_imported: bool = False


def _store_raw(settings: Settings, meeting_date: date, course: str | None, payload: dict[str, Any]) -> None:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        store_raw_response(session, settings.provider, meeting_date.isoformat(), course, payload)


def _record_refresh(
    settings: Settings,
    meeting_date: date,
    course: str | None,
    source: str,
    status: str,
    message: str | None = None,
) -> None:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        record_refresh_status(
            session,
            settings.provider,
            meeting_date.isoformat(),
            course,
            source,
            status,
            message,
        )


def get_refresh_statuses(settings: Settings, limit: int = 50) -> list[dict]:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        return list_refresh_statuses(session, limit=limit)


def get_scored_card_result(
    meeting_date: date, course: str | None, settings: Settings
) -> ScoredCardResult:
    if st is not None:
        return _cached_scored_card_result(
            meeting_date.isoformat(),
            course,
            settings.provider,
            settings.database_url,
            str(settings.sample_racecard_path),
        )
    return _load_scored_card_result(meeting_date, course, settings)


if st is not None:
    @st.cache_resource(
        ttl=int(THE_RACING_API_CONFIG.get("racecard_refresh_minutes", 60)) * 60,
        show_spinner=False,
    )
    def _cached_scored_card_result(
        meeting_date_iso: str,
        course: str | None,
        provider: str,
        database_url: str,
        sample_racecard_path: str,
    ) -> ScoredCardResult:
        settings = Settings(
            provider=provider,
            database_url=database_url,
            sample_racecard_path=Path(sample_racecard_path),
        )
        return _load_scored_card_result(date.fromisoformat(meeting_date_iso), course, settings)
else:
    def _cached_scored_card_result(
        meeting_date_iso: str,
        course: str | None,
        provider: str,
        database_url: str,
        sample_racecard_path: str,
    ) -> ScoredCardResult:
        settings = Settings(
            provider=provider,
            database_url=database_url,
            sample_racecard_path=Path(sample_racecard_path),
        )
        return _load_scored_card_result(date.fromisoformat(meeting_date_iso), course, settings)


def _load_scored_card_result(
    meeting_date: date, course: str | None, settings: Settings
) -> ScoredCardResult:
    provider = build_provider(settings)
    try:
        runners, raw = provider.fetch_racecard(meeting_date, course)
        _store_raw(settings, meeting_date, course, raw)
        _record_refresh(settings, meeting_date, course, "racecard", "success")
        runners, results_imported = _attach_results(provider, settings, meeting_date, course, runners)
        runners = _attach_horse_history(provider, settings, meeting_date, course, runners)
        return ScoredCardResult(
            scores=IanFormulaV31().score_runners(runners),
            provider=settings.provider,
            results_imported=results_imported,
        )
    except Exception as exc:
        error_payload = {
            "error": type(exc).__name__,
            "message": str(exc),
            "fallback": "mock",
        }
        _store_raw(settings, meeting_date, course, error_payload)
        _record_refresh(settings, meeting_date, course, "racecard", "error", str(exc))
        if settings.provider.lower() == "mock":
            raise

        mock_provider = MockRacingDataProvider(settings.sample_racecard_path)
        runners, raw = mock_provider.fetch_racecard(meeting_date, course)
        _store_raw(settings, meeting_date, course, raw)
        return ScoredCardResult(
            scores=IanFormulaV31().score_runners(runners),
            provider="mock",
            warning=(
                "Live Racing API data could not be loaded, so this view is using "
                "sample data. Check Streamlit logs for the API status code/details."
            ),
        )


def get_scored_card(meeting_date: date, course: str | None, settings: Settings) -> list[RunnerScore]:
    return get_scored_card_result(meeting_date, course, settings).scores


def _attach_results(
    provider,
    settings: Settings,
    meeting_date: date,
    course: str | None,
    runners: list[Runner],
) -> tuple[list[Runner], bool]:
    try:
        raw_results = provider.fetch_results(meeting_date)
    except Exception as exc:
        _store_raw(
            settings,
            meeting_date,
            course,
            {
                "error": type(exc).__name__,
                "message": str(exc),
                "source": "results",
            },
        )
        _record_refresh(settings, meeting_date, course, "results", "error", str(exc))
        return runners, False

    result_items = _flatten_results(raw_results)
    if not result_items:
        _record_refresh(settings, meeting_date, course, "results", "empty")
        return runners, False

    _store_raw(settings, meeting_date, course, {"results": raw_results})
    _record_refresh(settings, meeting_date, course, "results", "success")
    lookup = {_result_key(item): item for item in result_items}
    merged: list[Runner] = []
    matched = False
    for runner in runners:
        result = lookup.get(_runner_key(runner))
        if result is None:
            merged.append(runner)
            continue
        matched = True
        merged.append(
            replace(
                runner,
                source_payload={
                    **runner.source_payload,
                    "result_position": result.get("position"),
                    "result_payload": result,
                },
            )
        )
    return merged, matched


def _attach_horse_history(
    provider,
    settings: Settings,
    meeting_date: date,
    course: str | None,
    runners: list[Runner],
) -> list[Runner]:
    limit = int(THE_RACING_API_CONFIG.get("horse_history_limit", 6))
    max_runners = int(THE_RACING_API_CONFIG.get("horse_history_max_runners", 80))
    delay_seconds = _history_delay(settings)
    merged: list[Runner] = []
    raw_audit: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    fetched_count = 0

    for index, runner in enumerate(runners):
        if runner.is_non_runner or index >= max_runners:
            merged.append(runner)
            continue
        if delay_seconds and fetched_count:
            time.sleep(delay_seconds)
        try:
            raw_history = provider.fetch_horse_history(runner, limit=limit)
            fetched_count += 1
        except Exception as exc:
            errors.append({"horse": runner.horse, "error": type(exc).__name__, "message": str(exc)})
            merged.append(runner)
            continue

        history_items = _flatten_horse_history(raw_history)
        if not history_items:
            merged.append(runner)
            continue

        raw_audit.append({"horse": runner.horse, "history": raw_history})
        merged.append(
            replace(
                runner,
                source_payload={
                    **runner.source_payload,
                    "horse_history": history_items[:limit],
                    "horse_history_raw": raw_history,
                },
            )
        )

    if raw_audit:
        _store_raw(settings, meeting_date, course, {"horse_history": raw_audit})
        _record_refresh(
            settings,
            meeting_date,
            course,
            "horse_history",
            "success",
            f"{len(raw_audit)} runner histories refreshed",
        )
    if errors:
        _store_raw(settings, meeting_date, course, {"source": "horse_history", "errors": errors})
        _record_refresh(
            settings,
            meeting_date,
            course,
            "horse_history",
            "partial",
            f"{len(errors)} runner history errors",
        )
    if not raw_audit and not errors:
        _record_refresh(settings, meeting_date, course, "horse_history", "empty")
    return merged


def _flatten_horse_history(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for key in ("results", "data", "runs", "history", "race_results"):
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _history_delay(settings: Settings) -> float:
    if settings.provider.lower() == "mock":
        return 0.0
    try:
        return float(THE_RACING_API_CONFIG.get("horse_history_delay_seconds", 0.25))
    except (TypeError, ValueError):
        return 0.25


def _flatten_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for race in raw.get("results") or raw.get("data") or []:
        if not isinstance(race, dict):
            continue
        race_fields = {
            "date": race.get("date"),
            "course": race.get("course"),
            "off_time": race.get("off") or race.get("off_time"),
            "race_name": race.get("race_name"),
        }
        for runner in race.get("runners") or []:
            if not isinstance(runner, dict):
                continue
            rows.append(
                {
                    **race_fields,
                    "horse": runner.get("horse"),
                    "position": runner.get("position"),
                    "result_runner": runner,
                }
            )
    return rows


def _runner_key(runner: Runner) -> tuple[str, str, str, str]:
    return (
        runner.meeting_date.isoformat(),
        _normalise(runner.course),
        _normalise_time(runner.off_time),
        _normalise_horse(runner.horse),
    )


def _result_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("date") or ""),
        _normalise(str(item.get("course") or "")),
        _normalise_time(str(item.get("off_time") or "")),
        _normalise_horse(str(item.get("horse") or "")),
    )


def _normalise(value: str) -> str:
    text = " ".join(value.lower().strip().split())
    return text.replace(" (gb)", "").replace(" (ire)", "")


def _normalise_horse(value: str) -> str:
    text = _normalise(value)
    for suffix in (" (gb)", " (ire)", " (fr)", " (usa)"):
        text = text.replace(suffix, "")
    return text


def _normalise_time(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    parts = text.split(":")
    if len(parts) >= 2:
        try:
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
        except ValueError:
            return text
    return text
