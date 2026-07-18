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
    list_model_snapshots,
    list_refresh_statuses,
    make_session_factory,
    record_refresh_status,
    store_model_snapshots,
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


def get_model_snapshots(settings: Settings, limit: int = 1000) -> list[dict]:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        return list_model_snapshots(session, limit=limit)


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
    scorer = IanFormulaV31()
    try:
        runners, raw = provider.fetch_racecard(meeting_date, course)
        _store_raw(settings, meeting_date, course, raw)
        _record_refresh(settings, meeting_date, course, "racecard", "success")
        runners = _attach_horse_history(provider, settings, meeting_date, course, runners)
        runners = _attach_v5_evidence(provider, settings, meeting_date, course, runners)
        _store_pre_result_snapshots(settings, settings.provider, scorer.score_runners(runners))
        runners, results_imported = _attach_results(provider, settings, meeting_date, course, runners)
        return ScoredCardResult(
            scores=scorer.score_runners(runners),
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
        fallback_scores = scorer.score_runners(runners)
        _store_pre_result_snapshots(settings, "mock", fallback_scores)
        return ScoredCardResult(
            scores=fallback_scores,
            provider="mock",
            warning=(
                "Live Racing API data could not be loaded, so this view is using "
                "sample data. Check Streamlit logs for the API status code/details."
            ),
        )


def _store_pre_result_snapshots(settings: Settings, provider: str, scores: list[RunnerScore]) -> None:
    session_factory = make_session_factory(settings.database_url)
    with session_factory() as session:
        store_model_snapshots(session, provider, scores)


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


def _attach_v5_evidence(
    provider,
    settings: Settings,
    meeting_date: date,
    course: str | None,
    runners: list[Runner],
) -> list[Runner]:
    if settings.provider.lower() == "mock":
        return runners
    if str(THE_RACING_API_CONFIG.get("v5_evidence_enabled", "true")).lower() not in {"1", "true", "yes", "y"}:
        return runners

    max_runners = int(THE_RACING_API_CONFIG.get("v5_evidence_max_runners", 40))
    delay_seconds = _v5_evidence_delay()
    merged: list[Runner] = []
    raw_audit: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    fetched_count = 0

    calls = (
        ("odds_history", provider.fetch_odds_history),
        ("horse_profile", provider.fetch_horse_profile),
        ("trainer_course_analysis", provider.fetch_trainer_course_analysis),
        ("trainer_distance_analysis", provider.fetch_trainer_distance_analysis),
        ("trainer_jockey_analysis", provider.fetch_trainer_jockey_analysis),
        ("jockey_course_analysis", provider.fetch_jockey_course_analysis),
        ("jockey_trainer_analysis", provider.fetch_jockey_trainer_analysis),
    )

    for index, runner in enumerate(runners):
        if runner.is_non_runner or index >= max_runners:
            merged.append(runner)
            continue
        evidence: dict[str, Any] = {}
        for source, fetcher in calls:
            if delay_seconds and fetched_count:
                time.sleep(delay_seconds)
            try:
                payload = fetcher(runner)
                fetched_count += 1
            except Exception as exc:
                errors.append(
                    {
                        "horse": runner.horse,
                        "source": source,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            if payload:
                evidence[source] = payload

        if not evidence:
            merged.append(runner)
            continue

        derived = _derive_v5_evidence(runner, evidence)
        raw_audit.append({"horse": runner.horse, "evidence": evidence})
        merged.append(
            replace(
                runner,
                source_payload={
                    **runner.source_payload,
                    "v5_requested_evidence": sorted(evidence),
                    "v5_evidence_raw": evidence,
                    **derived,
                },
            )
        )

    if raw_audit:
        _store_raw(settings, meeting_date, course, {"v5_evidence": raw_audit})
        _record_refresh(
            settings,
            meeting_date,
            course,
            "v5_evidence",
            "success",
            f"{len(raw_audit)} runner evidence bundles refreshed",
        )
    if errors:
        _store_raw(settings, meeting_date, course, {"source": "v5_evidence", "errors": errors})
        _record_refresh(
            settings,
            meeting_date,
            course,
            "v5_evidence",
            "partial",
            f"{len(errors)} V5 evidence errors",
        )
    if not raw_audit and not errors:
        _record_refresh(settings, meeting_date, course, "v5_evidence", "empty")
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


def _v5_evidence_delay() -> float:
    try:
        return float(THE_RACING_API_CONFIG.get("v5_evidence_delay_seconds", 0.25))
    except (TypeError, ValueError):
        return 0.25


def _derive_v5_evidence(runner: Runner, evidence: dict[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    odds = evidence.get("odds_history")
    opening = _opening_odds_from_api(odds)
    if opening is not None:
        derived["opening_odds_decimal"] = opening
        derived["odds_history"] = odds.get("odds") if isinstance(odds, dict) else odds

    trainer_course = _best_analysis_row(evidence.get("trainer_course_analysis"), "courses", "course", runner.course)
    trainer_distance = _best_analysis_row(evidence.get("trainer_distance_analysis"), "distances", "dist", runner.distance)
    trainer_jockey = _best_analysis_row(evidence.get("trainer_jockey_analysis"), "jockeys", "jockey", runner.jockey)
    jockey_course = _best_analysis_row(evidence.get("jockey_course_analysis"), "courses", "course", runner.course)
    jockey_trainer = _best_analysis_row(evidence.get("jockey_trainer_analysis"), "trainers", "trainer", runner.trainer)

    for prefix, row in (
        ("trainer_course", trainer_course),
        ("trainer_distance", trainer_distance),
        ("trainer_jockey", trainer_jockey),
        ("jockey_course", jockey_course),
        ("jockey_trainer", jockey_trainer),
    ):
        if row:
            derived[f"{prefix}_win_pct"] = _analysis_percent(row.get("win_%"))
            derived[f"{prefix}_ae"] = _analysis_float(row.get("a/e"))
            derived[f"{prefix}_runners"] = row.get("runners") or row.get("rides")

    trainer_ae_values = [
        value
        for value in (
            derived.get("trainer_course_ae"),
            derived.get("trainer_distance_ae"),
            derived.get("trainer_jockey_ae"),
        )
        if value is not None
    ]
    trainer_pct_values = [
        value
        for value in (
            derived.get("trainer_course_win_pct"),
            derived.get("trainer_distance_win_pct"),
            derived.get("trainer_jockey_win_pct"),
        )
        if value is not None
    ]
    if trainer_ae_values:
        derived["trainer_ae"] = round(sum(trainer_ae_values) / len(trainer_ae_values), 3)
    if trainer_pct_values:
        derived["trainer_win_pct"] = round(sum(trainer_pct_values) / len(trainer_pct_values), 2)
    if jockey_course:
        derived["jockey_course_win_pct"] = _analysis_percent(jockey_course.get("win_%"))
        derived["jockey_course_ae"] = _analysis_float(jockey_course.get("a/e"))
    if jockey_trainer:
        derived["jockey_trainer_win_pct"] = _analysis_percent(jockey_trainer.get("win_%"))
        derived["jockey_trainer_ae"] = _analysis_float(jockey_trainer.get("a/e"))

    return {key: value for key, value in derived.items() if value not in (None, "", [])}


def _opening_odds_from_api(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("odds")
    if not isinstance(rows, list):
        return None
    for bookmaker in rows:
        if not isinstance(bookmaker, dict):
            continue
        history = bookmaker.get("history")
        if isinstance(history, list) and history:
            first = history[0]
            if isinstance(first, dict):
                odds = _decimal_odds(first.get("decimal") or first.get("fractional"))
                if odds is not None:
                    return odds
        odds = _decimal_odds(bookmaker.get("decimal") or bookmaker.get("fractional"))
        if odds is not None:
            return odds
    return None


def _best_analysis_row(payload: Any, collection: str, label_key: str, expected: str | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get(collection)
    if not isinstance(rows, list):
        return None
    expected_text = _normalise(str(expected or ""))
    if expected_text:
        for row in rows:
            if isinstance(row, dict) and _normalise(str(row.get(label_key) or "")) == expected_text:
                return row
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _analysis_percent(value: Any) -> float | None:
    number = _analysis_float(value)
    if number is None:
        return None
    return round(number * 100, 2) if number <= 1 else round(number, 2)


def _analysis_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().replace("%", ""))
    except ValueError:
        return None


def _decimal_odds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0:
                return None
            return float(num) / denominator + 1.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


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
