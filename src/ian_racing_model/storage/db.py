from __future__ import annotations

from datetime import datetime, timezone
import json
from threading import Lock

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class RawApiResponse(Base):
    __tablename__ = "raw_api_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    meeting_date: Mapped[str] = mapped_column(String(10), nullable=False)
    course: Mapped[str | None] = mapped_column(String(120))
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RefreshStatus(Base):
    __tablename__ = "refresh_status"
    __table_args__ = (UniqueConstraint("refresh_key", name="uq_refresh_status_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    refresh_key: Mapped[str] = mapped_column(String(240), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    meeting_date: Mapped[str] = mapped_column(String(10), nullable=False)
    course: Mapped[str | None] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"
    __table_args__ = (UniqueConstraint("snapshot_key", name="uq_model_snapshot_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_key: Mapped[str] = mapped_column(String(360), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    meeting_date: Mapped[str] = mapped_column(String(10), nullable=False)
    course: Mapped[str] = mapped_column(String(120), nullable=False)
    off_time: Mapped[str] = mapped_column(String(20), nullable=False)
    race_name: Mapped[str] = mapped_column(String(240), nullable=False)
    horse: Mapped[str] = mapped_column(String(180), nullable=False)
    race_type: Mapped[str | None] = mapped_column(String(80))
    race_class: Mapped[str | None] = mapped_column(String(80))
    surface: Mapped[str | None] = mapped_column(String(80))
    going: Mapped[str | None] = mapped_column(String(120))
    distance: Mapped[str | None] = mapped_column(String(80))
    field_size: Mapped[int | None] = mapped_column(Integer)
    current_odds: Mapped[str | None] = mapped_column(String(80))
    total_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(40), nullable=False)
    win_probability: Mapped[float | None] = mapped_column(Float)
    place_probability: Mapped[float | None] = mapped_column(Float)
    fair_win_odds: Mapped[float | None] = mapped_column(Float)
    fair_place_odds: Mapped[float | None] = mapped_column(Float)
    win_value_edge: Mapped[float | None] = mapped_column(Float)
    place_value_edge: Mapped[float | None] = mapped_column(Float)
    component_scores: Mapped[str] = mapped_column(Text, nullable=False)
    red_flags: Mapped[str] = mapped_column(Text, nullable=False)
    data_quality_warnings: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


_CREATE_LOCK = Lock()


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, future=True)
    with _CREATE_LOCK:
        try:
            Base.metadata.create_all(engine, checkfirst=True)
        except OperationalError as exc:
            if "already exists" not in str(exc).lower():
                raise
    return sessionmaker(engine, expire_on_commit=False, future=True)


def store_raw_response(
    session: Session,
    provider: str,
    meeting_date: str,
    course: str | None,
    payload: dict,
) -> None:
    session.add(
        RawApiResponse(
            provider=provider,
            meeting_date=meeting_date,
            course=course,
            payload=json.dumps(payload, sort_keys=True),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
    )
    session.commit()


def refresh_key(
    provider: str,
    meeting_date: str,
    course: str | None,
    source: str,
) -> str:
    return "|".join([provider, meeting_date, course or "all", source])


def record_refresh_status(
    session: Session,
    provider: str,
    meeting_date: str,
    course: str | None,
    source: str,
    status: str,
    message: str | None = None,
) -> None:
    key = refresh_key(provider, meeting_date, course, source)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    existing = session.execute(
        select(RefreshStatus).where(RefreshStatus.refresh_key == key)
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            RefreshStatus(
                refresh_key=key,
                provider=provider,
                meeting_date=meeting_date,
                course=course,
                source=source,
                status=status,
                message=message,
                refreshed_at=now,
            )
        )
    else:
        existing.status = status
        existing.message = message
        existing.refreshed_at = now
    session.commit()


def list_refresh_statuses(session: Session, limit: int = 50) -> list[dict]:
    rows = session.execute(
        select(RefreshStatus).order_by(RefreshStatus.refreshed_at.desc()).limit(limit)
    ).scalars()
    return [
        {
            "source": row.source,
            "status": row.status,
            "provider": row.provider,
            "meeting_date": row.meeting_date,
            "course": row.course or "All UK courses",
            "message": row.message or "",
            "refreshed_at": row.refreshed_at.isoformat(sep=" ", timespec="seconds"),
        }
        for row in rows
    ]


def store_model_snapshots(
    session: Session,
    provider: str,
    scores: list,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for item in scores:
        runner = item.runner
        key = snapshot_key(provider, runner.meeting_date.isoformat(), runner.course, runner.off_time, runner.race_name, runner.horse)
        existing = session.execute(
            select(ModelSnapshot).where(ModelSnapshot.snapshot_key == key)
        ).scalar_one_or_none()
        values = {
            "provider": provider,
            "meeting_date": runner.meeting_date.isoformat(),
            "course": runner.course,
            "off_time": runner.off_time,
            "race_name": runner.race_name,
            "horse": runner.horse,
            "race_type": runner.race_type,
            "race_class": runner.race_class,
            "surface": runner.surface,
            "going": runner.going,
            "distance": runner.distance,
            "field_size": runner.field_size,
            "current_odds": runner.current_odds,
            "total_score": item.total_score,
            "confidence": item.confidence,
            "recommendation": item.recommendation,
            "win_probability": item.win_probability,
            "place_probability": item.place_probability,
            "fair_win_odds": item.fair_win_odds,
            "fair_place_odds": item.fair_place_odds,
            "win_value_edge": item.win_value_edge,
            "place_value_edge": item.place_value_edge,
            "component_scores": json.dumps(
                {component.name: component.score for component in item.components},
                sort_keys=True,
            ),
            "red_flags": json.dumps(item.red_flags),
            "data_quality_warnings": json.dumps(item.data_quality_warnings),
            "snapshot_at": now,
        }
        if existing is None:
            session.add(ModelSnapshot(snapshot_key=key, **values))
        else:
            for field, value in values.items():
                setattr(existing, field, value)
    session.commit()


def list_model_snapshots(session: Session, limit: int = 1000) -> list[dict]:
    rows = session.execute(
        select(ModelSnapshot).order_by(ModelSnapshot.snapshot_at.desc()).limit(limit)
    ).scalars()
    return [
        {
            "snapshot_at": row.snapshot_at.isoformat(sep=" ", timespec="seconds"),
            "provider": row.provider,
            "meeting_date": row.meeting_date,
            "course": row.course,
            "off_time": row.off_time,
            "race": row.race_name,
            "horse": row.horse,
            "race_type": row.race_type or "Unknown",
            "race_class": row.race_class or "Unknown",
            "surface": row.surface or "Unknown",
            "going": row.going or "Unknown",
            "distance": row.distance or "Unknown",
            "field_size": row.field_size,
            "odds": row.current_odds or "Unavailable",
            "total_score": row.total_score,
            "confidence": row.confidence,
            "recommendation": row.recommendation,
            "win_probability": row.win_probability,
            "place_probability": row.place_probability,
            "fair_win_odds": row.fair_win_odds,
            "fair_place_odds": row.fair_place_odds,
            "win_value_edge": row.win_value_edge,
            "place_value_edge": row.place_value_edge,
            "red_flags": ", ".join(json.loads(row.red_flags or "[]")),
            "warnings": ", ".join(json.loads(row.data_quality_warnings or "[]")),
        }
        for row in rows
    ]


def snapshot_key(provider: str, meeting_date: str, course: str, off_time: str, race_name: str, horse: str) -> str:
    return "|".join([provider, meeting_date, course, off_time, race_name, horse]).lower()
