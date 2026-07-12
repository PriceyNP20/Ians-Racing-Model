from __future__ import annotations

from datetime import datetime, timezone
import json
from threading import Lock

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, create_engine, select
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
