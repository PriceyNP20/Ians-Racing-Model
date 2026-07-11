from __future__ import annotations

from datetime import datetime, timezone
import json
from threading import Lock

from sqlalchemy import DateTime, Integer, String, Text, create_engine
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
