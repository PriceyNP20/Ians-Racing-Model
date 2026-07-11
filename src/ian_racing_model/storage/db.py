from __future__ import annotations

from datetime import datetime, timezone
import json

from sqlalchemy import DateTime, Integer, String, Text, create_engine
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


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False, future=True)


def store_raw_response(session: Session, provider: str, meeting_date: str, course: str | None, payload: dict) -> None:
    session.add(RawApiResponse(provider=provider, meeting_date=meeting_date, course=course, payload=json.dumps(payload, sort_keys=True), created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    session.commit()
