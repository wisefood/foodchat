"""
SQLite persistence layer for FoodChat sessions.

Uses SQLAlchemy (sync) with a single-file SQLite database.
Designed to be a drop-in replacement for the in-memory store, with
an easy path to PostgreSQL by swapping DATABASE_URL.

Tables
------
sessions   — one row per chat session
messages   — one row per conversation turn, linked to a session
meal_plans — full plan payload (daily or weekly) linked to a message
feedback   — thumbs up/down + optional comment per assistant message
"""

import json
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session as DBSession, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./foodchat.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

# Enable foreign keys for SQLite
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    member_id = Column(String, nullable=False, index=True)
    user_profile = Column(Text, nullable=False)          # JSON blob
    active_context = Column(Text, nullable=True)         # JSON blob or NULL
    state = Column(String, default="ready")
    max_messages = Column(Integer, default=200)
    created_at = Column(DateTime, default=datetime.utcnow)


class MessageRow(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String, nullable=True)
    plan_id = Column(String, nullable=True)  # FK into meal_plans.id
    timestamp = Column(DateTime, default=datetime.utcnow)


class MealPlanRow(Base):
    __tablename__ = "meal_plans"

    id = Column(String, primary_key=True)           # MealPlan.id / WeeklyMealPlan.id
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    plan_type = Column(String, nullable=False)       # "daily" | "weekly"
    payload = Column(Text, nullable=False)           # full plan as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    member_id = Column(String, nullable=False, index=True)
    rating = Column(String, nullable=False)          # "up" | "down"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables if they don't exist. Called at app startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> DBSession:
    """Yield a database session. Use as a context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Low-level helpers used by SessionService                            #
# ------------------------------------------------------------------ #

def db_create_session(db: DBSession, session_id: str, member_id: str, user_profile: dict) -> SessionRow:
    row = SessionRow(
        session_id=session_id,
        member_id=member_id,
        user_profile=json.dumps(user_profile),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def db_get_session(db: DBSession, session_id: str) -> Optional[SessionRow]:
    return db.query(SessionRow).filter(SessionRow.session_id == session_id).first()


def db_get_session_scoped(db: DBSession, session_id: str, member_id: str) -> Optional[SessionRow]:
    """Fetch session only if it belongs to member_id — enforces user scoping."""
    return (
        db.query(SessionRow)
        .filter(SessionRow.session_id == session_id, SessionRow.member_id == member_id)
        .first()
    )


def db_delete_session(db: DBSession, session_id: str, member_id: str) -> bool:
    """Delete session only if it belongs to member_id."""
    row = db_get_session_scoped(db, session_id, member_id)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True


def db_get_member_sessions(db: DBSession, member_id: str) -> list[SessionRow]:
    return db.query(SessionRow).filter(SessionRow.member_id == member_id).all()


def db_add_message(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    intent: Optional[str] = None,
    plan_id: Optional[str] = None,
) -> MessageRow:
    row = MessageRow(
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        plan_id=plan_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def db_get_messages(
    db: DBSession,
    session_id: str,
    before_id: Optional[int] = None,
    limit: int = 20,
) -> list[MessageRow]:
    """
    Cursor-based pagination — returns `limit` messages before `before_id`,
    ordered oldest-first so the caller can prepend to the top of the chat UI.

    If before_id is None, returns the most recent `limit` messages.
    """
    q = db.query(MessageRow).filter(MessageRow.session_id == session_id)
    if before_id is not None:
        q = q.filter(MessageRow.id < before_id)
    rows = q.order_by(MessageRow.id.desc()).limit(limit).all()
    return list(reversed(rows))  # return oldest-first


def db_update_active_context(db: DBSession, session_id: str, active_context: Optional[dict]) -> None:
    row = db.query(SessionRow).filter(SessionRow.session_id == session_id).first()
    if row:
        row.active_context = json.dumps(active_context) if active_context else None
        db.commit()


def db_update_state(db: DBSession, session_id: str, state: str) -> None:
    row = db.query(SessionRow).filter(SessionRow.session_id == session_id).first()
    if row:
        row.state = state
        db.commit()


# ------------------------------------------------------------------ #
# Meal plan helpers                                                    #
# ------------------------------------------------------------------ #

def db_save_meal_plan(
    db: DBSession,
    plan_id: str,
    session_id: str,
    plan_type: str,
    payload: dict,
) -> MealPlanRow:
    row = MealPlanRow(
        id=plan_id,
        session_id=session_id,
        plan_type=plan_type,
        payload=json.dumps(payload),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def db_get_meal_plan(db: DBSession, plan_id: str) -> Optional[MealPlanRow]:
    return db.query(MealPlanRow).filter(MealPlanRow.id == plan_id).first()


def db_get_session_meal_plans(
    db: DBSession, session_id: str, plan_type: Optional[str] = None
) -> list[MealPlanRow]:
    q = db.query(MealPlanRow).filter(MealPlanRow.session_id == session_id)
    if plan_type:
        q = q.filter(MealPlanRow.plan_type == plan_type)
    return q.order_by(MealPlanRow.created_at.asc()).all()


# ------------------------------------------------------------------ #
# Feedback helpers                                                     #
# ------------------------------------------------------------------ #

def db_upsert_feedback(
    db: DBSession,
    message_id: int,
    session_id: str,
    member_id: str,
    rating: str,
    comment: Optional[str] = None,
) -> FeedbackRow:
    """Create or update feedback for a message (one per member per message)."""
    existing = (
        db.query(FeedbackRow)
        .filter(FeedbackRow.message_id == message_id, FeedbackRow.member_id == member_id)
        .first()
    )
    if existing:
        existing.rating = rating
        existing.comment = comment
        db.commit()
        db.refresh(existing)
        return existing

    row = FeedbackRow(
        message_id=message_id,
        session_id=session_id,
        member_id=member_id,
        rating=rating,
        comment=comment,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def db_get_feedback(db: DBSession, message_id: int) -> list[FeedbackRow]:
    return db.query(FeedbackRow).filter(FeedbackRow.message_id == message_id).all()


def db_get_message_by_id(db: DBSession, message_id: int) -> Optional[MessageRow]:
    return db.query(MessageRow).filter(MessageRow.id == message_id).first()
