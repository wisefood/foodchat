"""
SQLite persistence layer for FoodChat sessions.

Uses SQLAlchemy (sync) with a single-file SQLite database.
Designed to be a drop-in replacement for the in-memory store, with
an easy path to PostgreSQL by swapping DATABASE_URL.

Tables
------
sessions   — one row per chat session
messages   — one row per conversation turn, linked to a session
meal_plans — full plan payload (daily or weekly) linked to a session
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
    user_profile = Column(Text, nullable=False)           # JSON blob
    # Two independent canvas contexts (JSON blobs or NULL)
    daily_canvas = Column(Text, nullable=True)
    weekly_canvas = Column(Text, nullable=True)
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
    plan_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class MealPlanRow(Base):
    __tablename__ = "meal_plans"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    plan_type = Column(String, nullable=False)       # "daily" | "weekly"
    version = Column(Integer, nullable=False, default=1)
    parent_id = Column(String, nullable=True)        # id of previous version, NULL for v1
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
    _migrate_existing_db()


def _migrate_existing_db() -> None:
    """
    Idempotent column migrations for databases created before this version.
    Adds any missing columns without touching existing data.
    """
    import sqlalchemy as sa
    with engine.connect() as conn:
        inspector = sa.inspect(engine)

        # sessions table migrations
        existing_session_cols = {c["name"] for c in inspector.get_columns("sessions")}
        session_migrations = [
            ("daily_canvas", "TEXT"),
            ("weekly_canvas", "TEXT"),
        ]
        for col_name, col_type in session_migrations:
            if col_name not in existing_session_cols:
                conn.execute(sa.text(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}"))

        # sessions backward compat: drop active_context if it exists (no data needed)
        # — we leave it in place to avoid destructive migration; it's simply ignored

        # meal_plans table migrations
        existing_plan_cols = {c["name"] for c in inspector.get_columns("meal_plans")}
        plan_migrations = [
            ("version", "INTEGER NOT NULL DEFAULT 1"),
            ("parent_id", "TEXT"),
        ]
        for col_name, col_type in plan_migrations:
            if col_name not in existing_plan_cols:
                conn.execute(sa.text(f"ALTER TABLE meal_plans ADD COLUMN {col_name} {col_type}"))

        conn.commit()


def get_db() -> DBSession:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------ #
# Session helpers                                                      #
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
    return (
        db.query(SessionRow)
        .filter(SessionRow.session_id == session_id, SessionRow.member_id == member_id)
        .first()
    )


def db_delete_session(db: DBSession, session_id: str, member_id: str) -> bool:
    row = db_get_session_scoped(db, session_id, member_id)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True


def db_get_member_sessions(db: DBSession, member_id: str) -> list[SessionRow]:
    return db.query(SessionRow).filter(SessionRow.member_id == member_id).all()


def db_update_canvases(
    db: DBSession,
    session_id: str,
    daily_canvas: Optional[dict],
    weekly_canvas: Optional[dict],
) -> None:
    row = db.query(SessionRow).filter(SessionRow.session_id == session_id).first()
    if row:
        if daily_canvas is not None:
            row.daily_canvas = json.dumps(daily_canvas)
        if weekly_canvas is not None:
            row.weekly_canvas = json.dumps(weekly_canvas)
        db.commit()


def db_update_state(db: DBSession, session_id: str, state: str) -> None:
    row = db.query(SessionRow).filter(SessionRow.session_id == session_id).first()
    if row:
        row.state = state
        db.commit()


# ------------------------------------------------------------------ #
# Message helpers                                                      #
# ------------------------------------------------------------------ #

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
    """
    q = db.query(MessageRow).filter(MessageRow.session_id == session_id)
    if before_id is not None:
        q = q.filter(MessageRow.id < before_id)
    rows = q.order_by(MessageRow.id.desc()).limit(limit).all()
    return list(reversed(rows))


def db_get_message_by_id(db: DBSession, message_id: int) -> Optional[MessageRow]:
    return db.query(MessageRow).filter(MessageRow.id == message_id).first()


# ------------------------------------------------------------------ #
# Meal plan helpers                                                    #
# ------------------------------------------------------------------ #

def db_save_meal_plan(
    db: DBSession,
    plan_id: str,
    session_id: str,
    plan_type: str,
    payload: dict,
    version: int = 1,
    parent_id: Optional[str] = None,
) -> MealPlanRow:
    row = MealPlanRow(
        id=plan_id,
        session_id=session_id,
        plan_type=plan_type,
        version=version,
        parent_id=parent_id,
        payload=json.dumps(payload),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def db_get_meal_plan(db: DBSession, plan_id: str) -> Optional[MealPlanRow]:
    return db.query(MealPlanRow).filter(MealPlanRow.id == plan_id).first()


def db_get_session_meal_plans(
    db: DBSession,
    session_id: str,
    plan_type: Optional[str] = None,
) -> list[MealPlanRow]:
    q = db.query(MealPlanRow).filter(MealPlanRow.session_id == session_id)
    if plan_type:
        q = q.filter(MealPlanRow.plan_type == plan_type)
    return q.order_by(MealPlanRow.created_at.asc()).all()


def db_get_plan_lineage(
    db: DBSession,
    session_id: str,
    root_id: str,
    plan_type: str,
) -> list[MealPlanRow]:
    """
    Return all versions in a plan's lineage (same root), ordered oldest-first.
    Walks the parent_id chain via a recursive CTE for SQLite 3.35+ / Postgres.
    Falls back to a Python-side walk on older SQLite.
    """
    import sqlalchemy as sa
    try:
        # Recursive CTE — works on SQLite ≥ 3.35 and PostgreSQL
        cte = sa.text("""
            WITH RECURSIVE lineage(id) AS (
                SELECT :root_id
                UNION ALL
                SELECT mp.id
                FROM meal_plans mp
                JOIN lineage l ON mp.parent_id = l.id
                WHERE mp.session_id = :session_id AND mp.plan_type = :plan_type
            )
            SELECT mp.* FROM meal_plans mp
            JOIN lineage l ON mp.id = l.id
            ORDER BY mp.created_at ASC
        """)
        result = db.execute(cte, {"root_id": root_id, "session_id": session_id, "plan_type": plan_type})
        rows = result.fetchall()
        # Map to MealPlanRow-like objects via ORM
        plan_ids = [r[0] for r in rows]
        return (
            db.query(MealPlanRow)
            .filter(MealPlanRow.id.in_(plan_ids))
            .order_by(MealPlanRow.created_at.asc())
            .all()
        )
    except Exception:
        # Python-side fallback: fetch all plans for session and walk manually
        all_plans = db_get_session_meal_plans(db, session_id, plan_type)
        id_map = {p.id: p for p in all_plans}
        result_ids: list[str] = []
        # Forward walk from root
        current_id: Optional[str] = root_id
        while current_id and current_id in id_map:
            result_ids.append(current_id)
            # Find child
            child = next((p for p in all_plans if p.parent_id == current_id), None)
            current_id = child.id if child else None
        return [id_map[i] for i in result_ids if i in id_map]


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


# ------------------------------------------------------------------ #
# Legacy shim — kept so any direct callers of db_update_active_context
# don't crash until they are updated.
# ------------------------------------------------------------------ #

def db_update_active_context(db: DBSession, session_id: str, active_context: Optional[dict]) -> None:
    """Deprecated — use db_update_canvases instead. No-op stub."""
    pass
