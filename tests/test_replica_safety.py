"""M5 — replica/restart safety of the persistence layer.

These tests simulate the multi-replica scenario the Postgres move enables:
a mutation arriving at a replica whose in-memory cache has never seen the
session must load-through from the DB instead of raising.
"""

import uuid
from datetime import datetime, timezone

from conftest import make_candidates
from services.session_service import SessionService, _aware


class TestLoadThroughMutation:
    def test_add_message_on_cold_replica(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)

        cold = SessionService()          # fresh replica, empty cache
        message = cold.add_message(session.session_id, "user", "hello from replica B")
        assert message.content == "hello from replica B"

        # And a third replica sees the write
        third = SessionService()
        restored = third.get_session(session.session_id)
        assert restored.conversation[-1].content == "hello from replica B"

    def test_plan_mutation_on_cold_replica(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("r1"), "v1", {})

        cold = SessionService()
        refined = cold.refine_meal_plan(session.session_id, make_candidates("r2"), "v2", {})
        assert refined.version == 2

    def test_clarification_state_on_cold_replica(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)

        cold = SessionService()
        cold.set_clarification_state(session.session_id, {"kind": "foodscholar", "qa_thread_id": "t"})
        assert SessionService().get_session(session.session_id).state == "clarifying"

    def test_canvas_clear_persists(self, session_service, sample_profile):
        """db_update_canvases must NULL a cleared canvas (pre-M5 it skipped None)."""
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("c"), "r", {})
        assert session.daily_canvas is not None

        # Clear in memory, re-persist both canvases
        session.daily_canvas = None
        session_service._persist_canvases(session.session_id, session)

        restored = SessionService().get_session(session.session_id)
        assert restored.daily_canvas is None


class TestTimezoneAwareness:
    def test_aware_coercion_of_legacy_naive(self):
        naive = datetime(2026, 4, 1, 12, 0, 0)
        aware = _aware(naive)
        assert aware.tzinfo is not None
        assert _aware(None) is None
        already = datetime.now(timezone.utc)
        assert _aware(already) is already

    def test_new_timestamps_are_aware_and_sortable_with_loaded(self, session_service, sample_profile):
        """Mixing loaded rows and fresh objects in sort keys must not raise."""
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("a"), "r", {})

        restored = SessionService().get_session(session.session_id)
        session_service2 = SessionService()
        session_service2._sessions[session.session_id] = restored
        session_service2.add_meal_plan(session.session_id, make_candidates("b"), "r2", {})

        # sorted() raises TypeError when naive and aware mix — must not happen
        history = session_service2.get_daily_plan_history(session.session_id)
        assert len(history) == 2
        assert all(p.created_at.tzinfo is not None for p in history)
