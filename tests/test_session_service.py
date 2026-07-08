"""SessionService — lifecycle, ownership, canvas versioning, clarification persistence."""

import uuid

from conftest import make_candidates


def _new_session(session_service, sample_profile):
    member_id = f"member-{uuid.uuid4()}"
    return session_service.create_session(member_id, sample_profile), member_id


class TestSessionLifecycle:
    def test_create_and_get(self, session_service, sample_profile):
        session, member_id = _new_session(session_service, sample_profile)
        loaded = session_service.get_session(session.session_id)
        assert loaded is not None
        assert loaded.member_id == member_id
        assert loaded.user_profile["diet"] == ["vegetarian"]

    def test_owner_scoping_denies_other_member(self, session_service, sample_profile):
        session, _ = _new_session(session_service, sample_profile)
        assert session_service.get_session(session.session_id, member_id="intruder") is None

    def test_delete_requires_owner(self, session_service, sample_profile):
        session, member_id = _new_session(session_service, sample_profile)
        assert not session_service.delete_session(session.session_id, "intruder")
        assert session_service.delete_session(session.session_id, member_id)
        assert session_service.get_session(session.session_id) is None

    def test_reload_from_db_after_cache_loss(self, session_service, sample_profile):
        """A fresh service instance (empty cache) must restore from the DB —
        this is the restart scenario."""
        from services.session_service import SessionService

        session, member_id = _new_session(session_service, sample_profile)
        session_service.add_message(session.session_id, "user", "hello")
        session_service.add_message(session.session_id, "assistant", "hi!")

        fresh = SessionService()
        restored = fresh.get_session(session.session_id, member_id=member_id)
        assert restored is not None
        assert [m.content for m in restored.conversation] == ["hello", "hi!"]


class TestPlanCanvas:
    def test_add_then_refine_builds_lineage(self, session_service, sample_profile):
        session, _ = _new_session(session_service, sample_profile)
        v1 = session_service.add_meal_plan(session.session_id, make_candidates("a"), "first", {})
        v2 = session_service.refine_meal_plan(session.session_id, make_candidates("b"), "second", {})

        assert v1.version == 1 and v1.parent_id is None
        assert v2.version == 2 and v2.parent_id == v1.id
        assert session.daily_canvas.current_id == v2.id
        assert session.daily_canvas.root_id == v1.id
        assert session.get_current_daily_plan().id == v2.id

    def test_refine_without_canvas_creates_fresh_plan(self, session_service, sample_profile):
        session, _ = _new_session(session_service, sample_profile)
        plan = session_service.refine_meal_plan(session.session_id, make_candidates(), "r", {})
        assert plan.version == 1 and plan.parent_id is None

    def test_new_plan_opens_fresh_canvas(self, session_service, sample_profile):
        session, _ = _new_session(session_service, sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("a"), "first", {})
        v1b = session_service.add_meal_plan(session.session_id, make_candidates("c"), "restart", {})
        assert v1b.version == 1
        assert session.daily_canvas.root_id == v1b.id

    def test_weekly_plan_versioning(self, session_service, sample_profile):
        session, _ = _new_session(session_service, sample_profile)
        entries = [{"day": 1, "meal_idx": 0, "meal_type": "breakfast", "recipe": {}, "reward": 1.0}]
        w1 = session_service.add_weekly_meal_plan(session.session_id, entries)
        w2 = session_service.refine_weekly_meal_plan(session.session_id, entries)
        assert (w1.version, w2.version) == (1, 2)
        assert w2.parent_id == w1.id
        assert session.weekly_canvas.current_id == w2.id

    def test_member_current_plans_picks_newest_canvas(self, session_service, sample_profile):
        """Dashboard lookup: the member's most recently planned session wins."""
        member_id = f"member-{uuid.uuid4()}"
        older = session_service.create_session(member_id, sample_profile)
        session_service.add_meal_plan(older.session_id, make_candidates("old"), "r", {})

        newer = session_service.create_session(member_id, sample_profile)
        entries = [{"day": 1, "meal_idx": 0, "meal_type": "breakfast", "recipe": {}, "reward": 1.0}]
        weekly = session_service.add_weekly_meal_plan(newer.session_id, entries)

        best = session_service.get_member_current_plans(member_id)
        assert best.session_id == newer.session_id
        assert best.active_canvas.plan_type == "weekly"
        assert best.get_current_weekly_plan().id == weekly.id

    def test_member_current_plans_none_without_plans(self, session_service, sample_profile):
        member_id = f"member-{uuid.uuid4()}"
        session_service.create_session(member_id, sample_profile)
        assert session_service.get_member_current_plans(member_id) is None

    def test_plans_survive_reload(self, session_service, sample_profile):
        from services.session_service import SessionService

        session, member_id = _new_session(session_service, sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates(), "persisted", {"llm_score": 4})

        fresh = SessionService()
        restored = fresh.get_session(session.session_id, member_id=member_id)
        assert len(restored.meal_plans) == 1
        assert restored.meal_plans[0].reasoning == "persisted"
        assert restored.meal_plans[0].llm_score == 4
        assert restored.daily_canvas is not None


class TestClarificationPersistence:
    def test_state_roundtrip_survives_restart(self, session_service, sample_profile):
        """The core M0 guarantee: clarification state is data, not a generator."""
        from services.session_service import SessionService

        session, member_id = _new_session(session_service, sample_profile)
        state = {
            "original_query": "plan something",
            "profile": sample_profile,
            "origin_intent": "daily_plan",
            "phase": "collect",
            "pending_topics": ["cooking time"],
            "current_question": "How much time do you have to cook?",
            "transcript": [],
            "conflict_note": None,
        }
        session_service.set_clarification_state(session.session_id, state)
        assert session.state == "clarifying"

        fresh = SessionService()  # simulate restart
        restored = fresh.get_session(session.session_id, member_id=member_id)
        assert restored.state == "clarifying"
        assert restored.clarification == state

        fresh.clear_clarification_state(session.session_id)
        again = SessionService().get_session(session.session_id)
        assert again.state == "ready"
        assert again.clarification is None


class TestMessageLimit:
    def test_limit_enforced(self, session_service, sample_profile):
        import pytest

        session, _ = _new_session(session_service, sample_profile)
        session.max_messages = 2
        session_service.add_message(session.session_id, "user", "one")
        session_service.add_message(session.session_id, "assistant", "two")
        assert session.is_at_message_limit
        with pytest.raises(RuntimeError):
            session_service.add_message(session.session_id, "user", "three")
