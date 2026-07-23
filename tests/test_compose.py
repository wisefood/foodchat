"""Manual mode — hand-picked recipes become seed anchors, FoodChat fills the rest.

No LLM calls: the chat service is a recording fake; only the orchestrator
wiring runs (the seed resolution/pinning it delegates to is covered by
test_seeded_planning.py).
"""

import uuid

import pytest

from conftest import make_candidates
from services.orchestrator_service import OrchestratorService


class RecordingChatService:
    def __init__(self, session_service):
        self.session_service = session_service
        self.calls = []

    def process_plan_request(self, session_id, message, is_refinement=False,
                             seeds=None, skip_clarification=False):
        self.calls.append({
            "message": message, "is_refinement": is_refinement,
            "seeds": seeds, "skip_clarification": skip_clarification,
        })
        self.session_service.add_message(session_id, "user", message)
        plan = self.session_service.add_meal_plan(
            session_id, make_candidates("m"), "manual", {}
        )
        self.session_service.add_message(session_id, "assistant", "Filled it out.")
        return "Filled it out.", False, plan


class RecordingWeeklyService:
    def __init__(self, session_service):
        self.session_service = session_service
        self.calls = []

    def process_message(self, session_id, message, is_refinement=False, seeds=None):
        self.calls.append({
            "message": message, "is_refinement": is_refinement, "seeds": seeds,
        })
        self.session_service.add_message(session_id, "user", message)
        self.session_service.add_message(session_id, "assistant", "Week planned.")
        from types import SimpleNamespace
        return "Week planned.", SimpleNamespace(id="wp-1", version=1, parent_id=None)


def make_orchestrator(session_service):
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    orch.chat_service = RecordingChatService(session_service)
    orch.weekly_plan_service = RecordingWeeklyService(session_service)
    orch.foodscholar_service = None
    orch.seed_service = None
    orch.plan_analyst = None
    orch.memory_service = None
    orch.edit_service = None
    orch.orchestrator = None
    return orch


PICKS = [{"meal_type": "breakfast", "recipe_id": "rw-42", "title": "Overnight oats"}]


class TestComposePlan:
    def test_picks_become_slot_seeds_without_clarification(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)

        turn = orch.compose_plan(session.session_id, session.member_id, PICKS)

        call = orch.chat_service.calls[0]
        assert call["skip_clarification"] is True
        assert call["is_refinement"] is False
        assert call["seeds"] == [
            {"recipe_id": "rw-42", "meal_type": "breakfast", "name": "Overnight oats"},
        ]
        assert "Complete my meal plan" in call["message"]
        assert turn.intent == "daily_plan"
        assert turn.meal_plan is not None
        # Fresh daily plan → the slider card rides along as usual
        assert turn.plan_parameters is not None

    def test_chat_text_travels_with_the_picks(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)

        orch.compose_plan(
            session.session_id, session.member_id, PICKS,
            message="fill out the rest, keep it light",
        )
        assert orch.chat_service.calls[0]["message"] == "fill out the rest, keep it light"

    def test_weekly_picks_carry_their_day_to_exact_slots(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)

        turn = orch.compose_plan(
            session.session_id, session.member_id,
            [
                {"meal_type": "dinner", "recipe_id": "rw-7", "title": "Pastitsio", "day": 2},
                {"meal_type": "lunch", "recipe_id": "rw-9", "title": "Fakes", "day": 5},
            ],
            plan_type="weekly",
        )

        call = orch.weekly_plan_service.calls[0]
        assert call["is_refinement"] is False
        assert call["seeds"] == [
            {"recipe_id": "rw-7", "meal_type": "dinner", "name": "Pastitsio", "day": 2},
            {"recipe_id": "rw-9", "meal_type": "lunch", "name": "Fakes", "day": 5},
        ]
        assert "for this week" in call["message"]
        assert turn.intent == "weekly_plan"
        assert turn.weekly_meal_plan is not None
        assert orch.chat_service.calls == []

    def test_ownership_enforced(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        with pytest.raises(ValueError):
            orch.compose_plan(session.session_id, "someone-else", PICKS)

    def test_message_limit_short_circuits(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        session.max_messages = 0
        orch = make_orchestrator(session_service)
        turn = orch.compose_plan(session.session_id, session.member_id, PICKS)
        assert turn.at_message_limit
        assert orch.chat_service.calls == []


class TestManualPickPersistence:
    """Hand-picked dishes must survive text refinements — and die honestly."""

    def test_picks_reinjected_on_refinement(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        orch.compose_plan(session.session_id, session.member_id, PICKS)

        orch._handle_plan(session.session_id, "make it lighter", "refine_plan",
                          is_refinement=True)
        refine_call = orch.chat_service.calls[1]
        assert refine_call["seeds"] == [
            {"recipe_id": "rw-42", "meal_type": "breakfast", "name": "Overnight oats"},
        ]

    def test_explicit_seeds_beat_stored_picks(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        orch.compose_plan(session.session_id, session.member_id, PICKS)

        explicit = [{"name": "moussaka"}]
        orch._handle_plan(session.session_id, "work in moussaka", "refine_plan",
                          is_refinement=True, seeds=explicit)
        assert orch.chat_service.calls[1]["seeds"] == explicit

    def test_fresh_plan_clears_the_lineage(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        orch.compose_plan(session.session_id, session.member_id, PICKS)

        orch._handle_plan(session.session_id, "plan my day from scratch", "daily_plan",
                          is_refinement=False)
        assert session.user_profile.get("manual_picks", {}).get("daily", []) == []
        # A refinement after the fresh plan carries no stale picks
        orch._handle_plan(session.session_id, "lighter", "refine_plan", is_refinement=True)
        assert orch.chat_service.calls[2]["seeds"] is None

    def test_verified_edit_unpins_the_swapped_slot(self, session_service, sample_profile):
        from types import SimpleNamespace
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        orch.compose_plan(session.session_id, session.member_id, PICKS)

        plan = session_service.refine_meal_plan(session.session_id, make_candidates("e"), "r", {})
        session_service.add_message(session.session_id, "assistant", "Swapped it.")
        outcome = SimpleNamespace(
            text="Swapped it.", needs_clarification=False,
            meal_plan=plan, weekly_meal_plan=None,
            changed_slots=[{"meal_type": "breakfast", "day": None,
                            "old": {}, "new": {}, "directive": "lighter", "verified": True}],
        )
        orch._turn_from_edit(session.session_id, outcome)
        assert session.user_profile.get("manual_picks", {}).get("daily", []) == []

    def test_weekly_picks_reinjected_on_weekly_refinement(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        orch = make_orchestrator(session_service)
        orch.compose_plan(
            session.session_id, session.member_id,
            [{"meal_type": "dinner", "recipe_id": "rw-7", "title": "Pastitsio", "day": 2}],
            plan_type="weekly",
        )

        orch._handle_weekly(session.session_id, "less meat overall", "refine_plan",
                            is_refinement=True)
        refine_call = orch.weekly_plan_service.calls[1]
        assert refine_call["seeds"] == [
            {"recipe_id": "rw-7", "meal_type": "dinner", "name": "Pastitsio", "day": 2},
        ]
