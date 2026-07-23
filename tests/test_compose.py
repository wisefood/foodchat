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


def make_orchestrator(session_service):
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    orch.chat_service = RecordingChatService(session_service)
    orch.weekly_plan_service = None
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
