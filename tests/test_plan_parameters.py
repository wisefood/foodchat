"""Interactive plan-parameter card — sanitization, card payload, apply flow.

No LLM calls: the chat service is a fake that records how it was invoked;
only the real card/sanitize/describe logic and orchestrator wiring run.
"""

import uuid

from conftest import make_candidates
from services import plan_parameters
from services.orchestrator_service import OrchestratorService


class RecordingChatService:
    """process_plan_request stand-in that returns a stored plan and records
    the exact call (message, refinement flag, clarification skip)."""

    def __init__(self, session_service):
        self.session_service = session_service
        self.calls = []

    def process_plan_request(self, session_id, message, is_refinement=False,
                             seeds=None, skip_clarification=False):
        self.calls.append({
            "message": message,
            "is_refinement": is_refinement,
            "skip_clarification": skip_clarification,
        })
        self.session_service.add_message(session_id, "user", message)
        if is_refinement:
            plan = self.session_service.refine_meal_plan(
                session_id, make_candidates("v2"), "refined", {}
            )
        else:
            plan = self.session_service.add_meal_plan(
                session_id, make_candidates("v1"), "fresh", {}
            )
        self.session_service.add_message(session_id, "assistant", "Here you go.")
        return "Here you go.", False, plan


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


def make_session(session_service, sample_profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))


# --------------------------------------------------------------------- #
# Pure card logic                                                         #
# --------------------------------------------------------------------- #


class TestSanitize:
    def test_clamps_and_snaps_scale_values(self):
        assert plan_parameters.sanitize({"cooking_time": 23}) == {"cooking_time": 25}
        assert plan_parameters.sanitize({"cooking_time": 500}) == {"cooking_time": 90}
        assert plan_parameters.sanitize({"cooking_time": -5}) == {"cooking_time": 10}
        assert plan_parameters.sanitize({"cooking_time": "45"}) == {"cooking_time": 45}

    def test_drops_unknown_keys_and_invalid_choices(self):
        assert plan_parameters.sanitize({"spiciness": 3}) == {}
        assert plan_parameters.sanitize({"difficulty": "impossible"}) == {}
        assert plan_parameters.sanitize({"cooking_time": "soon"}) == {}
        assert plan_parameters.sanitize(None) == {}

    def test_accepts_valid_choices(self):
        values = {"difficulty": "easy", "goal": "high_protein"}
        assert plan_parameters.sanitize(values) == values


class TestCardAndText:
    def test_card_reflects_applied_profile_values(self):
        card = plan_parameters.build_card({"plan_parameters": {"cooking_time": 20}})
        by_key = {p["key"]: p for p in card["parameters"]}
        assert set(by_key) == {"cooking_time", "difficulty", "goal"}
        assert by_key["cooking_time"]["value"] == 20
        assert by_key["difficulty"]["value"] is None

    def test_describe_is_deterministic_and_ordered(self):
        text = plan_parameters.describe({"goal": "weight_loss", "cooking_time": 20})
        assert text == (
            "Adjust my meal plan to these settings: "
            "keep cooking time under 20 minutes per meal; "
            "aim for lighter, lower-calorie meals for weight loss."
        )

    def test_history_line_names_every_value(self):
        line = plan_parameters.history_line({"cooking_time": 20, "difficulty": "easy"})
        assert "Cooking time: 20 min" in line
        assert "Difficulty: Easy" in line


# --------------------------------------------------------------------- #
# Orchestrator wiring                                                     #
# --------------------------------------------------------------------- #


class TestCardAttachment:
    def test_fresh_daily_plan_carries_the_card(self, session_service, sample_profile):
        session = make_session(session_service, sample_profile)
        orch = make_orchestrator(session_service)

        turn = orch._handle_plan(session.session_id, "plan my day", "daily_plan",
                                 is_refinement=False)
        assert turn.plan_parameters is not None
        keys = [p["key"] for p in turn.plan_parameters["parameters"]]
        assert keys == ["cooking_time", "difficulty", "goal"]

    def test_text_refinement_does_not_re_attach_the_card(self, session_service, sample_profile):
        session = make_session(session_service, sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates(), "r", {})
        orch = make_orchestrator(session_service)

        turn = orch._handle_plan(session.session_id, "less pasta", "refine_plan",
                                 is_refinement=True)
        assert turn.plan_parameters is None


class TestApplyPlanParameters:
    def test_apply_refines_active_daily_plan_without_clarification(
        self, session_service, sample_profile
    ):
        session = make_session(session_service, sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates(), "r", {})
        orch = make_orchestrator(session_service)

        turn = orch.apply_plan_parameters(
            session.session_id, session.member_id,
            {"cooking_time": 20, "difficulty": "easy"},
        )

        call = orch.chat_service.calls[0]
        assert call["is_refinement"] is True
        assert call["skip_clarification"] is True
        assert "under 20 minutes" in call["message"]
        assert turn.intent == "refine_plan"
        assert turn.meal_plan is not None

        # The response card shows the values as current settings
        by_key = {p["key"]: p for p in turn.plan_parameters["parameters"]}
        assert by_key["cooking_time"]["value"] == 20
        assert by_key["difficulty"]["value"] == "easy"

        # Values persist on the profile and land in the known-facts history
        assert session.user_profile["plan_parameters"] == {
            "cooking_time": 20, "difficulty": "easy",
        }
        assert "Cooking time: 20 min" in session.user_profile["history"]

    def test_apply_without_canvas_generates_fresh_plan(self, session_service, sample_profile):
        session = make_session(session_service, sample_profile)
        orch = make_orchestrator(session_service)

        turn = orch.apply_plan_parameters(
            session.session_id, session.member_id, {"goal": "balanced"},
        )
        assert orch.chat_service.calls[0]["is_refinement"] is False
        assert turn.intent == "daily_plan"

    def test_apply_merges_with_previously_applied_values(self, session_service, sample_profile):
        session = make_session(session_service, sample_profile)
        orch = make_orchestrator(session_service)

        orch.apply_plan_parameters(session.session_id, session.member_id,
                                   {"cooking_time": 20})
        turn = orch.apply_plan_parameters(session.session_id, session.member_id,
                                          {"goal": "energy"})
        by_key = {p["key"]: p for p in turn.plan_parameters["parameters"]}
        assert by_key["cooking_time"]["value"] == 20
        assert by_key["goal"]["value"] == "energy"

    def test_apply_enforces_session_ownership(self, session_service, sample_profile):
        session = make_session(session_service, sample_profile)
        orch = make_orchestrator(session_service)

        import pytest
        with pytest.raises(ValueError):
            orch.apply_plan_parameters(session.session_id, "someone-else",
                                       {"cooking_time": 20})
