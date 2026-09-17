"""An answer to a clarifying question is a statement, not just an answer.

Reported from the canvas:

    > I've got spinach and half a jar of olives to use up
    < Would you like a quick dinner idea, a full-day plan, or a few options?
    > just a dinner
    < [a full day: breakfast, lunch and dinner]

"it's like it doesn't listen to what i say".

`continue_clarification` went straight to `_generate_and_store`. The intake —
the one pass that hears shape, diet, pantry, cooking time and facets — ran on
the turn that ASKED the question and never on the turn that answered it. So the
shape fell back to the default three meals, and everything else the member said
in that sentence went with it.

Every test here stubs the extractors to SUCCEED, because a clarification answer
is exactly where a member says something new, and offline they never fire.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                              # noqa: E402
from models.planning_state import PlanningStateDelta               # noqa: E402
from services import turn_intake                                   # noqa: E402
from services.chat_service import ChatService                      # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_turn():
    turn_intake.forget()
    yield
    turn_intake.forget()


@pytest.fixture
def clarifying(session_service, sample_profile, monkeypatch):
    """A session mid-clarification, with the generation step captured."""
    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    session_service.set_clarification_state(session.session_id, {
        "original_query": "I've got spinach and half a jar of olives to use up",
        "profile": dict(sample_profile),
        "origin_intent": "daily_plan",
        "phase": "collect",
        "pending_topics": [],
        "current_question": "Would you like a dinner idea or a full day?",
        "transcript": [],
        "conflict_note": None,
    })

    svc = ChatService.__new__(ChatService)
    svc.session_service = session_service

    # The clarifier is satisfied by the answer and hands back a profile.
    class _Clarifier:
        @staticmethod
        def step(state, message):
            return type("O", (), {
                "needs_clarification": False,
                "question": "",
                "state": state,
                "final_query": "dinner using spinach and olives",
                "profile": dict(sample_profile),
                "collected_facts": [],
            })()

    svc.clarifier = _Clarifier()

    captured = {}

    def _generate(session_id, final_query, profile, is_refinement):
        captured["profile"] = profile
        return "ok", False, None

    svc._generate_and_store = _generate
    return svc, session.session_id, captured


def _extractor_says(monkeypatch, spec: PlanSpec):
    import services.planning_delta as planning_delta

    monkeypatch.setattr(
        planning_delta, "extract_state_delta",
        lambda text, extractor=None: PlanningStateDelta(spec=spec),
    )


class TestTheAnswerIsHeard:
    def test_just_a_dinner_plans_a_dinner(self, clarifying, monkeypatch):
        """The reported symptom: one meal asked for, three delivered."""
        svc, session_id, captured = clarifying
        _extractor_says(monkeypatch, PlanSpec(meals=("dinner",)))

        svc.continue_clarification(session_id, "just a dinner")

        assert captured["profile"]["_plan_spec"]["meals"] == ["dinner"]

    def test_a_pantry_stated_in_the_answer_reaches_the_plan(self, clarifying, monkeypatch):
        """Same silence, different sentence — and this one is the whole reason
        the member was in a clarification at all."""
        svc, session_id, captured = clarifying
        import services.pantry_service as pantry_service

        monkeypatch.setattr(
            pantry_service, "extract_pantry_delta",
            lambda message, extractor=None: PlanningStateDelta(
                pantry_add=("spinach", "olives")
            ),
        )
        svc.continue_clarification(session_id, "just a dinner, using the spinach")

        assert set(captured["profile"]["_pantry"]) == {"spinach", "olives"}

    def test_a_diet_stated_in_the_answer_reaches_every_fetch(self, clarifying, monkeypatch):
        svc, session_id, captured = clarifying
        import services.diet_intent as diet_intent

        monkeypatch.setattr(
            diet_intent, "extract_diet_delta",
            lambda message, extractor=None: PlanningStateDelta(diet_tags=("gluten_free",)),
        )
        svc.continue_clarification(session_id, "just a dinner, I'm coeliac")

        assert captured["profile"]["_diet_tags"] == ["gluten_free"]

    def test_the_answer_is_persisted_as_standing_state(self, clarifying, monkeypatch,
                                                       session_service):
        """Not only this plan: "just a dinner" holds until they say otherwise,
        the same as it would from any other turn."""
        svc, session_id, _ = clarifying
        _extractor_says(monkeypatch, PlanSpec(meals=("dinner",)))

        svc.continue_clarification(session_id, "just a dinner")

        assert session_service.get_planning_state(session_id).spec.meals == ("dinner",)

    def test_a_question_that_is_still_unanswered_changes_nothing(self, clarifying,
                                                                 monkeypatch):
        """Only a SETTLED clarification plans. A follow-up question must not
        generate a plan on the way past."""
        svc, session_id, captured = clarifying

        class _StillAsking:
            @staticmethod
            def step(state, message):
                return type("O", (), {
                    "needs_clarification": True,
                    "question": "For how many people?",
                    "state": state,
                })()

        svc.clarifier = _StillAsking()
        text, needs, plan, _intent = svc.continue_clarification(session_id, "dinner")

        assert needs is True and plan is None
        assert "profile" not in captured, "it generated a plan mid-question"
