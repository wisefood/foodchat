"""OrchestratorService — clarification fall-through + preference_update routing.

No LLM calls: the classifier, chat service, extractor, and memory service are
all fakes; only the real routing logic is exercised.
"""

import uuid

from conftest import make_candidates
from services.edit_service import EditService
from services.orchestrator_service import OrchestratorService


class QueuedClassifier:
    """Returns queued intents; records every message it classifies."""

    def __init__(self, *intents):
        self.intents = list(intents)
        self.calls = []

    def classify(self, message, history):
        self.calls.append(message)
        return {"intent": self.intents.pop(0), "target_plan_type": None}


class AckChatService:
    """Smalltalk stand-in that logs messages like the real ChatService."""

    REPLY = "Noted — no chicken from here on."

    def __init__(self, session_service):
        self.session_service = session_service

    def process_smalltalk(self, session_id, message):
        self.session_service.add_message(session_id, "user", message)
        self.session_service.add_message(session_id, "assistant", self.REPLY)
        return self.REPLY, False, None


class AlwaysAmbiguousExtractor:
    """Never resolves a slot — every extraction asks for clarification."""

    def extract(self, message, plan_type):
        return {"meal_type": None, "day": None, "directive": "different",
                "needs_slot_clarification": True,
                "question": "Which meal should I swap?"}


class ChickenNudger:
    """Memory-service stand-in: nudges whenever chicken is mentioned."""

    def suggest(self, session, message):
        if "chicken" in message:
            return [{"kind": "dislike", "value": "chicken"}]
        return []


def make_orchestrator(session_service, classifier):
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    orch.chat_service = AckChatService(session_service)
    orch.weekly_plan_service = None
    orch.foodscholar_service = None
    orch.seed_service = None
    orch.plan_analyst = None
    orch.memory_service = ChickenNudger()
    orch.edit_service = EditService(session_service, client=None,
                                    extractor=AlwaysAmbiguousExtractor())
    orch.orchestrator = classifier
    return orch


def test_valid_intents_cover_all_routed_intents():
    """plan_question was routed but rejected by the validator (pre-fix bug);
    preference_update is new — both must survive classification."""
    from agents import OrchestratorAgent
    assert "plan_question" in OrchestratorAgent.VALID_INTENTS
    assert "preference_update" in OrchestratorAgent.VALID_INTENTS


class TestPreferenceMidClarification:
    def test_preference_reply_escapes_edit_interrogation(self, session_service, sample_profile):
        """The demo transcript bug: 'just remember i dont like chicken' while
        an edit-slot question is pending must be acknowledged (with a memory
        nudge attached), not answered with another 'which meal?'."""
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("cur"), "r", {})

        classifier = QueuedClassifier("edit_plan_slot", "preference_update")
        orch = make_orchestrator(session_service, classifier)

        first = orch.process(session.session_id, session.member_id, "swap something for me")
        assert first.needs_clarification
        assert session.state == "clarifying"

        second = orch.process(session.session_id, session.member_id,
                              "just remember i dont like chicken")
        assert second.intent == "preference_update"
        assert second.content == AckChatService.REPLY
        # Nudges now run on clarification turns too (consent-gated write)
        assert second.memory_suggestions == [{"kind": "dislike", "value": "chicken"}]
        # The fall-through re-classified the message as a fresh turn
        assert classifier.calls == ["swap something for me",
                                    "just remember i dont like chicken"]
        assert session.state == "ready"
