"""FoodScholar bridge — answer, clarification round-trip, graceful degradation."""

import uuid

import httpx
import pytest

from services.foodscholar_service import (
    FoodScholarService,
    FoodScholarTurn,
    UNAVAILABLE_MESSAGE,
)


class FakeClient:
    """FoodScholarClient stand-in returning queued /qa/ask payloads."""

    base_url = "http://foodscholar.test"

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def ask(self, question, member_id, qa_thread_id=None,
            clarification_answer=None, clarification_id=None):
        self.calls.append({
            "question": question, "member_id": member_id,
            "qa_thread_id": qa_thread_id,
            "clarification_answer": clarification_answer,
            "clarification_id": clarification_id,
        })
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


ANSWER_PAYLOAD = {
    "needs_clarification": False,
    "primary_answer": {
        "answer": "Keto is generally **not recommended** for teenagers…",
        "confidence": "medium",
        "citations": [
            {"source_type": "article", "source_title": "Ketogenic diets in adolescents",
             "source_url": "/foodscholar/urn:article:123", "display_label": None},
            {"source_type": "guideline", "source_title": "Belgian dietary guidelines",
             "source_url": None, "display_label": "G1"},
        ],
    },
}

CLARIFICATION_PAYLOAD = {
    "needs_clarification": True,
    "qa_thread_id": "thread-42",
    "clarification": {
        "id": "clar-1",
        "question": "Are you asking about a specific age?",
        "options": [{"label": "13–15", "value": "13-15"}, {"label": "16–19", "value": "16-19"}],
    },
}


def _session(session_service, sample_profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)


class TestAnswerPath:
    def test_answer_with_attribution(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        svc = FoodScholarService(session_service, client=FakeClient(ANSWER_PAYLOAD))

        turn = svc.process_question(session.session_id, "is keto safe for teens?")

        assert isinstance(turn, FoodScholarTurn)
        assert not turn.needs_clarification
        assert "not recommended" in turn.text
        assert turn.attribution.source == "foodscholar"
        assert turn.attribution.confidence == "medium"
        assert len(turn.attribution.citations) == 2
        assert turn.attribution.citations[1].label == "G1"
        assert turn.attribution.learn_more_url.startswith("/foodscholar?q=is%20keto")
        # Both turns recorded in the conversation
        roles = [m.role for m in session.conversation[-2:]]
        assert roles == ["user", "assistant"]


class TestClarificationRoundTrip:
    def test_question_then_answer_resumes_thread(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        client = FakeClient(CLARIFICATION_PAYLOAD, ANSWER_PAYLOAD)
        svc = FoodScholarService(session_service, client=client)

        first = svc.process_question(session.session_id, "is keto safe for teens?")
        assert first.needs_clarification
        assert "specific age" in first.text
        assert "13–15" in first.text  # options rendered into the chat text
        assert session.state == "clarifying"
        assert session.clarification["kind"] == "foodscholar"
        assert session.clarification["qa_thread_id"] == "thread-42"

        second = svc.continue_clarification(session.session_id, "16-19")
        assert not second.needs_clarification
        assert second.attribution is not None
        assert session.state == "ready" and session.clarification is None
        # The resume call carried the thread + original question + free-text answer
        resume = client.calls[1]
        assert resume["qa_thread_id"] == "thread-42"
        assert resume["question"] == "is keto safe for teens?"
        assert resume["clarification_answer"] == "16-19"
        assert resume["clarification_id"] == "clar-1"

    def test_clarification_state_survives_restart(self, session_service, sample_profile):
        """The pending FoodScholar thread must be resumable from the DB."""
        from services.session_service import SessionService

        session = _session(session_service, sample_profile)
        svc = FoodScholarService(session_service, client=FakeClient(CLARIFICATION_PAYLOAD))
        svc.process_question(session.session_id, "is keto ok?")

        fresh_sessions = SessionService()  # simulate restart
        restored = fresh_sessions.get_session(session.session_id)
        assert restored.state == "clarifying"
        assert restored.clarification["kind"] == "foodscholar"

        svc2 = FoodScholarService(fresh_sessions, client=FakeClient(ANSWER_PAYLOAD))
        turn = svc2.continue_clarification(session.session_id, "any age")
        assert turn.attribution is not None


class TestDegradation:
    def test_unreachable_foodscholar_never_raises(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        svc = FoodScholarService(
            session_service,
            client=FakeClient(httpx.ConnectError("connection refused")),
        )
        turn = svc.process_question(session.session_id, "is keto ok?")
        assert turn.text == UNAVAILABLE_MESSAGE
        assert turn.attribution is None
        assert not turn.needs_clarification
        assert session.conversation[-1].content == UNAVAILABLE_MESSAGE


class TestOrchestratorRouting:
    """nutrition_question and foodscholar-clarification turns route correctly."""

    def _orchestrator(self, session_service, fs_service):
        from services.orchestrator_service import OrchestratorService

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service
        orch.chat_service = None          # must not be touched in these paths
        orch.weekly_plan_service = None
        orch.memory_service = None        # nudges off in these routing tests
        orch.foodscholar_service = fs_service

        class FixedClassifier:
            def classify(self, message, history):
                return {"intent": "nutrition_question", "target_plan_type": None}

        orch.orchestrator = FixedClassifier()
        return orch

    def test_nutrition_question_returns_attributed_turn(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        svc = FoodScholarService(session_service, client=FakeClient(ANSWER_PAYLOAD))
        orch = self._orchestrator(session_service, svc)

        turn = orch.process(session.session_id, session.member_id, "is keto safe?")
        assert turn.intent == "nutrition_question"
        assert turn.attribution is not None
        assert turn.meal_plan is None

    def test_clarifying_foodscholar_session_bypasses_classifier(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        client = FakeClient(CLARIFICATION_PAYLOAD, ANSWER_PAYLOAD)
        svc = FoodScholarService(session_service, client=client)
        orch = self._orchestrator(session_service, svc)

        orch.process(session.session_id, session.member_id, "is keto safe?")

        # Classifier would now blow up if consulted — replace with a tripwire.
        class Tripwire:
            def classify(self, *_):
                raise AssertionError("classifier must be skipped while clarifying")

        orch.orchestrator = Tripwire()
        turn = orch.process(session.session_id, session.member_id, "for a 17-year-old")
        assert turn.intent == "nutrition_question"
        assert turn.attribution is not None
