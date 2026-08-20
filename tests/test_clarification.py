"""ClarificationManager — state-machine transitions with faked LLM clients.

No test here calls an LLM: the manager is built via __new__ and its clients
are replaced with stubs, so we exercise exactly the transition logic that the
old generator implementation could not persist.
"""

import json

from services.clarification import (
    ClarificationManager,
    ClarificationOutcome,
    ClarificationState,
)


class FakeLLM:
    """Stands in for a pooled ChatGroq client; returns queued .content payloads."""

    def __init__(self, *payloads):
        self._payloads = list(payloads)

    def invoke(self, _messages, config=None):
        class R:
            pass

        r = R()
        payload = self._payloads.pop(0)
        r.content = payload if isinstance(payload, str) else json.dumps(payload)
        return r


class FakeReconciler:
    def __init__(self, result):
        self._result = result

    def reconcile(self, query, profile):
        return self._result


def make_manager(
    reconcile_result=None,
    checker_payloads=(),
    profile_payloads=(),
    question_payloads=(),
    reformulate_payloads=(),
) -> ClarificationManager:
    mgr = ClarificationManager.__new__(ClarificationManager)
    mgr.reconciler = FakeReconciler(reconcile_result or {"needs_clarification": False})
    mgr.query_checker = FakeLLM(*checker_payloads)
    mgr.profile_checker = FakeLLM(*profile_payloads)
    mgr.question_generator = FakeLLM(*question_payloads)
    mgr.reformulator = FakeLLM(*reformulate_payloads)
    return mgr


PROFILE = {"diet": ["vegan"], "allergies": [], "preferences": ["quick meals"]}


class TestStartOutcomes:
    def test_specific_query_passes_through(self):
        mgr = make_manager(
            reconcile_result={"needs_clarification": False},
            checker_payloads=[{"response": "YES"}],
        )
        outcome = mgr.start("vegan dinner with lentils under 30 minutes", PROFILE, "daily_plan")
        assert not outcome.needs_clarification
        assert outcome.final_query == "vegan dinner with lentils under 30 minutes"
        # Reconciliation keys ride along on the outcome profile
        assert outcome.profile["needs_clarification"] is False

    def test_dietary_conflict_asks_warning_first(self):
        mgr = make_manager(
            reconcile_result={
                "needs_clarification": True,
                "has_dietary_conflict": True,
                "conflict_explanation": "You are vegan but asked for beef — proceed?",
                "missing_info": [],
            },
            reformulate_payloads=[{"reformulated_query": "final query"}],
        )
        outcome = mgr.start("beef stew please", PROFILE, "daily_plan")
        assert outcome.needs_clarification
        assert "vegan" in outcome.question
        assert outcome.state.phase == "conflict"

        # Answering the conflict (no pending topics) finalizes
        done = mgr.step(outcome.state, "yes, make an exception")
        assert done.final_query == "final query"
        assert done.state is None

    def test_missing_info_asks_at_most_one_topic(self):
        """Interrogation cap: even when the reconciler emits several topics,
        only the first is asked — parameter-style topics live on the slider
        card now, so a single food-direction question is the ceiling."""
        mgr = make_manager(
            reconcile_result={
                "needs_clarification": True,
                "has_dietary_conflict": False,
                "missing_info": ["food preferences or cravings", "budget"],
            },
            question_payloads=["Any cravings for today?"],
            reformulate_payloads=[{"reformulated_query": "quick cheap vegan plan"}],
        )
        outcome = mgr.start("plan my day", PROFILE, "refine_plan")
        assert outcome.question == "Any cravings for today?"
        assert outcome.state.origin_intent == "refine_plan"
        assert outcome.state.pending_topics == ["food preferences or cravings"]

        final = mgr.step(outcome.state, "something greek")
        assert final.question is None
        assert final.final_query == "quick cheap vegan plan"
        assert final.state is None or not final.state.pending_topics
        assert final.collected_facts == [
            "Q: Any cravings for today?\nA: something greek",
        ]

    def test_vague_query_with_thin_profile_asks(self):
        mgr = make_manager(
            reconcile_result={"needs_clarification": False},
            checker_payloads=[{"response": "NO"}],
            profile_payloads=[{"response": "NO", "suggestions": ["favorite cuisines"]}],
            question_payloads=["Any cuisines you love?"],
            reformulate_payloads=[{"reformulated_query": "italian vegan plan"}],
        )
        outcome = mgr.start("food please", PROFILE, "daily_plan")
        assert outcome.question == "Any cuisines you love?"
        final = mgr.step(outcome.state, "italian")
        assert final.final_query == "italian vegan plan"

    def test_vague_query_with_rich_profile_reformulates_silently(self):
        mgr = make_manager(
            reconcile_result={"needs_clarification": False},
            checker_payloads=[{"response": "NO"}],
            profile_payloads=[{"response": "YES", "suggestions": []}],
            reformulate_payloads=[{"reformulated_query": "vegan plan matching profile"}],
        )
        outcome = mgr.start("food please", PROFILE, "daily_plan")
        assert not outcome.needs_clarification
        assert outcome.final_query == "vegan plan matching profile"


class TestStateSerialization:
    def test_json_roundtrip(self):
        state = ClarificationState(
            original_query="q", profile=PROFILE, origin_intent="daily_plan",
            phase="collect", pending_topics=["budget"],
            current_question="How much?", transcript=[{"question": "a?", "answer": "b"}],
        )
        restored = ClarificationState.from_json(state.to_json())
        assert restored == state

    def test_resume_after_restart(self):
        """step() works on a state deserialized by a different manager instance."""
        state = ClarificationState(
            original_query="plan my day", profile=PROFILE, origin_intent="daily_plan",
            phase="collect", pending_topics=["cooking time"],
            current_question="How long can you cook?",
        )
        revived = ClarificationState.from_json(state.to_json())

        mgr = make_manager(reformulate_payloads=[{"reformulated_query": "resumed"}])
        outcome = mgr.step(revived, "an hour")
        assert outcome.final_query == "resumed"


class TestCollectedFactsRemembered:
    def test_final_outcome_carries_collected_facts(self):
        mgr = make_manager(
            reconcile_result={
                "needs_clarification": True,
                "has_dietary_conflict": False,
                "missing_info": ["cooking time"],
            },
            question_payloads=["How long can you cook?"],
            reformulate_payloads=[{"reformulated_query": "quick plan"}],
        )
        outcome = mgr.start("plan my day", PROFILE, "daily_plan")
        final = mgr.step(outcome.state, "30 minutes on weekdays")
        assert final.final_query == "quick plan"
        assert final.collected_facts == [
            "Q: How long can you cook?\nA: 30 minutes on weekdays",
        ]

    def test_answers_land_in_session_profile_history(self, session_service, sample_profile):
        """The whole point: what the user answered is never re-asked in-session."""
        import uuid
        from services.chat_service import ChatService
        from services.session_service import SessionService

        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        state = ClarificationState(
            original_query="plan my day", profile=dict(sample_profile),
            origin_intent="daily_plan", phase="collect",
            pending_topics=["cooking time"],
            current_question="How long can you cook?",
        )
        session_service.set_clarification_state(session.session_id, state.to_dict())

        svc = ChatService.__new__(ChatService)
        svc.session_service = session_service
        svc.clarifier = make_manager(
            reformulate_payloads=[{"reformulated_query": "quick plan"}],
        )

        class NoPlanPipeline:
            def generate(self, *a, **k):
                return []

        svc.pipeline = NoPlanPipeline()

        class NoSignals:
            def get_signals(self, member_id):
                from services.feedback_service import FeedbackSignals
                return FeedbackSignals()

        svc.feedback_service = NoSignals()

        svc.continue_clarification(session.session_id, "30 minutes tops")

        assert "30 minutes tops" in session.user_profile["history"]
        # Survives restart — the reconciler's known facts see it next time
        restored = SessionService().get_session(session.session_id)
        assert "30 minutes tops" in restored.user_profile["history"]


class TestClarificationStateIsPersistable:
    """A turn that asks a question must be able to STORE what it asked.

    The profile snapshot rides inside `ClarificationState` and is json.dumps-ed
    onto the session row. It once carried a live `PlanSpec` dataclass, so any
    turn that happened to clarify died with "Object of type PlanSpec is not
    JSON serializable" — intermittently, because whether a turn clarifies is
    an LLM decision. Anything the plan flow stashes in that snapshot has to
    survive the dump.
    """

    def _clarifying_service(self, session_service):
        from services.chat_service import ChatService

        svc = ChatService.__new__(ChatService)
        svc.session_service = session_service
        svc.clarifier = make_manager(
            reconcile_result={
                "needs_clarification": True,
                "has_dietary_conflict": False,
                "missing_info": ["cooking time"],
            },
            question_payloads=["How long can you cook?"],
        )
        svc.seed_service = None  # no seeds on this path
        return svc

    def test_plan_turn_that_clarifies_persists_cleanly(
        self, session_service, sample_profile, monkeypatch
    ):
        import importlib
        import uuid

        from models.plan_spec import PlanSpec
        from models.planning_state import PlanningStateDelta
        from services.session_service import SessionService

        # import_module, not `from services import chat_service`: the package's
        # singleton placeholders shadow submodule names on attribute lookup.
        chat_service_module = importlib.import_module("services.chat_service")

        # Shape extraction is an LLM call; pin it so the turn is offline and
        # the standing spec is definitely non-default — the exact combination
        # that used to raise.
        monkeypatch.setattr(
            chat_service_module, "extract_state_delta",
            lambda *a, **k: PlanningStateDelta(
                spec=PlanSpec(num_days=3, meals=("lunch",),
                              plates={"lunch": ("main", "salad")}),
            ),
        )

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        svc = self._clarifying_service(session_service)

        text, needs, plan = svc.process_plan_request(
            session.session_id, "plan 3 days with a salad on the side",
        )

        assert needs is True and plan is None and text
        # The write actually reached the DB, and what it wrote is JSON.
        restored = SessionService().get_session(session.session_id)
        assert restored.state == "clarifying"
        json.dumps(restored.clarification)

    def test_stored_spec_still_drives_the_structured_path(self):
        """The round-trip must preserve the shape, not just serialize."""
        from models.plan_spec import PlanSpec

        spec = PlanSpec(num_days=3, meals=("lunch",), plates={"lunch": ("main", "salad")})
        revived = PlanSpec.coerce(json.loads(json.dumps(spec.to_dict())))

        assert revived == spec
        assert not revived.is_default          # still routes to plan_structured
        assert PlanSpec.coerce(spec) is spec    # a live instance passes through
