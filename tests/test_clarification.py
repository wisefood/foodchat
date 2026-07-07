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

    def invoke(self, _messages):
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

    def test_missing_info_collects_each_topic(self):
        mgr = make_manager(
            reconcile_result={
                "needs_clarification": True,
                "has_dietary_conflict": False,
                "missing_info": ["cooking time", "budget"],
            },
            question_payloads=["How long can you cook?", "What's your budget?"],
            reformulate_payloads=[{"reformulated_query": "quick cheap vegan plan"}],
        )
        outcome = mgr.start("plan my day", PROFILE, "refine_plan")
        assert outcome.question == "How long can you cook?"
        assert outcome.state.origin_intent == "refine_plan"

        second = mgr.step(outcome.state, "30 minutes")
        assert second.question == "What's your budget?"
        assert second.state.transcript[0] == {
            "question": "How long can you cook?", "answer": "30 minutes",
        }

        final = mgr.step(second.state, "20 euros")
        assert final.final_query == "quick cheap vegan plan"

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
