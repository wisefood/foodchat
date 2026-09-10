"""
Every turn hears the member, not just the ones that were going to plan.

The four extractors — shape, pantry, diet, facets — used to live inside the
handlers, so what got heard depended on where the turn was routed:

    daily / refine      all four
    weekly              two (pantry, and its own copy of the diet extraction)
    edit, question,     none
    tool, smalltalk

Which makes this conversation lose a hard constraint:

    "swap Tuesday's dinner"          → edit turn
    "…by the way I'm coeliac now"    → still an edit turn, extracted nothing
    "plan next week"                 → weekly, gluten-free never stated

The parameterised test below is the one that matters: it routes the same
sentence through every intent the classifier can return and asserts the diet
landed each time. Delete the intake call from `_classify_and_route` and all of
them fail.
"""

from __future__ import annotations

import sys
import time
import uuid

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                              # noqa: E402
from models.planning_state import PlanningStateDelta               # noqa: E402
from services import turn_intake                                   # noqa: E402
from services.orchestrator_service import OrchestratorService      # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_turn():
    """No memo carried in from another test — the memo is per turn."""
    turn_intake.forget()
    yield
    turn_intake.forget()


@pytest.fixture
def pinned(monkeypatch):
    """All four extractors, offline, each recording what it was shown."""
    import services.diet_intent as diet_intent
    import services.intent_facets as intent_facets
    import services.pantry_service as pantry_service
    import services.planning_delta as planning_delta

    seen: dict[str, list[str]] = {k: [] for k in ("spec", "pantry", "diet", "facets")}

    def _pin(module, name, key, delta):
        def fake(message, *a, **k):
            seen[key].append(message)
            return delta() if callable(delta) else delta
        monkeypatch.setattr(module, name, fake)

    _pin(planning_delta, "extract_state_delta", "spec",
         PlanningStateDelta(spec=PlanSpec(num_days=2)))
    _pin(pantry_service, "extract_pantry_delta", "pantry",
         PlanningStateDelta(pantry_add=("spinach",)))
    _pin(diet_intent, "extract_diet_delta", "diet",
         PlanningStateDelta(diet_tags=("gluten_free",)))
    _pin(intent_facets, "extract_facet_delta", "facets",
         PlanningStateDelta(cuisines=("thai",)))
    return seen


@pytest.fixture
def session(session_service, sample_profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)


def _intake(session_service, session_id, message):
    """Always the fixture's own service — two SessionService instances keep
    separate caches, so a write through one is invisible to the other."""
    return turn_intake.intake(session_id, message, session_service=session_service)


# ── the merge itself ─────────────────────────────────────────────────────

class TestOnePassHearsEverything:
    def test_all_four_land_on_the_standing_state(self, session_service, session, pinned):
        state = _intake(session_service, session.session_id, "gluten free thai, 2 days, got spinach")
        assert state.diet_tags == ("gluten_free",)
        assert state.pantry == ("spinach",)
        assert state.cuisines == ("thai",)
        assert state.spec.num_days == 2

    def test_it_is_persisted_for_the_next_turn(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "gluten free please")
        turn_intake.forget()
        stored = session_service.get_planning_state(session.session_id)
        assert stored.diet_tags == ("gluten_free",)

    def test_every_extractor_sees_the_members_own_words(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "make it thai")
        for key, messages in pinned.items():
            assert messages == ["make it thai"], key

    def test_an_empty_message_asks_nobody(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "   ")
        assert all(not v for v in pinned.values())

    def test_nothing_heard_leaves_the_state_untouched(self, session_service, session,
                                                      monkeypatch):
        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())

        before = session_service.get_planning_state(session.session_id)
        after = _intake(session_service, session.session_id, "thanks, looks great")
        assert after == before


class TestResetIsAppliedFirst:
    def test_start_over_but_i_still_have_the_spinach(self, session_service, session,
                                                     monkeypatch):
        """A reset wipes the state. Anything said in the SAME breath must
        survive it, which only works if the reset merges first."""
        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(diet_tags=("vegan",))
            ),
        )
        turn_intake.forget()

        monkeypatch.setattr(planning_delta, "extract_state_delta",
                            lambda *a, **k: PlanningStateDelta(reset=True))
        monkeypatch.setattr(pantry_service, "extract_pantry_delta",
                            lambda *a, **k: PlanningStateDelta(pantry_add=("spinach",)))
        monkeypatch.setattr(diet_intent, "extract_diet_delta",
                            lambda *a, **k: PlanningStateDelta())
        monkeypatch.setattr(intent_facets, "extract_facet_delta",
                            lambda *a, **k: PlanningStateDelta())

        state = _intake(session_service, session.session_id, "start over — I have spinach")
        assert state.diet_tags == (), "the reset should have cleared the old diet"
        assert state.pantry == ("spinach",), "the reset ate this turn's own statement"


class TestItRunsThemTogether:
    def test_four_slow_extractors_cost_about_one(self, session_service, session, monkeypatch):
        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        def slow(*_a, **_k):
            time.sleep(0.4)
            return PlanningStateDelta()

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, slow)

        started = time.monotonic()
        _intake(session_service, session.session_id, "anything")
        elapsed = time.monotonic() - started
        # Sequentially this is 1.6s. The margin is wide on purpose — the claim
        # is "concurrent", not a latency budget.
        assert elapsed < 1.0, f"extractors ran in sequence ({elapsed:.2f}s)"

    def test_one_extractor_blowing_up_does_not_lose_the_others(self, session_service, session,
                                                               monkeypatch, pinned):
        import services.planning_delta as planning_delta

        def boom(*_a, **_k):
            raise RuntimeError("groq is down")

        monkeypatch.setattr(planning_delta, "extract_state_delta", boom)
        state = _intake(session_service, session.session_id, "gluten free thai")
        assert state.diet_tags == ("gluten_free",)
        assert state.cuisines == ("thai",)


class TestItIsPaidForOnce:
    def test_a_second_call_in_the_same_turn_is_free(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "make it thai")
        _intake(session_service, session.session_id, "make it thai")
        assert pinned["diet"] == ["make it thai"]

    def test_a_new_turn_asks_again(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "make it thai")
        turn_intake.forget()
        _intake(session_service, session.session_id, "make it thai")
        assert len(pinned["diet"]) == 2

    def test_a_different_message_asks_again(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "make it thai")
        _intake(session_service, session.session_id, "actually make it greek")
        assert len(pinned["diet"]) == 2


# ── in front of the router, whatever the router decides ──────────────────

class _Classifier:
    def __init__(self, intent):
        self.intent = intent

    def classify(self, message, history):
        return {"intent": self.intent, "target_plan_type": None}


def _orch(session_service, intent):
    """An orchestrator that classifies as `intent` and dispatches nowhere.

    `_route` is stubbed because what is under test is the seam, not the
    handlers: the claim is that the member's statement is recorded BEFORE the
    turn is dispatched, so it holds no matter which handler runs.
    """
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    orch.orchestrator = _Classifier(intent)
    orch._route = lambda *a, **k: "routed"
    return orch


EVERY_INTENT = [
    "daily_plan", "weekly_plan", "refine_plan", "edit_plan_slot",
    "switch_plan_type", "nutrition_question", "plan_question",
    "preference_update", "chat",
]


class TestEveryKindOfTurnHearsIt:
    @pytest.mark.parametrize("intent", EVERY_INTENT)
    def test_a_diet_stated_mid_conversation_lands(self, session_service, session,
                                                  pinned, intent):
        """The bug this seam exists to fix: "…by the way I'm coeliac now" said
        on an edit turn, a question turn or plain conversation used to be
        extracted by nobody."""
        orch = _orch(session_service, intent)
        orch._classify_and_route(session, session.session_id, "by the way I'm coeliac now")
        state = session_service.get_planning_state(session.session_id)
        assert state.diet_tags == ("gluten_free",), f"lost on a {intent} turn"

    @pytest.mark.parametrize("intent", EVERY_INTENT)
    def test_a_pantry_statement_lands_too(self, session_service, session, pinned, intent):
        orch = _orch(session_service, intent)
        orch._classify_and_route(session, session.session_id, "I've got spinach to use up")
        assert session_service.get_planning_state(session.session_id).pantry == ("spinach",)

    def test_the_foodscholar_bypass_hears_it_as_well(self, session_service, session, pinned):
        """The bypass returns before the classifier. Intake is in front of
        both, so a constraint stated while asking a nutrition question is not
        the price of asking it."""
        orch = _orch(session_service, "chat")
        orch._handle_nutrition_question = lambda *a, **k: "answered"
        orch._compose_scholar_question = lambda *a, **k: "q"
        orch._classify_and_route(session, session.session_id,
                                 "food scholar: is gluten free ok for kids?")
        assert session_service.get_planning_state(session.session_id).diet_tags == ("gluten_free",)


# ── the horizon follows the request; the shape is standing ───────────────
#
# Reported from the live demo: "daily plans are still weekly plans, just with
# another layout — locally I get 3 recipes." Locally is a fresh session. On the
# demo the member had planned a week earlier in the session, `num_days=7` sat
# on the standing spec, and "plan for today" said nothing the shape extractor
# could read — so the standing seven days went straight into a daily plan.

class TestAFreshDailyRequestIsOneDay:
    def _standing_week(self, session_service, session, monkeypatch):
        """A session whose standing spec carries a week, then a turn that
        says nothing about shape."""
        import services.planning_delta as planning_delta

        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(spec=PlanSpec(
                    num_days=7, plates={"dinner": ("main", "salad")},
                ))
            ),
        )
        # The extractor abstains — exactly what "plan for today" produces.
        monkeypatch.setattr(planning_delta, "extract_state_delta",
                            lambda *a, **k: PlanningStateDelta())
        return _intake(session_service, session.session_id, "plan for today")

    def test_the_standing_week_does_not_become_the_day(self, session_service, session, monkeypatch):
        state = self._standing_week(session_service, session, monkeypatch)
        assert state.spec.num_days == 7, "precondition: the week is standing"
        assert turn_intake.named_shape() is False

        planned = turn_intake.plan_horizon(state, is_refinement=False)
        assert planned.spec.num_days == 1

    def test_the_shape_itself_survives(self, session_service, session, monkeypatch):
        """Only the horizon moves. The salad beside dinner is a preference."""
        state = self._standing_week(session_service, session, monkeypatch)
        planned = turn_intake.plan_horizon(state, is_refinement=False)
        assert planned.spec.roles_for("dinner") == ("main", "salad")
        assert planned.spec.meals == state.spec.meals

    def test_a_refinement_keeps_its_days(self, session_service, session, monkeypatch):
        """'Make day 2 lighter' on a three-day plan is about that plan."""
        state = self._standing_week(session_service, session, monkeypatch)
        assert turn_intake.plan_horizon(state, is_refinement=True).spec.num_days == 7

    def test_a_turn_that_names_its_horizon_is_believed(self, session_service, session, monkeypatch):
        import services.planning_delta as planning_delta

        monkeypatch.setattr(planning_delta, "extract_state_delta",
                            lambda *a, **k: PlanningStateDelta(spec=PlanSpec(num_days=3)))
        state = _intake(session_service, session.session_id, "three days please")
        assert turn_intake.named_shape() is True
        assert turn_intake.plan_horizon(state, is_refinement=False).spec.num_days == 3

    def test_named_shape_is_per_turn(self, session_service, session, pinned):
        _intake(session_service, session.session_id, "2 days")
        assert turn_intake.named_shape() is True
        turn_intake.forget()
        assert turn_intake.named_shape() is False

    def test_a_one_day_shape_is_left_alone(self, session_service, session, monkeypatch):
        import services.planning_delta as planning_delta

        monkeypatch.setattr(planning_delta, "extract_state_delta",
                            lambda *a, **k: PlanningStateDelta())
        state = _intake(session_service, session.session_id, "something light")
        assert turn_intake.plan_horizon(state, is_refinement=False) is state
