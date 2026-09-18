"""
"Score my plan" is about the plan on the screen.

    > score my plan
    < There's no weekly plan in this conversation yet — ask for a weekly plan
      first and I'll have something to work with.

    > why not scoring daily ones as well?

A daily plan was open on the canvas at the time. Two correct refusals produced
one nonsense answer:

* `is_explicit_score_request` requires a **meal listing**, because the scorer
  was written for a plan a member PASTED. "score my plan" lists nothing, so
  the scoring route was never taken;
* the turn fell through to the tool selector, which reached for a weekly
  reader, and the weekly reader said the only true thing it knew.

Nothing in the scorer needed changing. Every metric reads `GroundedMeal`, and
a plan on the canvas is better grounded than a pasted one — its dishes ARE
catalogue recipes, so the lookup, the similarity threshold and the estimated
servings are all skipped. The missing piece was an adapter and a route.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from conftest import make_candidates                                # noqa: E402
from models.pasted_plan import (                                    # noqa: E402
    FROM_RECIPE,
    MATCHED,
    NUTRITION_FROM_RECIPE,
)
from models.session import DayPlan, Meal, MealCourse, MealPlan      # noqa: E402
from services.orchestrator_service import OrchestratorService       # noqa: E402
from services.plan_scorer import PlanScorerService                  # noqa: E402
from services.plan_scorer.canvas import from_canvas, scorer_slot    # noqa: E402


def _course(recipe_id: str, title: str, kcal: float = 400.0) -> MealCourse:
    return MealCourse(
        recipe_id=recipe_id, title=title, ingredients="oats, milk",
        directions="Cook it.",
        nutrition={"kcal": kcal, "protein_g": 12.0, "carbs_g": 40.0, "fat_g": 9.0},
    )


def _two_snack_day() -> MealPlan:
    """The day the member actually asked for: two snacks, in between."""
    meals = [
        Meal("breakfast", [_course("r-b", "Spiced lentil fritters")]),
        Meal("snack", [_course("r-s1", "Pea and mint toast")]),
        Meal("lunch", [_course("r-l", "Barley salad")]),
        Meal("snack_2", [_course("r-s2", "Apple and almond butter")]),
        Meal("dinner", [_course("r-d", "Feta-crusted salmon")]),
    ]
    return MealPlan.from_days([DayPlan(day=1, meals=meals)], reasoning="")


# ── the adapter ──────────────────────────────────────────────────────────

class TestReadingTheCanvas:
    def test_every_plate_is_matched_not_guessed_at(self):
        _plan, grounded = from_canvas(_two_snack_day(), "daily")
        assert len(grounded) == 5
        for dish in grounded:
            assert dish.state == MATCHED
            assert dish.ingredients_source == FROM_RECIPE
            assert dish.nutrition_source == NUTRITION_FROM_RECIPE
            assert dish.recipe_id

    def test_a_second_snack_is_still_a_snack(self):
        """Not `other`, which is where an unrecognised slot lands — an
        afternoon apple would have been counted in with the brunches."""
        assert scorer_slot("snack_2") == "snack"
        _plan, grounded = from_canvas(_two_snack_day(), "daily")
        assert [g.slot for g in grounded].count("snack") == 2

    def test_a_plate_with_no_nutrition_gets_none_not_an_estimate(self):
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("lunch", [MealCourse("r-x", "Mystery dish", "", "")]),
        ])], reasoning="")
        _pasted, grounded = from_canvas(plan, "daily")
        assert grounded[0].nutrition is None
        assert grounded[0].nutrition_source == ""

    def test_a_weekly_plan_reads_its_entries(self):
        class _Weekly:
            entries = [
                {"day": 1, "meal_type": "breakfast",
                 "recipe": {"recipe_id": "r-1", "title": "Porridge",
                            "nutrition": {"kcal": 300.0}}},
                {"day": 2, "meal_type": "dinner",
                 "recipe": {"recipe_id": "r-2", "title": "Chilli"}},
            ]

        pasted, grounded = from_canvas(_Weekly(), "weekly")
        assert pasted.plan_type == "weekly"
        assert len(pasted.days) == 2
        assert [g.day for g in grounded] == [1, 2]

    def test_the_days_are_counted_from_the_plan(self):
        pasted, _grounded = from_canvas(_two_snack_day(), "daily")
        assert len(pasted.days) == 1
        assert pasted.meal_count == 5


# ── the service ──────────────────────────────────────────────────────────

class _RecordingScorer:
    """The real scorer's shape, with none of its model calls."""

    def __init__(self):
        self.seen = None

    def score(self, plan_type, grounded, built, profile, context=None):
        from services.plan_scorer.scoring import ScoreResult

        self.seen = (plan_type, list(grounded))
        return ScoreResult(metrics=[], constraints=[])


class _EchoWriter:
    def __init__(self):
        self.facts = None

    def write(self, facts, message, fallback=""):
        self.facts = facts
        return "Scored."


def _scorer(session_service):
    service = PlanScorerService.__new__(PlanScorerService)
    service.session_service = session_service
    service.parser = None
    service.grounder = None          # nothing to ground: these ARE recipes
    service.scorer = _RecordingScorer()
    service.writer = _EchoWriter()
    return service


@pytest.fixture
def planned(session_service, sample_profile):
    session = session_service.create_session(f"m-{uuid.uuid4()}", sample_profile)
    session_service.add_meal_plan(session.session_id, make_candidates(), "")
    return session


class TestScoringTheCanvas:
    def test_a_daily_plan_is_scored(self, session_service, planned):
        service = _scorer(session_service)
        turn = service.score_canvas(planned.session_id, "daily")
        assert turn is not None
        assert service.scorer.seen[0] == "daily"
        assert len(service.scorer.seen[1]) == 3

    def test_it_never_touches_the_canvas(self, session_service, planned):
        before = session_service.get_session(planned.session_id).daily_canvas
        _scorer(session_service).score_canvas(planned.session_id, "daily")
        after = session_service.get_session(planned.session_id).daily_canvas
        assert after.current_id == before.current_id

    def test_no_plan_means_no_score_rather_than_a_refusal(
        self, session_service, sample_profile,
    ):
        """`None` routes the turn on. A member with no plan is talking about
        one they have not made, and the classifier is better placed to say so
        than a reader inventing an apology."""
        empty = session_service.create_session(f"m-{uuid.uuid4()}", sample_profile)
        assert _scorer(session_service).score_canvas(empty.session_id, "daily") is None

    def test_the_writer_is_told_whose_plan_this_is(self, session_service, planned):
        """The pasted path forbids offering a new plan, because replacing what
        somebody wrote is not what they asked for. On their own canvas that
        instruction makes the score a dead end."""
        service = _scorer(session_service)
        service.score_canvas(planned.session_id, "daily")
        assert service.writer.facts["action"] == "scored_own_plan"
        assert "Do not offer a new plan" not in service.writer.facts["instruction"]

    def test_the_payload_says_where_the_plan_came_from(self, session_service, planned):
        turn = _scorer(session_service).score_canvas(planned.session_id, "daily")
        assert turn.plan_score["source"] == "canvas"


# ── the route ────────────────────────────────────────────────────────────

class TestTheRoute:
    @pytest.mark.parametrize("message", [
        "score my plan",
        "rate this",
        "how does this look?",
        "what do you think of it",
        "evaluate my day",
    ])
    def test_these_reach_the_canvas_scorer(self, session_service, planned, message):
        orch = OrchestratorService.__new__(OrchestratorService)
        orch.plan_scorer = _scorer(session_service)
        session = session_service.get_session(planned.session_id)
        assert orch._maybe_score_canvas(session, planned.session_id, message) is not None

    @pytest.mark.parametrize("message", [
        # A request, not a question about what is there.
        "make it score better",
        "swap the lunch so it rates higher",
        # A pasted plan — `_handle_score_plan` owns that one, because it is
        # the path that can read the text.
        "rate this: breakfast: porridge with berries\nlunch: chicken noodle soup",
        # Nothing about scoring at all.
        "what's for dinner?",
    ])
    def test_these_do_not(self, session_service, planned, message):
        orch = OrchestratorService.__new__(OrchestratorService)
        orch.plan_scorer = _scorer(session_service)
        session = session_service.get_session(planned.session_id)
        assert orch._maybe_score_canvas(session, planned.session_id, message) is None

    def test_a_scorer_that_raises_routes_the_turn_on(self, session_service, planned):
        class _Broken:
            def score_canvas(self, *a, **k):
                raise RuntimeError("catalogue down")

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.plan_scorer = _Broken()
        session = session_service.get_session(planned.session_id)
        assert orch._maybe_score_canvas(session, planned.session_id, "score my plan") is None


class TestTheWiringItself:
    """`_maybe_score_canvas` passing its own tests proves nothing if the turn
    never reaches it — which is exactly how this bug shipped: the scorer could
    have scored that plan all along, and no route arrived at it."""

    def test_score_my_plan_never_reaches_the_classifier(
        self, session_service, planned, monkeypatch,
    ):
        from services import turn_intake

        turn_intake.forget()

        class _NeverAsked:
            def classify(self, message, history):
                raise AssertionError("the classifier was consulted")

        class _NoTool:
            def choose(self, *a, **k):
                return None

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service
        orch.plan_scorer = _scorer(session_service)
        orch.orchestrator = _NeverAsked()
        OrchestratorService._tool_selector = _NoTool()
        monkeypatch.setattr(
            OrchestratorService, "_maybe_use_tool",
            lambda self, *a, **k: pytest.fail("the tool selector answered first"),
        )

        session = session_service.get_session(planned.session_id)
        turn = orch._classify_and_route(session, planned.session_id, "score my plan")
        assert turn.intent == "score_plan"
        assert turn.plan_score["source"] == "canvas"
        turn_intake.forget()
        OrchestratorService._tool_selector = None
