"""
The structured path is a first-class path.

`plan_structured` builds the shape `PlanSpec` exists to serve — N days, N meals,
multi-plate meals. It was missing almost everything the classic three-meal path
had:

    no `query` parameter at all   the member's message never reached planning
    no apply_transparency         no reason chips, empty constraints ledger
    no fetch_details             plates rendered with no nutrition
    no overlay_plan              the member's own adapted recipes ignored
    favourites never sent        only pantry boost ids, or nothing
    is_refinement ignored        every refinement started a fresh canvas

It matters more than it looks: `apply_plan_parameters` routes through the
STANDING PlanSpec, so once a member has asked for any non-default shape, every
slider apply lands here.
"""

from __future__ import annotations

import inspect

import pytest

from models.session import DayPlan, Meal, MealCourse, MealPlan
from services.transparency import apply_transparency


def _plate(rid, title, role="main"):
    return MealCourse(recipe_id=rid, title=title, ingredients=f"{title} stuff",
                      directions="cook", role=role)


def _structured_plan():
    """Two days, and a dinner with two plates — the shape the legacy accessors
    cannot see past."""
    return MealPlan.from_days([
        DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("r1", "Porridge")]),
            Meal("lunch", [_plate("r2", "Soup")]),
            Meal("dinner", [_plate("r3", "Ragu"), _plate("r4", "Salad", "side")]),
        ]),
        DayPlan(day=2, meals=[
            Meal("breakfast", [_plate("r5", "Toast")]),
            Meal("dinner", [_plate("r6", "Curry")]),
        ]),
    ], reasoning="two days")


class _Rich:
    def __init__(self, kcal):
        self._kcal = kcal
        self.image_url = "http://img"

    def nutrition_dict(self):
        return {"kcal": self._kcal}


class TestTransparencyReachesEveryPlate:
    """It iterated the three legacy scalar accessors, which are a compatibility
    projection of day 1's MAIN plates — so every side, every dessert and every
    day after the first got nothing."""

    def test_every_plate_on_every_day_is_enriched(self):
        plan = _structured_plan()
        enrichment = {f"r{i}": _Rich(100 * i) for i in range(1, 7)}
        apply_transparency(plan, {"diet": []}, set(), enrichment)

        seen = {
            plate.recipe_id: plate.nutrition
            for day in plan.day_plans for meal in day.meals
            for plate in meal.plates if plate.recipe_id
        }
        assert set(seen) == {"r1", "r2", "r3", "r4", "r5", "r6"}
        assert all(n for n in seen.values()), "a plate was left without nutrition"

    def test_the_second_plate_of_a_meal_is_reached(self):
        """A side dish is exactly what the legacy accessors cannot see."""
        plan = _structured_plan()
        apply_transparency(plan, {"diet": []}, set(), {"r4": _Rich(120)})
        side = plan.day_plans[0].meals[2].plates[1]
        assert side.recipe_id == "r4"
        assert side.nutrition == {"kcal": 120}

    def test_a_second_day_is_reached(self):
        plan = _structured_plan()
        apply_transparency(plan, {"diet": []}, set(), {"r6": _Rich(300)})
        assert plan.day_plans[1].meals[1].plates[0].nutrition == {"kcal": 300}

    def test_every_plate_gets_reason_chips(self):
        plan = _structured_plan()
        apply_transparency(plan, {"diet": []}, {"r3"}, {})
        chips = {
            plate.recipe_id: plate.match_reasons
            for day in plan.day_plans for meal in day.meals
            for plate in meal.plates if plate.recipe_id
        }
        assert all(isinstance(v, list) for v in chips.values())
        assert any(r["kind"] == "pinned" for r in chips["r3"])

    def test_the_ledger_and_summary_are_populated(self):
        plan = _structured_plan()
        apply_transparency(plan, {"diet": ["vegetarian"], "allergies": ["nuts"]},
                           set(), {})
        assert plan.constraints_applied
        assert plan.personalization_summary is not None

    def test_a_legacy_plan_still_works_identically(self):
        """`day_plans` projects three scalars into one day, so the same
        iteration must cover the old shape too."""
        plan = MealPlan.from_courses(
            [_plate("a", "B"), _plate("b", "L"), _plate("c", "D")],
            "legacy", {}, version=1, parent_id=None,
        )
        apply_transparency(plan, {"diet": []}, set(),
                           {"a": _Rich(1), "b": _Rich(2), "c": _Rich(3)})
        assert plan.breakfast.nutrition == {"kcal": 1}
        assert plan.dinner.nutrition == {"kcal": 3}

    def test_blank_slots_are_skipped_not_crashed(self):
        """`from_days` inserts blank MealCourses for absent legacy slots."""
        plan = _structured_plan()
        apply_transparency(plan, {"diet": []}, set(), {})  # no enrichment at all
        assert plan.constraints_applied is not None


class TestThePipelineTakesTheQuery:
    def test_plan_structured_accepts_a_query(self):
        from services.planning_pipeline import PlanningPipeline

        params = inspect.signature(PlanningPipeline.plan_structured).parameters
        assert "query" in params, "the member's message must reach planning"

    def test_the_caller_passes_it(self):
        """Read from the AST, not by slicing to the first `)`.

        The text version broke the moment another keyword argument's value
        contained a bracketed expression — which says nothing about whether the
        query reaches planning, only about where a paren happens to fall.
        """
        import ast

        from services.chat_service import ChatService

        tree = ast.parse(inspect.getsource(ChatService._generate_structured).lstrip())
        passed = {
            kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "plan_structured"
            for kw in node.keywords
        }
        assert "query" in passed, "the member's message must reach planning"

    def test_the_caller_asks_for_something_new(self):
        """Same call, and the reason the assertion above needed rewriting: a
        fresh plan must exclude what the member was just served, or asking
        twice returns the same plan — RecipeWrangler's order is deterministic
        and the grader runs at temperature 0."""
        import ast

        from services.chat_service import ChatService

        tree = ast.parse(inspect.getsource(ChatService._generate_structured).lstrip())
        passed = {
            kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "plan_structured"
            for kw in node.keywords
        }
        assert "avoid_recent" in passed

    def test_real_favourites_are_boosted_not_just_pantry_ids(self):
        from services.planning_pipeline import PlanningPipeline

        src = inspect.getsource(PlanningPipeline.plan_structured)
        head = src[:src.find("PLANNER.plan_meals(")]
        assert "favorite_recipe_ids" in head, "the member's own favourites"

    def test_a_declined_favourites_offer_is_honoured_here_too(self):
        from services.planning_pipeline import PlanningPipeline

        src = inspect.getsource(PlanningPipeline.plan_structured)
        assert "use_favorites" in src


class TestTheCallerDoesTheRest:
    @pytest.mark.parametrize("needle,why", [
        ("CANDIDATES.fetch_details", "plates rendered with no nutrition"),
        ("apply_transparency", "no chips and an empty ledger"),
        ("overlay_plan", "the member's adapted recipes were ignored"),
        ("refine_prepared_meal_plan", "every refinement lost its lineage"),
        ("split_ledger", "a relaxed constraint could be claimed as honoured"),
    ])
    def test_the_missing_step_is_now_present(self, needle, why):
        from services.chat_service import ChatService

        src = inspect.getsource(ChatService._generate_structured)
        assert needle in src, why

    def test_transparency_runs_before_the_pantry_annotation(self):
        """Otherwise the pantry chips are overwritten rather than appended to."""
        from services.chat_service import ChatService

        src = inspect.getsource(ChatService._generate_structured)
        assert src.find("apply_transparency") < src.find("annotate_daily_plan")

    def test_feedback_history_is_no_longer_dropped(self):
        from services.chat_service import ChatService

        src = inspect.getsource(ChatService._generate_structured)
        assert "feedback_history" in src


class TestRefinementKeepsItsLineage:
    def test_a_refinement_bumps_the_version_and_keeps_the_root(self, session_service,
                                                               sample_profile):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        first = session_service.add_prepared_meal_plan(
            session.session_id, _structured_plan()
        )
        assert first.version == 1

        second = session_service.refine_prepared_meal_plan(
            session.session_id, _structured_plan()
        )
        assert second.version == 2, "a refinement must not restart at v1"
        assert second.parent_id == first.id, "lineage must be walkable"

    def test_the_days_survive_the_refinement(self):
        """`refine_meal_plan` rebuilds from three courses, which flattens
        `days` — the exact shape this path exists to escape."""
        import uuid

        from services.session_service import SessionService

        svc = SessionService()
        session = svc.create_session(f"member-{uuid.uuid4()}", {"diet": []})
        svc.add_prepared_meal_plan(session.session_id, _structured_plan())
        refined = svc.refine_prepared_meal_plan(
            session.session_id, _structured_plan()
        )
        assert refined.days is not None
        assert len(refined.days) == 2
        assert len(refined.days[0].meals[2].plates) == 2, "the side plate was lost"

    def test_refining_with_no_prior_plan_is_a_first_plan(self):
        import uuid

        from services.session_service import SessionService

        svc = SessionService()
        session = svc.create_session(f"member-{uuid.uuid4()}", {"diet": []})
        plan = svc.refine_prepared_meal_plan(session.session_id, _structured_plan())
        assert plan.version == 1 and plan.parent_id is None
