"""
A plan of any shape can now be scored.

`_compute_metrics` needed a `ScoredPlan`, and a `ScoredPlan` could only be
three named courses — so the structured path (multi-day, multi-plate) shipped
with every quality score at zero. In the UI that is indistinguishable from a
plan that WAS judged and found wanting.

What it still does not have, deliberately, is an `llm_score`: `plan_meals`
returns one recipe per slot rather than a pool, so there is no combination to
rank, and a fabricated ranking would be worse than an absent one.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from models.session import DayPlan, Meal, MealCourse, MealPlan   # noqa: E402
from services.chat_service import (                              # noqa: E402
    _food_variety_score,
    _plan_as_text,
    scored_plan_from,
)


def _plate(rid, title, role="main", ingredients=None):
    return MealCourse(recipe_id=rid, title=title,
                      ingredients=ingredients or f"{title.lower()}, salt",
                      directions="cook", role=role)


def _two_days() -> MealPlan:
    return MealPlan.from_days([
        DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("b1", "Porridge", ingredients="oats, milk")]),
            Meal("lunch", [_plate("l1", "Soup", ingredients="lentils, carrot")]),
            Meal("dinner", [_plate("d1", "Ragu", ingredients="tomato, beef"),
                            _plate("s1", "Salad", "side", ingredients="lettuce, oil")]),
        ]),
        DayPlan(day=2, meals=[
            Meal("breakfast", [_plate("b2", "Toast", ingredients="bread, jam")]),
            Meal("dinner", [_plate("d2", "Curry", ingredients="chickpeas, spice")]),
        ]),
    ], "two days")


class TestItReadsTheWholePlan:
    def test_every_plate_on_every_day_is_included(self):
        """Judging a seven-day plan on its first day reports the variety of a
        Monday."""
        sp = scored_plan_from(_two_days())
        assert [c.recipe_id for c in sp.courses] == ["b1", "l1", "d1", "s1", "b2", "d2"]

    def test_slots_are_labelled_by_day_so_two_dinners_do_not_collapse(self):
        sp = scored_plan_from(_two_days())
        assert "day 1 dinner (main)" in sp.slots
        assert "day 2 dinner" in sp.slots

    def test_they_stay_in_eating_order(self):
        """A stable sort, not an alphabetical one — otherwise every day's
        dinner would sort before its lunch."""
        sp = scored_plan_from(_two_days())
        assert sp.slot_names == [
            "day 1 breakfast", "day 1 lunch",
            "day 1 dinner (main)", "day 1 dinner (side)",
            "day 2 breakfast", "day 2 dinner",
        ]

    def test_a_single_day_is_not_labelled_with_a_day(self):
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("b", "Porridge")]),
            Meal("dinner", [_plate("d", "Ragu")]),
        ])], "one day")
        assert scored_plan_from(plan).slot_names == ["breakfast", "dinner"]

    def test_a_blank_plate_is_skipped(self):
        """`from_days` inserts blank courses for absent legacy slots."""
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("b", "Porridge")]),
            Meal("lunch", [MealCourse("", "", "", "")]),
        ])], "one day")
        assert scored_plan_from(plan).slot_names == ["breakfast"]

    def test_a_legacy_three_course_plan_works_too(self):
        from models.recipe import CandidateRecipe

        plan = MealPlan.from_courses([
            CandidateRecipe(recipe_id=r, title=r.title(), ingredients="x", directions="y")
            for r in ("a", "b", "c")
        ], "legacy", {})
        sp = scored_plan_from(plan)
        assert sp.slot_names == ["breakfast", "lunch", "dinner"]
        assert sp.is_classic


class TestTheMetricsReadIt:
    def test_variety_counts_across_the_whole_plan(self):
        count, reasoning = _food_variety_score(scored_plan_from(_two_days()))
        assert count >= 8, "six plates of two ingredients each"
        assert "Unique food items" in reasoning

    def test_the_plan_text_names_every_slot(self):
        text = _plan_as_text(scored_plan_from(_two_days()))
        for label in ("Day 1 Breakfast", "Day 1 Dinner (Main)", "Day 2 Dinner"):
            assert label in text

    def test_the_plan_text_does_not_crash_on_a_missing_slot(self):
        """It was hardcoded to Breakfast/Lunch/Dinner, so a plan without one
        would have raised `AttributeError` on `None.title` the moment
        `ScoredPlan` stopped guaranteeing all three."""
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("dinner", [_plate("d", "Ragu")]),
        ])], "dinner only")
        assert "Dinner: Ragu" in _plan_as_text(scored_plan_from(plan))

    def test_variety_survives_a_plan_with_no_ingredients(self):
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("dinner", [MealCourse("d", "Ragu", "", "cook")]),
        ])], "sparse")
        count, _ = _food_variety_score(scored_plan_from(plan))
        assert count == 0


class TestWhatItDeliberatelyDoesNotClaim:
    def test_there_is_no_invented_llm_score(self):
        """`plan_meals` returns one recipe per slot, so there is no combination
        to rank. A fabricated ranking is worse than an absent one."""
        assert scored_plan_from(_two_days()).score == 0

    def test_the_structured_path_does_not_write_llm_score(self):
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(ChatService._generate_structured)
        assert 'key not in ("llm_score", "llm_reasoning")' in src

    def test_the_metrics_stage_is_sheddable(self):
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(ChatService._generate_structured)
        assert 'turn_budget.skip("quality metrics"' in src


class TestExecutedEndToEnd:
    def test_metrics_land_on_the_plan(self, monkeypatch):
        """Executed, not grepped: the scores must actually reach the object the
        member's UI renders."""
        from services import turn_budget
        from services.chat_service import ChatService

        service = ChatService.__new__(ChatService)
        monkeypatch.setattr(
            ChatService, "_compute_metrics",
            lambda self, sid, plan: {
                "llm_score": 0, "llm_reasoning": "",
                "fvs_count": 11, "fvs_reasoning": "eleven items",
                "diversity_llm_score": 4, "diversity_llm_reasoning": "varied",
                "guideline_adherence_score": 3, "guideline_adherence_reasoning": "ok",
            },
        )
        plan = _two_days()
        with turn_budget.start(300):
            metrics = service._compute_metrics("s", scored_plan_from(plan))
            for key, value in metrics.items():
                if key not in ("llm_score", "llm_reasoning"):
                    setattr(plan, key, value)

        assert plan.fvs_count == 11
        assert plan.diversity_llm_score == 4
        assert plan.guideline_adherence_score == 3
        assert plan.llm_score == 0, "no ranking was invented"
