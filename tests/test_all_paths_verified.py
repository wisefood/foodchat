"""
Every planning path measures what it produced.

`PlanBrief` -> `PlanStrategist` -> `plan_verifier` were wired into ONE path:
`_generate_structured`, which a member reaches only by asking for an unusual
SHAPE. A plain "plan my day" goes through the classic path, and a week goes
through the weekly planner, and neither had a brief, a strategist, or a single
measured constraint. They reported the request back as though it were a result
— the exact failure the verifier exists to end — on the two paths that carry
almost all the traffic.

The weekly path holds a flat list of `{day, meal_idx, meal_type, recipe}`
dicts rather than a `MealPlan`. Rather than teach the verifier a second shape,
and have two definitions of "every plate in a plan" drift apart, the entries
are adapted into the one it already reads.
"""

from __future__ import annotations

import inspect
import sys

import pytest

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                          # noqa: E402
from models.recipe import RecipeEnrichment                        # noqa: E402
from services import plan_verifier                                # noqa: E402
from services.weekly_plan_service import _as_meal_plan            # noqa: E402


def _entry(day, idx, slot, rid, title, ingredients="", nutrition=None):
    return {
        "day": day, "meal_idx": idx, "meal_type": slot,
        "recipe": {
            "recipe_id": rid, "recipe_title": title,
            "recipe_ingredients": ingredients or f"{title.lower()}, salt",
            "recipe_directions": "cook",
            "nutrition": nutrition,
        },
    }


def _week(days=2):
    out = []
    for day in range(1, days + 1):
        for idx, slot in enumerate(("breakfast", "lunch", "dinner")):
            out.append(_entry(day, idx, slot, f"r{day}{idx}", f"Dish {day}{idx}"))
    return out


# ── the weekly adapter ────────────────────────────────────────────────────

class TestTheWeeklyAdapter:
    def test_it_produces_the_shape_the_verifier_reads(self):
        plan = _as_meal_plan(_week(2))
        assert plan is not None
        assert len(plan.day_plans) == 2
        assert [m.meal_type for m in plan.day_plans[0].meals] == [
            "breakfast", "lunch", "dinner"
        ]

    def test_every_entry_becomes_a_plate(self):
        plan = _as_meal_plan(_week(3))
        plates = [p for d in plan.day_plans for m in d.meals for p in m.plates]
        assert len(plates) == 9

    def test_days_and_meals_keep_their_order(self):
        plan = _as_meal_plan(list(reversed(_week(2))))
        assert [d.day for d in plan.day_plans] == [1, 2]
        assert [m.meal_type for m in plan.day_plans[0].meals] == [
            "breakfast", "lunch", "dinner"
        ]

    def test_two_entries_in_one_slot_become_two_plates(self):
        """Latent — the weekly planner is a fixed 7x3 walk — but the adapter
        must not be the thing that loses a plate when it stops being."""
        entries = [
            _entry(1, 0, "dinner", "main", "Ragu"),
            _entry(1, 1, "dinner", "side", "Salad"),
        ]
        plan = _as_meal_plan(entries)
        assert len(plan.day_plans[0].meals[0].plates) == 2

    def test_an_entry_with_no_recipe_id_is_skipped(self):
        entries = _week(1) + [_entry(1, 3, "snack", "", "Nothing")]
        plan = _as_meal_plan(entries)
        plates = [p for d in plan.day_plans for m in d.meals for p in m.plates]
        assert len(plates) == 3

    def test_an_empty_week_adapts_to_nothing_rather_than_raising(self):
        assert _as_meal_plan([]) is None

    def test_the_ingredient_text_survives(self):
        """The pantry and allergen checks both read it."""
        plan = _as_meal_plan([_entry(1, 0, "dinner", "r", "Satay", "peanuts, chicken")])
        assert "peanuts" in plan.day_plans[0].meals[0].plates[0].ingredients


# ── the verifier reads a week ─────────────────────────────────────────────

class TestVerifyingAWeek:
    def test_an_allergen_anywhere_in_the_week_is_caught(self):
        entries = _week(2)
        entries.append(_entry(2, 3, "snack", "bad", "Satay", "peanuts, chicken"))
        report = plan_verifier.verify(
            _as_meal_plan(entries), {"allergens": ["peanuts"]}, {},
        )
        check = report.get("allergens")
        assert check.status == plan_verifier.FAILED
        assert check.offenders == ("bad",)

    def test_a_clean_week_passes_over_all_of_it(self):
        report = plan_verifier.verify(
            _as_meal_plan(_week(3)), {"allergens": ["peanuts"]}, {},
        )
        check = report.get("allergens")
        assert check.status == plan_verifier.PASSED
        assert check.of == 9, "every meal of every day was looked at"

    def test_the_diet_check_reads_the_whole_week(self):
        entries = _week(2)
        rich = {
            f"r{d}{i}": RecipeEnrichment(recipe_id=f"r{d}{i}", title="x",
                                         diet_tags=["vegetarian"])
            for d in (1, 2) for i in range(3)
        }
        rich["r21"] = RecipeEnrichment(recipe_id="r21", title="x", diet_tags=["omnivore"])
        report = plan_verifier.verify(
            _as_meal_plan(entries), {"diet": ["vegetarian"]}, rich,
        )
        check = report.get("diet")
        assert check.status == plan_verifier.FAILED
        assert check.offenders == ("r21",)

    def test_calories_are_scored_per_day_not_per_week(self):
        """A week's total against a daily target would fail every plan."""
        entries = _week(2)
        rich = {
            f"r{d}{i}": RecipeEnrichment(recipe_id=f"r{d}{i}", title="x", kcal=700)
            for d in (1, 2) for i in range(3)
        }
        check = plan_verifier.verify(
            _as_meal_plan(entries), {"kcal_target": 2100}, rich,
        ).get("calories")
        assert check.status == plan_verifier.PASSED
        assert check.of == 2, "two days were scored"


# ── all three paths ───────────────────────────────────────────────────────

class TestEveryPathIsWired:
    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_the_daily_paths_build_a_brief_and_verify(self, method):
        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert "_brief_for(" in src, f"{method} has no brief"
        assert "plan_verifier.verify(" in src, f"{method} does not measure"
        assert "brief.to_requested()" in src, f"{method} checks something else"

    def test_the_weekly_path_verifies(self):
        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        assert "plan_verifier.verify(" in src
        assert "_as_meal_plan(" in src

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_measured_rows_are_appended_not_substituted(self, method):
        """The declarative rows carry `source` — which diner a constraint is
        there for — which a measurement cannot know. Both, in that order."""
        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert "list(meal_plan.constraints_applied or []) + report.as_ledger_rows()" in src

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_a_failed_check_reaches_the_reply(self, method):
        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert '"verified_problems"' in src

    def test_the_weekly_reply_gets_them_too(self):
        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        assert '"verified_problems"' in src

    def test_weekly_verification_never_costs_the_week(self):
        """It describes a plan that already exists."""
        from services.weekly_plan_service import WeeklyPlanService

        src = inspect.getsource(WeeklyPlanService.process_message)
        verify_at = src.index("plan_verifier.verify(")
        assert "except Exception" in src[verify_at:verify_at + 900]


class TestOneContract:
    def test_all_three_check_against_the_same_brief_shape(self):
        """`to_requested()` is the single definition of what gets checked, so
        the three paths cannot drift into three ideas of the same constraint."""
        requested = PlanBrief(
            allergens=("peanuts",), diet=("vegetarian",), pantry=("zucchini",),
        ).to_requested()
        plan = _as_meal_plan(_week(1))
        names = {c.name for c in plan_verifier.verify(plan, requested, {}).checks}
        assert names == {"allergens", "diet", "pantry"}
