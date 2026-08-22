"""
The weekly planner is no longer stuck at seven days and three meals.

`environment.py` walked a literal 7 days x ["breakfast", "lunch", "dinner"],
with the 7 and the 3 written as `> 7` and `>= 3` inside `step()`, and
`planner.py` looped `TOTAL_SLOTS = 21`. So "plan me three days", "two weeks",
and "a week with a snack" were all unbuildable on this path — while `PlanSpec`,
the type that exists to express exactly that, never reached it.

The tracker mattered as much as the walk. Every target was a daily figure times
a literal 7, so a 3-day plan would have been handed a 14,000 kcal budget and
three meat meals — the calorie steering would have done nothing for the whole
plan, and three meat meals across three days is every dinner.

The dangerous case is the guard: `PlanSpec.default()` is ONE day, and the
default is what a spec looks like when the extractor found no shape in the
message. Honouring it would turn "plan my week" into a single day.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                                    # noqa: E402
from services.weekly_planner.environment import WeeklyMealPlanEnv        # noqa: E402
from services.weekly_planner.state_tracking import (                     # noqa: E402
    DEFAULT_WEEKLY_MEAT_LIMIT,
    WeeklyNutritionalTracker,
)


def _env(spec=None, profile=None):
    return WeeklyMealPlanEnv(
        user_profile=profile or {"diet": [], "preferences": []},
        action_space=object(),
        reward_calculator=object(),
        spec=spec,
    )


# ── the default is unchanged ──────────────────────────────────────────────

class TestTheDefaultShape:
    def test_no_spec_is_still_a_week_of_three_meals(self):
        env = _env()
        assert env.num_days == 7
        assert env.meal_types == ["breakfast", "lunch", "dinner"]
        assert env.total_slots == 21

    def test_a_DEFAULT_spec_does_not_shrink_the_week(self):
        """`PlanSpec.default()` is one day, and it is what a spec looks like
        when the extractor found no shape in the message. Honouring it would
        turn "plan my week" into a single day whenever the words did not happen
        to name a number."""
        env = _env(PlanSpec.default())
        assert env.num_days == 7, "a default spec must not shrink the week"
        assert env.total_slots == 21

    def test_a_one_day_spec_is_also_refused_here(self):
        env = _env(PlanSpec(num_days=1, meals=("lunch", "dinner")))
        assert env.num_days == 7
        # The MEALS still land — a one-day spec that names slots is naming
        # slots, whatever it says about days.
        assert env.meal_types == ["lunch", "dinner"]


# ── the shapes that were unbuildable ─────────────────────────────────────

class TestOtherShapes:
    @pytest.mark.parametrize("days", [2, 3, 5, 10, 14])
    def test_a_different_horizon_is_walked(self, days):
        env = _env(PlanSpec(num_days=days, meals=("breakfast", "lunch", "dinner")))
        assert env.num_days == days
        assert env.total_slots == days * 3

    def test_a_week_with_a_snack_is_four_slots_a_day(self):
        env = _env(PlanSpec(
            num_days=7, meals=("breakfast", "lunch", "snack", "dinner"),
        ))
        assert env.meal_types == ["breakfast", "lunch", "snack", "dinner"]
        assert env.total_slots == 28

    def test_two_meals_a_day_is_fourteen_slots(self):
        env = _env(PlanSpec(num_days=7, meals=("lunch", "dinner")))
        assert env.total_slots == 14


# ── the walk actually advances that way ──────────────────────────────────

class _Rewards:
    """Stands in for the reward calculator; the walk is what is under test."""

    def calculate_step_reward(self, *_a, **_k):
        return 1.0


class _Actions:
    def mark_selected(self, *_a, **_k):
        return None


class TestTheWalk:
    """Driven through the real `step()`, not a reimplementation of its
    arithmetic — a test that re-derives the advance logic would pass even if
    `step()` still counted to 21."""

    def _walk(self, spec=None, limit=200):
        env = WeeklyMealPlanEnv(
            user_profile={"diet": [], "preferences": []},
            action_space=_Actions(),
            reward_calculator=_Rewards(),
            spec=spec,
        )
        seen = []
        for n in range(limit):
            if env.done:
                break
            seen.append((env.current_day, env.meal_types[env.current_meal_idx]))
            env.step({
                "recipe_id": f"r{n}", "recipe_title": f"Dish {n}",
                "recipe_ingredients": "lentils, carrot", "recipe_directions": "cook",
            })
        return env, seen

    def test_a_three_day_plan_stops_after_nine_slots(self):
        env, seen = self._walk(PlanSpec(num_days=3,
                                        meals=("breakfast", "lunch", "dinner")))
        assert len(seen) == 9
        assert seen[0] == (1, "breakfast")
        assert seen[-1] == (3, "dinner")
        assert env.done
        assert len(env.plan) == 9, "and it recorded nine entries"

    def test_a_snack_slot_is_visited_every_day(self):
        _env_, seen = self._walk(PlanSpec(num_days=2,
                                          meals=("breakfast", "snack", "dinner")))
        assert seen == [
            (1, "breakfast"), (1, "snack"), (1, "dinner"),
            (2, "breakfast"), (2, "snack"), (2, "dinner"),
        ]

    def test_the_recorded_entries_carry_the_right_slot(self):
        """`step()` writes `meal_type` from `self.meal_types[idx]`, so an
        off-by-one here would label a snack as a dinner downstream."""
        env, _ = self._walk(PlanSpec(num_days=2, meals=("lunch", "snack")))
        assert [(e["day"], e["meal_type"]) for e in env.plan] == [
            (1, "lunch"), (1, "snack"), (2, "lunch"), (2, "snack"),
        ]

    def test_the_default_still_walks_twenty_one(self):
        env, seen = self._walk()
        assert len(seen) == 21 and len(env.plan) == 21

    def test_a_finished_walk_refuses_another_step(self):
        env, _ = self._walk(PlanSpec(num_days=1, meals=("lunch",)))
        with pytest.raises(RuntimeError):
            env.step({"recipe_id": "x", "recipe_title": "X"})

    def test_the_state_reports_the_right_slot(self):
        env = _env(PlanSpec(num_days=2, meals=("lunch", "dinner")))  # noqa: F841
        state = env._get_state()
        assert state["meal_type"] == "lunch"
        env.current_meal_idx = 1
        assert env._get_state()["meal_type"] == "dinner"


# ── the tracker scales with the horizon ──────────────────────────────────

class TestTheTrackerScales:
    @pytest.mark.parametrize("days,expected", [(1, 2000), (3, 6000), (7, 14000), (14, 28000)])
    def test_the_calorie_budget_follows_the_horizon(self, days, expected):
        """A 3-day plan on a 7-day budget never thinks it is near the limit, so
        the calorie steering does nothing for the whole plan."""
        t = WeeklyNutritionalTracker({"diet": [], "preferences": []}, None, days)
        assert t.targets["calories"] == pytest.approx(expected)

    def test_a_stated_target_scales_too(self):
        t = WeeklyNutritionalTracker(
            {"diet": [], "preferences": ["1800 calories target"]}, None, 3,
        )
        assert t.targets["calories"] == pytest.approx(5400)

    @pytest.mark.parametrize("days,expected", [(1, 1), (3, 1), (7, 3), (14, 6)])
    def test_the_meat_limit_scales(self, days, expected):
        """Three meat meals across three days is every dinner, which is not the
        limit anyone meant."""
        t = WeeklyNutritionalTracker({"diet": [], "preferences": []}, None, days)
        assert t.targets["meat_limit"] == expected

    def test_it_never_rounds_down_to_vegetarian(self):
        """Rounding a short plan's limit to zero would silently turn an
        omnivore's plan vegetarian."""
        for days in range(1, 8):
            t = WeeklyNutritionalTracker({"diet": [], "preferences": []}, None, days)
            assert t.targets["meat_limit"] >= 1

    def test_a_vegetarian_is_still_zero_at_every_horizon(self):
        for days in (1, 3, 7, 14):
            t = WeeklyNutritionalTracker({"diet": ["vegetarian"], "preferences": []},
                                         None, days)
            assert t.targets["meat_limit"] == 0

    def test_the_seven_day_default_is_unchanged(self):
        """Every existing caller passes no horizon."""
        t = WeeklyNutritionalTracker({"diet": [], "preferences": []})
        assert t.targets["calories"] == pytest.approx(2000 * 7)
        assert t.targets["meat_limit"] == DEFAULT_WEEKLY_MEAT_LIMIT

    def test_the_env_gives_its_tracker_the_same_horizon(self):
        env = _env(PlanSpec(num_days=3, meals=("breakfast", "lunch", "dinner")))
        assert env.tracker.num_days == 3
        assert env.tracker.targets["calories"] == pytest.approx(6000)

    def test_a_reset_keeps_the_horizon(self):
        env = _env(PlanSpec(num_days=3, meals=("breakfast", "lunch", "dinner")))
        env.reset()
        assert env.tracker.num_days == 3
        assert env.num_days == 3


# ── the planner no longer counts to 21 ───────────────────────────────────

class TestThePlannerFollowsTheEnv:
    def test_it_asks_the_env_for_its_size(self):
        import inspect

        from services.weekly_planner import planner

        src = inspect.getsource(planner.WeeklyPlanner.generate_full_plan)
        assert 'getattr(self.env, "total_slots"' in src

    def test_the_constant_remains_only_as_a_fallback(self):
        from services.weekly_planner.planner import TOTAL_SLOTS

        assert TOTAL_SLOTS == 21

    def test_the_service_passes_the_spec(self):
        import inspect

        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        assert "spec=state.spec" in src
