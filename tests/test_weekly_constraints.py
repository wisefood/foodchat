"""M6 — weekly constraints actually steer selection (meat limit, calories).

These are the regression tests for the pre-M6 bug where constraint
penalties were computed after each pick and never influenced anything.
All deterministic — no LLM, no network.
"""

from services.weekly_planner.environment import WeeklyMealPlanEnv
from services.weekly_planner.planner import WeeklyPlanner
from services.weekly_planner.reward_logic import (
    RewardCalculator,
    apply_hard_constraints,
    constraint_score,
)
from services.weekly_planner.state_tracking import WeeklyNutritionalTracker


def _meat(day, idx, kcal=None):
    action = {"recipe_id": f"meat-{day}-{idx}", "recipe_title": "Beef Stew",
              "recipe_ingredients": "beef, potato", "recipe_directions": "d"}
    if kcal is not None:
        action["nutrition"] = {"kcal": kcal}
    return action


def _veg(day, idx, kcal=None):
    action = {"recipe_id": f"veg-{day}-{idx}", "recipe_title": "Lentil Bowl",
              "recipe_ingredients": "lentils, rice", "recipe_directions": "d"}
    if kcal is not None:
        action["nutrition"] = {"kcal": kcal}
    return action


class MixedActions:
    """One meat + one veg candidate per slot."""

    def __init__(self, kcal_meat=None, kcal_veg=None):
        self.kcal_meat, self.kcal_veg = kcal_meat, kcal_veg

    def get_candidate_actions(self, meal_type, state):
        day, idx = state["day"], state["meal_idx"]
        return [_meat(day, idx, self.kcal_meat), _veg(day, idx, self.kcal_veg)]

    def mark_selected(self, recipe_id):
        pass


def _plan(profile, action_space, scorer=None):
    env = WeeklyMealPlanEnv(
        user_profile=profile,
        action_space=action_space,
        reward_calculator=RewardCalculator(),
    )
    return WeeklyPlanner(env).generate_full_plan(user_query="week", scorer=scorer), env


class TestMeatLimit:
    def test_meat_limit_enforced_even_when_scorer_prefers_meat(self):
        # The scorer loves beef; the hard constraint must still cap it.
        meat_lover = lambda c, chosen: 5.0 if "beef" in c["recipe_ingredients"] else 0.0
        entries, env = _plan(
            {"preferences": ["meat limit 2"], "diet": []},
            MixedActions(), scorer=meat_lover,
        )
        assert len(entries) == 21
        meat_count = sum(1 for e in entries if "beef" in e["recipe"]["recipe_ingredients"])
        assert meat_count == 2
        assert env.tracker.meat_meals_count == 2

    def test_all_meat_pool_relaxes_instead_of_failing(self):
        class MeatOnly:
            def get_candidate_actions(self, meal_type, state):
                return [_meat(state["day"], state["meal_idx"])]

            def mark_selected(self, recipe_id):
                pass

        entries, _ = _plan({"preferences": ["meat limit 1"], "diet": []}, MeatOnly())
        assert len(entries) == 21  # completed despite an unsatisfiable limit

    def test_vegetarian_diet_means_zero_meat_limit(self):
        tracker = WeeklyNutritionalTracker({"diet": ["vegetarian"], "preferences": []})
        assert tracker.targets["meat_limit"] == 0

    def test_default_limit_and_preference_override(self):
        assert WeeklyNutritionalTracker({"preferences": []}).targets["meat_limit"] == 3
        assert WeeklyNutritionalTracker(
            {"preferences": ["max 2 meat meals per week"]}
        ).targets["meat_limit"] == 2
        assert WeeklyNutritionalTracker(
            {"preferences": ["meat limit 5"]}
        ).targets["meat_limit"] == 5

    def test_pescatarian_fish_not_counted(self):
        from models.session import MealCourse

        tracker = WeeklyNutritionalTracker({"diet": ["pescatarian"], "preferences": []})
        tracker.update_tracker(MealCourse("r1", "Grilled Salmon", "salmon, lemon", "d"))
        assert tracker.meat_meals_count == 0
        tracker.update_tracker(MealCourse("r2", "Beef Stew", "beef", "d"))
        assert tracker.meat_meals_count == 1

    def test_veg_tag_overrides_keyword_detection(self):
        from models.session import MealCourse

        tracker = WeeklyNutritionalTracker({"preferences": []})
        tracker.update_tracker(
            MealCourse("r1", "Beef-style Tofu", "tofu", "d"), tags=["vegetarian"]
        )
        assert tracker.meat_meals_count == 0


class TestCalorieBudget:
    def test_tracker_accumulates_enrichment_style_nutrition(self):
        from models.session import MealCourse

        tracker = WeeklyNutritionalTracker({"preferences": ["2000 calories target"]})
        tracker.update_tracker(
            MealCourse("r", "Bowl", "rice", "d"),
            nutrition_info={"kcal": 500.0, "protein_g": 30.0},
        )
        assert tracker.weekly_calories == 500.0
        assert tracker.weekly_protein == 30.0
        assert tracker.get_status()["remaining"]["calories"] == 14000.0 - 500.0

    def test_selection_keeps_week_within_budget(self):
        # 400 kcal veg vs 1200 kcal meat, no preference scorer. While the
        # 1200 kcal option exceeds the fair per-slot share it must lose
        # deterministically (first 14 slots); once the untouched budget makes
        # both fit, either is acceptable — but the week can never exceed the
        # target. Pre-M6, selection ignored calories entirely (uniform random
        # here would average 16800 kcal against a 14000 budget).
        entries, env = _plan(
            {"preferences": ["2000 calories target"], "diet": []},
            MixedActions(kcal_meat=1200.0, kcal_veg=400.0),
        )
        chosen = [e["recipe"]["recipe_id"] for e in entries]
        assert all(rid.startswith("veg-") for rid in chosen[:14])
        assert env.tracker.weekly_calories <= env.tracker.targets["calories"]

    def test_constraint_score_neutral_without_nutrition(self):
        tracker = WeeklyNutritionalTracker({"preferences": []})
        assert constraint_score(_veg(1, 0), tracker, 21) == 0.0

    def test_constraint_score_penalizes_overage(self):
        tracker = WeeklyNutritionalTracker({"preferences": ["2000 calories target"]})
        light = constraint_score(_veg(1, 0, kcal=400.0), tracker, 21)
        heavy = constraint_score(_meat(1, 0, kcal=1200.0), tracker, 21)
        assert light == 0.0
        assert heavy < 0.0


class TestHardConstraintFilter:
    def test_filter_drops_meat_once_limit_spent(self):
        from models.session import MealCourse

        tracker = WeeklyNutritionalTracker({"preferences": ["meat limit 1"]})
        pool = [_meat(1, 0), _veg(1, 0)]
        assert apply_hard_constraints(pool, tracker) == pool  # limit not hit yet

        tracker.update_tracker(MealCourse("r", "Beef Stew", "beef", "d"))
        filtered = apply_hard_constraints(pool, tracker)
        assert [c["recipe_id"] for c in filtered] == ["veg-1-0"]

    def test_filter_relaxes_when_pool_is_all_meat(self):
        from models.session import MealCourse

        tracker = WeeklyNutritionalTracker({"preferences": ["meat limit 1"]})
        tracker.update_tracker(MealCourse("r", "Beef Stew", "beef", "d"))
        pool = [_meat(1, 0), _meat(1, 1)]
        assert apply_hard_constraints(pool, tracker) == pool


class TestDeterministicReward:
    def test_step_reward_is_negative_penalty_no_llm(self):
        calc = RewardCalculator()
        tracker = WeeklyNutritionalTracker({"preferences": ["meat limit 1"]})
        assert calc.calculate_step_reward(_veg(1, 0), tracker, []) == 0.0

        from models.session import MealCourse

        tracker.update_tracker(MealCourse("a", "Beef Stew", "beef", "d"))
        tracker.update_tracker(MealCourse("b", "Pork Chops", "pork", "d"))
        # One over the limit -> -15
        assert calc.calculate_step_reward(_meat(1, 0), tracker, []) == -15.0
