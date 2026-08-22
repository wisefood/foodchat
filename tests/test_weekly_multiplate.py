"""
A weekly dinner can be a main and a salad, and survive a refinement.

The renderer has been plate-aware for a while — `foodchat.vue` builds one cell
per ENTRY grouped by slot, with a comment saying the producer did not exist yet.
This is that producer.

Weekly needs it for a case the structured path cannot serve. A FRESH multi-plate
request is routed to the structured path (the orchestrator checks
`delta.spec.plates` and sends it there), because RecipeWrangler plans N days
natively. But a multi-plate week that already EXISTS and is being refined comes
back through this walk — and before this, the walk had no notion of a plate, so
every refinement flattened the shape the member had asked for back to single
dishes.

What is tested here is the walk itself, through the real `step()`:

* an action is a whole MEAL, and the MDP still steps once per meal;
* the plan comes out as one row per plate, which is the shape everything
  downstream already reads;
* the tracker counts the whole table once — not the main alone, and not twice.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                                    # noqa: E402
from models.recipe import CandidateRecipe                                # noqa: E402
from services.weekly_planner.environment import WeeklyMealPlanEnv        # noqa: E402
from services.weekly_planner.planner import WeeklyPlanner               # noqa: E402
from services.weekly_planner.reward_logic import (                       # noqa: E402
    RewardCalculator,
    is_meat_candidate,
)


SPEC = PlanSpec(
    num_days=2, meals=("lunch", "dinner"), plates={"dinner": ("main", "side")},
)


def _c(rid, title, ingredients, kcal=300.0):
    return CandidateRecipe(
        recipe_id=rid, title=title, ingredients=ingredients, directions="cook",
        nutrition={"kcal": kcal, "protein_g": 10.0},
    )


class _Space:
    """A real `RecipeActionSpace` with the fetch replaced, so the composition,
    the action shape and the exclusion bookkeeping are all the live code."""

    def __init__(self, profile=None, spec=SPEC, per_role=3):
        from services.weekly_planner.action_adapter import RecipeActionSpace

        self.inner = RecipeActionSpace(
            profile or {"diet": [], "allergies": []}, spec=spec,
        )
        self.per_role = per_role
        self.days_fetched: list[int] = []
        self.excluded_at_fetch: list[list[str]] = []
        # Pre-seed the role cache per day so no HTTP call is made, using the
        # real cache the real code reads.
        self._spec = spec

    def _fill(self, day: int) -> None:
        self.days_fetched.append(day)
        self.excluded_at_fetch.append(list(self.inner._selected_ids))
        taken = set(self.inner._selected_ids)
        pools = {}
        for slot in self._spec.meals:
            for role in self._spec.roles_for(slot):
                pools[(slot, role)] = [
                    _c(f"d{day}-{slot}-{role}-{n}",
                       f"{role.title()} {slot} {day}{n}",
                       f"{role}stuff{n}")
                    for n in range(self.per_role)
                    if f"d{day}-{slot}-{role}-{n}" not in taken
                ]
        self.inner._role_cache[day] = pools
        # The single-dish path reads a different cache. Filled too, so a
        # single-plate spec exercises the untouched code with no network call.
        by_slot: dict = {}
        for (slot, _role), candidates in pools.items():
            by_slot.setdefault(slot, []).extend(candidates)
        self.inner._day_cache[day] = by_slot

    def get_candidate_actions(self, meal_type, current_state):
        day = current_state.get("day", 1)
        if day not in self.inner._role_cache:
            self._fill(day)
        return self.inner.get_candidate_actions(meal_type, current_state)

    def mark_selected(self, recipe_id):
        self.inner.mark_selected(recipe_id)


def _run(spec=SPEC, profile=None, space=None):
    space = space or _Space(profile, spec)
    env = WeeklyMealPlanEnv(
        user_profile=profile or {"diet": [], "preferences": []},
        action_space=space,
        reward_calculator=RewardCalculator(),
        spec=spec,
    )
    return WeeklyPlanner(env).generate_full_plan(user_query="a week"), env, space


# ── the action is a meal ────────────────────────────────────────────────────

class TestAnActionIsAWholeMeal:
    def test_a_two_plate_dinner_offers_compositions_not_dishes(self):
        space = _Space()
        actions = space.get_candidate_actions("dinner", {"day": 1})
        assert actions
        assert all(len(a["plates"]) == 2 for a in actions)
        assert all(
            [p["role"] for p in a["plates"]] == ["main", "side"] for a in actions
        )

    def test_a_single_plate_slot_still_offers_one_plate(self):
        space = _Space()
        actions = space.get_candidate_actions("lunch", {"day": 1})
        assert all(len(a["plates"]) == 1 for a in actions)

    def test_the_top_level_fields_are_the_main(self):
        """Every existing consumer — the preference scorer, the pinning check,
        the stored entry — reads these and must keep reading what it read."""
        action = _Space().get_candidate_actions("dinner", {"day": 1})[0]
        main = next(p for p in action["plates"] if p["role"] == "main")
        assert action["recipe_id"] == main["recipe_id"]
        assert action["recipe_title"] == main["recipe_title"]

    def test_the_nutrition_is_the_whole_meal(self):
        """Scoring a main-plus-side meal on the main alone lets every side
        through the calorie budget unmeasured."""
        action = _Space().get_candidate_actions("dinner", {"day": 1})[0]
        assert action["nutrition"]["kcal"] == pytest.approx(600.0)
        assert action["nutrition"]["complete"] is True

    def test_a_meal_missing_macros_says_so(self):
        space = _Space()
        space._fill(1)
        space.inner._role_cache[1][("dinner", "side")] = [
            CandidateRecipe("bare", "Bare side", "leaves", "serve"),
        ]
        action = space.inner.get_candidate_actions("dinner", {"day": 1})[0]
        assert action["nutrition"]["complete"] is False

    def test_meat_in_a_side_counts_against_the_meat_limit(self):
        """A limit that only inspects mains is a limit with a hole in it."""
        action = {
            "recipe_title": "Green salad",
            "recipe_ingredients": "lettuce, cucumber",
            "meal_ingredients": "lettuce, cucumber ; smoked bacon lardons",
        }
        assert is_meat_candidate(action) is True

    def test_a_dish_action_is_unaffected(self):
        assert is_meat_candidate(
            {"recipe_title": "Bean stew", "recipe_ingredients": "beans"},
        ) is False


# ── the walk ────────────────────────────────────────────────────────────────

class TestTheWalkProducesPlates:
    def test_one_row_per_plate(self):
        """2 days x (1-plate lunch + 2-plate dinner) = 6 rows over 4 meals."""
        plan, env, _ = _run()
        assert len(plan) == 6
        assert env.slots_filled == 4
        assert env.total_slots == 4

    def test_the_plates_of_a_meal_share_a_day_a_slot_and_a_meal_idx(self):
        """That is exactly what the renderer groups on."""
        plan, _, _ = _run()
        dinner = [r for r in plan if r["day"] == 1 and r["meal_type"] == "dinner"]
        assert len(dinner) == 2
        assert {r["meal_idx"] for r in dinner} == {1}
        assert [r["role"] for r in dinner] == ["main", "side"]

    def test_the_reward_belongs_to_the_meal_not_to_each_plate(self):
        """Repeating it would make a two-plate dinner look twice as well-scored
        to anything that sums the column."""
        plan, _, _ = _run()
        dinner = [r for r in plan if r["day"] == 1 and r["meal_type"] == "dinner"]
        assert dinner[1]["reward"] == 0.0

    def test_every_plate_is_excluded_from_later_days(self):
        """Marking the main alone would let the same salad appear as a side on
        Tuesday and as a main on Friday."""
        _, _, space = _run()
        # Day 1 is a one-plate lunch plus a two-plate dinner: three plates,
        # all of them spent by the time day 2 is fetched.
        assert len(space.excluded_at_fetch[1]) == 3

    def test_no_recipe_repeats_across_the_plan(self):
        plan, _, _ = _run()
        ids = [r["recipe"]["recipe_id"] for r in plan]
        assert len(ids) == len(set(ids))

    def test_the_tracker_counts_each_meal_once(self):
        """Once per meal, with the whole table's calories — not once per plate
        (which doubles the day) and not the main alone (which hides the side)."""
        plan, env, _ = _run()
        # 4 meals: two 1-plate lunches at 300, two 2-plate dinners at 600.
        assert env.tracker.get_status()["cumulative"]["calories"] == pytest.approx(1800.0)

    def test_the_calorie_scorer_never_runs_out_of_slots(self):
        """`slots_remaining` used to be `total_slots - len(env.plan)`, and
        `env.plan` now holds one row per PLATE — so it would reach zero and go
        negative halfway through a composed week."""
        plan, env, _ = _run()
        assert env.slots_filled <= env.total_slots

    def test_variety_sees_every_plate(self):
        """Counting the main alone would let the same side reappear all week
        without the variety penalty noticing."""
        import inspect

        from services.weekly_planner import planner as planner_module

        source = inspect.getsource(planner_module.WeeklyPlanner.generate_full_plan)
        assert 'chosen_recipe.get("plates")' in source


# ── nothing about the single-plate week changes ─────────────────────────────

class TestTheOrdinaryWeekIsUntouched:
    def test_no_spec_means_the_old_single_dish_path(self):
        from services.weekly_planner.action_adapter import RecipeActionSpace

        space = RecipeActionSpace({"diet": []})
        assert space.multi_plate is False

    def test_an_all_single_plate_spec_also_takes_the_old_path(self):
        """`multi_plate` is about PLATES, not about days or meals — a three-day
        plan with one dish per meal must not start paying for compositions."""
        from services.weekly_planner.action_adapter import RecipeActionSpace

        space = RecipeActionSpace({"diet": []}, spec=PlanSpec(num_days=3))
        assert space.multi_plate is False

    def test_a_dish_walk_still_produces_one_row_per_slot(self):
        spec = PlanSpec(num_days=2, meals=("lunch", "dinner"))
        plan, env, _ = _run(spec=spec, space=_Space(spec=spec))
        assert len(plan) == 4 == env.total_slots
        # Byte-for-byte the rows this walk always produced: no `role` key at
        # all, which the API model defaults to "main". A single-plate week must
        # not start carrying multi-plate bookkeeping it has no use for.
        assert all("role" not in r for r in plan)
        assert all(set(r) == {"day", "meal_idx", "meal_type", "recipe", "reward"}
                   for r in plan)

    def test_the_composition_is_llm_free(self):
        """This loop used to make one Groq call per committed slot — 21 per
        week — to grade a recipe already locked in, and it was removed for
        exactly that reason. Adding seven judge calls back would undo it."""
        import inspect

        from services.weekly_planner import action_adapter

        source = inspect.getsource(action_adapter.RecipeActionSpace._composed_actions)
        assert "meal_composer.judge" not in source
        assert "compose" in source


# ── the shape reaches the stored plan ──────────────────────────────────────

class TestItReachesWhatIsStored:
    def test_the_adapter_the_verifier_reads_keeps_the_roles(self):
        from services.weekly_plan_service import _as_meal_plan

        plan, _, _ = _run()
        adapted = _as_meal_plan(plan)
        dinner = next(
            m for m in adapted.day_plans[0].meals if m.meal_type == "dinner"
        )
        assert [p.role for p in dinner.plates] == ["main", "side"]

    def test_the_api_response_carries_the_role(self):
        """Pydantic's default is extra='ignore', so an unmodelled field is
        silently dropped and the UI would label the plates "Dinner 1" and
        "Dinner 2"."""
        from routers.foodchat_router import WeeklyMealPlanEntryResponse

        plan, _, _ = _run()
        row = next(r for r in plan if r.get("role") == "side")
        assert WeeklyMealPlanEntryResponse(**row).role == "side"

    def test_an_old_entry_without_a_role_reads_as_a_main(self):
        from routers.foodchat_router import WeeklyMealPlanEntryResponse

        legacy = {"day": 1, "meal_idx": 0, "meal_type": "breakfast",
                  "recipe": {"recipe_id": "r"}, "reward": 0.0}
        assert WeeklyMealPlanEntryResponse(**legacy).role == "main"
