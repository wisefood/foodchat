"""A recipe may come back — but only where, when, and as often as it may.

Before M9 the weekly planner excluded every committed recipe from every
later fetch, so a repeat was impossible at the source. That reads as 21
shopping lists rather than a week: nobody eats seven different breakfasts.

Breakfast now has a slot-scoped cooldown. Everything below exists to keep
that from becoming an excuse — the failure mode is not "too few repeats",
it is a thin candidate pool quietly producing a repetitive week that the
plan then describes as a feature. So every repeat carries the day it
repeats and the authority it repeated under, and a duplicate that carries
neither is reported as exactly that.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe  # noqa: E402
from services.weekly_planner import action_adapter  # noqa: E402
from services.weekly_planner.action_adapter import (  # noqa: E402
    MAX_APPEARANCES,
    REPEAT_MIN_GAP_DAYS,
    RecipeActionSpace,
)
from services.weekly_planner.environment import WeeklyMealPlanEnv  # noqa: E402
from services.weekly_planner.explainability import (  # noqa: E402
    REPEAT_KIND,
    build_weekly_explainability,
    repeat_facts,
    variety_metrics,
)
from services.weekly_planner.planner import (  # noqa: E402
    IngredientBasket,
    WeeklyPlanner,
    build_preference_scorer,
)
from services.weekly_planner.reward_logic import RewardCalculator  # noqa: E402

BREAKFASTS = [("b1", "Oats"), ("b2", "Shakshuka"), ("b3", "Toast"), ("b4", "Congee")]
LUNCHES = [(f"l{i}", f"Lunch{i}") for i in range(9)]
DINNERS = [(f"d{i}", f"Dinner{i}") for i in range(9)]


def _cand(pair, ingredients="carrot, spinach"):
    return CandidateRecipe(
        recipe_id=pair[0], title=pair[1], ingredients=ingredients, directions="cook"
    )


@pytest.fixture
def offline(monkeypatch):
    """The real action space, with only its network edges stubbed."""
    calls = {"exclusions": [], "pantry": []}

    def fake_pool(*, profile, allergens, diet, cuisines, exclude_recipe_ids,
                  limit_per_slot):
        excluded = set(exclude_recipe_ids or [])
        calls["exclusions"].append(sorted(excluded))
        return {
            "breakfast": [_cand(p) for p in BREAKFASTS if p[0] not in excluded],
            "lunch": [_cand(p) for p in LUNCHES if p[0] not in excluded],
            "dinner": [_cand(p) for p in DINNERS if p[0] not in excluded],
        }

    def fake_pantry(profile, pantry, **kwargs):
        calls["pantry"].append(list(pantry))
        return {}

    monkeypatch.setattr(action_adapter, "_fetch_candidate_pool", fake_pool)
    monkeypatch.setattr(action_adapter.CANDIDATES, "fetch_details", lambda ids: {})
    monkeypatch.setattr(
        action_adapter.CANDIDATES, "split_cuisines", lambda likes: ([], list(likes or []))
    )
    from services import pantry_service

    monkeypatch.setattr(pantry_service, "fetch_pantry_candidates", fake_pantry)
    return calls


def _space(offline, **profile):
    profile.setdefault("diet", [])
    profile.setdefault("preferences", [])
    profile.setdefault("plan_parameters", {})
    return RecipeActionSpace(profile, additional_diet=[], pantry=())


def _ids(space, day, meal_type):
    return [a["recipe_id"] for a in space.get_candidate_actions(meal_type, {"day": day})]


def _action(space, day, meal_type, recipe_id):
    return next(
        a for a in space.get_candidate_actions(meal_type, {"day": day})
        if a["recipe_id"] == recipe_id
    )


class TestTheCooldown:
    def test_a_served_breakfast_is_gone_the_next_day(self, offline):
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")

        assert "b1" not in _ids(space, 2, "breakfast")

    def test_it_comes_back_after_the_gap_carrying_the_day_it_repeats(self, offline):
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")

        action = _action(space, 1 + REPEAT_MIN_GAP_DAYS, "breakfast", "b1")

        assert action["repeat_of_day"] == 1

    def test_it_does_not_come_back_a_third_time(self, offline):
        """Two servings is a routine. More is the planner out of ideas.

        Days are fixed rather than derived from MAX_APPEARANCES: a fixture
        computed from the constant under test moves with it, and this
        assertion held even with the cap raised to 99.
        """
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")
        space.mark_committed("b1", 3, "breakfast")

        assert MAX_APPEARANCES == 2, "the fixed days below encode this"
        assert "b1" not in _ids(space, 5, "breakfast")

    def test_a_breakfast_never_reappears_as_lunch(self, offline):
        """A breakfast may come back as breakfast. Moving it is another claim."""
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")

        assert "b1" not in _ids(space, 4, "lunch")

    def test_lunch_and_dinner_keep_the_original_never_repeat_rule(self, offline):
        space = _space(offline)
        space.mark_committed("l0", 1, "lunch")
        space.mark_committed("d0", 1, "dinner")

        assert "l0" not in _ids(space, 5, "lunch")
        assert "d0" not in _ids(space, 5, "dinner")

    def test_a_pinned_or_downvoted_recipe_has_no_way_back(self, offline):
        """`mark_selected` still means never, cooldown or not."""
        space = _space(offline)
        space.mark_selected("b1")
        space.mark_committed("b1", 1, "breakfast")

        assert "b1" not in _ids(space, 4, "breakfast")

    def test_a_fresh_recipe_carries_no_repeat_label(self, offline):
        space = _space(offline)

        assert "repeat_of_day" not in _action(space, 1, "breakfast", "b1")


class TestTheCooldownCostsNoExtraRequests:
    def test_the_day_fetch_keeps_ids_a_slot_could_still_repeat(self, offline):
        """One fetch serves all three of a day's slots, so it excludes only
        what NO slot could use; the per-slot rule is applied afterwards. If
        this regressed to a per-slot fetch the cooldown would still work and
        the plan would silently cost three times the RecipeWrangler calls."""
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")
        space.mark_committed("l0", 1, "lunch")

        _ids(space, 1 + REPEAT_MIN_GAP_DAYS, "breakfast")
        excluded = offline["exclusions"][-1]

        assert "b1" not in excluded, "breakfast could legally repeat it"
        assert "l0" in excluded, "lunch never repeats"

    def test_one_pool_fetch_per_day_not_per_slot(self, offline):
        space = _space(offline)
        before = len(offline["exclusions"])
        for meal in ("breakfast", "lunch", "dinner"):
            _ids(space, 3, meal)

        assert len(offline["exclusions"]) - before == 1


class TestWhenTheSourceIgnoresTheExclusion:
    """`exclude_recipe_ids` is a request to RecipeWrangler, not a guarantee.

    If a committed, downvoted or pinned recipe comes back in the pool anyway,
    the per-slot rule is the only thing between it and the member's plate. So
    these assert against a source that returns everything, every time.
    """

    @pytest.fixture
    def stubborn(self, monkeypatch, offline):
        def fake_pool(*, profile, allergens, diet, cuisines, exclude_recipe_ids,
                      limit_per_slot):
            return {
                "breakfast": [_cand(p) for p in BREAKFASTS],
                "lunch": [_cand(p) for p in LUNCHES],
                "dinner": [_cand(p) for p in DINNERS],
            }

        monkeypatch.setattr(action_adapter, "_fetch_candidate_pool", fake_pool)
        return offline

    def test_a_downvoted_or_pinned_recipe_is_still_dropped(self, stubborn):
        space = _space(stubborn)
        space.mark_selected("b1")
        space.mark_committed("b1", 1, "breakfast")

        assert "b1" not in _ids(space, 4, "breakfast")

    def test_a_recipe_inside_its_cooldown_is_still_dropped(self, stubborn):
        space = _space(stubborn)
        space.mark_committed("b1", 1, "breakfast")

        assert "b1" not in _ids(space, 2, "breakfast")

    def test_a_third_serving_is_still_dropped(self, stubborn):
        space = _space(stubborn)
        space.mark_committed("b1", 1, "breakfast")
        space.mark_committed("b1", 3, "breakfast")

        assert "b1" not in _ids(space, 5, "breakfast")

    def test_a_lunch_is_still_dropped(self, stubborn):
        space = _space(stubborn)
        space.mark_committed("l0", 1, "lunch")

        assert "l0" not in _ids(space, 5, "lunch")


class TestARepeatOfferedIsRecordedEvenIfNotTaken:
    """"The week repeated nothing" and "nothing was offered to repeat" look
    identical from a stored plan, and they have opposite fixes — one in the
    scorer, one at RecipeWrangler. So the offer is recorded either way."""

    def test_the_offer_is_recorded_with_what_could_have_repeated(self, offline):
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")
        _ids(space, 3, "breakfast")

        assert space.selection_events == [{
            "type": "repeat_offered", "day": 3, "meal_type": "breakfast",
            "count": 1, "recipe_ids": ["b1"],
        }]

    def test_a_slot_with_nothing_to_repeat_records_nothing(self, offline):
        space = _space(offline)
        _ids(space, 1, "breakfast")

        assert space.selection_events == []

    def test_a_slot_inside_the_cooldown_records_nothing(self, offline):
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")
        _ids(space, 2, "breakfast")

        assert space.selection_events == []

    def test_it_is_recorded_once_per_slot(self, offline):
        space = _space(offline)
        space.mark_committed("b1", 1, "breakfast")
        for _ in range(3):
            _ids(space, 3, "breakfast")

        assert len(space.selection_events) == 1


class TestWhoAskedForIt:
    """The label that keeps a thin pool from being sold as a preference."""

    def test_a_starred_recipe_repeats_as_a_member_request(self, offline):
        space = _space(offline, favorite_recipe_ids=["b1"])
        space.mark_committed("b1", 1, "breakfast")

        action = _action(space, 3, "breakfast", "b1")

        assert action["repeat_source"] == action_adapter.REPEAT_MEMBER_REQUEST

    def test_anything_else_repeats_on_the_plans_own_authority(self, offline):
        space = _space(offline, favorite_recipe_ids=["b4"])
        space.mark_committed("b1", 1, "breakfast")

        action = _action(space, 3, "breakfast", "b1")

        assert action["repeat_source"] == action_adapter.REPEAT_PLAN


class TestTheScorerLetsARepeatCompete:
    PROFILE = {"diet": [], "preferences": [], "plan_parameters": {}}
    FRESH = {"recipe_id": "x", "recipe_title": "Lentil Bowl",
             "recipe_ingredients": "lentils, spinach"}
    REPEAT = {"recipe_id": "b1", "recipe_title": "Lentil Bowl",
              "recipe_ingredients": "lentils, spinach", "repeat_of_day": 1,
              "repeat_source": "plan"}

    def test_a_repeat_is_not_charged_for_resembling_itself(self):
        """The exact-title penalty is −2 per token and scales with how many
        words the recipe happens to be called — large enough that the cooldown
        could never be reached while it applied."""
        scorer = build_preference_scorer(self.PROFILE)
        chosen = ["Lentil Bowl"]

        assert scorer(self.REPEAT, chosen) > scorer(self.FRESH, chosen)

    def test_it_competes_on_equal_terms_with_a_new_dish(self):
        """`_REPEAT_PENALTY` is zero, and this is why it has to be.

        It shipped at 1.0 — "enough that a repeat does not win a coin flip".
        But a member with no favourites and no liked *ingredients* leaves
        almost every candidate scoring exactly 0.0, so any positive penalty is
        not a tiebreak, it is a veto: the repeat loses to every fresh
        candidate that exists and the cooldown can never fire. Observed on a
        live week: seven distinct breakfasts, `planned_repeats: 0`.

        Whether a repeat is legal is the cooldown's job and how often is the
        cap's. This scorer only has to let it into the tie pool.
        """
        scorer = build_preference_scorer(self.PROFILE)

        assert scorer(self.REPEAT, []) == scorer(self.FRESH, [])

    def test_a_bare_profile_leaves_almost_everything_tied(self):
        """The premise of the test above, asserted rather than assumed: two
        liked cuisines are removed from `food_likes` before the scorer sees
        them, so they boost nothing."""
        profile = dict(self.PROFILE, food_likes=["italian", "asian"])
        scorer = build_preference_scorer(profile)

        assert scorer(self.FRESH, []) == 0.0

    def test_it_still_pays_for_resembling_a_different_dish(self):
        scorer = build_preference_scorer(self.PROFILE)

        assert scorer(self.REPEAT, ["Lentil Salad"]) < scorer(self.REPEAT, [])

    def test_a_repeat_earns_no_reuse_bonus_from_its_own_ingredients(self):
        """It shares everything with its earlier serving, so at `strict` the
        reuse axis would have made repeating the cheapest way to score —
        observed: a strict week repeated at the first legal opportunity."""
        strict = dict(self.PROFILE, plan_parameters={"food_waste": "strict"})
        scorer = build_preference_scorer(strict)
        basket = IngredientBasket()
        basket.add("lentils, spinach", 1)

        assert scorer(self.REPEAT, [], basket, 3) < scorer(self.FRESH, [], basket, 3)

    def test_a_starred_repeat_still_wins_its_slot(self):
        """The penalty must not overrule what the member actually asked for."""
        scorer = build_preference_scorer(dict(self.PROFILE, favorite_recipe_ids=["b1"]))

        assert scorer(self.REPEAT, []) > scorer(self.FRESH, [])


class TestItIsWrittenDown:
    @staticmethod
    def _entries(pairs):
        """pairs: [(day, meal_type, recipe_id, title, repeat_of_day, source)]"""
        out = []
        for day, meal, rid, title, of_day, source in pairs:
            recipe = {"recipe_id": rid, "recipe_title": title,
                      "recipe_ingredients": "lentils, spinach"}
            if of_day:
                recipe["repeat_of_day"] = of_day
                recipe["repeat_source"] = source
            out.append({"day": day, "meal_idx": 0, "meal_type": meal,
                        "recipe": recipe, "reward": 0.0})
        return out

    WEEK = [
        (1, "breakfast", "b1", "Oats", None, None),
        (2, "breakfast", "b2", "Toast", None, None),
        (3, "breakfast", "b1", "Oats", 1, "member_request"),
        (5, "breakfast", "b2", "Toast", 2, "plan"),
    ]

    def test_repeats_are_counted_and_split_by_authority(self):
        facts = repeat_facts(self._entries(self.WEEK))

        assert facts["count"] == 2
        assert facts["by_source"] == {"member_request": 1, "plan": 1}
        assert facts["min_gap_days"] == 2
        assert facts["max_appearances"] == 2
        assert facts["unexplained"] == 0

    def test_the_two_kinds_of_repeat_get_different_chips(self):
        entries = self._entries(self.WEEK)
        build_weekly_explainability(entries, {"preferences": []})

        def chip(index):
            return next(
                r for r in entries[index]["recipe"]["match_reasons"]
                if r["kind"] == REPEAT_KIND
            )

        assert chip(2)["source"] == "member_request"
        assert "favorite" in chip(2)["label"]
        assert chip(3)["source"] == "plan"
        assert "the same breakfast as Tuesday" == chip(3)["label"]

    def test_a_fresh_meal_gets_no_repeat_chip(self):
        entries = self._entries(self.WEEK)
        build_weekly_explainability(entries, {"preferences": []})

        kinds = [r["kind"] for r in entries[0]["recipe"]["match_reasons"]]
        assert REPEAT_KIND not in kinds

    def test_variety_reports_planned_repeats_apart_from_monotony(self):
        metrics = variety_metrics(self._entries(self.WEEK))

        assert metrics["distinct_recipes"] == 2
        assert metrics["planned_repeats"] == 2
        assert metrics["unexplained_repeats"] == 0
        assert "planned repeat" in metrics["reasoning"]

    def test_a_duplicate_nobody_sanctioned_is_not_dressed_up_as_one(self):
        """A pinned dish or a slot edit can put a duplicate on the plate
        without passing the cooldown. Folding it into the planned-repeat
        count is how a thin pool gets presented as a preference."""
        week = list(self.WEEK)
        week[3] = (5, "breakfast", "b2", "Toast", None, None)
        metrics = variety_metrics(self._entries(week))

        assert metrics["planned_repeats"] == 1
        assert metrics["unexplained_repeats"] == 1
        assert "planned repeat" not in metrics["reasoning"]

    def test_the_ledger_row_names_both_authorities_and_the_measured_gap(self):
        result = build_weekly_explainability(self._entries(self.WEEK), {"preferences": []})
        row = next(
            r for r in result["constraints_applied"] if "repeat meals" in r["constraint"]
        )

        assert row["status"] == "satisfied"
        assert row["source"] == "your favourites and the plan"
        assert "1 you starred" in row["detail"]
        assert "1 the plan's own" in row["detail"]
        assert "2 day(s) apart" in row["detail"]

    def test_a_repeat_outside_the_policy_is_reported_as_violated(self):
        """Measured against the entries, not asserted from the constants."""
        week = list(self.WEEK)
        week[2] = (2, "breakfast", "b1", "Oats", 1, "plan")  # adjacent days
        result = build_weekly_explainability(self._entries(week), {"preferences": []})
        row = next(
            r for r in result["constraints_applied"] if "repeat meals" in r["constraint"]
        )

        assert row["status"] == "violated"

    def test_an_unexplained_duplicate_gets_its_own_violated_row(self):
        week = list(self.WEEK)
        week[3] = (5, "breakfast", "b2", "Toast", None, None)
        result = build_weekly_explainability(self._entries(week), {"preferences": []})
        row = next(
            r for r in result["constraints_applied"]
            if "recorded reason" in r["constraint"]
        )

        assert row["status"] == "violated" and row["source"] == "the plan"

    def test_the_justification_the_member_reads_says_it_too(self):
        result = build_weekly_explainability(self._entries(self.WEEK), {"preferences": []})

        assert "2 meal(s) repeat earlier in the week" in result["reasoning"]
        assert "1 you'd starred" in result["reasoning"]
        assert "1 the plan's own choice" in result["reasoning"]

    def test_a_repeat_is_not_also_billed_as_ingredient_reuse(self):
        """A meal that IS an earlier meal shares every ingredient with it.
        "the same breakfast as Monday" and "also uses Monday's rolled oats"
        on one card say the same thing twice, and counting it would inflate
        the reuse figure by the repeat count."""
        from services.weekly_planner.explainability import (
            SHARED_INGREDIENT_KIND,
            annotate_shared_ingredients,
            shared_ingredient_facts,
        )

        entries = self._entries(self.WEEK)
        for entry in entries:
            entry["recipe"]["recipe_ingredients"] = "rolled oats, blueberries"

        assert shared_ingredient_facts(entries)["entries"].keys() == {1}
        annotate_shared_ingredients(entries)
        kinds = [r["kind"] for r in (entries[2]["recipe"].get("match_reasons") or [])]
        assert SHARED_INGREDIENT_KIND not in kinds

    def test_a_flag_left_behind_by_an_edit_is_not_a_repeat(self):
        """`edit_service` replaces a slot with a freshly built recipe dict,
        clearing the flag on the slot it edits and leaving the other serving
        still claiming to repeat a day that no longer has it. Checked against
        the plan, not believed."""
        week = list(self.WEEK)
        week[0] = (1, "breakfast", "b9", "Something Else", None, None)
        facts = repeat_facts(self._entries(week))

        assert facts["count"] == 1, "only the Toast repeat survives"
        assert facts["by_source"] == {"plan": 1}

    def test_a_week_with_no_repeats_says_nothing_about_them(self):
        entries = self._entries(self.WEEK[:2])
        result = build_weekly_explainability(entries, {"preferences": []})

        assert "repeat" not in result["reasoning"]
        assert not [
            r for r in result["constraints_applied"] if "repeat" in r["constraint"]
        ]


class TestAcrossAWholeWeek:
    """The end-to-end claim, through the real action space."""

    @staticmethod
    def _week(offline, **profile):
        profile.setdefault("diet", [])
        profile.setdefault("preferences", [])
        profile.setdefault("plan_parameters", {})
        space = RecipeActionSpace(profile, additional_diet=[], pantry=())
        env = WeeklyMealPlanEnv(profile, space, RewardCalculator())
        entries = WeeklyPlanner(env).generate_full_plan(
            user_query="a week", scorer=build_preference_scorer(profile),
        )
        assert len(entries) == 21
        return entries, env

    def test_lunch_and_dinner_are_still_all_distinct(self, offline):
        entries, _ = self._week(offline)
        for meal in ("lunch", "dinner"):
            ids = [e["recipe"]["recipe_id"] for e in entries if e["meal_type"] == meal]
            assert len(ids) == len(set(ids)), meal

    def test_breakfast_repeats_stay_spaced_and_capped(self, offline):
        entries, _ = self._week(offline)
        days: dict = {}
        for entry in entries:
            if entry["meal_type"] == "breakfast":
                days.setdefault(entry["recipe"]["recipe_id"], []).append(entry["day"])
        for recipe_id, served in days.items():
            assert len(served) <= MAX_APPEARANCES, recipe_id
            assert all(
                b - a >= REPEAT_MIN_GAP_DAYS for a, b in zip(sorted(served), sorted(served)[1:])
            ), (recipe_id, served)

    def test_a_thin_pool_repeats_rather_than_failing_the_plan(self, offline):
        """Four breakfasts across seven days was an unfillable week before the
        cooldown existed — `PlanGenerationError` on day 5."""
        entries, _ = self._week(offline)

        assert len([e for e in entries if e["meal_type"] == "breakfast"]) == 7

    def test_every_repeat_that_happened_was_recorded_when_it_happened(self, offline):
        entries, env = self._week(offline)
        logged = {
            (e["day"], e["recipe_id"])
            for e in env.selection_events if e["type"] == "repeat_allowed"
        }
        served = {
            (e["day"], e["recipe"]["recipe_id"])
            for e in entries if e["recipe"].get("repeat_of_day")
        }

        assert logged == served
