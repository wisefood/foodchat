"""The sourcing half of cross-day ingredient reuse.

The scoring half shipped first and can only reorder the pool it is handed.
A day's pool was fetched knowing nothing about what the week had already
bought, so reuse happened when a shared ingredient turned up by luck —
scoring cannot put the second dill recipe in the pool, only prefer it once
it is there.

This is the other half: before each new day is fetched, the planner offers
the action space the ingredients still worth another meal, and the action
space decides — on the member's food-waste setting, because the search
costs real round-trips — whether to go looking for them. Both outcomes are
recorded, so "why did this week reuse nothing" is answerable from the
stored plan.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe  # noqa: E402
from services.weekly_planner import action_adapter  # noqa: E402
from services.weekly_planner.action_adapter import (  # noqa: E402
    DERIVED_PANTRY_ITEMS,
    RecipeActionSpace,
)
from services.weekly_planner.explainability import (  # noqa: E402
    build_weekly_explainability,
)
from services.weekly_planner.planner import (  # noqa: E402
    _MAX_REWARDED_USES,
    _MIN_REUSE_GAP_DAYS,
    IngredientBasket,
)


class TestWhatIsWorthAnotherMeal:
    """`reusable_items` is what the search is spent on, so it answers to the
    same two rules the reuse bonus scores by. Sourcing something the scorer
    then penalises would buy latency and a worse week."""

    def test_it_names_the_whole_ingredient_not_the_words_it_splits_into(self):
        """A token is fine for scoring and useless as a search term: nobody
        sells "self", and nobody would recognise it on a card either."""
        basket = IngredientBasket()
        basket.add("self raising flour, 3 cups broccoli florets", 1)

        assert basket.reusable_items(1 + _MIN_REUSE_GAP_DAYS) == ["broccoli florets"]

    def test_nothing_is_offered_before_the_gap_has_passed(self):
        basket = IngredientBasket()
        basket.add("cabbage, walnuts", 1)

        assert basket.reusable_items(1 + _MIN_REUSE_GAP_DAYS - 1) == []

    def test_an_ingredient_that_has_had_its_run_is_not_offered_again(self):
        basket = IngredientBasket()
        for use in range(_MAX_REWARDED_USES):
            basket.add("cabbage, walnuts", 1 + use * _MIN_REUSE_GAP_DAYS)

        assert basket.reusable_items(7) == []

    def test_staples_are_never_worth_a_round_trip(self):
        basket = IngredientBasket()
        basket.add("olive oil, salt, plain flour", 1)

        assert basket.reusable_items(5) == []

    def test_the_offer_is_capped(self):
        basket = IngredientBasket()
        basket.add("cabbage, walnuts, avocado, capsicum, courgette, radish", 1)

        assert len(basket.reusable_items(5, limit=2)) == 2

    def test_the_members_own_items_are_left_to_their_own_search(self):
        basket = IngredientBasket()
        basket.add("cabbage, walnuts", 1)

        assert basket.reusable_items(5, exclude_stems={"cabbage"}) == ["walnuts"]

    def test_an_empty_week_offers_nothing(self):
        assert IngredientBasket().reusable_items(1) == []


BREAKFASTS = [
    ("b1", "Oats", "rolled oats, blueberries"),
    ("b2", "Toast", "avocado, sourdough"),
    ("b3", "Congee", "jasmine rice, spring onion"),
    ("b4", "Muesli", "rolled oats, grated apple"),
]
LUNCHES = [(f"l{i}", f"Lunch{i}", "cabbage, walnuts, capsicum") for i in range(9)]
DINNERS = [(f"d{i}", f"Dinner{i}", "lentils, courgette, radish") for i in range(9)]


@pytest.fixture
def offline(monkeypatch):
    calls = {"pantry": [], "merges": []}

    def fake_pool(*, profile, allergens, diet, cuisines, exclude_recipe_ids,
                  limit_per_slot):
        excluded = set(exclude_recipe_ids or [])
        make = lambda rows: [  # noqa: E731
            CandidateRecipe(recipe_id=r[0], title=r[1], ingredients=r[2], directions="d")
            for r in rows if r[0] not in excluded
        ]
        return {"breakfast": make(BREAKFASTS), "lunch": make(LUNCHES),
                "dinner": make(DINNERS)}

    def fake_pantry(profile, pantry, **kwargs):
        calls["pantry"].append(list(pantry))
        return {}

    def fake_merge(base, pantry_pools, pantry, limit_per_slot):
        calls["merges"].append(list(pantry))
        return base

    from services import pantry_service

    monkeypatch.setattr(action_adapter, "_fetch_candidate_pool", fake_pool)
    monkeypatch.setattr(action_adapter.CANDIDATES, "fetch_details", lambda ids: {})
    monkeypatch.setattr(
        action_adapter.CANDIDATES, "split_cuisines", lambda likes: ([], list(likes or []))
    )
    monkeypatch.setattr(pantry_service, "fetch_pantry_candidates", fake_pantry)
    monkeypatch.setattr(pantry_service, "merge_pantry_pool", fake_merge)
    return calls


def _space(waste="strict", pantry=(), **profile):
    profile.setdefault("diet", [])
    profile.setdefault("preferences", [])
    profile["plan_parameters"] = {"food_waste": waste} if waste else {}
    return RecipeActionSpace(profile, additional_diet=[], pantry=pantry)


def _fetch(space, day, items=("cabbage", "walnuts")):
    space.offer_derived_pantry(list(items), day)
    return space.get_candidate_actions("lunch", {"day": day})


class TestWhenTheSearchIsSpent:
    def test_strict_goes_looking_for_what_the_week_already_buys(self, offline):
        _fetch(_space("strict"), 3)

        assert offline["pantry"] == [["cabbage", "walnuts"]]

    @pytest.mark.parametrize("waste", ["off", "reuse", None])
    def test_anything_below_strict_does_not(self, offline, waste):
        """The search is one request per ingredient per day, paid by every
        member on a plan. It belongs behind the setting that asks for it."""
        _fetch(_space(waste), 3)

        assert offline["pantry"] == []

    def test_the_offer_is_capped_at_what_the_latency_budget_allows(self, offline):
        _fetch(_space("strict"), 3, items=[f"item{i}" for i in range(10)])

        assert len(offline["pantry"][0]) == DERIVED_PANTRY_ITEMS

    def test_an_ingredient_the_member_named_is_not_searched_for_twice(self, offline):
        space = _space("strict", pantry=("cabbage",))
        _fetch(space, 3, items=("cabbage", "walnuts"))

        derived, stated = offline["pantry"]
        assert derived == ["walnuts"], "the member's cabbage has its own fan-out"
        assert stated == ["cabbage"]

    def test_the_members_own_pantry_still_ranks_the_pool_last(self, offline):
        """Both merges sort coverage-first, so whichever runs last decides the
        top of the day's pool. What the member told us they have must outrank
        what the plan inferred."""
        space = _space("strict", pantry=("aubergine",))
        _fetch(space, 3, items=("cabbage",))

        assert offline["merges"] == [["cabbage"], ["aubergine"]]

    def test_nothing_is_searched_for_before_anything_is_reusable(self, offline):
        space = _space("strict")
        space.get_candidate_actions("lunch", {"day": 1})

        assert offline["pantry"] == []


class TestTheDecisionIsRecorded:
    def test_a_day_that_searched_says_what_it_searched_for(self, offline):
        space = _space("strict")
        _fetch(space, 3)

        assert space.selection_events == [
            {"type": "derived_pantry_sourced", "day": 3, "items": ["cabbage", "walnuts"]}
        ]

    def test_a_setting_that_skipped_it_says_so_too(self, offline):
        """Otherwise "this week reused nothing" is indistinguishable from
        "this week found nothing to reuse"."""
        space = _space("off")
        _fetch(space, 3)

        assert space.selection_events == [{
            "type": "derived_pantry_skipped", "day": 3, "waste_mode": "off",
            "items": ["cabbage", "walnuts"],
        }]

    def test_the_skip_is_noted_once_not_once_a_day(self, offline):
        space = _space("off")
        for day in (3, 4, 5, 6):
            _fetch(space, day)

        assert len(space.selection_events) == 1

    def test_the_ledger_reports_the_days_that_searched(self, offline):
        events = [
            {"type": "derived_pantry_sourced", "day": 3, "items": ["cabbage"]},
            {"type": "derived_pantry_sourced", "day": 5, "items": ["cabbage", "walnuts"]},
        ]
        result = build_weekly_explainability(
            [], {"preferences": []}, selection_events=events,
        )
        row = next(
            r for r in result["constraints_applied"]
            if "already buys" in r["constraint"]
        )

        assert row["status"] == "satisfied"
        assert row["source"] == "food-waste setting"
        assert "2 day(s) also searched for cabbage, walnuts" in row["detail"]

    def test_searching_is_never_reported_as_reuse(self, offline):
        """Whether any of it was chosen is a different measurement, taken over
        the finished week by `shared_ingredient_facts`. A search that found
        nothing usable must not read as a saving."""
        events = [{"type": "derived_pantry_sourced", "day": 3, "items": ["cabbage"]}]
        result = build_weekly_explainability(
            [], {"preferences": []}, selection_events=events,
        )
        row = next(
            r for r in result["constraints_applied"]
            if "already buys" in r["constraint"]
        )

        assert "measured separately" in row["detail"]

    def test_a_skipped_week_gets_no_ledger_row_at_all(self, offline):
        """Nothing was applied, so there is no constraint to report — the
        event carries that trace instead."""
        events = [{"type": "derived_pantry_skipped", "day": 3, "waste_mode": "off",
                   "items": ["cabbage"]}]
        result = build_weekly_explainability(
            [], {"preferences": []}, selection_events=events,
        )

        assert not [
            r for r in result["constraints_applied"] if "already buys" in r["constraint"]
        ]


class TestThePlannerOffersWhatItHasBought:
    def test_the_offer_is_made_once_per_day_with_that_days_basket(self, offline):
        """Duck-typed on purpose, so an action space without the hook (the
        fakes in these tests, and the one the edit path builds) is untouched."""
        from services.weekly_planner.environment import WeeklyMealPlanEnv
        from services.weekly_planner.planner import WeeklyPlanner, build_preference_scorer
        from services.weekly_planner.reward_logic import RewardCalculator

        offers = []

        class Spy(RecipeActionSpace):
            def offer_derived_pantry(self, items, day):
                offers.append((day, list(items)))
                super().offer_derived_pantry(items, day)

        profile = {"diet": [], "preferences": [], "plan_parameters": {"food_waste": "strict"}}
        space = Spy(profile, additional_diet=[], pantry=())
        env = WeeklyMealPlanEnv(profile, space, RewardCalculator())
        WeeklyPlanner(env).generate_full_plan(
            user_query="a week", scorer=build_preference_scorer(profile),
        )

        assert [day for day, _ in offers] == [1, 2, 3, 4, 5, 6, 7]
        assert offers[0][1] == [], "nothing has been bought before day 1"
        # Some day, not a particular one. Whether Sunday still has anything
        # left to reuse depends on which recipes happened to land where — by
        # then every ingredient in this fixture may have had its two meals,
        # which is the cap doing its job, not the offer failing.
        assert any(items for _day, items in offers), "nothing was ever offered"
