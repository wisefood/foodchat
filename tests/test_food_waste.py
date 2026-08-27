"""The food-waste setting, and the difference it must make.

Food waste is a dimension of the *plan*: a week where Monday's leftover half
cabbage reappears in Wednesday's dinner wastes less than one where every meal
opens a new set of ingredients. The member controls it from the plan ribbon —
off / reuse / strict — and the weekly planner honours it through the scorer,
which is the only place it can: weekly selection runs without an LLM, so a
preference that never becomes a number there does not exist.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from services import plan_parameters  # noqa: E402
from services.weekly_planner.planner import (  # noqa: E402
    IngredientBasket,
    build_preference_scorer,
    perishable_tokens,
)


class TestPerishableTokens:
    def test_staples_do_not_count_as_reuse(self):
        """Two meals sharing olive oil have not saved a wilting vegetable."""
        tokens = perishable_tokens("2 tbsp olive oil, salt, pepper, 1 aubergine")

        assert "aubergine" in tokens
        assert {"olive", "salt", "pepper"} & tokens == set()

    def test_measurement_noise_is_not_an_ingredient(self):
        tokens = perishable_tokens("2 cups chopped spinach, finely grated carrot")

        assert "spinach" in tokens and "carrot" in tokens
        assert {"cups", "chopped", "finely", "grated"} & tokens == set()

    @pytest.mark.parametrize("junk", ["", None, "1 2 3", "salt"])
    def test_nothing_perishable_is_empty_not_an_error(self, junk):
        assert perishable_tokens(junk) == set()


class TestWasteScoring:
    CABBAGE = {"recipe_id": "a", "recipe_title": "Braised dish",
               "recipe_ingredients": "half a cabbage, carrots, cream"}
    UNRELATED = {"recipe_id": "b", "recipe_title": "Different dish",
                 "recipe_ingredients": "quinoa, pomegranate, feta"}

    @staticmethod
    def _profile(waste):
        return {"plan_parameters": {"food_waste": waste}}

    def test_off_means_overlap_is_worth_nothing(self):
        scorer = build_preference_scorer(self._profile("off"))
        basket = frozenset({"cabbage", "carrots"})

        assert scorer(self.CABBAGE, [], basket) == scorer(self.UNRELATED, [], basket)

    def test_reuse_prefers_the_meal_that_shares_the_basket(self):
        scorer = build_preference_scorer(self._profile("reuse"))
        basket = frozenset({"cabbage", "carrots"})

        assert scorer(self.CABBAGE, [], basket) > scorer(self.UNRELATED, [], basket)

    def test_strict_outweighs_reuse(self):
        basket = frozenset({"cabbage", "carrots"})
        reuse = build_preference_scorer(self._profile("reuse"))(self.CABBAGE, [], basket)
        strict = build_preference_scorer(self._profile("strict"))(self.CABBAGE, [], basket)

        assert strict > reuse

    def test_reuse_does_not_outrank_a_favourite(self):
        """The weights are ordered on purpose: reuse nudges, favourites win."""
        profile = {**self._profile("reuse"), "favorite_recipe_ids": ["b"]}
        scorer = build_preference_scorer(profile)
        basket = frozenset({"cabbage", "carrots", "cream"})

        assert scorer(self.UNRELATED, [], basket) > scorer(self.CABBAGE, [], basket)

    def test_overlap_bonus_is_capped(self):
        """One busy ingredient list must not ride to the top on volume."""
        scorer = build_preference_scorer(self._profile("strict"))
        many = {"recipe_id": "c", "recipe_title": "Everything stew",
                "recipe_ingredients": "cabbage carrots leeks parsnips swede turnips kale beets"}
        basket = frozenset(
            {"cabbage", "carrots", "leeks", "parsnips", "swede", "turnips", "kale", "beets"}
        )

        assert scorer(many, [], basket) <= 1.6 * 4

    def test_an_empty_basket_scores_no_one(self):
        """Day 1 has no reuse to reward — nothing chosen yet."""
        scorer = build_preference_scorer(self._profile("strict"))

        assert scorer(self.CABBAGE, [], frozenset()) == 0.0

    def test_legacy_two_argument_call_still_works(self):
        """Scorers predate the waste axis; old call sites must not break."""
        scorer = build_preference_scorer(self._profile("off"))

        assert scorer(self.CABBAGE, []) == 0.0


class TestIngredientBasket:
    def test_it_remembers_every_day_an_ingredient_lands_on(self):
        basket = IngredientBasket()
        basket.add("half a cabbage, carrots", day=1)
        basket.add("cabbage soup", day=3)

        assert basket.days_used("cabbage") == [1, 3]
        assert basket.days_used("carrots") == [1]

    def test_an_ingredient_never_eaten_has_no_history(self):
        assert IngredientBasket().days_used("cabbage") == []

    def test_tokens_is_the_flat_basket_the_old_scorers_expect(self):
        """Unstemmed: a pre-M8 scorer intersects it with its own raw tokens."""
        basket = IngredientBasket()
        basket.add("cabbage, carrots, olive oil, salt", day=1)

        assert basket.tokens() == frozenset({"cabbage", "carrots"})

    def test_singular_and_plural_are_the_same_ingredient(self):
        """Observed in a real plan: "tomatoes" and "tomato" were stored apart,
        so a member was served tomatoes on all seven days unpenalised."""
        basket = IngredientBasket()
        basket.add("chopped tomatoes", day=1)
        basket.add("one tomato", day=2)

        assert basket.days_used("tomato") == [1, 2]
        assert basket.days_used("tomatoes") == [1, 2]

    def test_it_finds_a_pantry_item_by_the_pantry_matcher_rules(self):
        basket = IngredientBasket()
        basket.add("2 tomatoes, olive oil", day=1)
        basket.add("lentils and spinach", day=2)
        basket.add("tomato passata", day=4)

        assert basket.days_matching("tomato") == [1, 4]
        assert basket.days_matching("cabbage") == []


class TestReuseSpacing:
    """When an ingredient comes back matters as much as whether it does.

    Two days later is reuse — the cabbage was bought once and cooked twice.
    The same evening, or the next day, is just cabbage again, and the flat
    basket this replaces could not tell the difference: it rewarded both.

    Nothing here models a shelf life. We do not record purchase dates or
    expiry, so no gap is ever "too old to count" — the spacing is about the
    week being worth eating.
    """

    CABBAGE = TestWasteScoring.CABBAGE
    UNRELATED = TestWasteScoring.UNRELATED

    @staticmethod
    def _basket(*days):
        basket = IngredientBasket()
        for day in days:
            basket.add("half a cabbage, carrots, cream", day)
        return basket

    @staticmethod
    def _scorer(waste="strict", **profile):
        return build_preference_scorer({"plan_parameters": {"food_waste": waste}, **profile})

    def test_a_two_day_gap_is_reuse_and_scores_positively(self):
        scorer = self._scorer()

        assert scorer(self.CABBAGE, [], self._basket(1), 3) > scorer(
            self.UNRELATED, [], self._basket(1), 3
        )

    def test_the_same_day_is_monotony_not_reuse(self):
        """Cabbage at lunch and again at dinner saves nothing worth eating."""
        scorer = self._scorer()

        assert scorer(self.CABBAGE, [], self._basket(2), 2) < scorer(
            self.UNRELATED, [], self._basket(2), 2
        )

    def test_the_next_day_is_still_repetition(self):
        """A smaller penalty than the same day, but a penalty."""
        scorer = self._scorer()

        assert -3.0 < scorer(self.CABBAGE, [], self._basket(2), 3) < 0
        assert scorer(self.CABBAGE, [], self._basket(2), 3) > scorer(
            self.CABBAGE, [], self._basket(2), 2
        )

    def test_a_wide_gap_is_still_reuse_because_no_shelf_life_is_modelled(self):
        """Nothing records when the cabbage was bought or when it goes off.

        A five-day gap is reuse for the same reason a two-day gap is: one
        cabbage on the shopping list instead of two. Treating it as expired
        would be a claim the state cannot back.
        """
        scorer = self._scorer()

        assert scorer(self.CABBAGE, [], self._basket(1), 6) == scorer(
            self.CABBAGE, [], self._basket(1), 3
        )
        assert scorer(self.CABBAGE, [], self._basket(1), 6) > scorer(
            self.UNRELATED, [], self._basket(1), 6
        )

    def test_a_third_appearance_stops_being_reuse(self):
        """Buy once, cook twice. Three times is just the same week of food."""
        scorer = self._scorer()
        spent = self._basket(1, 3)

        assert scorer(self.CABBAGE, [], spent, 5) < scorer(self.UNRELATED, [], spent, 5)

    def test_monotony_applies_even_when_food_waste_is_off(self):
        """`off` means sharing earns nothing — never "repeat freely"."""
        scorer = self._scorer(waste="off")

        assert scorer(self.CABBAGE, [], self._basket(2), 2) < scorer(
            self.UNRELATED, [], self._basket(2), 2
        )

    def test_off_still_pays_nothing_for_a_well_spaced_overlap(self):
        scorer = self._scorer(waste="off")

        assert scorer(self.CABBAGE, [], self._basket(1), 3) == scorer(
            self.UNRELATED, [], self._basket(1), 3
        )

    def test_monotony_does_not_outrank_a_favourite(self):
        """Same ordering rule as the reuse bonus: it nudges, it does not veto."""
        scorer = self._scorer(favorite_recipe_ids=["a"])

        assert scorer(self.CABBAGE, [], self._basket(2), 2) > 0

    def test_the_penalty_is_capped_so_long_lists_are_not_punished(self):
        """Uncapped, this would prefer recipes with short ingredient lists."""
        scorer = self._scorer()
        many = {"recipe_id": "c", "recipe_title": "Everything stew",
                "recipe_ingredients": "cabbage carrots leeks parsnips swede turnips kale"}
        basket = IngredientBasket()
        basket.add(many["recipe_ingredients"], 2)

        assert scorer(many, [], basket, 2) == -3.0

    def test_a_flat_basket_keeps_the_older_behaviour(self):
        """The old three-argument contract: reward overlap, no timing."""
        scorer = self._scorer()
        flat = frozenset({"cabbage", "carrots"})

        assert scorer(self.CABBAGE, [], flat) > scorer(self.UNRELATED, [], flat)


class TestStatedPantrySpacing:
    """A stated pantry is a request to use items up, not to eat them daily.

    Regression for a real plan: a member said "I already have tomatoes,
    pasta, cabbage, eggplants" and got tomatoes in 15 of 21 meals, on every
    single day. The +3-per-item pantry boost (up to +6) outranks the −3.0
    monotony cap on its own, so the spacing had no way to bite.
    """

    TOMATO = {"recipe_id": "t", "recipe_title": "T1",
              "recipe_ingredients": "chopped tomatoes, onion"}
    OTHER = {"recipe_id": "o", "recipe_title": "O1",
             "recipe_ingredients": "lentils, spinach"}

    @staticmethod
    def _scorer():
        return build_preference_scorer(
            {"plan_parameters": {"food_waste": "off"}}, pantry=("tomatoes",),
        )

    @staticmethod
    def _basket(*days):
        basket = IngredientBasket()
        for day in days:
            basket.add("chopped tomatoes, onion", day)
        return basket

    def test_an_unused_pantry_item_is_boosted(self):
        scorer = self._scorer()

        assert scorer(self.TOMATO, [], IngredientBasket(), 1) > scorer(
            self.OTHER, [], IngredientBasket(), 1
        )

    def test_the_boost_does_not_repeat_on_the_next_day(self):
        scorer = self._scorer()

        assert scorer(self.TOMATO, [], self._basket(1), 2) < scorer(
            self.OTHER, [], self._basket(1), 2
        )

    def test_the_boost_returns_after_a_gap(self):
        scorer = self._scorer()

        assert scorer(self.TOMATO, [], self._basket(1), 3) > scorer(
            self.OTHER, [], self._basket(1), 3
        )

    def test_the_boost_stops_once_the_item_has_been_used_up(self):
        """Two meals is what "use up my tomatoes" buys. Not a tomato week."""
        scorer = self._scorer()

        assert scorer(self.TOMATO, [], self._basket(1, 3), 5) < scorer(
            self.OTHER, [], self._basket(1, 3), 5
        )

    def test_a_singular_recipe_still_counts_against_a_plural_pantry_item(self):
        basket = IngredientBasket()
        basket.add("one tomato, onion", 1)
        scorer = self._scorer()

        assert scorer(self.TOMATO, [], basket, 2) < scorer(self.OTHER, [], basket, 2)


class _CabbageOrSomethingNew:
    """Every slot offers the cabbage plate, or one nobody has eaten yet.

    Titles are deliberately single short tokens so the title-overlap variety
    penalty stays out of it — this fake isolates the ingredient axis.
    """

    def __init__(self):
        self.n = 0

    def get_candidate_actions(self, meal_type, state):
        self.n += 1
        return [
            {"recipe_id": f"cabbage-{self.n}", "recipe_title": f"C{self.n}",
             "recipe_ingredients": "half a cabbage, carrots, cream",
             "recipe_directions": "d"},
            {"recipe_id": f"new-{self.n}", "recipe_title": f"N{self.n}",
             "recipe_ingredients": "veg" + "a" * self.n, "recipe_directions": "d"},
        ]

    def mark_selected(self, recipe_id):
        pass


class TestSpacingAcrossTheWeek:
    """The end-to-end claim: the cabbage comes back, but not tomorrow."""

    @staticmethod
    def _cabbage_days():
        from services.weekly_planner.environment import WeeklyMealPlanEnv
        from services.weekly_planner.planner import WeeklyPlanner
        from services.weekly_planner.reward_logic import RewardCalculator

        profile = {
            "diet": [], "preferences": [],
            # Tips every otherwise-tied slot to the cabbage plate, so the
            # spacing below is the scorer's doing and not a coin flip.
            "food_likes": ["carrots"],
            "plan_parameters": {"food_waste": "strict"},
        }
        env = WeeklyMealPlanEnv(
            user_profile=profile,
            action_space=_CabbageOrSomethingNew(),
            reward_calculator=RewardCalculator(),
        )
        entries = WeeklyPlanner(env).generate_full_plan(
            user_query="a week", scorer=build_preference_scorer(profile),
        )
        assert len(entries) == 21
        return [e["day"] for e in entries
                if "cabbage" in e["recipe"]["recipe_ingredients"]]

    def test_the_shared_ingredient_comes_back(self):
        assert len(self._cabbage_days()) >= 2

    def test_it_never_lands_twice_in_a_day(self):
        days = self._cabbage_days()

        assert len(days) == len(set(days))

    def test_it_never_lands_on_adjacent_days(self):
        days = sorted(set(self._cabbage_days()))

        assert all(b - a >= 2 for a, b in zip(days, days[1:])), days


class TestScorerArity:
    """Adding the day must not break a scorer that never asked for one."""

    @staticmethod
    def _run(scorer):
        from services.weekly_planner.environment import WeeklyMealPlanEnv
        from services.weekly_planner.planner import WeeklyPlanner
        from services.weekly_planner.reward_logic import RewardCalculator

        env = WeeklyMealPlanEnv(
            user_profile={"diet": [], "preferences": []},
            action_space=_CabbageOrSomethingNew(),
            reward_calculator=RewardCalculator(),
        )
        return WeeklyPlanner(env).generate_full_plan(user_query="w", scorer=scorer)

    def test_a_two_argument_scorer_is_called_with_two(self):
        seen = []

        def old(candidate, chosen_titles):
            seen.append(len(chosen_titles))
            return 0.0

        assert len(self._run(old)) == 21
        assert seen

    def test_a_three_argument_scorer_still_receives_a_flat_basket(self):
        seen = []

        def pre_spacing(candidate, chosen_titles, chosen_ingredients=frozenset()):
            seen.append(chosen_ingredients)
            return 0.0

        assert len(self._run(pre_spacing)) == 21
        assert all(isinstance(b, frozenset) for b in seen)
        # It still grows as the week fills — the flat basket, just without days.
        assert seen[-1] > seen[0]

    def test_a_four_argument_scorer_receives_the_day_being_planned(self):
        seen = []

        def day_aware(candidate, chosen_titles, basket, current_day=0):
            seen.append(current_day)
            return 0.0

        self._run(day_aware)

        assert min(seen) == 1 and max(seen) == 7
        assert seen == sorted(seen)


class TestParameterPlumbing:
    def test_the_card_offers_the_control(self):
        card = plan_parameters.build_card({})
        keys = [p["key"] for p in card["parameters"]]

        assert "food_waste" in keys

    def test_sanitize_accepts_the_modes_and_rejects_junk(self):
        assert plan_parameters.sanitize({"food_waste": "reuse"}) == {"food_waste": "reuse"}
        assert plan_parameters.sanitize({"food_waste": "maximum"}) == {}

    def test_waste_mode_defaults_off(self):
        assert plan_parameters.waste_mode({}) == "off"
        assert plan_parameters.waste_mode({"food_waste": "nonsense"}) == "off"
        assert plan_parameters.waste_mode({"food_waste": "strict"}) == "strict"

    def test_describe_says_it_in_words(self):
        """The daily path hears the setting as prose in the grader prompt."""
        text = plan_parameters.describe({"food_waste": "reuse"})

        assert "food waste" in text
