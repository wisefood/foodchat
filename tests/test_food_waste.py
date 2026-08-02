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
