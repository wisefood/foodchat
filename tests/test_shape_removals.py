"""Taking something OUT of a plan.

Reported from the canvas, twice in a row:

    > remove the snack from midday
    < Done — I swapped the lunch: "Spanish-style chorizo frittata" → "Oat Snack Cakes"

    > hmmm better before lunch, i will have lunch later today
    < Done — I swapped the breakfast: "Tofu Scramble" → "Sunday Lunch" · 82 → 1907 kcal

Nothing could remove a meal, so both messages classified as slot edits, and an
edit can only replace the dish on a slot. The leftover words — "snack",
"lunch" — were then searched as RECIPE TITLES, and the corpus obligingly
contains dishes named after meals. A member's word for WHEN they eat came back
as WHAT they eat, at four times the calories, reported as "Done".

Two fixes meet here: the shape reader learned to remove, and the edit path
stopped reading a structural phrase as a dish name.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                        # noqa: E402
from services import shape_intent                            # noqa: E402
from services.edit_service import _names_a_dish              # noqa: E402

FULL = PlanSpec(meals=("breakfast", "lunch", "snack", "dinner"),
                plates={"dinner": ("main", "salad")})


class TestRemovingAMeal:
    @pytest.mark.parametrize("message", [
        "remove the snack from midday",
        "remove the snack",
        "drop the snack",
        "no snack today",
        "get rid of the snack",
        "take out the snack",
        "cancel the snack",
        "i don't want the snack",
    ])
    def test_the_phrasings_a_member_uses(self, message):
        spec, notes = shape_intent.removals(message, FULL)
        assert notes == ["removed snack"], message
        assert spec.meals == ("breakfast", "lunch", "dinner")

    def test_removing_a_plate_leaves_the_meal(self, ):
        spec, notes = shape_intent.removals("take out the salad from dinner", FULL)
        assert notes == ["removed the salad from dinner"]
        assert spec.meals == FULL.meals
        assert spec.roles_for("dinner") == ("main",)


class TestItRefusesToDestroy:
    def test_the_last_meal_cannot_go(self):
        """"Cancel the plan" is a different request and this is not it."""
        one = PlanSpec(meals=("dinner",))
        spec, notes = shape_intent.removals("remove dinner", one)
        assert notes == [] and spec.meals == ("dinner",)

    def test_a_main_cannot_go(self):
        """A meal with no main is a side dish pretending to be dinner."""
        spec = FULL.without_plate("dinner", "main")
        assert spec.roles_for("dinner") == ("main", "salad")

    def test_a_meal_the_plan_never_had_is_not_news(self):
        spec, notes = shape_intent.removals("remove the brunch", FULL)
        assert notes == [] and spec.meals == FULL.meals

    def test_a_snack_eaten_is_not_a_snack_removed(self):
        """The verb has to be a removal verb, next to the target."""
        spec, notes = shape_intent.removals(
            "i had a snack earlier, plan me dinner", FULL,
        )
        assert notes == [] and spec.meals == FULL.meals

    def test_adding_is_never_removing(self):
        spec, notes = shape_intent.removals("add a snack", FULL)
        assert notes == []

    def test_a_plate_with_no_meal_named_is_left_alone(self):
        """Guessing which meal loses its salad is worse than asking."""
        spec, notes = shape_intent.removals("remove the salad", FULL)
        assert notes == [] and spec.plates == FULL.plates


class TestAStructuralPhraseIsNotADish:
    """The half that made the swaps destructive rather than merely wrong."""

    @pytest.mark.parametrize("phrase", [
        "lunch", "snack", "breakfast", "the snack", "my lunch",
        "before lunch", "later today", "midday", "after dinner",
        "the second meal",
    ])
    def test_it_is_never_searched_as_a_recipe_title(self, phrase):
        assert _names_a_dish(phrase) is False, phrase

    @pytest.mark.parametrize("phrase", [
        "oat snack cakes",        # survives the word "snack" inside it
        "sunday lunch roast",     # survives "lunch"
        "spaghetti puttanesca",
        "something with feta",
    ])
    def test_a_real_dish_still_is_one(self, phrase):
        assert _names_a_dish(phrase) is True, phrase
