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


class TestOneSentenceCanDoTwoThings:
    """"Add a snack before my lunch on all days" is one request and the second
    half reached nothing: the router skips its reorder branch whenever the shape
    grew — an addition that mentions an order is still an addition — and nothing
    else applied the order. The snack was added and landed after lunch, by the
    eating-order rule it was explicitly asked to break."""

    def test_the_addition_and_the_order_both_land(self):
        from services import plan_navigation, shape_intent

        spec = PlanSpec(num_days=3, meals=("breakfast", "lunch", "dinner"))
        grown, added = shape_intent.additions("add a snack before my lunch on all days", spec)
        assert added == ["added snack"]
        assert grown.meals == ("breakfast", "lunch", "snack", "dinner"), (
            "precondition: eating order puts a snack after lunch"
        )

        move = plan_navigation.reorder_request("add a snack before my lunch on all days")
        moved = grown.reorder(move[0], **{move[1]: move[2]})

        assert moved.meals == ("breakfast", "snack", "lunch", "dinner")


class TestHowManyDays:
    """"Switch to daily from three days plan" said one day, and the three-day
    plan came back three days long: the LLM extractor may abstain, and on a
    REFINEMENT `plan_horizon` deliberately leaves the days alone."""

    @pytest.mark.parametrize("message,days", [
        ("switch to daily from three days plan", 1),
        ("switch to daily", 1),
        ("just one day", 1),
        ("make it one day", 1),
        ("just today", 1),
        ("plan me three days", 3),
        ("plan for 5 days", 5),
        ("for 5 days", 5),
        ("three days please", 3),
        ("make it 2 days", 2),
        ("weekly plan", 7),
    ])
    def test_a_horizon_that_was_asked_for(self, message, days):
        from services import shape_intent

        assert shape_intent.horizon(message) == days

    @pytest.mark.parametrize("message", [
        "i have three days of leftovers",
        "we ate the same thing for three days",
        "the kids were off school for three days",
        "add a snack",
        "something lighter for lunch",
    ])
    def test_a_number_merely_mentioned_is_not_a_horizon(self, message):
        """The first version of this read "i have three days of leftovers" as a
        three-day plan — the exact false positive its own docstring warned
        about, written in the same breath as the warning. A count has to be
        ASKED for."""
        from services import shape_intent

        assert shape_intent.horizon(message) is None


class TestACuisineIsNotADishName:
    """Reported: "greek breakfast?" → "Done — I swapped the breakfast: Soy
    banana bran muffins → Greek chicken", answered with "this is a main
    dish....."

    The named-dish search runs with NO course-type filter, on purpose: someone
    asking for apple pie at breakfast has decided pie is breakfast food. A
    cuisine is not that — it describes a style, every slot has one, and letting
    it override the slot's courses lands a title search on the first Greek thing
    in the corpus whatever meal it belongs to.
    """

    @pytest.fixture(autouse=True)
    def _vocabulary(self, monkeypatch):
        """The live vocabulary, pinned — offline it is unreachable and the
        function then answers False, which is the old behaviour."""
        from services.candidates_client import CANDIDATES

        known = {"greek", "italian", "thai", "mexican", "hungarian"}
        monkeypatch.setattr(
            CANDIDATES, "split_cuisines",
            lambda likes: ([w for w in likes if w in known],
                           [w for w in likes if w not in known]),
        )

    @pytest.mark.parametrize("word", ["greek", "Greek", " italian ", "thai"])
    def test_a_cuisine_is_recognised(self, word):
        from services.edit_service import _names_a_cuisine

        assert _names_a_cuisine(word) is True

    @pytest.mark.parametrize("word", [
        "apple pie", "shakshuka", "something with feta", "lighter", "",
    ])
    def test_a_real_dish_or_directive_is_not(self, word):
        from services.edit_service import _names_a_cuisine

        assert _names_a_cuisine(word) is False

    def test_the_search_branch_asks_both_questions(self):
        """The named-dish search must be gated on BOTH, or the cuisine check
        is a function nothing calls."""
        import inspect

        from services.edit_service import EditService

        source = inspect.getsource(EditService._find_replacement)
        assert "_names_a_cuisine(predicate.directive)" in source
        assert "not _names_a_cuisine" in source
