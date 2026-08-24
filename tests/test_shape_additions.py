"""
"Add breakfast." "And a salad on the side."

Adding was the one thing nothing in the system could do, and a member found
both halves of it in one sitting.

    add breakfast as well i dont have one
      -> "Day 1 doesn't have a breakfast in this plan — it has lunch, dinner.
          Which of those should I change?"
    okay yes but my day doesnt have breakfast
      -> the same sentence again

That is not the edit path being unhelpful: an edit REPLACES the dish on a slot,
and they are asking for a slot that does not exist. There was no way out of the
loop.

The quieter half, same cause:

    also lighter lunch and add a salad as well there for side
      -> lunch swapped for something lighter, and no salad

The swap was classified, executed and reported. The addition was heard by
nobody, so the reply was a confident answer to half the request.

Both are shape changes, and `PlanSpec` is the type that holds shape — so they
have to reach the spec. Deterministic, over closed sets (the slots
RecipeWrangler fills, the roles a plate can have), so there is nothing to
invent and an unrecognised word leaves the turn exactly as it was.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import MAX_PLATES_PER_MEAL, PlanSpec        # noqa: E402
from services import shape_intent                                 # noqa: E402

# The plan the member actually had.
PLAN = PlanSpec(meals=("lunch", "dinner"), plates={"dinner": ("main", "side")})


def add(message, spec=PLAN, focus=None):
    return shape_intent.additions(message, spec, focus_slot=focus)


class TestTheTwoThatWereReported:
    def test_add_breakfast(self):
        spec, changed = add("add breakfast as well i dont have one")
        assert changed == ["added breakfast"]
        assert spec.meals == ("breakfast", "lunch", "dinner")

    def test_the_follow_up_when_the_first_attempt_failed(self):
        """"my day doesnt have breakfast" is what someone says after "add
        breakfast" was answered with a question. It is a statement that the plan
        LACKS something, which is a request to add it."""
        spec, changed = add("okay yes but my day doesnt have breakfast")
        assert changed == ["added breakfast"]
        assert "breakfast" in spec.meals

    def test_lighter_lunch_and_a_salad_is_both(self):
        spec, changed = add(
            "also lighter lunch and add a salad as well there for side",
            focus="lunch",
        )
        assert changed == ["added a salad to lunch"]
        assert spec.roles_for("lunch") == ("main", "salad")

    def test_for_side_is_not_a_second_plate(self):
        """It says WHERE the salad goes. Read as another plate it gave lunch
        three courses for a request that named one."""
        spec, _ = add("add a salad as well there for side", focus="lunch")
        assert spec.roles_for("lunch") == ("main", "salad")


class TestMealsKeepTheirOrder:
    def test_breakfast_lands_before_lunch(self):
        spec, _ = add("add breakfast")
        assert spec.meals == ("breakfast", "lunch", "dinner")

    def test_a_snack_lands_where_a_snack_goes(self):
        spec, _ = add("add a snack")
        assert spec.meals == ("lunch", "snack", "dinner")


class TestSlotOrPlate:
    """`side`, `dessert` and `drink` are each a meal in their own right and a
    plate on another meal. Read the wrong way round, "add a side to lunch"
    produced a meal called "side"."""

    def test_named_meal_means_a_plate(self):
        spec, changed = add("add a side to lunch")
        assert changed == ["added a side to lunch"]
        assert spec.roles_for("lunch") == ("main", "side")
        assert "side" not in spec.meals

    def test_no_meal_named_means_a_course_of_its_own(self):
        spec, changed = add("add a dessert")
        assert changed == ["added dessert"]
        assert "dessert" in spec.meals

    def test_a_named_meal_wins_over_the_focus(self):
        spec, _ = add("add a dessert to dinner", focus="lunch")
        assert "dessert" in spec.roles_for("dinner")
        assert "dessert" not in spec.roles_for("lunch")

    def test_a_plate_on_a_meal_the_plan_lacks_implies_the_meal(self):
        spec, _ = shape_intent.additions(
            "add a salad to lunch", PlanSpec(meals=("dinner",)),
        )
        assert "lunch" in spec.meals
        assert spec.roles_for("lunch") == ("main", "salad")


class TestItNeverAddsWhatWasRefused:
    @pytest.mark.parametrize("message", [
        "no dessert please",
        "without a starter",
        "drop the salad",
        "remove the side",
        "skip breakfast",
        "forget the dessert",
        "not a salad",
    ])
    def test_a_refusal_adds_nothing(self, message):
        assert add(message, focus="lunch")[1] == []

    def test_a_refusal_of_one_thing_does_not_veto_another(self):
        """A clause-wide negation check read "add a salad but no dessert" as
        entirely negative."""
        spec, changed = add("add a salad but no dessert", focus="lunch")
        assert changed == ["added a salad to lunch"]
        assert "dessert" not in spec.roles_for("lunch")
        assert "dessert" not in spec.meals

    def test_a_reason_is_not_a_refusal(self):
        """"i dont have one" is WHY they are asking, not a refusal — and a
        clause-wide check read the "dont" and dropped the request."""
        assert add("add breakfast as well i dont have one")[1] == ["added breakfast"]


class TestItAddsNothingItWasNotAsked:
    @pytest.mark.parametrize("message", [
        "make the lunch lighter",
        "swap the dinner for something quicker",
        "i dont have time for this",
        "we dont have nuts in",
        "plan my day",
        "thanks, that looks great",
        "breakfast was lovely yesterday",
        "",
    ])
    def test_nothing_added(self, message):
        assert add(message)[1] == []

    def test_a_meal_already_on_the_plan_is_not_re_added(self):
        assert add("add lunch")[1] == []

    def test_a_plate_already_on_the_meal_is_not_re_added(self):
        assert add("add a side to dinner")[1] == []

    def test_an_unnamed_plate_with_no_focus_is_left_alone(self):
        """Better than putting it on a meal the member did not choose."""
        assert add("add a salad")[1] == []

    def test_applying_twice_changes_nothing(self):
        once, _ = add("add breakfast")
        twice, changed = add("add breakfast", spec=once)
        assert changed == [] and twice.meals == once.meals


class TestTheShapeStaysBuildable:
    def test_it_will_not_exceed_the_plate_limit(self):
        full = PlanSpec(
            meals=("dinner",),
            plates={"dinner": ("main", "side", "dessert")[:MAX_PLATES_PER_MEAL]},
        )
        spec, changed = shape_intent.additions("add a drink to dinner", full)
        assert changed == []
        assert len(spec.roles_for("dinner")) == MAX_PLATES_PER_MEAL

    def test_an_unknown_course_is_not_invented(self):
        assert add("add a sandwich course to lunch")[1] == []

    def test_the_spec_is_never_mutated_in_place(self):
        """`PlanSpec` is frozen and callers hold the old one."""
        before = PLAN.describe()
        add("add breakfast")
        assert PLAN.describe() == before


class TestItReachesTheTurnAndTheRouter:
    def test_intake_grows_the_standing_spec(self, session_service, sample_profile,
                                            monkeypatch):
        import uuid

        from models.planning_state import PlanningStateDelta
        from services import turn_intake

        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(spec=PLAN),
            ),
        )
        turn_intake.forget()

        state = turn_intake.intake(
            session.session_id, "add breakfast as well i dont have one",
            session_service=session_service,
        )
        assert "breakfast" in state.spec.meals
        assert turn_intake.added_shape() == ["added breakfast"]

    def test_a_growing_turn_re_plans_instead_of_swapping_a_slot(self):
        """The wiring that matters. Without it the addition still reaches the
        spec and the turn still goes to the editor, which can only swap."""
        import ast
        import inspect

        from services.orchestrator_service import OrchestratorService

        src = inspect.getsource(OrchestratorService._route)
        block = src[src.index('if intent == "edit_plan_slot"'):]
        block = block[:block.index('if intent == "nutrition_question"')]
        tree = ast.parse("if True:\n" + "\n".join(
            "    " + line for line in block.splitlines()
        ))
        calls = {
            getattr(n.func, "attr", None) for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        assert "added_shape" in calls, "the router never asks whether the shape grew"
        assert "_handle_plan" in calls or "_handle_weekly" in calls

    def test_the_dead_end_sentence_offers_to_add(self):
        """It used to send the member in a circle: they asked for a meal the
        plan does not have and were asked to pick one it does."""
        from services.edit_service import EditService
        from models.session import DayPlan, Meal, MealCourse

        days = [DayPlan(day=1, meals=[
            Meal("lunch", [MealCourse("l", "Soup", "x", "y")]),
            Meal("dinner", [MealCourse("d", "Stew", "x", "y")]),
        ])]
        text = EditService._structured_miss_text(days, 1, "breakfast")
        assert "add breakfast" in text
        assert "Which of those should I change?" not in text


class TestPhrasingsFoundByProbing:
    """The gaps a written-down list of cases does not find.

    Each of these came from running plausible member phrasings through the
    reader and looking at what it did — which is how the "should have" and the
    plural came to light, both of them silently doing nothing.
    """

    @pytest.mark.parametrize("message,expected", [
        ("can you add a breakfast", "added breakfast"),
        ("i'd like a snack too", "added snack"),
        ("lunch should have a soup as well", "added a soup to lunch"),
        ("dinner needs a salad", "added a salad to dinner"),
        ("also a drink with dinner", "added a drink to dinner"),
        ("throw in a dessert after dinner", "added a dessert to dinner"),
    ])
    def test_phrasings_that_mean_add(self, message, expected):
        assert add(message)[1] == [expected]

    def test_a_plural_is_the_same_request(self):
        """"Add two sides to dinner" matched nothing at all, because the reader
        looked for `side` and the member wrote `sides`."""
        bare = PlanSpec(meals=("lunch", "dinner"))
        spec, changed = shape_intent.additions("add two sides to dinner", bare)
        assert changed == ["added a side to dinner"]
        assert spec.roles_for("dinner") == ("main", "side")

    @pytest.mark.parametrize("message,focus", [
        ("what's for breakfast?", None),
        ("the breakfast is too heavy", None),
        ("i had breakfast already", None),
        ("replace breakfast", None),
        ("just breakfast today", None),
        ("only lunch and dinner", None),
        ("less salad please", "lunch"),
        ("swap the salad for something else", "lunch"),
    ])
    def test_phrasings_that_do_not(self, message, focus):
        """A question about a meal, a complaint about one, a memory of one and a
        request to REPLACE one all mention a slot and none of them is an
        addition. "Only" and "just" are narrowings, which the shape extractor
        owns — this reader deliberately stays out of them."""
        assert add(message, focus=focus)[1] == []


class TestWhatThisReaderDoesNotDo:
    """Written down because the alternative is someone assuming it does."""

    def test_it_cannot_remove_a_meal(self):
        """"Instead of" is a replacement, and this reader only adds — so
        "give me a brunch instead of lunch" adds the brunch and leaves the
        lunch. Half the request, and the half it can express."""
        spec, changed = add("give me a brunch instead of lunch")
        assert changed == ["added brunch"]
        assert "lunch" in spec.meals

    def test_a_bare_slot_name_is_not_a_request(self):
        """"Breakfast please" is plausibly one. So is "what's for breakfast?",
        and nothing in the words separates them — so neither fires, and the
        dead-end sentence tells the member the phrasing that does."""
        assert add("breakfast please")[1] == []
