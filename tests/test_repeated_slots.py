"""
Two snacks means two snacks.

    > plan my day include two snack as well in-between
    < Breakfast 08:00 · Lunch 13:00 · Snack 16:00 · Dinner 19:30
      "Planned 1 day — breakfast; lunch; snack; dinner from your preferences."

One snack, and a summary that describes the day as though that were what was
asked for. Nothing in the pipeline was wrong on its own: `additions` heard
"snack" and added one, which is its job. The number had nowhere to land —
`PlanSpec.meals` is a tuple of slot NAMES, every writer refused a name it
already held, and `plates`/`slot_food_groups` are dicts keyed by that name. A
second snack could not be represented, so no reader looked for one.

These tests cover the three places a second snack could still be lost:

* the SHAPE, which now names instances (`snack_2`) and places repeats in the
  gaps between anchor meals — which is what "in-between" means;
* the REQUEST and the reply pairing, because RecipeWrangler is asked for the
  kind (it has never heard of `snack_2`) and both entries echo back as
  `snack` — so a pool keyed on the echo merged them and lost one;
* the ROUND TRIP, because a spec that cannot survive `to_dict` is a spec that
  lasts until the first clarifying question.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.plan_spec import (                                     # noqa: E402
    MAX_MEALS_PER_DAY,
    PlanSpec,
    slot_instance,
    slot_kind,
)
from models.recipe import slot_sort_key                            # noqa: E402
from services import shape_intent, turn_intake                     # noqa: E402
from services.plan_client import PlanClient                        # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_turn():
    turn_intake.forget()
    yield
    turn_intake.forget()


# ── the shape ────────────────────────────────────────────────────────────

class TestTheShapeCanHoldTwo:
    def test_two_snacks_land_between_the_meals(self):
        spec = PlanSpec().with_meal_count("snack", 2)
        assert spec.meals == ("breakfast", "snack", "lunch", "snack_2", "dinner")

    def test_one_snack_goes_where_it_always_did(self):
        """`add a snack` and `add 1 snack` must describe the same day."""
        assert PlanSpec().with_meal_count("snack", 1).meals == \
            PlanSpec().with_meal("snack").meals

    def test_growing_from_one_renumbers_by_position(self):
        """The morning snack is `snack`, whichever was asked for first."""
        spec = PlanSpec().with_meal("snack").with_meal_count("snack", 2)
        assert spec.meals.index("snack") < spec.meals.index("snack_2")
        assert spec.meals.index("snack") < spec.meals.index("lunch")

    def test_cutting_back_keeps_the_earlier_one(self):
        spec = PlanSpec().with_meal_count("snack", 2).with_meal_count("snack", 1)
        assert spec.instances_of("snack") == ("snack",)
        assert spec.meals.index("snack") < spec.meals.index("lunch")

    def test_plates_and_tastes_travel_with_a_renumbered_slot(self):
        spec = (
            PlanSpec()
            .with_meal("snack")
            .with_food_groups("snack", ["fruit"])
            .with_plate("snack", "drink")
            .with_meal_count("snack", 2)
        )
        # That snack moved from first-and-only to second in the day.
        assert spec.slot_food_groups["snack_2"] == ("fruit",)
        assert "drink" in spec.roles_for("snack_2")
        assert spec.slot_food_groups.get("snack") is None

    def test_the_day_has_a_ceiling(self):
        spec = PlanSpec().with_meal_count("snack", 4)
        assert len(spec.meals) <= MAX_MEALS_PER_DAY

    def test_describe_speaks_the_kind_not_the_id(self):
        said = PlanSpec().with_meal_count("snack", 2).describe()
        assert "snack_2" not in said
        assert said.count("snack") == 2

    def test_an_instance_sorts_beside_its_kind(self):
        """Not after the drinks, which is where an unknown slot ties."""
        assert slot_sort_key("snack_2") == slot_sort_key("snack")

    def test_kind_and_instance_round_trip(self):
        assert slot_kind(slot_instance("snack", 3)) == "snack"
        assert slot_instance("snack", 1) == "snack"


class TestItSurvivesBeingStored:
    def test_a_two_snack_shape_comes_back_whole(self):
        spec = PlanSpec().with_meal_count("snack", 2)
        assert PlanSpec.from_spec(spec.to_dict()).meals == spec.meals

    def test_a_taste_scoped_to_a_slot_comes_back_too(self):
        """`slot_food_groups` was never serialised: "fruit for the snack"
        survived exactly until the turn that asked a clarifying question."""
        spec = PlanSpec().with_food_groups("snack", ["fruit"]) \
            if "snack" in PlanSpec().meals else \
            PlanSpec().with_meal("snack").with_food_groups("snack", ["fruit"])
        assert PlanSpec.from_spec(spec.to_dict()).slot_food_groups == \
            spec.slot_food_groups


# ── the reader ───────────────────────────────────────────────────────────

class TestReadingHowMany:
    @pytest.mark.parametrize("message", [
        "plan my day include two snack as well in-between",
        "plan my day with 2 snacks",
        "i want a couple of snacks",
        "two snacks please",
    ])
    def test_two_is_heard(self, message):
        spec, notes = shape_intent.slot_counts(message, PlanSpec())
        assert len(spec.instances_of("snack")) == 2, message
        assert notes

    @pytest.mark.parametrize("message", [
        "add a snack",                      # no number — `additions` adds one
        "plan for three days",              # a count of days, not of meals
        "no snacks please",                 # a refusal
        "a main and two sides for dinner",  # a count of PLATES
        "some snacks would be nice",        # not a written number
    ])
    def test_these_are_not_a_meal_count(self, message):
        spec, notes = shape_intent.slot_counts(message, PlanSpec())
        assert notes == [], message
        assert spec.instances_of("snack") == (), message

    def test_the_note_says_what_landed_not_what_was_asked(self):
        """A day has a ceiling, and a note claiming four snacks when three
        were built would put a falsehood into the member's own summary."""
        _spec, notes = shape_intent.slot_counts("give me four snacks", PlanSpec())
        assert notes and notes[0].startswith("3 snacks")
        assert "asked for 4" in notes[0]


class TestTheWiringItself:
    """`slot_counts` passing its own tests proves nothing if nobody calls it."""

    def test_intake_hears_the_number(self, session_service, sample_profile,
                                     monkeypatch):
        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta
        from models.planning_state import PlanningStateDelta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        state = turn_intake.intake(
            session.session_id,
            "plan my day include two snack as well in-between",
            session_service=session_service,
        )
        assert len(state.spec.instances_of("snack")) == 2


# ── the request and the reply ────────────────────────────────────────────

class TestRecipeWranglerNeverSeesAnInstance:
    def test_the_request_asks_for_the_kind(self):
        spec = PlanSpec().with_meal_count("snack", 2)
        slots = [entry["slot"] for entry in spec.to_request_slots()]
        assert slots == ["breakfast", "snack", "lunch", "snack", "dinner"]

    def test_the_role_sequence_keeps_the_instance(self):
        """Positional pairing is what puts the two snacks back where they go."""
        spec = PlanSpec().with_meal_count("snack", 2)
        assert [slot for slot, _role in spec.role_sequence()] == list(spec.meals)

    def test_two_snacks_come_back_as_two_pools(self):
        """Keyed on the echoed slot, both entries said `snack` and the second
        pool overwrote the first — a plan with one snack and no complaint."""
        spec = PlanSpec().with_meal_count("snack", 2)
        envelope = {
            "days": [{
                "day": 1,
                "slots": [
                    {"slot": slot_kind(slot),
                     "recipes": [{"recipe_id": f"r-{index}", "title": f"Dish {index}"}]}
                    for index, (slot, _role) in enumerate(spec.role_sequence())
                ],
            }],
        }
        pools = PlanClient.to_role_pools(envelope, spec)
        assert ("snack", "main") in pools[1]
        assert ("snack_2", "main") in pools[1]
        assert pools[1][("snack", "main")][0].recipe_id != \
            pools[1][("snack_2", "main")][0].recipe_id
