"""PlanSpec — the shape of a plan, in FoodChat's own vocabulary.

`DYNAMIC_MEALS_PLAN.md` §4.1 specifies this shape, and Phase 1 already landed
the model it produces (`Meal.plates`, `DayPlan`, `MealCourse.role`). What the
plan named as the blocker was a question about RecipeWrangler: *"check whether
`candidates_client` can filter by dish_type/course; if not, this is the one
RW-side dependency."*

It can. `/api/v2/tools/plan_meals` takes a `course_types` override per slot, so
a `side` plate becomes a real query for salads and soups rather than a
keyword-biased guess. These tests pin the translation across that boundary and
the two things easiest to get quietly wrong:

**One request per plate.** RecipeWrangler ORs a slot's `course_types`, so a
single entry naming main-dish and salad returns *one* recipe that is either.
A two-plate meal asked for that way comes back with a plate missing and nothing
to indicate it.

**Roles stay on this side.** FoodChat persists `main`/`side`/`dessert`; the
corpus is annotated with course types. Letting either vocabulary cross would
put the other system's taxonomy into stored plans permanently.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import (  # noqa: E402
    DEFAULT_MEALS,
    MAX_DAYS,
    MAX_MEALS_PER_DAY,
    ROLE_COURSE_TYPES,
    ROLES,
    PlanSpec,
)


class TestDefault:
    def test_default_is_todays_shape(self):
        spec = PlanSpec.default()

        assert spec.num_days == 1
        assert spec.meals == DEFAULT_MEALS
        assert spec.is_default

    def test_default_asks_for_one_main_per_meal(self):
        assert PlanSpec.default().to_request_slots() == [
            {"slot": "breakfast", "count": 1, "course_types": ["main-dish"]},
            {"slot": "lunch", "count": 1, "course_types": ["main-dish"]},
            {"slot": "dinner", "count": 1, "course_types": ["main-dish"]},
        ]

    @pytest.mark.parametrize(
        "spec",
        [
            PlanSpec(num_days=3),
            PlanSpec(meals=("lunch", "dinner")),
            PlanSpec(plates={"dinner": ("main", "side")}),
        ],
    )
    def test_anything_else_is_not_default(self, spec):
        """`is_default` gates the legacy graded path, which composes one recipe
        per slot and cannot express a second plate."""
        assert not spec.is_default

    def test_every_meal_has_a_main_by_default(self):
        assert PlanSpec.default().roles_for("lunch") == ("main",)


class TestMultiPlate:
    def test_a_two_plate_meal_becomes_two_requests(self):
        """The OR-filter trap: one entry with two course sets returns one dish."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})

        assert spec.to_request_slots() == [
            {"slot": "dinner", "count": 1, "course_types": ["main-dish"]},
            {"slot": "dinner", "count": 1, "course_types": ["side", "salad", "soup"]},
        ]

    def test_plates_keep_the_same_slot_name(self):
        """So the response regroups into one `Meal` on the way home."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "dessert")})

        assert {entry["slot"] for entry in spec.to_request_slots()} == {"dinner"}

    def test_a_side_maps_to_several_course_types(self):
        """A side can be a salad, a soup or a side.

        Asking for only one would empty the plate on a corpus that annotated it
        as another — and unlike a multi-plate meal, these are alternatives, so
        the OR is the right semantics here.
        """
        assert set(ROLE_COURSE_TYPES["side"]) == {"side", "salad", "soup"}

    def test_role_sequence_matches_the_request_order(self):
        """The response carries the slot but not the role, so the two are zipped
        back by position — they must be generated in the same order."""
        spec = PlanSpec(meals=("lunch", "dinner"), plates={"dinner": ("main", "side")})

        assert spec.role_sequence() == [
            ("lunch", "main"),
            ("dinner", "main"),
            ("dinner", "side"),
        ]
        assert len(spec.role_sequence()) == len(spec.to_request_slots())

    def test_total_plates_counts_across_days(self):
        spec = PlanSpec(num_days=3, meals=("dinner",), plates={"dinner": ("main", "side")})

        assert spec.total_plates == 6


class TestKcalSplit:
    """The honest-nutrition rule: main + side is one meal, not two."""

    def test_a_single_plate_takes_the_whole_meal(self):
        assert PlanSpec.default().kcal_split("lunch") == {"main": 1.0}

    def test_weights_sum_to_one(self):
        """Otherwise a two-plate meal silently spends more than a meal's budget."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})

        assert sum(spec.kcal_split("dinner").values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_for_any_role_combination(self):
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "dessert")})

        assert sum(spec.kcal_split("dinner").values()) == pytest.approx(1.0)

    def test_the_main_takes_the_larger_share(self):
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        split = spec.kcal_split("dinner")

        assert split["main"] > split["side"]


class TestFromSpec:
    """Input arrives from an LLM, so everything is untrusted."""

    def test_builds_from_a_valid_spec(self):
        spec = PlanSpec.from_spec(
            {"num_days": 3, "meals": ["lunch", "dinner"], "plates": {"dinner": ["main", "side"]}}
        )

        assert spec.num_days == 3
        assert spec.meals == ("lunch", "dinner")
        assert spec.roles_for("dinner") == ("main", "side")

    def test_unknown_slots_are_dropped(self):
        """RecipeWrangler would accept `elevenses`, match nothing, and return an
        empty meal — which reads as a broken planner, not an unknown word."""
        assert PlanSpec.from_spec({"meals": ["lunch", "elevenses"]}).meals == ("lunch",)

    def test_unknown_roles_are_dropped(self):
        spec = PlanSpec.from_spec({"plates": {"dinner": ["main", "amuse-bouche"]}})

        assert spec.roles_for("dinner") == ("main",)

    def test_a_meal_is_always_anchored_by_a_main(self):
        """A spec asking only for a side describes a side dish, not a meal."""
        spec = PlanSpec.from_spec({"meals": ["dinner"], "plates": {"dinner": ["side"]}})

        assert spec.roles_for("dinner") == ("main", "side")

    def test_plates_can_imply_the_meal(self):
        """"dinner should have a side" implies dinner is in the plan."""
        spec = PlanSpec.from_spec({"plates": {"dinner": ["main", "side"]}})

        assert "dinner" in spec.meals

    def test_a_single_main_is_not_recorded_as_an_override(self):
        """It is the default, and storing it would make `is_default` false for a
        spec that describes exactly the default."""
        assert PlanSpec.from_spec({"plates": {"lunch": ["main"]}}).plates == {}

    def test_nothing_usable_falls_back_to_the_default(self):
        """A wrong number of meals is correctable next turn; no plan is not."""
        assert PlanSpec.from_spec({"meals": ["brunch-o-clock"]}).meals == DEFAULT_MEALS

    @pytest.mark.parametrize("junk", [None, "lunch", 42, [], {"meals": "dinner"}])
    def test_garbage_does_not_raise(self, junk):
        assert PlanSpec.from_spec(junk).meals

    @pytest.mark.parametrize(
        "days,expected", [(0, 1), (-3, 1), (99, MAX_DAYS), ("x", 1), (5, 5)]
    )
    def test_days_are_clamped(self, days, expected):
        assert PlanSpec.from_spec({"num_days": days}).num_days == expected

    def test_meals_per_day_are_capped(self):
        """A forty-plate day is neither renderable as cards nor speakable."""
        spec = PlanSpec.from_spec({"meals": list(
            ("breakfast", "brunch", "lunch", "dinner", "snack", "dessert", "side", "drink")
        )})

        assert len(spec.meals) == MAX_MEALS_PER_DAY

    def test_duplicate_meals_are_collapsed(self):
        assert PlanSpec.from_spec({"meals": ["lunch", "lunch"]}).meals == ("lunch",)


class TestDescribe:
    def test_describes_a_multi_plate_meal(self):
        text = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")}).describe()

        assert "main + side" in text

    def test_describes_multiple_days(self):
        assert "3 days" in PlanSpec(num_days=3).describe()

    def test_a_single_plate_meal_reads_as_just_its_name(self):
        assert PlanSpec(meals=("lunch",)).describe().endswith("lunch")


class TestVocabularyBoundary:
    def test_every_role_maps_to_course_types(self):
        """A role with no mapping produces an empty filter, which matches the
        whole corpus — a dessert plate would come back as any recipe at all."""
        for role in ROLES:
            assert ROLE_COURSE_TYPES.get(role), f"{role} has no course types"

    def test_course_types_never_appear_in_a_spec(self):
        """RecipeWrangler's taxonomy stops at the client boundary; letting it
        into a spec would put it into every persisted plan."""
        spec = PlanSpec.from_spec({"plates": {"dinner": ["main", "main-dish"]}})

        assert spec.roles_for("dinner") == ("main",)
