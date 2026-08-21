"""
The plan for the plan.

The pipeline went straight from a message to a search: reconcile, splat some
filters, grade what comes back. Nothing wrote down what the plan was FOR, which
is why nothing could check whether it succeeded — the ledger reported the
request back to the member and called it a result.

A brief is that missing artefact, and it has two jobs beyond tidiness:

* it is built deterministically, so a reasoning step is an improvement rather
  than a dependency — a strategist that fails leaves a working plan;
* it defines the contract with the verifier, so "what we asked for" and "what
  we check" cannot drift into two hand-maintained lists.

The tests that matter most here are the ones about what a strategist is NOT
allowed to do.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_brief import DEFAULT_RELAXATION_ORDER, PlanBrief    # noqa: E402
from models.plan_spec import PlanSpec                                 # noqa: E402
from models.planning_state import PlanningState, PlanningStateDelta   # noqa: E402


VOCAB = {
    "cuisines": ["thai", "greek", "italian"],
    "moods": ["light", "hearty", "comforting"],
    "flavor_profiles": ["spicy", "fresh"],
    "food_groups": ["vegetables", "legumes"],
}


class TestDeterministicBuild:
    def test_hard_constraints_come_from_the_profile(self):
        brief = PlanBrief.build({
            "allergies": ["peanuts"], "diet": ["vegetarian"],
        })
        assert "peanuts" in brief.allergens
        assert "vegetarian" in brief.diet

    def test_a_diet_stated_in_chat_joins_the_profile_one(self):
        state = PlanningState().merge(PlanningStateDelta(diet_tags=("gluten_free",)))
        brief = PlanBrief.build({"diet": ["vegetarian"], "_diet_tags": ["gluten_free"]}, state)
        assert set(brief.diet) >= {"vegetarian", "gluten_free"}

    def test_a_goal_becomes_claim_tags(self):
        brief = PlanBrief.build({"plan_parameters": {"goal": "high_protein"}})
        assert "high_protein" in brief.claim_tags

    def test_a_time_budget_is_carried(self):
        brief = PlanBrief.build({"plan_parameters": {"cooking_time": 30}})
        assert brief.max_minutes == 30

    @pytest.mark.parametrize("value", [None, 0, -5, "soon"])
    def test_an_unusable_time_budget_is_none_not_zero(self, value):
        brief = PlanBrief.build({"plan_parameters": {"cooking_time": value}})
        assert brief.max_minutes is None

    def test_the_pantry_and_the_anchors_ride_along(self):
        state = PlanningState().merge(PlanningStateDelta(
            pantry_add=("zucchini",), anchors={"breakfast": "pie-1"},
        ))
        brief = PlanBrief.build({"_pantry": ["zucchini"]}, state)
        assert brief.pantry == ("zucchini",)
        assert brief.anchors == {"breakfast": "pie-1"}


class TestTheCalorieTarget:
    def test_a_stated_target_wins(self):
        assert PlanBrief.build({"calorie_target": 1800}).kcal_target == 1800

    def test_no_target_and_no_goal_means_no_target(self):
        """A number the member never set must not be reported as theirs — the
        weekly tracker's habit of rendering an invented 2000 with
        `source: "calorie target"` is exactly this mistake."""
        assert PlanBrief.build({}).kcal_target is None

    def test_a_goal_brings_the_default(self):
        """A goal IS the member asking to be planned against something."""
        brief = PlanBrief.build({"plan_parameters": {"goal": "weight_loss"}})
        assert brief.kcal_target == 2000.0

    def test_the_days_budget_is_split_across_meals_and_plates(self):
        """`PlanSpec.kcal_split` has been correct and unused since it was
        written. Without it a two-plate dinner is budgeted as two whole meals,
        which is how a "light" plan comes back at 3,000 calories."""
        spec = PlanSpec(num_days=1, meals=("breakfast", "lunch", "dinner"),
                        plates={"dinner": ("main", "dessert")})
        brief = PlanBrief.build({"calorie_target": 2100}, None, spec)
        assert brief.kcal_by_slot["breakfast"] == {"main": 700.0}
        dinner = brief.kcal_by_slot["dinner"]
        assert set(dinner) == {"main", "dessert"}
        assert sum(dinner.values()) == pytest.approx(700.0, abs=0.2)

    def test_no_target_means_no_split(self):
        spec = PlanSpec(num_days=1, meals=("breakfast", "lunch", "dinner"))
        assert PlanBrief.build({}, None, spec).kcal_by_slot == {}


class TestTheNutriScoreFloor:
    def test_a_quality_goal_sets_one(self):
        assert PlanBrief.build({"plan_parameters": {"goal": "balanced"}}).min_nutri_score == "c"

    def test_a_composition_goal_does_not(self):
        """A score floor would quietly exclude perfectly good high-protein
        dishes for an unrelated reason."""
        assert PlanBrief.build({"plan_parameters": {"goal": "high_protein"}}).min_nutri_score is None

    def test_a_stated_floor_wins(self):
        brief = PlanBrief.build({
            "min_nutri_score": "B", "plan_parameters": {"goal": "balanced"},
        })
        assert brief.min_nutri_score == "b"


# ── what a strategist may and may not do ─────────────────────────────────

class TestStrategyIsValidated:
    def test_a_real_facet_is_accepted(self):
        brief = PlanBrief().with_strategy({"moods": ["light"]}, VOCAB)
        assert brief.moods == ("light",)

    def test_an_invented_facet_is_dropped(self):
        """The search ANDs facet values and never relaxes an unknown one, so an
        invented mood does not narrow the search — it empties it, and the
        member is told no meals exist."""
        brief = PlanBrief().with_strategy({"moods": ["energising"]}, VOCAB)
        assert brief.moods == ()

    def test_a_mix_keeps_only_the_real_one(self):
        brief = PlanBrief().with_strategy({"cuisines": ["thai", "atlantean"]}, VOCAB)
        assert brief.cuisines == ("thai",)

    def test_no_vocabulary_means_no_facets_accepted(self):
        """With no live list there is no value that is safe to send."""
        assert PlanBrief().with_strategy({"moods": ["light"]}, {}).moods == ()

    def test_it_adds_to_what_was_already_standing(self):
        brief = PlanBrief(moods=("hearty",)).with_strategy({"moods": ["light"]}, VOCAB)
        assert set(brief.moods) == {"hearty", "light"}

    def test_a_claim_tag_the_corpus_carries_is_accepted(self):
        brief = PlanBrief().with_strategy({"claim_tags": ["high_protein"]}, VOCAB)
        assert "high_protein" in brief.claim_tags

    def test_an_alias_is_normalised(self):
        brief = PlanBrief().with_strategy({"claim_tags": ["low-carb"]}, VOCAB)
        assert brief.claim_tags == ("low_calorie",)

    def test_an_invented_claim_tag_is_dropped(self):
        assert PlanBrief().with_strategy({"claim_tags": ["keto_friendly"]}, VOCAB).claim_tags == ()


class TestStrategyCannotTouchSafety:
    """A reasoning step may decide HOW to search. It may not decide to drop an
    allergen."""

    def test_allergens_are_not_adjustable(self):
        brief = PlanBrief(allergens=("peanuts",)).with_strategy(
            {"allergens": [], "moods": ["light"]}, VOCAB)
        assert brief.allergens == ("peanuts",)

    def test_diet_is_not_adjustable(self):
        brief = PlanBrief(diet=("vegetarian",)).with_strategy(
            {"diet": ["omnivore"]}, VOCAB)
        assert brief.diet == ("vegetarian",)

    def test_a_relaxation_order_cannot_gain_a_step(self):
        brief = PlanBrief().with_strategy(
            {"relaxation_order": ["allergens", "moods"]}, VOCAB)
        assert "allergens" not in brief.relaxation_order

    def test_reordering_is_allowed_and_stays_complete(self):
        brief = PlanBrief().with_strategy(
            {"relaxation_order": ["cuisines", "moods"]}, VOCAB)
        assert brief.relaxation_order[:2] == ("cuisines", "moods")
        assert set(brief.relaxation_order) == set(DEFAULT_RELAXATION_ORDER)

    @pytest.mark.parametrize("kcal", [400, 9000, "lots", None, 0])
    def test_an_implausible_calorie_target_is_ignored(self, kcal):
        """A strategist proposing 400 or 9000 has made an arithmetic error, and
        a nutrition assistant should not build a plan around one."""
        assert PlanBrief().with_strategy({"kcal_target": kcal}, VOCAB).kcal_target is None

    def test_a_plausible_one_is_taken(self):
        assert PlanBrief().with_strategy({"kcal_target": 1800}, VOCAB).kcal_target == 1800.0

    def test_an_empty_proposal_changes_nothing(self):
        brief = PlanBrief(moods=("hearty",), kcal_target=1900.0)
        assert brief.with_strategy({}, VOCAB) is brief

    def test_a_rationale_is_kept_and_bounded(self):
        brief = PlanBrief().with_strategy({"rationale": "x" * 900}, VOCAB)
        assert len(brief.rationale) == 500


class TestTheVerifierContract:
    def test_to_requested_carries_every_checkable_constraint(self):
        brief = PlanBrief(
            allergens=("peanuts",), diet=("vegetarian",), moods=("light",),
            claim_tags=("high_protein",), kcal_target=2000.0, max_minutes=30,
            min_nutri_score="c", pantry=("zucchini",), anchors={"lunch": "r1"},
        )
        requested = brief.to_requested()
        for key in ("allergens", "diet", "moods", "tags", "kcal_target",
                    "max_minutes", "min_nutri_score", "pantry", "anchors"):
            assert key in requested, key

    def test_the_verifier_reports_on_what_the_brief_asked_for(self):
        """One definition, so the two lists cannot drift apart."""
        from models.session import DayPlan, Meal, MealCourse, MealPlan
        from services import plan_verifier

        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("lunch", [MealCourse("r1", "Soup", "carrot", "cook")]),
        ])], "one")
        brief = PlanBrief(allergens=("peanuts",), pantry=("zucchini",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        assert {c.name for c in report.checks} == {"allergens", "pantry"}

    def test_claim_tags_ride_the_tags_key_not_diet(self):
        """They are on ZERO recipes as diet tags: sending one as a diet filter
        empties every slot."""
        requested = PlanBrief(claim_tags=("high_protein",)).to_requested()
        assert requested["tags"] == ["high_protein"]
        assert requested["diet"] == []

    def test_facet_kwargs_splat_into_a_plan_meals_call(self):
        kwargs = PlanBrief(cuisines=("thai",), claim_tags=("high_protein",)).facet_kwargs()
        assert set(kwargs) == {"cuisines", "moods", "flavor_profiles",
                               "food_groups", "tags"}
        assert kwargs["cuisines"] == ["thai"]


class TestDescribe:
    def test_it_names_what_is_in_force(self):
        brief = PlanBrief(diet=("vegetarian",), allergens=("peanuts",),
                          moods=("light",), kcal_target=1800.0, pantry=("zucchini",))
        text = brief.describe()
        for word in ("vegetarian", "peanuts", "light", "1800", "zucchini"):
            assert word in text

    def test_an_empty_brief_says_so_rather_than_returning_nothing(self):
        assert PlanBrief().describe() == "no standing constraints"
