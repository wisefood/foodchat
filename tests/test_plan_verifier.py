"""
Measured, not declared.

Every constraint FoodChat reported came from the REQUEST: the ledger walked
`profile["diet"]` and rendered each value as satisfied because it had been
sent. That is a statement about what was asked, dressed as a statement about
what arrived — and the difference has already cost three real failures:

* 26 of 37 gateway dietary groups dropped before the fetch, reported
  `hard / satisfied` regardless;
* `low-carb` sent as a diet filter, present on zero recipes, so the search was
  guaranteed empty and the fallback plan still "honoured" it;
* the weekly tracker's invented 2000 kcal reported as the member's own target.

This module measures what is on the plates. These tests pin the measurements,
and — just as much — pin the honesty of the cases it CANNOT measure.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import RecipeEnrichment                       # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan    # noqa: E402
from services import plan_verifier as pv                          # noqa: E402


def _plate(rid, title, ingredients="", role="main"):
    return MealCourse(recipe_id=rid, title=title,
                      ingredients=ingredients or f"{title} things",
                      directions="cook", role=role)


def _plan(*plates_per_day):
    """One day per argument, each a list of (rid, title[, ingredients])."""
    days = []
    for index, plates in enumerate(plates_per_day, start=1):
        meals = [
            Meal(slot, [_plate(*spec)])
            for slot, spec in zip(("breakfast", "lunch", "dinner"), plates)
        ]
        days.append(DayPlan(day=index, meals=meals))
    return MealPlan.from_days(days, "test")


def _rich(rid, **kw):
    return RecipeEnrichment(recipe_id=rid, title=kw.pop("title", rid), **kw)


# ── allergens ─────────────────────────────────────────────────────────────

class TestAllergens:
    def test_a_labelled_allergen_is_caught(self):
        plan = _plan([("a", "Toast"), ("b", "Satay"), ("c", "Stew")])
        rich = {"b": _rich("b", allergens=["peanuts"])}
        check = pv.verify(plan, {"allergens": ["peanuts"]}, rich).get("allergens")
        assert check.status == pv.FAILED
        assert check.offenders == ("b",)

    def test_an_unlabelled_allergen_is_caught_in_the_ingredients(self):
        """The corpus labels unevenly. Trusting the label alone means trusting
        a recipe nobody got round to labelling."""
        plan = _plan([("a", "Toast"), ("b", "Thai curry", "peanuts, coconut"), ("c", "Stew")])
        check = pv.verify(plan, {"allergens": ["peanuts"]}, {}).get("allergens")
        assert check.status == pv.FAILED and check.offenders == ("b",)

    def test_a_clean_plan_passes_and_says_how_many_it_looked_at(self):
        plan = _plan([("a", "Toast"), ("b", "Soup"), ("c", "Stew")])
        check = pv.verify(plan, {"allergens": ["peanuts"]}, {}).get("allergens")
        assert check.status == pv.PASSED
        assert "3 dishes" in check.detail

    def test_an_allergen_failure_blocks(self):
        plan = _plan([("a", "Satay", "peanuts")])
        report = pv.verify(plan, {"allergens": ["peanuts"]}, {})
        assert report.blocking and report.blocking[0].name == "allergens"

    def test_no_allergens_asked_means_no_row(self):
        assert pv.verify(_plan([("a", "Toast")]), {}, {}).get("allergens") is None


# ── diet: the check the vegetarian failure needed ─────────────────────────

class TestDiet:
    def test_a_dish_without_the_tag_is_caught(self):
        """The plan said `vegetarian: satisfied` because the word had been
        sent. Nothing looked at what came back."""
        plan = _plan([("a", "Salad"), ("b", "Beef stew"), ("c", "Risotto")])
        rich = {
            "a": _rich("a", diet_tags=["vegetarian"]),
            "b": _rich("b", diet_tags=["gluten_free"]),
            "c": _rich("c", diet_tags=["vegetarian"]),
        }
        check = pv.verify(plan, {"diet": ["vegetarian"]}, rich).get("diet")
        assert check.status == pv.FAILED
        assert check.offenders == ("b",)

    def test_all_labelled_and_correct_passes(self):
        plan = _plan([("a", "Salad"), ("b", "Soup"), ("c", "Risotto")])
        rich = {r: _rich(r, diet_tags=["vegetarian"]) for r in ("a", "b", "c")}
        assert pv.verify(plan, {"diet": ["vegetarian"]}, rich).get("diet").status == pv.PASSED

    def test_a_variant_tag_satisfies_its_base(self):
        """The corpus carries `vegetarian_or_vegan`; a member asking for
        vegetarian is served by it."""
        plan = _plan([("a", "Salad")])
        rich = {"a": _rich("a", diet_tags=["vegetarian_or_vegan"])}
        assert pv.verify(plan, {"diet": ["vegetarian"]}, rich).get("diet").status == pv.PASSED

    def test_unlabelled_dishes_are_partial_not_violations(self):
        """Treating "unlabelled" as "not vegetarian" would condemn most of a
        corpus that annotates unevenly."""
        plan = _plan([("a", "Salad"), ("b", "Mystery"), ("c", "Risotto")])
        rich = {"a": _rich("a", diet_tags=["vegetarian"]),
                "c": _rich("c", diet_tags=["vegetarian"])}
        check = pv.verify(plan, {"diet": ["vegetarian"]}, rich).get("diet")
        assert check.status == pv.PARTIAL
        assert "2 of 3" in check.detail

    def test_nothing_labelled_at_all_is_unknown_not_satisfied(self):
        """The old ledger said satisfied here. It knew nothing."""
        plan = _plan([("a", "Salad"), ("b", "Soup"), ("c", "Stew")])
        check = pv.verify(plan, {"diet": ["vegetarian"]}, {}).get("diet")
        assert check.status == pv.UNKNOWN
        assert "carry dietary labels to check against" in check.detail

    def test_a_diet_failure_blocks(self):
        plan = _plan([("a", "Beef stew")])
        rich = {"a": _rich("a", diet_tags=["omnivore"])}
        assert pv.verify(plan, {"diet": ["vegetarian"]}, rich).blocking


# ── claims, calories, time, score ─────────────────────────────────────────

class TestClaims:
    def test_partial_is_the_normal_answer(self):
        """"High protein" over a week means the plan leans that way, not that
        every dish qualifies."""
        plan = _plan([("a", "A"), ("b", "B"), ("c", "C")])
        rich = {"a": _rich("a", tags=["high_protein"]), "b": _rich("b", tags=[]),
                "c": _rich("c", tags=["high_protein"])}
        check = pv.verify(plan, {"tags": ["high_protein"]}, rich).get("claims")
        assert check.status == pv.PARTIAL and "2 of 3" in check.detail

    def test_none_matching_is_a_failure(self):
        plan = _plan([("a", "A")])
        rich = {"a": _rich("a", tags=["quick"])}
        assert pv.verify(plan, {"tags": ["high_protein"]}, rich).get("claims").status == pv.FAILED

    def test_a_claim_never_blocks_a_plan(self):
        plan = _plan([("a", "A")])
        rich = {"a": _rich("a", tags=[])}
        assert not pv.verify(plan, {"tags": ["high_protein"]}, rich).blocking

    def test_slugs_read_as_words(self):
        plan = _plan([("a", "A")])
        check = pv.verify(plan, {"tags": ["30_minutes_or_less"]},
                          {"a": _rich("a", tags=[])}).get("claims")
        assert "30 minutes or less" in check.detail


class TestCalories:
    def test_a_day_within_tolerance_passes(self):
        plan = _plan([("a", "A"), ("b", "B"), ("c", "C")])
        rich = {r: _rich(r, kcal=700) for r in ("a", "b", "c")}
        check = pv.verify(plan, {"kcal_target": 2000}, rich).get("calories")
        assert check.status == pv.PASSED

    def test_a_day_well_over_is_reported_with_the_number(self):
        plan = _plan([("a", "A"), ("b", "B"), ("c", "C")])
        rich = {r: _rich(r, kcal=1500) for r in ("a", "b", "c")}
        check = pv.verify(plan, {"kcal_target": 2000}, rich).get("calories")
        assert check.status == pv.FAILED
        assert "2500 kcal over" in check.detail

    def test_it_scores_each_day_separately(self):
        plan = _plan(
            [("a", "A"), ("b", "B"), ("c", "C")],
            [("d", "D"), ("e", "E"), ("f", "F")],
        )
        rich = {r: _rich(r, kcal=700) for r in ("a", "b", "c")}
        rich |= {r: _rich(r, kcal=2000) for r in ("d", "e", "f")}
        check = pv.verify(plan, {"kcal_target": 2000}, rich).get("calories")
        assert check.status == pv.FAILED
        assert "1 of 2 days" in check.detail and "day 2" in check.detail

    def test_a_partial_day_says_the_total_is_a_floor(self):
        """Otherwise a plan with one unprofiled dish reads as under target when
        it may not be."""
        plan = _plan([("a", "A"), ("b", "B"), ("c", "C")])
        rich = {"a": _rich("a", kcal=700), "b": _rich("b", kcal=700)}
        check = pv.verify(plan, {"kcal_target": 1400}, rich).get("calories")
        assert "floor" in check.detail

    def test_no_macros_at_all_is_unknown_not_zero(self):
        """"0 kcal, 100% under target" is a measurement of nothing presented as
        a measurement of the plan."""
        plan = _plan([("a", "A"), ("b", "B"), ("c", "C")])
        check = pv.verify(plan, {"kcal_target": 2000}, {}).get("calories")
        assert check.status == pv.UNKNOWN

    @pytest.mark.parametrize("target", [None, 0, "", "not a number"])
    def test_an_unusable_target_produces_no_row(self, target):
        plan = _plan([("a", "A")])
        assert pv.verify(plan, {"kcal_target": target}, {}).get("calories") is None


class TestTimeAndScore:
    def test_a_dish_over_the_limit_is_caught(self):
        plan = _plan([("a", "A"), ("b", "B")])
        rich = {"a": _rich("a", duration=20), "b": _rich("b", duration=95)}
        check = pv.verify(plan, {"max_minutes": 30}, rich).get("cooking time")
        assert check.status == pv.FAILED and check.offenders == ("b",)
        assert "95" in check.detail

    def test_untimed_dishes_do_not_count_as_passing(self):
        plan = _plan([("a", "A")])
        check = pv.verify(plan, {"max_minutes": 30}, {}).get("cooking time")
        assert check.status == pv.UNKNOWN

    def test_a_worse_nutri_score_is_caught(self):
        plan = _plan([("a", "A"), ("b", "B")])
        rich = {"a": _rich("a", nutri_score_label="B"),
                "b": _rich("b", nutri_score_label="D")}
        check = pv.verify(plan, {"min_nutri_score": "c"}, rich).get("nutri-score")
        assert check.status == pv.FAILED and check.offenders == ("b",)

    def test_the_boundary_counts_as_meeting_it(self):
        plan = _plan([("a", "A")])
        rich = {"a": _rich("a", nutri_score_label="C")}
        assert pv.verify(plan, {"min_nutri_score": "c"}, rich).get("nutri-score").status == pv.PASSED


class TestPantryAndPicks:
    def test_partial_pantry_use_is_reported_as_partial(self):
        plan = _plan([("a", "Zucchini fritters", "zucchini, egg"), ("b", "Soup", "carrot")])
        check = pv.verify(plan, {"pantry": ["zucchini", "spinach"]}, {}).get("pantry")
        assert check.status == pv.PARTIAL
        assert "spinach" in check.detail

    def test_using_nothing_is_a_failure(self):
        plan = _plan([("a", "Soup", "carrot, stock")])
        assert pv.verify(plan, {"pantry": ["zucchini"]}, {}).get("pantry").status == pv.FAILED

    def test_a_named_dish_that_vanished_is_caught(self):
        """The apple pie: served, then regenerated away a turn later, with
        nothing reporting that it had gone."""
        plan = _plan([("a", "Muffins"), ("b", "Soup"), ("c", "Stew")])
        check = pv.verify(plan, {"anchors": {"breakfast": "pie-1"}}, {}).get("your picks")
        assert check.status == pv.FAILED and "breakfast" in check.detail

    def test_a_named_dish_that_is_present_passes(self):
        plan = _plan([("pie-1", "Apple pie"), ("b", "Soup"), ("c", "Stew")])
        check = pv.verify(plan, {"anchors": {"breakfast": "pie-1"}}, {}).get("your picks")
        assert check.status == pv.PASSED


# ── the honesty of what it cannot measure ────────────────────────────────

class TestFacetsAreNotClaimed:
    def test_a_facet_is_reported_unverified_never_satisfied(self):
        """The four families are Elasticsearch-only annotations; the details
        endpoint reads Neo4j and never sees them. Claiming `satisfied` would be
        the same lie the dietary-group ledger told, in a new place."""
        plan = _plan([("a", "Pad thai")])
        check = pv.verify(plan, {"cuisines": ["thai"], "moods": ["light"]}, {}).get(
            "taste preferences")
        assert check.status == pv.UNVERIFIED
        assert "thai" in check.detail and "light" in check.detail

    def test_it_says_why_rather_than_just_shrugging(self):
        plan = _plan([("a", "A")])
        check = pv.verify(plan, {"moods": ["hearty"]}, {}).get("taste preferences")
        assert "no annotation to check" in check.detail

    def test_no_facets_asked_means_no_row(self):
        assert pv.verify(_plan([("a", "A")]), {}, {}).get("taste preferences") is None

    def test_an_unverified_facet_is_not_a_failure(self):
        plan = _plan([("a", "A")])
        report = pv.verify(plan, {"cuisines": ["thai"]}, {})
        assert not report.failed and not report.blocking


# ── the report itself ─────────────────────────────────────────────────────

class TestTheReport:
    def test_it_only_reports_what_was_asked_for(self):
        plan = _plan([("a", "A")])
        assert pv.verify(plan, {}, {}).checks == []

    def test_offenders_are_deduplicated_across_checks(self):
        plan = _plan([("a", "Slow satay", "peanuts")])
        rich = {"a": _rich("a", duration=200, allergens=["peanuts"])}
        report = pv.verify(plan, {"allergens": ["peanuts"], "max_minutes": 30}, rich)
        assert report.offenders == ["a"]

    def test_the_ledger_rows_use_the_states_the_ui_renders(self):
        plan = _plan([("a", "Beef", "beef")])
        rich = {"a": _rich("a", diet_tags=["omnivore"])}
        rows = pv.verify(plan, {"diet": ["vegetarian"], "cuisines": ["thai"]},
                         rich).as_ledger_rows()
        states = {row["status"] for row in rows}
        assert states <= {"satisfied", "violated", "relaxed", "unsupported"}

    def test_every_row_says_it_was_measured(self):
        """The old ledger's `source` was the profile field it came from, which
        is what made it a statement about the request."""
        plan = _plan([("a", "A")])
        rows = pv.verify(plan, {"allergens": ["peanuts"]}, {}).as_ledger_rows()
        assert all(row["source"] == "measured on the plan" for row in rows)

    def test_safety_rows_are_hard_and_the_rest_are_soft(self):
        plan = _plan([("a", "A")])
        rows = pv.verify(plan, {
            "allergens": ["peanuts"], "diet": ["vegetarian"],
            "max_minutes": 30, "pantry": ["zucchini"],
        }, {}).as_ledger_rows()
        by_name = {row["constraint"]: row["type"] for row in rows}
        assert by_name["allergens"] == "hard" and by_name["diet"] == "hard"
        assert by_name["cooking time"] == "soft" and by_name["pantry"] == "soft"

    def test_describe_is_never_empty(self):
        assert pv.describe(pv.verify(_plan([("a", "A")]), {}, {})) == "nothing to verify"
        assert "allergens" in pv.describe(
            pv.verify(_plan([("a", "A")]), {"allergens": ["nuts"]}, {}))

    def test_it_reads_every_day_and_every_plate(self):
        """A verifier that checks day 1's mains is the bug it exists to catch."""
        plan = MealPlan.from_days([
            DayPlan(day=1, meals=[Meal("dinner", [_plate("a", "Main"),
                                                  _plate("b", "Satay", "peanuts", "side")])]),
            DayPlan(day=2, meals=[Meal("dinner", [_plate("c", "Other")])]),
        ], "two days")
        check = pv.verify(plan, {"allergens": ["peanuts"]}, {}).get("allergens")
        assert check.status == pv.FAILED and check.of == 3

    def test_it_survives_a_legacy_three_course_plan(self):
        from models.recipe import CandidateRecipe

        plan = MealPlan.from_courses([
            CandidateRecipe(recipe_id=r, title=r, ingredients="x", directions="y")
            for r in ("a", "b", "c")
        ], "legacy", {})
        assert pv.verify(plan, {"allergens": ["peanuts"]}, {}).get("allergens").of == 3
