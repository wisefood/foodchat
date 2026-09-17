"""
A constraint the plan did not enforce must never be reported as satisfied.

`normalize_diet_tags` drops 26 of the gateway's 37 dietary groups — RecipeWrangler
has no diet tag for `peanut_free`, `halal`, `kosher`, `keto`, `low_sodium` and the
rest. The ledger then rendered EVERY raw profile diet value as
`type: hard, status: satisfied`. So a member who selected `peanut_free` in their
profile was shown a plan header asserting a peanut-free guarantee, with no filter
behind it and no allergen backstop, because the backstop keys on plain-English
allergen names and never saw the slug.

Nothing here can invent an upstream filter. What it can do is stop pretending,
and route the free-from slugs to the ingredient screen that does cover them.
"""

from __future__ import annotations

import pytest

from services.candidates_client import (
    FREE_FROM_TO_ALLERGEN,
    classify_diet_tags,
    free_from_allergens,
    normalize_diet_tags,
    screening_allergens,
)
from services.transparency import constraints_ledger


def _diet_rows(profile):
    return [r for r in constraints_ledger(profile) if r["source"] == "dietary group"]


class TestNeverClaimAnUnenforcedConstraint:
    @pytest.mark.parametrize("slug", [
        "peanut_free", "egg_free", "soy_free", "shellfish_free", "fish_free",
        "sesame_free", "halal", "kosher", "jain", "keto", "paleo", "whole30",
        "low_sodium", "sugar_free", "no_added_sugar", "high_fiber",
        "low_cholesterol", "low_calorie", "diabetic_friendly", "raw_vegan",
        "plant_based", "lacto_vegetarian", "ovo_vegetarian",
        "lacto_ovo_vegetarian", "buddhist_vegetarian",
    ])
    def test_an_unfilterable_diet_is_never_satisfied(self, slug):
        rows = _diet_rows({"diet": [slug]})
        assert rows, f"{slug} produced no ledger row at all"
        assert rows[0]["status"] == "unsupported", (
            f"{slug} claims {rows[0]['status']!r} with no filter behind it"
        )

    @pytest.mark.parametrize("slug", ["vegetarian", "vegan", "pescatarian",
                                      "gluten_free", "dairy_free", "nut_free"])
    def test_a_real_filter_still_reports_satisfied(self, slug):
        rows = _diet_rows({"diet": [slug]})
        assert rows[0]["status"] == "satisfied", slug
        assert normalize_diet_tags([slug]), f"{slug} should be filterable"

    def test_a_non_restrictive_label_is_reported_as_the_label_it_is(self):
        """"omnivore" is the absence of a restriction, so it is never a
        satisfied HARD constraint.

        It does keep a row. I had it dropped entirely — a constraints ledger
        listing a non-constraint is a chip the member cannot act on — but
        every value keeping a row is the stronger invariant: nothing a member
        set disappears without an answer, and "a description of how you eat"
        IS the answer.
        """
        rows = constraints_ledger({"diet": ["omnivore"], "preferences": []})
        row = next(r for r in rows if r["constraint"] == "omnivore")

        assert row["type"] == "soft" and row["status"] == "satisfied"
        assert "not a recipe filter" in row["detail"]
        # And the classifier agrees with the row: neither forwarded as a
        # filter nor reported as something we failed to enforce.
        assert classify_diet_tags(["omnivore"]) == ([], [])
    def test_the_row_explains_itself(self):
        rows = _diet_rows({"diet": ["halal"]})
        assert "no filter for this" in rows[0]["detail"]

    def test_a_mixed_profile_reports_each_row_on_its_own_merits(self):
        rows = {r["constraint"]: r["status"]
                for r in _diet_rows({"diet": ["vegetarian", "peanut_free", "omnivore"]})}
        assert rows["vegetarian"] == "satisfied"
        assert rows["peanut_free"] == "unsupported"


class TestFreeFromReachesTheBackstop:
    """It cannot become an RW filter, but it can become an ingredient screen."""

    @pytest.mark.parametrize("slug,allergen", [
        ("peanut_free", "peanuts"),
        ("egg_free", "eggs"),
        ("shellfish_free", "shellfish"),
        ("soy_free", "soy"),
        ("fish_free", "fish"),
        ("sesame_free", "sesame"),
        ("gluten_free", "gluten"),
        ("dairy_free", "dairy"),
    ])
    def test_the_slug_implies_its_allergen(self, slug, allergen):
        assert free_from_allergens([slug]) == [allergen]

    def test_screening_unions_stated_and_implied(self):
        got = screening_allergens({"allergies": ["nuts"], "diet": ["peanut_free"]})
        assert "nuts" in got and "peanuts" in got

    def test_it_does_not_duplicate_what_the_member_already_stated(self):
        got = screening_allergens({"allergies": ["peanuts"], "diet": ["peanut_free"]})
        assert got.count("peanuts") == 1

    def test_every_mapped_allergen_is_expandable(self):
        """A mapping to a name the synonym table doesn't know would screen
        nothing — silently reintroducing the hole this closes."""
        from services.candidates_client import ALLERGEN_SYNONYMS

        for slug, allergen in FREE_FROM_TO_ALLERGEN.items():
            assert allergen in ALLERGEN_SYNONYMS, f"{slug} -> {allergen} not expandable"

    def test_a_plate_containing_the_allergen_is_caught(self):
        from services.candidates_client import allergen_conflict

        allergens = screening_allergens({"allergies": [], "diet": ["peanut_free"]})
        assert allergen_conflict("Satay Noodles peanut butter noodles", allergens)
        assert not allergen_conflict("Tomato Soup tomato basil", allergens)


class TestSplitLedgerLeavesItAlone:
    """Whether the REPLY mentions an unsupported value depends on whether
    anything else covers it — which is the only question that separates
    over-alarming a member from leaving them uninformed."""

    def test_one_with_a_backstop_is_neither_honored_nor_apologised_for(self):
        """Claiming it honoured is the lie this status exists to stop. Saying
        "couldn't honour peanut_free" would over-alarm a member whose peanuts
        ARE screened out of every plate. The ledger row carries the nuance."""
        from services.transparency import split_ledger

        honored, not_honored = split_ledger([
            {"constraint": "vegetarian", "status": "satisfied"},
            {"constraint": "peanut_free", "status": "unsupported",
             "covered_by": "peanuts"},
        ])
        assert honored == ["vegetarian"]
        assert not_honored == []

    def test_one_with_nothing_behind_it_reaches_the_reply(self):
        """`halal` has no filter AND no backstop. Leaving it out of both lists
        means the only place the member could learn that is a chip — so the
        reply lists it as honoured by omission, which is the same failure in
        its quiet direction."""
        from services.transparency import split_ledger

        honored, not_honored = split_ledger([
            {"constraint": "vegetarian", "status": "satisfied"},
            {"constraint": "halal", "status": "unsupported"},
        ])
        assert honored == ["vegetarian"]
        assert not_honored == ["halal"]

    def test_the_rows_the_ledger_builds_carry_that_distinction(self):
        """End to end, not on hand-written rows: the classifier decides which
        of the two a real profile value is."""
        from services.transparency import constraints_ledger, split_ledger

        _honored, not_honored = split_ledger(constraints_ledger(
            {"diet": ["peanut_free", "halal"], "preferences": []}
        ))
        assert not_honored == ["halal"]
