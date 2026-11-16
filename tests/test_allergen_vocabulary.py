"""Canonical allergen identifiers reach every FoodChat safety path.

The gateway folds a profile's allergies to RecipeWrangler's identifiers
(`milk`, `tree_nut`, `crustacean_shellfish` ...). FoodChat's backstops were
keyed on the spellings the old picker stored ("dairy", "nuts", "shellfish");
`ALLERGEN_ALIASES` folds one onto the other. These tests hold every EU
identifier to having ingredient words behind it, every safety path (text
screen, plan verifier, seed gate) to reading both spellings, and every
sentence a member reads to saying "tree nut", not `tree_nut`.
"""
import pytest

from services import plan_verifier as pv
from services.candidates_client import (
    ALLERGEN_SYNONYMS,
    FREE_FROM_TO_ALLERGEN,
    allergen_conflict,
    allergen_label_match,
    canonical_allergen,
    display_allergen,
)
from services.transparency import constraints_ledger

CANONICAL = (
    "gluten", "crustacean_shellfish", "egg", "fish", "peanut", "soy", "milk",
    "tree_nut", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
    "wheat",
)


class TestCanonicalIdsExpand:
    @pytest.mark.parametrize("cid", CANONICAL)
    def test_every_canonical_id_has_ingredient_words(self, cid):
        assert ALLERGEN_SYNONYMS.get(canonical_allergen(cid)), f"{cid} would screen nothing"

    @pytest.mark.parametrize("cid, text", [
        ("tree_nut", "Almond and pear tart"),
        ("crustacean_shellfish", "Garlic prawns with rice"),
        ("molluscs", "Steamed mussels in white wine"),
        ("milk", "Mozzarella salad"),
        ("egg", "Lemon meringue pie"),
        ("sulphites", "Dried apricots (sodium metabisulphite)"),
        ("celery", "Celeriac remoulade"),
        ("lupin", "Lupini bean snack"),
        ("mustard", "Chicken with a Dijon glaze"),
        ("wheat", "Bulgur salad"),
        ("soy", "Miso-glazed aubergine"),
        ("tree_nut", "Marzipan stollen"),
    ])
    def test_a_canonical_profile_still_catches_the_dish(self, cid, text):
        assert allergen_conflict(text, [cid])

    def test_legacy_spellings_keep_working(self):
        assert allergen_conflict("Satay noodles with peanut butter", ["nuts"])
        assert allergen_conflict("Cheddar omelette", ["dairy"]) and allergen_conflict("Brie tart", ["milk"])

    def test_free_from_slugs_still_expand(self):
        for slug, allergen in FREE_FROM_TO_ALLERGEN.items():
            assert allergen in ALLERGEN_SYNONYMS, f"{slug} -> {allergen}"


    @pytest.mark.parametrize("profile, label", [
        ("nuts", "tree_nut"), ("milk", "dairy"), ("dairy", "milk"),
        ("shellfish", "crustacean_shellfish"), ("crustacean_shellfish", "shellfish"),
        ("gluten", "wheat"), ("wheat", "gluten"), ("sulfites", "sulphites"),
    ])
    def test_labels_meet_across_spellings(self, profile, label):
        assert allergen_label_match(profile, [label])

    def test_a_crustacean_allergy_is_not_refused_mussels_by_label(self):
        assert not allergen_label_match("crustacean_shellfish", ["molluscs"])


class TestTheVerifierReadsBothSpellings:
    def _plan(self):
        from test_plan_verifier import _plan, _rich
        return _plan, _rich

    def test_a_legacy_profile_meets_a_canonical_label(self):
        _plan, _rich = self._plan()
        plan = _plan([("a", "Toast"), ("b", "Romesco"), ("c", "Stew")])
        rich = {"b": _rich("b", allergens=["tree_nut"])}
        check = pv.verify(plan, {"allergens": ["nuts"]}, rich).get("allergens")
        assert check.status == pv.FAILED and check.offenders == ("b",)

    def test_a_canonical_profile_meets_an_unlabelled_ingredient(self):
        _plan, _rich = self._plan()
        plan = _plan([("a", "Toast"), ("b", "Pesto pasta", "basil, pine nuts, parmesan"), ("c", "Stew")])
        check = pv.verify(plan, {"allergens": ["tree_nut"]}, {}).get("allergens")
        assert check.status == pv.FAILED and check.offenders == ("b",)

    def test_the_report_reads_as_words(self):
        _plan, _rich = self._plan()
        plan = _plan([("a", "Toast"), ("b", "Soup")])
        check = pv.verify(plan, {"allergens": ["tree_nut"]}, {}).get("allergens")
        assert check.status == pv.PASSED and "tree nut" in check.detail and "tree_nut" not in check.detail


class TestDisplay:
    def test_identifiers_read_as_words(self):
        assert display_allergen("tree_nut") == "tree nut"
        assert display_allergen("crustacean_shellfish") == "crustacean shellfish"
        assert display_allergen("peanuts") == "peanuts"

    def test_the_ledger_does_not_print_identifiers(self):
        rows = constraints_ledger({"allergies": ["tree_nut", "peanuts"]})
        constraints = {row["constraint"] for row in rows}
        assert "no tree nut" in constraints and "no peanuts" in constraints

    def test_the_no_plan_message_does_not_print_identifiers(self):
        from services.chat_service import no_plan_message
        text = no_plan_message({"allergies": ["crustacean_shellfish"]})
        assert "crustacean shellfish" in text and "crustacean_shellfish" not in text
