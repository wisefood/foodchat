"""RecipeCandidatesClient — payload construction, response parsing, tag mapping."""

import json

import httpx
import pytest

from services import candidates_client as cc
from services.candidates_client import (
    RecipeCandidatesClient,
    allergen_conflict,
    normalize_diet_tags,
)


class TestNormalizeDietTags:
    def test_maps_known_variants(self):
        assert normalize_diet_tags(["Gluten-Free", "nut-free"]) == ["gluten_free", "nut_free"]

    def test_drops_nutrition_claims_that_no_recipe_carries(self):
        # high-protein/low-carb/low-fat live on RecipeWrangler's claim field,
        # not diet_tags — zero recipes carry them, so as a filter they empty
        # every slot. They were mapped straight through until the corpus was
        # censused.
        assert normalize_diet_tags(["high_protein", "low-carb", "low_fat"]) == []

    def test_drops_non_restrictive_labels(self):
        assert normalize_diet_tags(["omnivore", "mediterranean", "balanced"]) == []

    def test_drops_unknown_tags(self):
        assert normalize_diet_tags(["keto-carnivore-hybrid"]) == []

    def test_accepts_string_and_none(self):
        assert normalize_diet_tags("vegan") == ["vegan"]
        assert normalize_diet_tags(None) == []


class _CapturingClient:
    """Replaces httpx.Client: captures the request, returns a canned response."""

    captured: dict = {}
    response_payload: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        _CapturingClient.captured = {"url": url, "json": json}
        request = httpx.Request("POST", url)
        return httpx.Response(200, json=_CapturingClient.response_payload, request=request)


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(cc.httpx, "Client", _CapturingClient)
    _CapturingClient.captured = {}
    _CapturingClient.response_payload = {"results": {}}
    return _CapturingClient


class TestAllergenConflict:
    """Synonym-expanded, word-boundary allergen matching."""

    def test_category_expands_to_specific_ingredients(self):
        # The live incident: "tree nuts" allergy vs an almond dish RW tagged nut_free
        assert allergen_conflict("Almond crumbed chicken", ["Tree Nuts"]) == "almond"
        assert allergen_conflict("garlic prawns with rice", ["Shellfish"]) == "prawn"

    def test_direct_allergen_name_matches(self):
        assert allergen_conflict("peanut butter cookies", ["peanuts"]) == "peanut"

    def test_word_boundaries_prevent_false_positives(self):
        # "eggplant" must not trip an egg allergy; "creamy" must not trip dairy
        assert allergen_conflict("grilled eggplant with creamy tahini-free dressing",
                                 ["eggs"]) is None
        assert allergen_conflict("creamy-style oat drink", ["dairy"]) is None

    def test_plural_forms_match(self):
        assert allergen_conflict("topped with toasted almonds", ["tree nuts"]) == "almond"

    def test_no_allergies_never_conflicts(self):
        assert allergen_conflict("almond walnut shrimp", []) is None


class TestGatewayAllergenNames:
    """The gateway writes EU-list slugs, not the synonym table's keys.

    Live: a member allergic to `milk` was planned yoghurt, and one allergic to
    `crustacean_shellfish` was told a shrimp salad was clear — the backstop
    searched for the slug itself and nothing else.
    """

    @pytest.mark.parametrize("allergy, text, term", [
        ("milk", "veggie burrito bowl with yoghurt", "yoghurt"),
        ("milk", "cheese and crackers", "cheese"),
        ("crustacean_shellfish", "shrimp and avocado salad", "shrimp"),
        ("molluscs", "steamed mussels in white wine", "mussel"),
        ("peanut", "chicken with peanuts", "peanut"),
        ("egg", "scrambled eggs on toast", "egg"),
        ("tree_nut", "pesto with cashews", "cashew"),
        ("soybeans", "crispy tofu stir fry", "tofu"),
        ("sesame_seeds", "houmous with tahini", "tahini"),
    ])
    def test_a_gateway_name_reaches_the_ingredients(self, allergy, text, term):
        assert allergen_conflict(text, [allergy]) == term

    @pytest.mark.parametrize("name", [
        "milk", "egg", "eggs", "peanut", "peanuts", "tree_nut", "tree_nuts", "nuts",
        "crustacean_shellfish", "crustaceans", "molluscs", "shellfish", "fish",
        "soy", "soybeans", "sesame", "sesame_seeds", "gluten", "celery", "lupin",
    ])
    def test_every_known_name_expands_beyond_itself(self, name):
        """A name that only matches its own spelling is the bug coming back."""
        assert cc.canonical_allergen(name) in cc.ALLERGEN_SYNONYMS

    def test_every_alias_points_at_a_real_key(self):
        assert set(cc.ALLERGEN_ALIASES.values()) <= set(cc.ALLERGEN_SYNONYMS)

    def test_a_crustacean_allergy_does_not_refuse_molluscs(self):
        assert allergen_conflict("grilled squid with lemon", ["crustacean_shellfish"]) is None
        assert allergen_conflict("grilled squid with lemon", ["shellfish"]) == "squid"

    def test_plant_dairy_is_not_milk(self):
        assert allergen_conflict("coconut milk curry with peanut butter", ["milk"]) is None
        assert allergen_conflict("coconut milk curry with peanut butter", ["peanut"]) == "peanut"
        assert allergen_conflict("oat milk porridge with butter", ["dairy"]) == "butter"

    def test_an_unknown_allergen_is_still_searched_by_name(self):
        assert allergen_conflict("mustard-glazed carrots", ["Mustard"]) == "mustard"
        assert allergen_conflict("kiwi_fruit salad with kiwi fruit", ["kiwi_fruit"])

    def test_a_recipe_label_is_compared_canonically(self):
        assert cc.allergen_label_match("milk", ["dairy"])
        assert cc.allergen_label_match("crustacean_shellfish", ["shellfish"])
        assert cc.allergen_label_match("shellfish", ["mollusc"])
        assert not cc.allergen_label_match("crustacean_shellfish", ["molluscs"])
        assert not cc.allergen_label_match("milk", ["egg"])

    def test_the_verifier_no_longer_passes_yoghurt_for_a_milk_allergy(self):
        from types import SimpleNamespace

        from services import plan_verifier

        plates = [SimpleNamespace(recipe_id="r1", title="Burrito bowl", ingredients="rice, yoghurt")]
        check = plan_verifier._check_allergens(plates, {"allergens": ["milk"]}, {})

        assert check.status == plan_verifier.FAILED and check.offenders == ("r1",)


class TestAllergenBackstopMoved:
    """The backstop now runs in `plan_client.to_candidates`.

    It used to live inside `fetch_candidates`, which is gone: every slot pool
    comes from `/api/v2/tools/plan_meals` now. The check itself was not dropped
    — the reason for it stands, since the corpus has tagged almond dishes
    `nut_free` in production — it just moved to where the candidates arrive.
    See `test_plan_client.py::TestAllergenBackstop`.
    """

    def test_the_matcher_still_lives_here(self):
        """`allergen_conflict` stays in this module; only its caller moved."""
        from services.candidates_client import allergen_conflict

        assert allergen_conflict("almond cake", ["tree nuts"])
        assert allergen_conflict("green salad", ["tree nuts"]) is None

