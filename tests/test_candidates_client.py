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
        assert normalize_diet_tags(["Gluten-Free", "high_protein"]) == ["gluten_free", "high-protein"]

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

