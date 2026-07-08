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


class TestFetchCandidates:
    def test_builds_contractual_payload(self, capture):
        client = RecipeCandidatesClient(base_url="http://rw.test")
        client.fetch_candidates(
            allergens=["peanuts"],
            diet=["vegetarian", "omnivore"],
            include_ingredients=["chickpeas"],
            exclude_ingredients=["olives"],
            exclude_recipe_ids=["r-1"],
            limit_per_slot=4,
            randomize=True,
        )
        sent = capture.captured
        assert sent["url"] == "http://rw.test/api/v1/recipes/foodchat_candidates"
        assert sent["json"]["user_profile"] == {"allergies": ["peanuts"], "diet": ["vegetarian"]}
        assert sent["json"]["constraints"]["exclude_recipe_ids"] == ["r-1"]
        assert sent["json"]["quotas"] == {"breakfast": 4, "lunch": 4, "dinner": 4}
        assert sent["json"]["randomize"] is True

    def test_parses_slots_into_typed_candidates(self, capture):
        capture.response_payload = {
            "results": {
                "breakfast": [{"recipe_id": 7, "title": "Oats", "ingredients": "oats", "directions": "cook"}],
                "lunch": [],
                # dinner key missing entirely — must still yield an (empty) slot
            }
        }
        result = RecipeCandidatesClient(base_url="http://rw.test").fetch_candidates()
        assert set(result.keys()) == {"breakfast", "lunch", "dinner"}
        assert result["breakfast"][0].recipe_id == "7"  # coerced to str
        assert result["breakfast"][0].title == "Oats"
        assert result["lunch"] == [] and result["dinner"] == []


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


class TestAllergenBackstop:
    """fetch_candidates drops candidates whose text contradicts upstream filters."""

    def test_poisoned_tag_candidate_is_dropped(self, capture):
        capture.response_payload = {
            "results": {
                "breakfast": [
                    # RW believed this was nut_free (broken graph tags)
                    {"recipe_id": 1, "title": "Almond crumbed chicken",
                     "ingredients": "chicken, almond meal", "directions": "fry"},
                    {"recipe_id": 2, "title": "Oat porridge",
                     "ingredients": "oats, banana", "directions": "simmer"},
                ],
                "lunch": [], "dinner": [],
            }
        }
        result = RecipeCandidatesClient(base_url="http://rw.test").fetch_candidates(
            allergens=["tree nuts"]
        )
        assert [c.title for c in result["breakfast"]] == ["Oat porridge"]

    def test_no_allergens_passes_everything(self, capture):
        capture.response_payload = {
            "results": {
                "breakfast": [{"recipe_id": 1, "title": "Almond granola",
                               "ingredients": "almonds", "directions": "bake"}],
                "lunch": [], "dinner": [],
            }
        }
        result = RecipeCandidatesClient(base_url="http://rw.test").fetch_candidates()
        assert len(result["breakfast"]) == 1
