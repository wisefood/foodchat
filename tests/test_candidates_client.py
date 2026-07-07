"""RecipeCandidatesClient — payload construction, response parsing, tag mapping."""

import json

import httpx
import pytest

from services import candidates_client as cc
from services.candidates_client import RecipeCandidatesClient, normalize_diet_tags


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
