"""
Pantry-driven planning (food waste) — LLM-free unit tests.

Covers: PlanningState pantry merge semantics, the deterministic coverage
matcher, pool merging/ranking, per-item fetch degradation, badge/ledger
annotation on both plan shapes, the weekly scorer boost, the extraction gate,
and the `uses_ingredient` edit predicate. No network, no LLM (per conftest).
"""

import pytest

from models.planning_state import PlanningState, PlanningStateDelta
from models.recipe import CandidateRecipe
from services import pantry_service
from services.pantry_service import (
    coverage_facts,
    describe_coverage,
    extract_pantry_delta,
    matched_items,
    merge_pantry_pool,
    normalize_items,
)


# --------------------------------------------------------------------- #
# PlanningState pantry semantics
# --------------------------------------------------------------------- #

class TestPantryState:
    def test_add_and_carry(self):
        state = PlanningState().merge(
            PlanningStateDelta(pantry_add=("zucchini", "spinach"))
        )
        assert state.pantry == ("zucchini", "spinach")
        # Silence is not a retraction: an empty delta leaves the pantry alone.
        state = state.merge(PlanningStateDelta())
        assert state.pantry == ("zucchini", "spinach")

    def test_remove_and_dedupe(self):
        state = PlanningState(pantry=("zucchini", "spinach"))
        state = state.merge(
            PlanningStateDelta(pantry_add=("spinach", "feta"), pantry_remove=("zucchini",))
        )
        assert state.pantry == ("spinach", "feta")

    def test_reset_clears_pantry(self):
        state = PlanningState(pantry=("zucchini",))
        assert state.merge(PlanningStateDelta(reset=True)).pantry == ()

    def test_round_trip(self):
        state = PlanningState(pantry=("zucchini", "ground beef"))
        restored = PlanningState.from_dict(state.to_dict())
        assert restored.pantry == ("zucchini", "ground beef")

    def test_delta_is_empty(self):
        assert PlanningStateDelta().is_empty
        assert not PlanningStateDelta(pantry_add=("x",)).is_empty
        assert not PlanningStateDelta(pantry_remove=("x",)).is_empty

    def test_describe_names_pantry(self):
        assert "zucchini" in PlanningState(pantry=("zucchini",)).describe()


# --------------------------------------------------------------------- #
# The matcher (source of truth for every user-facing claim)
# --------------------------------------------------------------------- #

class TestMatcher:
    def test_word_boundaries(self):
        assert matched_items("basmati rice, water", ["rice"]) == ["rice"]
        assert matched_items("the price of things", ["rice"]) == []

    def test_plural_tolerance(self):
        assert matched_items("three carrots, diced", ["carrot"]) == ["carrot"]

    def test_multiword_items(self):
        assert matched_items("500g ground beef, onion", ["ground beef"]) == ["ground beef"]

    def test_normalize(self):
        assert normalize_items([" Zucchini ", "zucchini", "", None, "Feta"]) == (
            "zucchini", "feta",
        )


# --------------------------------------------------------------------- #
# Pool merge + coverage ranking
# --------------------------------------------------------------------- #

def _cand(rid, title, ingredients):
    return CandidateRecipe(rid, title, ingredients, "cook it")


class TestMergePool:
    def test_coverage_first_capped_deduped(self):
        base = {"lunch": [
            _cand("b1", "Plain pasta", "pasta, oil"),
            _cand("b2", "Zucchini bake", "zucchini, cheese"),
        ]}
        pantry_pools = {"lunch": [
            _cand("p1", "Zucchini feta pie", "zucchini, feta"),
            _cand("b1", "Plain pasta", "pasta, oil"),  # duplicate id
        ]}
        merged = merge_pantry_pool(base, pantry_pools, ["zucchini", "feta"], 3)
        ids = [c.recipe_id for c in merged["lunch"]]
        assert ids[0] == "p1"          # two matches beats one
        assert ids[1] == "b2"          # base recipe with a match ranks next
        assert ids.count("b1") == 1    # deduped
        assert len(ids) <= 3

    def test_upstream_order_kept_among_equals(self):
        base = {"dinner": [
            _cand("a", "Stew", "potato"),
            _cand("b", "Soup", "carrot"),
        ]}
        merged = merge_pantry_pool(base, {}, ["zucchini"], 8)
        assert [c.recipe_id for c in merged["dinner"]] == ["a", "b"]


class TestFetchDegradation:
    def test_failure_is_best_effort(self, monkeypatch):
        # Every per-item call failing must yield empty pools, not an error.
        from services import plan_client

        def boom(self, **kwargs):
            raise RuntimeError("down")

        monkeypatch.setattr(plan_client.PlanClient, "plan_meals", boom)
        pools = pantry_service.fetch_pantry_candidates(
            {"allergies": [], "diet": []}, ["zucchini"],
        )
        assert all(v == [] for v in pools.values())

    def test_no_items_no_calls(self):
        assert pantry_service.fetch_pantry_candidates({}, []) == {}


# --------------------------------------------------------------------- #
# Coverage facts + honest description
# --------------------------------------------------------------------- #

class TestCoverage:
    def test_used_and_unused(self):
        facts = coverage_facts(
            [("Zucchini bake", "zucchini, cheese"), ("Oatmeal", "oats, milk")],
            ["zucchini", "durian"],
        )
        assert facts["used"] == {"zucchini": ["Zucchini bake"]}
        assert facts["unused"] == ["durian"]
        assert facts["used_count"] == 1 and facts["total"] == 2

    def test_describe_owns_the_misses(self):
        facts = coverage_facts([("Oatmeal", "oats")], ["durian"])
        note = describe_coverage(facts)
        assert "durian" in note and "couldn't" in note

    def test_describe_empty(self):
        assert describe_coverage(None) == ""
        assert describe_coverage({"items": []}) == ""


# --------------------------------------------------------------------- #
# Badges on both plan shapes
# --------------------------------------------------------------------- #

class TestAnnotation:
    def _legacy_plan(self):
        from datetime import datetime, timezone
        from models.session import MealCourse, MealPlan

        def course(rid, title, ingredients):
            return MealCourse(rid, title, ingredients, "cook")

        return MealPlan(
            id="p1", created_at=datetime.now(timezone.utc),
            breakfast=course("b", "Oatmeal", "oats, milk"),
            lunch=course("l", "Zucchini pie", "zucchini, feta"),
            dinner=course("d", "Stew", "potato"),
            reasoning="",
        )

    def test_daily_badges_and_ledger(self):
        plan = self._legacy_plan()
        facts = pantry_service.annotate_daily_plan(plan, ["zucchini", "durian"])
        kinds = [r["kind"] for r in (plan.lunch.match_reasons or [])]
        labels = " ".join(r["label"] for r in plan.lunch.match_reasons)
        assert "pantry" in kinds
        assert "zucchini" in labels and pantry_service.BADGE_FOODWASTE in labels
        # One chip per matching course, and no claim about the item's history:
        # nothing records whether a pantry item is a leftover or fresh.
        assert len(plan.lunch.match_reasons) == 1
        assert "leftover" not in labels.lower()
        assert not plan.breakfast.match_reasons  # no false badge
        row = plan.constraints_applied[-1]
        assert row["source"] == "your pantry" and row["status"] == "relaxed"
        assert facts["unused"] == ["durian"]

    def test_daily_no_pantry_no_touch(self):
        plan = self._legacy_plan()
        assert pantry_service.annotate_daily_plan(plan, []) is None
        assert plan.constraints_applied in ([], None) or all(
            r.get("source") != "your pantry" for r in plan.constraints_applied
        )

    def test_weekly_entries_appended_not_overwritten(self):
        entries = [{
            "day": 1, "meal_idx": 1, "meal_type": "lunch",
            "recipe": {
                "recipe_id": "x", "recipe_title": "Zucchini pie",
                "recipe_ingredients": "zucchini, feta",
                "match_reasons": [{"kind": "favorite", "label": "one of your favorites"}],
            },
        }]
        explainability = {"constraints_applied": []}
        facts = pantry_service.annotate_weekly_entries(
            entries, ["zucchini"], explainability=explainability,
        )
        reasons = entries[0]["recipe"]["match_reasons"]
        assert reasons[0]["kind"] == "favorite"  # kept
        assert any(r["kind"] == "pantry" for r in reasons)
        assert explainability["constraints_applied"][-1]["status"] == "satisfied"
        assert facts["used"] == {"zucchini": ["Zucchini pie"]}


# --------------------------------------------------------------------- #
# Weekly scorer boost
# --------------------------------------------------------------------- #

class TestScorerBoost:
    def test_pantry_boost_capped(self):
        from services.weekly_planner.planner import build_preference_scorer

        scorer = build_preference_scorer(
            {"favorite_recipe_ids": [], "food_likes": []},
            pantry=("zucchini", "feta", "spinach"),
        )
        none = scorer({"recipe_id": "1", "recipe_title": "Stew",
                       "recipe_ingredients": "potato"}, [])
        one = scorer({"recipe_id": "2", "recipe_title": "Bake",
                      "recipe_ingredients": "zucchini, oil"}, [])
        three = scorer({"recipe_id": "3", "recipe_title": "Pie",
                        "recipe_ingredients": "zucchini, feta, spinach"}, [])
        assert one - none == 3.0
        assert three - none == 6.0  # capped at two items

    def test_no_pantry_scores_unchanged(self):
        from services.weekly_planner.planner import build_preference_scorer

        scorer = build_preference_scorer({"favorite_recipe_ids": [], "food_likes": []})
        assert scorer({"recipe_id": "1", "recipe_title": "Bake",
                       "recipe_ingredients": "zucchini"}, []) == 0.0


# --------------------------------------------------------------------- #
# Extraction gate + delta building (extractor faked — no LLM)
# --------------------------------------------------------------------- #

class _FakeExtractor:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def extract(self, message):
        self.calls += 1
        return self.payload


class TestExtractionGate:
    def test_gated_messages_skip_the_llm(self):
        fake = _FakeExtractor({"have": ["zucchini"], "used_up": []})
        delta = extract_pantry_delta("plan me a healthy week", extractor=fake)
        assert fake.calls == 0 and delta.is_empty

    def test_pantry_phrasing_reaches_extractor(self):
        fake = _FakeExtractor({"have": ["Zucchini", "spinach"], "used_up": []})
        delta = extract_pantry_delta(
            "I have zucchini and spinach in the fridge", extractor=fake,
        )
        assert fake.calls == 1
        assert delta.pantry_add == ("zucchini", "spinach")

    def test_an_adverb_between_subject_and_verb_still_reaches_it(self):
        """"I already have avocado, tomatoes and pasta" — about the most
        natural way anyone says this — missed the gate entirely, because the
        pattern required "I" and "have" adjacent. The plan then ignored the
        member's fridge while the reply still thanked them for it.
        """
        fake = _FakeExtractor({"have": ["avocado", "tomatoes", "pasta"], "used_up": []})
        delta = extract_pantry_delta(
            "Give me a weekly plan, I like Italian and Asian cuisine, and I "
            "already have avocado, tomatoes and pasta.",
            extractor=fake,
        )

        assert fake.calls == 1
        assert delta.pantry_add == ("avocado", "tomatoes", "pasta")

    @pytest.mark.parametrize("message", [
        "I already have avocado",
        "we still have some spinach",
        "I just got a big bag of carrots",
        "we have leeks",
        "I've got zucchini",
        "I only have pasta left",
    ])
    def test_the_ways_people_actually_say_it(self, message):
        fake = _FakeExtractor({"have": ["x"], "used_up": []})
        extract_pantry_delta(message, extractor=fake)

        assert fake.calls == 1, message

    @pytest.mark.parametrize("message", [
        "plan me a healthy week",
        "I like Italian food",
        "what should I cook on Tuesday",
        "make Wednesday lighter",
    ])
    def test_ordinary_turns_still_cost_no_llm_call(self, message):
        """The gate exists to keep the extractor off every message. Widening
        it must not widen it into everything."""
        fake = _FakeExtractor({"have": ["x"], "used_up": []})
        extract_pantry_delta(message, extractor=fake)

        assert fake.calls == 0, message

    def test_used_up_becomes_removal(self):
        fake = _FakeExtractor({"have": [], "used_up": ["zucchini"]})
        delta = extract_pantry_delta("I used up the zucchini", extractor=fake)
        assert delta.pantry_remove == ("zucchini",)

    def test_extractor_failure_changes_nothing(self):
        class Boom:
            def extract(self, message):
                raise RuntimeError("down")

        delta = extract_pantry_delta("I have zucchini", extractor=Boom())
        assert delta.is_empty


# --------------------------------------------------------------------- #
# Edit directive: "something with zucchini" is verifiable
# --------------------------------------------------------------------- #

class TestUsesIngredientPredicate:
    def test_classify(self):
        from services.edit_service import DirectivePredicate

        p = DirectivePredicate("something with zucchini")
        assert p.kind == "uses_ingredient" and p.tag == "zucchini"
        assert p.verifiable
        p = DirectivePredicate("one using the leftover chicken")
        assert p.kind == "uses_ingredient" and p.tag == "chicken"

    def test_diet_words_stay_diet(self):
        from services.edit_service import DirectivePredicate

        p = DirectivePredicate("with something vegetarian")
        assert p.kind == "diet_tag" and p.tag == "vegetarian"

    def test_generic_stays_unverified(self):
        from services.edit_service import DirectivePredicate

        assert DirectivePredicate("something else").kind == "unverified"
        assert DirectivePredicate("lighter").kind == "lighter"
