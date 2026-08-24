"""
Every control on the settings card changes the plan, or it is not there.

The question a member eventually asks out loud: "energy boost isn't supported,
so why have it?" Three separate answers, and only one of them was "it is
supported":

* **Effort (was Difficulty).** `grep -ri difficulty` across RecipeWrangler
  returns nothing — no field, no tag, no vocabulary. So Easy/Medium/Elaborate
  was a control where two of three options changed the plan in no way at all.
  What the corpus does carry is `5_ingredients_or_less`, which is a real form of
  "keep it simple", so the control is now that and "Elaborate" is gone rather
  than kept as decoration.
* **Food waste.** Real, and wired on the weekly path only. Its own docstring
  claimed the daily path heard it "as prose via `describe`" — but `describe`
  builds the message for a slider APPLY, so a member with reuse standing got it
  once and every later "plan my day" ignored it.
* **Goal.** Actually supported, all four. "Energy boost" is `moods: [hearty]`
  plus the `high_protein` and `high_fibre` claim tags, and every one of those
  values is in RecipeWrangler's own published vocabulary. It looked unsupported
  because the plan came back identical either way — which was the repetition
  bug, not this one.

This file is the guard against the first case coming back: an option that
filters nothing, ranks nothing and says nothing.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from services import intent_facets, plan_parameters               # noqa: E402

# RecipeWrangler's published vocabulary, copied from
# `recipe_wrangler/catalog/vocabularies.py`. Copied deliberately: this asserts
# what the corpus carries, and reading it from the live service would make the
# test pass whenever the service was unreachable.
LIVE_MOODS = {
    "comfort", "light", "hearty", "fresh", "indulgent", "quick", "festive",
    "warming", "refreshing",
}
LIVE_TAGS = {
    "30_minutes_or_less", "healthy_and_nutritious", "high_protein", "low_fat",
    "5_ingredients_or_less", "high_fibre", "low_calorie",
}


def _choices():
    return [d for d in plan_parameters.PARAMETER_DEFS if d["kind"] == "choice"]


def _effect(key: str, value: str) -> dict:
    """What choosing this option actually does.

    Three honest ways for a control to act, and the test below accepts any of
    them. `filters` narrows the pool; `ranks` reorders it without excluding
    anything (food waste is a property of a COMBINATION — whether three meals
    share a bunch of coriander — so it can only ever be a ranking signal); and
    a value may be a deliberate neutral, which is what "Any" and "Off" mean.
    """
    if key == "goal":
        return {
            "filters": bool(
                intent_facets.facets_for_goal(value)
                or intent_facets.claim_tags_for(goal=value)
            ),
            "ranks": False,
        }
    if key == "difficulty":
        return {
            "filters": bool(intent_facets.claim_tags_for(difficulty=value)),
            "ranks": False,
        }
    if key == "food_waste":
        return {
            "filters": False,
            "ranks": plan_parameters.waste_mode({"food_waste": value}) != "off",
        }
    return {"filters": False, "ranks": False}


class TestEveryOptionDoesSomething:
    def test_no_option_is_decoration(self):
        """An option is allowed to apply no FILTER — "Any" and "Off" are honest
        answers. What it may not do is apply nothing anywhere: no filter, no
        ranking signal, and no explanation."""
        inert = []
        for definition in _choices():
            for option in definition["options"]:
                value = option["value"]
                effect = _effect(definition["key"], value)
                if effect["filters"] or effect["ranks"]:
                    continue
                # Neither. Then it must be a deliberate neutral, and the only
                # neutral values are the ones that mean "no preference".
                if value in ("medium", "off", "balanced"):
                    continue
                inert.append(f"{definition['key']}={value}")
        assert not inert, (
            "these change the plan in no way and say nothing about it: "
            f"{inert}"
        )

    def test_elaborate_is_gone(self):
        """There is no "elaborate" annotation in the corpus to ask for. Keeping
        the option and applying nothing was the lie."""
        effort = next(d for d in _choices() if d["key"] == "difficulty")
        assert [o["value"] for o in effort["options"]] == ["easy", "medium"]

    def test_a_stored_value_from_before_the_change_is_harmless(self):
        """Profiles carry `difficulty: "hard"`. `sanitize` drops it, so the card
        falls back to its default rather than showing an option that no longer
        exists."""
        assert plan_parameters.sanitize({"difficulty": "hard"}) == {}
        assert intent_facets.claim_tags_for(difficulty="hard") == []

    def test_simple_does_not_duplicate_the_time_control(self):
        """Two controls writing the same filter is how they end up disagreeing
        about what the member asked for."""
        tags = intent_facets.claim_tags_for(difficulty="easy")
        assert "30_minutes_or_less" not in tags
        assert tags == ["5_ingredients_or_less"]


class TestNothingIsSentThatTheCorpusLacks:
    """An invented facet value is not a narrow filter — RecipeWrangler rejects
    it and reports it, so the member's choice quietly does nothing."""

    @pytest.mark.parametrize("goal", ["weight_loss", "balanced", "high_protein", "energy"])
    def test_every_goal_maps_onto_published_vocabulary(self, goal):
        facets = intent_facets.facets_for_goal(goal)
        for mood in facets.get("moods", []):
            assert mood in LIVE_MOODS, f"{goal} asks for a mood the corpus lacks"
        for tag in intent_facets.claim_tags_for(goal=goal):
            assert tag in LIVE_TAGS, f"{goal} asks for a tag the corpus lacks"

    def test_energy_boost_is_supported(self):
        """Named because it is the one that was asked about."""
        assert intent_facets.facets_for_goal("energy") == {"moods": ["hearty"]}
        assert set(intent_facets.claim_tags_for(goal="energy")) == {
            "high_protein", "high_fibre",
        }

    def test_every_effort_tag_is_published_too(self):
        for value in ("easy", "medium"):
            for tag in intent_facets.claim_tags_for(difficulty=value):
                assert tag in LIVE_TAGS


class TestFoodWasteReachesTheDailyPath:
    """It had exactly one reader — the weekly scorer — while the card offered it
    on both canvases."""

    def _query(self, waste, monkeypatch):
        import services.planning_pipeline as module

        seen = {}

        class _Grader:
            def grade_daily_plans(self, query, *a, **k):
                seen["query"] = query
                return []

        monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: {
            slot: [__import__("models.recipe", fromlist=["CandidateRecipe"])
                   .CandidateRecipe(f"{slot}-0", slot.title(), "beans", "cook")]
            for slot in ("breakfast", "lunch", "dinner")
        })
        monkeypatch.setattr(module, "CANDIDATES", type("C", (), {
            "split_cuisines": lambda self, likes: ([], list(likes or [])),
        })())

        pipeline = module.PlanningPipeline.__new__(module.PlanningPipeline)
        pipeline.grader = _Grader()
        pipeline.generate(
            "plan my day",
            {"allergies": [], "plan_parameters": {"food_waste": waste}},
        )
        return seen.get("query", "")

    def test_reuse_reaches_the_grader(self, monkeypatch):
        assert "food waste" in self._query("reuse", monkeypatch).lower()

    def test_strict_asks_for_the_trade_explicitly(self, monkeypatch):
        """Reuse pulls against variety. The member chose which side."""
        query = self._query("strict", monkeypatch).lower()
        assert "shopping list" in query and "less varied" in query

    def test_off_says_nothing(self, monkeypatch):
        assert self._query("off", monkeypatch) == "plan my day"

    def test_it_is_a_ranking_signal_not_a_filter(self):
        """No recipe is excluded for it: sharing ingredients is a property of a
        combination, not of a dish."""
        import inspect

        source = inspect.getsource(plan_parameters.waste_mode)
        assert "grader" in source
        pipeline_src = inspect.getsource(
            __import__("services.planning_pipeline", fromlist=["PlanningPipeline"])
            .PlanningPipeline.generate
        )
        waste_block = pipeline_src[pipeline_src.find("waste = plan_parameters"):]
        assert "grader_query" in waste_block[:400]
        assert "exclude" not in waste_block[:400]
