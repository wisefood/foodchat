"""The `/api/v2/tools` client and the pipeline's fallback onto it.

Two things are worth pinning here.

The **semantic mismatch** between the endpoints. `include_ingredients` ranks in
`foodchat_candidates` and *requires* in `plan_meals`. Forwarding a member's
liked ingredients to the second one turned "likes chickpeas" into "every
breakfast must contain chickpeas", which matches nothing — the fallback
produced no plan at all, precisely when it was the last thing standing between
the user and an apology. A regression here is silent: the code runs, the
request succeeds, and the plan is empty.

The **fallback contract**. It exists so a model outage costs the ranking rather
than the plan. That only holds if it is actually reached on every failure shape
the grader has — raising, and returning nothing.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe, ScoredPlan  # noqa: E402
from services.plan_client import PlanClient  # noqa: E402


def envelope(**overrides):
    """A minimal `plan_meals` response."""
    base = {
        "days": [
            {
                "day": 1,
                "slots": [
                    {
                        "slot": "breakfast",
                        "recipes": [
                            {
                                "recipe_id": "b1",
                                "title": "Porridge",
                                "nutrition": {"calories": 300.0, "protein_g": 10.0},
                                "default_nutri_score": "A",
                            }
                        ],
                    },
                    {
                        "slot": "lunch",
                        "recipes": [{"recipe_id": "l1", "title": "Soup"}],
                    },
                    {
                        "slot": "dinner",
                        "recipes": [{"recipe_id": "d1", "title": "Stew"}],
                    },
                ],
                "nutrition_total": {"calories": 900.0, "complete": True},
            }
        ],
        "relaxations": [],
        "rejected_options": [],
    }
    base.update(overrides)
    return base


class TestToCandidates:
    def test_flattens_days_into_slots(self):
        by_slot = PlanClient.to_candidates(envelope())

        assert sorted(by_slot) == ["breakfast", "dinner", "lunch"]
        assert by_slot["breakfast"][0].title == "Porridge"

    def test_carries_nutrition_and_score(self):
        """Both are in the response; dropping them forced a second round trip."""
        recipe = PlanClient.to_candidates(envelope())["breakfast"][0]

        assert recipe.nutri_score == "A"
        # Renamed at this boundary: RecipeWrangler says `calories`, everything
        # downstream — `MealCourse.nutrition`, the meal-card chips, the day
        # totals — reads `kcal`. Passing the dict through unchanged rendered
        # every plan at 0 kcal against recipes whose calories were right there.
        assert recipe.nutrition["kcal"] == 300.0
        assert recipe.nutrition["protein_g"] == 10.0
        assert "calories" not in recipe.nutrition

    def test_the_renaming_survives_a_missing_macro(self):
        """A partial profile must not become a KeyError mid-plan."""
        env = envelope()
        env["days"][0]["slots"][0]["recipes"][0]["nutrition"] = {"calories": 100.0}
        recipe = PlanClient.to_candidates(env)["breakfast"][0]

        assert recipe.nutrition["kcal"] == 100.0
        assert recipe.nutrition["fat_g"] is None

    def test_missing_nutrition_is_none_not_an_error(self):
        """An unprofiled recipe is a real state, not a failure."""
        recipe = PlanClient.to_candidates(envelope())["lunch"][0]

        assert recipe.nutrition is None
        assert recipe.nutri_score is None

    def test_recipes_without_an_id_are_skipped(self):
        """An id-less card cannot be excluded from a later day or enriched."""
        broken = envelope()
        broken["days"][0]["slots"][0]["recipes"].append({"title": "No id"})

        assert len(PlanClient.to_candidates(broken)["breakfast"]) == 1

    def test_multiple_days_accumulate_per_slot(self):
        two = envelope()
        second = {
            "day": 2,
            "slots": [{"slot": "breakfast",
                       "recipes": [{"recipe_id": "b2", "title": "Eggs"}]}],
        }
        two["days"].append(second)

        assert [c.title for c in PlanClient.to_candidates(two)["breakfast"]] == [
            "Porridge",
            "Eggs",
        ]


class TestDescribeRelaxations:
    def test_reports_what_was_dropped(self):
        """A preference dropped silently is indistinguishable from one ignored."""
        env = envelope(relaxations=[
            {"day": 1, "slot": "breakfast", "dropped": "cuisines"}
        ])
        lines = PlanClient.describe_relaxations(env)

        assert len(lines) == 1
        assert "cuisine" in lines[0] and "breakfast" in lines[0]

    def test_reports_rejected_options(self):
        env = envelope(rejected_options=["cuisine:klingon"])

        assert any("klingon" in line for line in PlanClient.describe_relaxations(env))

    def test_nothing_dropped_reports_nothing(self):
        assert PlanClient.describe_relaxations(envelope()) == []


class TestPlanPayload:
    """What actually goes on the wire."""

    @pytest.fixture
    def sent(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return envelope()

        class FakeClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, url, json):
                captured["url"] = url
                captured["json"] = json
                return FakeResponse()

        import services.plan_client as module
        monkeypatch.setattr(module.httpx, "Client", FakeClient)
        return captured

    def test_hits_the_v2_tools_endpoint(self, sent):
        PlanClient(base_url="http://rw.test").plan_meals()

        assert sent["url"] == "http://rw.test/api/v2/tools/plan_meals"

    def test_slots_become_objects_not_a_dict(self, sent):
        """`slots` is a list of {slot, count}; a dict is a 422."""
        PlanClient(base_url="http://rw.test").plan_meals()

        assert isinstance(sent["json"]["slots"], list)
        assert sent["json"]["slots"][0] == {"slot": "breakfast", "count": 1}

    def test_optional_filters_are_omitted_when_unset(self, sent):
        """An absent filter must not become a constraint."""
        PlanClient(base_url="http://rw.test").plan_meals()

        assert "max_minutes" not in sent["json"]
        assert "min_nutri_score" not in sent["json"]

    def test_nutri_score_is_upper_cased(self, sent):
        PlanClient(base_url="http://rw.test").plan_meals(min_nutri_score="b")

        assert sent["json"]["min_nutri_score"] == "B"

    def test_relaxation_is_requested(self, sent):
        """Otherwise an over-constrained slot returns empty instead of widening."""
        PlanClient(base_url="http://rw.test").plan_meals()

        assert sent["json"]["allow_relaxation"] is True


class TestPipelineFallback:
    """What happens when the grader cannot rank the pool.

    The shape of this changed: `plan_meals` is no longer a fallback *source*,
    it is the only source. So the question is no longer "does it get called"
    (it always does) but "is its unranked order served when the grader fails,
    and left alone when the grader works".
    """

    @pytest.fixture
    def pipeline(self, monkeypatch):
        from services.planning_pipeline import PlanningPipeline

        p = PlanningPipeline.__new__(PlanningPipeline)

        class Candidates:
            def split_cuisines(self, likes):
                return [], list(likes or [])

        import services.planning_pipeline as module
        monkeypatch.setattr(module, "CANDIDATES", Candidates())

        pool = {
            "breakfast": [CandidateRecipe("b0", "B0", "", ""),
                          CandidateRecipe("b1", "B1", "", "")],
            "lunch": [CandidateRecipe("l0", "L0", "", "")],
            "dinner": [CandidateRecipe("d0", "D0", "", "")],
        }
        calls = []

        def fake_pool(**kwargs):
            calls.append(kwargs)
            return dict(pool)

        monkeypatch.setattr(module, "_fetch_candidate_pool", fake_pool)
        p.calls = calls
        p.pool = pool
        return p

    def test_the_pool_always_comes_from_the_planning_endpoint(self, pipeline):
        """No second source exists — this is the point of the switch."""
        class Good:
            def grade_daily_plans(self, query, candidates, profile, history):
                return [ScoredPlan(
                    breakfast=candidates["breakfast"][0],
                    lunch=candidates["lunch"][0],
                    dinner=candidates["dinner"][0],
                    score=5, reasoning="graded",
                )]

        pipeline.grader = Good()
        pipeline.generate("q", {"allergies": []})

        assert len(pipeline.calls) == 1

    def test_a_working_grader_result_is_used_unchanged(self, pipeline):
        class Good:
            def grade_daily_plans(self, query, candidates, profile, history):
                return [ScoredPlan(
                    breakfast=candidates["breakfast"][1],   # not the pool's first
                    lunch=candidates["lunch"][0],
                    dinner=candidates["dinner"][0],
                    score=5, reasoning="graded",
                )]

        pipeline.grader = Good()
        plans = pipeline.generate("q", {"allergies": []})

        assert plans[0].score == 5
        assert plans[0].breakfast.recipe_id == "b1", (
            "the grader's pick was replaced by the pool's default order"
        )

    def test_grader_exception_serves_the_unranked_pool(self, pipeline):
        """A model outage costs the ranking, not the plan."""
        class Boom:
            def grade_daily_plans(self, *a, **k):
                raise RuntimeError("groq down")

        pipeline.grader = Boom()
        plans = pipeline.generate("q", {"allergies": []})

        assert len(plans) == 1
        assert plans[0].score == 0
        assert plans[0].breakfast.recipe_id == "b0"

    def test_the_unranked_plan_says_it_is_unranked(self, pipeline):
        """Score 0 alone reads as "graded badly"; the reason disambiguates."""
        class Boom:
            def grade_daily_plans(self, *a, **k):
                raise RuntimeError("down")

        pipeline.grader = Boom()

        assert "grader unavailable" in pipeline.generate("q", {"allergies": []})[0].reasoning

    def test_grader_returning_nothing_serves_the_unranked_pool(self, pipeline):
        class Empty:
            def grade_daily_plans(self, *a, **k):
                return []

        pipeline.grader = Empty()

        assert len(pipeline.generate("q", {"allergies": []})) == 1

    def test_no_second_call_is_made_to_assemble_the_fallback(self, pipeline):
        """The pool is already in hand; re-fetching it was pure waste."""
        class Boom:
            def grade_daily_plans(self, *a, **k):
                raise RuntimeError("down")

        pipeline.grader = Boom()
        pipeline.generate("q", {"allergies": []})

        assert len(pipeline.calls) == 1

    def test_an_empty_slot_yields_no_plan(self, pipeline, monkeypatch):
        """`plan_meals` already relaxed everything it was allowed to, so a gap
        means the hard constraints genuinely admit nothing."""
        import services.planning_pipeline as module

        monkeypatch.setattr(module, "_fetch_candidate_pool",
                            lambda **kw: {"breakfast": [], "lunch": [], "dinner": []})
        pipeline.grader = None

        assert pipeline.generate("q", {"allergies": []}) == []

    def test_hard_constraints_reach_the_pool(self, pipeline):
        class Good:
            def grade_daily_plans(self, query, candidates, profile, history):
                return [ScoredPlan(
                    breakfast=candidates["breakfast"][0],
                    lunch=candidates["lunch"][0],
                    dinner=candidates["dinner"][0],
                    score=5, reasoning="",
                )]

        pipeline.grader = Good()
        pipeline.generate("q", {
            "allergies": ["peanut"], "diet": ["vegan"], "food_dislikes": ["liver"],
        })

        profile = pipeline.calls[0]["profile"]
        assert profile["allergies"] == ["peanut"]
        assert profile["diet"] == ["vegan"]
        assert profile["food_dislikes"] == ["liver"]

    def test_liked_ingredients_are_not_a_requirement(self, pipeline):
        """`plan_meals` requires every `include_ingredient`, so forwarding a
        member's likes would demand chickpeas in every breakfast."""
        class Good:
            def grade_daily_plans(self, query, candidates, profile, history):
                return [ScoredPlan(
                    breakfast=candidates["breakfast"][0],
                    lunch=candidates["lunch"][0],
                    dinner=candidates["dinner"][0],
                    score=5, reasoning="",
                )]

        pipeline.grader = Good()
        pipeline.generate("q", {"allergies": [], "food_likes": ["chickpeas"]})

        assert "include_ingredients" not in pipeline.calls[0]

class TestStructuredPlan:
    """Phase 2: a plan of any shape, returned in FoodChat's own model.

    The regrouping is the fragile part. RecipeWrangler echoes the slot but not
    the role — roles are FoodChat's vocabulary — so the two are zipped back
    together by position. If the request order and `role_sequence()` ever
    diverge, every plate gets the wrong role: a dessert labelled `main`, and a
    nutrition budget split against the wrong weights. Nothing raises.
    """

    @pytest.fixture
    def pipeline(self, monkeypatch):
        from models.plan_spec import PlanSpec
        from services.planning_pipeline import PlanningPipeline

        p = PlanningPipeline.__new__(PlanningPipeline)

        class Candidates:
            def split_cuisines(self, likes):
                return [], list(likes or [])

        import services.planning_pipeline as module
        monkeypatch.setattr(module, "CANDIDATES", Candidates())

        sent = {}

        def fake_plan_meals(**kwargs):
            sent.update(kwargs)
            spec: PlanSpec = kwargs["spec"]
            # Echo one recipe per requested plate, in request order — exactly
            # what the real service does.
            slots = [
                {"slot": entry["slot"], "recipes": [{
                    "recipe_id": f"r{i}", "title": f"R{i}",
                    "nutrition": {"calories": 100.0 * (i + 1), "protein_g": 5.0},
                    "default_nutri_score": "A",
                }]}
                for i, entry in enumerate(spec.to_request_slots())
            ]
            return {"days": [{"day": 1, "slots": slots}], "relaxations": [],
                    "rejected_options": []}

        class FakePlanner:
            plan_meals = staticmethod(fake_plan_meals)
            describe_relaxations = staticmethod(PlanClient.describe_relaxations)

        import services.plan_client as plan_module
        monkeypatch.setattr(plan_module, "PLANNER", FakePlanner())
        p.sent = sent
        return p

    def test_returns_a_real_meal_plan(self, pipeline):
        from models.plan_spec import PlanSpec
        from models.session import MealPlan

        plan = pipeline.plan_structured({"allergies": []}, PlanSpec.default())

        assert isinstance(plan, MealPlan)

    def test_plates_regroup_into_one_meal_with_the_right_roles(self, pipeline):
        from models.plan_spec import PlanSpec

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side", "dessert")})
        plan = pipeline.plan_structured({"allergies": []}, spec)

        dinner = plan.day_plans[0].meals[0]
        assert dinner.meal_type == "dinner"
        assert [plate.role for plate in dinner.plates] == ["main", "side", "dessert"]

    def test_the_legacy_scalar_fields_are_populated_from_the_mains(self, pipeline):
        """A great deal of code addresses `plan.dinner` by name."""
        from models.plan_spec import PlanSpec

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        plan = pipeline.plan_structured({"allergies": []}, spec)

        assert plan.dinner.recipe_id == plan.day_plans[0].meals[0].main.recipe_id
        assert plan.dinner.role == "main"

    def test_nutrition_is_renamed_into_foodchats_shape(self, pipeline):
        """RecipeWrangler says `calories`; everything here reads `kcal`."""
        from models.plan_spec import PlanSpec

        plan = pipeline.plan_structured({"allergies": []}, PlanSpec.default())
        nutrition = plan.day_plans[0].meals[0].plates[0].nutrition

        assert nutrition["kcal"] == 100.0
        assert "calories" not in nutrition

    def test_hard_constraints_are_forwarded(self, pipeline):
        from models.plan_spec import PlanSpec

        pipeline.plan_structured(
            {"allergies": ["peanut"], "diet": ["vegan"],
             "food_dislikes": ["liver"], "min_nutri_score": "B"},
            PlanSpec.default(),
        )

        assert pipeline.sent["allergens"] == ["peanut"]
        assert pipeline.sent["diet"] == ["vegan"]
        assert pipeline.sent["exclude_ingredients"] == ["liver"]
        assert pipeline.sent["min_nutri_score"] == "B"

    def test_an_empty_response_yields_no_plan(self, pipeline, monkeypatch):
        """Better than a day with meals silently missing."""
        from models.plan_spec import PlanSpec
        import services.plan_client as plan_module

        class Empty:
            plan_meals = staticmethod(lambda **k: {"days": []})
            describe_relaxations = staticmethod(lambda e: [])

        monkeypatch.setattr(plan_module, "PLANNER", Empty())

        assert pipeline.plan_structured({"allergies": []}, PlanSpec.default()) is None


class TestAllergenBackstop:
    """The client-side allergen check, now applied where candidates arrive.

    It moved here with `fetch_candidates`'s retirement. It is not redundant with
    the server filter: the corpus has tagged almond dishes `nut_free` in
    production, and one filter between a member and an allergen is one too few.
    The server side got considerably better during this work, which is a reason
    to keep checking rather than to stop.
    """

    def _envelope_with(self, title, ingredients):
        return {
            "days": [{"day": 1, "slots": [{
                "slot": "lunch",
                "recipes": [{"recipe_id": "x", "title": title,
                             "ingredients": ingredients, "directions": ""}],
            }]}],
            "relaxations": [], "rejected_options": [],
        }

    def test_a_candidate_contradicting_its_tags_is_dropped(self):
        env = self._envelope_with("Almond cake", "almonds, flour, sugar")

        assert PlanClient.to_candidates(env, allergens=["tree nuts"])["lunch"] == []

    def test_a_clean_candidate_survives(self):
        env = self._envelope_with("Tomato soup", "tomato, basil")

        assert len(PlanClient.to_candidates(env, allergens=["tree nuts"])["lunch"]) == 1

    def test_no_allergens_drops_nothing(self):
        env = self._envelope_with("Almond cake", "almonds")

        assert len(PlanClient.to_candidates(env)["lunch"]) == 1

    def test_the_check_reads_ingredients_not_just_the_title(self):
        """A title that hides its allergen is the case that matters."""
        env = self._envelope_with("Winter traybake", "cashews, squash, oil")

        assert PlanClient.to_candidates(env, allergens=["tree nuts"])["lunch"] == []
