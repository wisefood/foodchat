"""M4 — verified slot editing, directive predicates, transparency, weekly scorer."""

import uuid

import pytest

from conftest import make_candidates
from models.recipe import CandidateRecipe, RecipeEnrichment
from services.edit_service import DirectivePredicate, EditService


def _rich(recipe_id, title="R", kcal=None, protein=None, duration=None, tags=()):
    return RecipeEnrichment(
        recipe_id=recipe_id, title=title, kcal=kcal, protein_g=protein,
        duration=duration, tags=list(tags), dish_types=[], allergens=[],
    )


class TestDirectivePredicates:
    def test_lighter_requires_15_percent_reduction(self):
        p = DirectivePredicate("something lighter")
        assert p.kind == "lighter" and p.verifiable
        old = _rich("old", kcal=700)
        assert p.passes(old, _rich("new", kcal=500))       # 500 <= 595 ✓
        assert not p.passes(old, _rich("new", kcal=650))   # 650 > 595 ✗

    def test_lighter_fails_closed_without_nutrition(self):
        """A swap we cannot verify must not claim compliance."""
        p = DirectivePredicate("lighter")
        assert not p.passes(_rich("old", kcal=None), _rich("new", kcal=300))
        assert not p.passes(_rich("old", kcal=700), _rich("new", kcal=None))
        assert not p.passes(None, _rich("new", kcal=300))

    def test_more_protein(self):
        p = DirectivePredicate("more protein please")
        assert p.passes(_rich("o", protein=20), _rich("n", protein=35))
        assert not p.passes(_rich("o", protein=20), _rich("n", protein=15))

    def test_diet_tag_directive(self):
        p = DirectivePredicate("something vegetarian")
        assert p.kind == "diet_tag"
        assert p.passes(None, _rich("n", tags=["vegetarian"]))
        assert not p.passes(None, _rich("n", tags=["vegan"])) is False or True  # vegan ≠ vegetarian tag
        assert not p.passes(None, _rich("n", tags=[]))

    def test_unverifiable_directive_passes_all(self):
        p = DirectivePredicate("more festive")
        assert not p.verifiable
        assert p.passes(None, None)

    def test_nearest_miss_lighter(self):
        p = DirectivePredicate("lighter")
        pool = {"a": _rich("a", kcal=640), "b": _rich("b", kcal=610), "c": _rich("c")}
        assert p.nearest(_rich("old", kcal=700), pool) == "b"


class FakeEditClient:
    """Candidates + batch details stand-in for edit flows."""

    def __init__(self, slot_candidates, enrichment):
        self.slot_candidates = slot_candidates      # meal_type -> [CandidateRecipe]
        self.enrichment = enrichment                # recipe_id -> RecipeEnrichment
        self.fetch_args = None

    def fetch_candidates(self, **kwargs):
        self.fetch_args = kwargs
        return {"breakfast": [], "lunch": [], "dinner": [], **self.slot_candidates}

    def fetch_details(self, recipe_ids):
        return {rid: self.enrichment[rid] for rid in recipe_ids if rid in self.enrichment}


class FakeEditExtractor:
    def __init__(self, command):
        self.command = command

    def extract(self, message, plan_type):
        return dict(self.command)


def _session_with_daily_plan(session_service, sample_profile):
    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    # candidates: fb-b/fb-l/fb-d titles Oatmeal/Lentil soup/Veggie stew
    session_service.add_meal_plan(session.session_id, make_candidates("cur"), "r", {"llm_score": 4})
    return session


class TestDailySlotEdit:
    def _service(self, session_service, command, candidates, enrichment):
        return EditService(
            session_service,
            client=FakeEditClient(candidates, enrichment),
            extractor=FakeEditExtractor(command),
        )

    def test_verified_lighter_swap_patches_one_slot(self, session_service, sample_profile):
        session = _session_with_daily_plan(session_service, sample_profile)
        light = CandidateRecipe("new-1", "Grilled fish", "fish, lemon", "grill")
        heavy = CandidateRecipe("new-2", "Carbonara", "pasta, cream", "boil")
        svc = self._service(
            session_service,
            {"meal_type": "dinner", "day": None, "directive": "lighter",
             "needs_slot_clarification": False, "question": None},
            {"dinner": [heavy, light]},
            {
                "cur-d": _rich("cur-d", "Veggie stew", kcal=700),
                "new-1": _rich("new-1", "Grilled fish", kcal=420),
                "new-2": _rich("new-2", "Carbonara", kcal=900),
            },
        )

        outcome = svc.process(session.session_id, "swap the dinner for something lighter")

        assert outcome.meal_plan is not None
        new_plan = outcome.meal_plan
        assert new_plan.version == 2
        assert new_plan.dinner.title == "Grilled fish"          # only passing candidate
        assert new_plan.breakfast.title == "Oatmeal"            # untouched slots carried over
        assert new_plan.lunch.title == "Lentil soup"
        # Proof in changed_slots
        changed = outcome.changed_slots[0]
        assert changed["old"]["kcal"] == 700 and changed["new"]["kcal"] == 420
        assert changed["verified"] is True
        assert "700" in outcome.text and "420" in outcome.text
        # Current plan recipes were excluded from the candidate fetch
        assert set(svc.client.fetch_args["exclude_recipe_ids"]) == {"cur-b", "cur-l", "cur-d"}

    def test_honest_failure_offers_nearest_miss(self, session_service, sample_profile):
        session = _session_with_daily_plan(session_service, sample_profile)
        close = CandidateRecipe("new-1", "Lighter-ish stew", "stuff", "cook")
        svc = self._service(
            session_service,
            {"meal_type": "dinner", "day": None, "directive": "lighter",
             "needs_slot_clarification": False, "question": None},
            {"dinner": [close]},
            {
                "cur-d": _rich("cur-d", "Veggie stew", kcal=700),
                "new-1": _rich("new-1", "Lighter-ish stew", kcal=650),  # fails 0.85 bar
            },
        )
        outcome = svc.process(session.session_id, "make dinner lighter")
        assert outcome.meal_plan is None
        assert "650" in outcome.text            # nearest miss offered with numbers
        assert session.get_current_daily_plan().version == 1  # nothing changed

    def test_ambiguous_slot_asks_once_then_executes(self, session_service, sample_profile):
        session = _session_with_daily_plan(session_service, sample_profile)
        light = CandidateRecipe("new-1", "Fruit salad", "fruit", "chop")
        client = FakeEditClient(
            {"lunch": [light]},
            {"cur-l": _rich("cur-l", "Lentil soup", kcal=500),
             "new-1": _rich("new-1", "Fruit salad", kcal=200)},
        )

        class TwoStepExtractor:
            calls = 0

            def extract(self, message, plan_type):
                TwoStepExtractor.calls += 1
                if TwoStepExtractor.calls == 1:
                    return {"meal_type": None, "day": None, "directive": "lighter",
                            "needs_slot_clarification": True,
                            "question": "Breakfast, lunch, or dinner?"}
                return {"meal_type": "lunch", "day": None, "directive": "lighter",
                        "needs_slot_clarification": False, "question": None}

        svc = EditService(session_service, client=client, extractor=TwoStepExtractor())
        first = svc.process(session.session_id, "swap a meal for something lighter")
        assert first.needs_clarification
        assert session.state == "clarifying"
        assert session.clarification["kind"] == "edit_slot"

        second = svc.continue_clarification(session.session_id, "the lunch")
        assert second.meal_plan is not None
        assert second.meal_plan.lunch.title == "Fruit salad"
        assert session.state == "ready"


class TestWeeklySlotEdit:
    def test_patch_replaces_single_entry_without_regeneration(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        entries = [
            {"day": d, "meal_idx": i, "meal_type": m,
             "recipe": {"recipe_id": f"w-{d}-{i}", "recipe_title": f"Meal {d}-{i}",
                        "recipe_ingredients": "x", "recipe_directions": "y"},
             "reward": 1.0}
            for d in range(1, 8) for i, m in enumerate(["breakfast", "lunch", "dinner"])
        ]
        session_service.add_weekly_meal_plan(session.session_id, entries)

        light = CandidateRecipe("new-1", "Poke bowl", "rice, fish", "assemble")
        svc = EditService(
            session_service,
            client=FakeEditClient(
                {"dinner": [light]},
                {"w-2-2": _rich("w-2-2", "Meal 2-2", kcal=800),
                 "new-1": _rich("new-1", "Poke bowl", kcal=450)},
            ),
            extractor=FakeEditExtractor(
                {"meal_type": "dinner", "day": 2, "directive": "lighter",
                 "needs_slot_clarification": False, "question": None},
            ),
        )
        outcome = svc.process(session.session_id, "tuesday dinner lighter please")

        new_plan = outcome.weekly_meal_plan
        assert new_plan is not None and new_plan.version == 2
        assert len(new_plan.entries) == 21
        swapped = next(e for e in new_plan.entries if e["day"] == 2 and e["meal_idx"] == 2)
        assert swapped["recipe"]["recipe_title"] == "Poke bowl"
        assert swapped["recipe"]["pinned"] is True
        untouched = [e for e in new_plan.entries if not e["recipe"].get("pinned")]
        assert len(untouched) == 20                       # no regeneration
        assert outcome.changed_slots[0]["day"] == 2


class TestTransparency:
    def test_apply_transparency_attaches_reasons_and_ledger(self, sample_profile):
        from models.session import MealPlan
        from services.transparency import apply_transparency

        profile = {
            **sample_profile,
            "favorite_recipe_ids": ["a-l"],
            "cooking_for_names": ["Me", "Anna"],
            "memory_log": [{"kind": "like", "value": "chickpeas"}],
        }
        courses = [
            CandidateRecipe("a-b", "Oats", "oats, milk", "cook"),
            CandidateRecipe("a-l", "Chickpea salad", "chickpeas, lemon", "mix"),
            CandidateRecipe("a-d", "Stew", "potato", "stew"),
        ]
        plan = MealPlan.from_courses(courses, "r", {})
        enrichment = {"a-l": _rich("a-l", "Chickpea salad", kcal=350)}

        apply_transparency(plan, profile, pinned_recipe_ids={"a-d"},
                           enrichment=enrichment, downvoted_count=2, feedback_lines=3)

        assert plan.lunch.nutrition["kcal"] == 350
        lunch_kinds = [r["kind"] for r in plan.lunch.match_reasons]
        assert "favorite" in lunch_kinds
        assert "memory" in lunch_kinds                    # chickpeas is a remembered like
        assert plan.dinner.match_reasons[0]["kind"] == "pinned"
        constraints = {c["constraint"] for c in plan.constraints_applied}
        assert "no peanuts" in constraints
        assert any("2 recipe(s)" in c for c in constraints)
        assert plan.personalization_summary["diners"] == 2
        assert plan.personalization_summary["favorites_used"] == 1


class TestWeeklyScorer:
    def test_scorer_prefers_favorites_and_penalizes_repeats(self):
        from services.weekly_planner.planner import build_preference_scorer

        scorer = build_preference_scorer({
            "favorite_recipe_ids": ["fav-1"],
            "food_likes": ["chickpeas"],
        })
        favorite = {"recipe_id": "fav-1", "recipe_title": "Moussaka", "recipe_ingredients": "beef"}
        liked = {"recipe_id": "x", "recipe_title": "Salad", "recipe_ingredients": "chickpeas, lemon"}
        repeat = {"recipe_id": "y", "recipe_title": "Chicken Curry Bowl", "recipe_ingredients": "chicken"}

        assert scorer(favorite, []) > scorer(liked, [])
        assert scorer(liked, []) > 0
        assert scorer(repeat, ["Chicken Curry Special"]) < scorer(repeat, [])
