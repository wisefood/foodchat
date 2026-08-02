"""Seeded planning (M2) — seed resolution, placement, pinning, favorites offer."""

import uuid

import pytest

from models.recipe import CandidateRecipe, ResolvedRecipe
from services.seed_service import SeedService


def _resolved(recipe_id, title, dish_types=("dinner",), allergens=(), ingredients="stuff"):
    return ResolvedRecipe(
        recipe=CandidateRecipe(recipe_id, title, ingredients, "cook it"),
        dish_types=list(dish_types), allergens=list(allergens), tags=[],
    )


class FakeRecipeClient:
    """Autocomplete + detail stand-in keyed by lowercase dish name."""

    def __init__(self, recipes: dict[str, ResolvedRecipe]):
        self.recipes = recipes  # name -> ResolvedRecipe
        self.by_id = {r.recipe.recipe_id: r for r in recipes.values()}

    def autocomplete(self, name, limit=5):
        hit = self.recipes.get(name.lower())
        return [(hit.recipe.recipe_id, hit.recipe.title)] if hit else []

    def fetch_recipe(self, recipe_id):
        return self.by_id.get(recipe_id)


PASTITSIO = _resolved("r-past", "Pastitsio", dish_types=("dinner",))
FAKES = _resolved("r-fakes", "Fakes (Greek lentil soup)", dish_types=("lunch", "dinner"))
PB_TOAST = _resolved("r-pb", "Peanut Butter Toast", dish_types=("breakfast",),
                     allergens=["peanuts"], ingredients="bread, peanut butter")

CLIENT = FakeRecipeClient({
    "pastitsio": PASTITSIO,
    "fakes": FAKES,
    "peanut butter toast": PB_TOAST,
})


def make_seed_service():
    svc = SeedService.__new__(SeedService)
    svc.client = CLIENT
    svc.extractor = None  # extract_seeds not used in these tests
    return svc


class TestResolution:
    def test_resolves_named_dishes(self, sample_profile):
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "pastitsio"}, {"name": "fakes"}], sample_profile)
        assert [r.status for r in res] == ["resolved", "resolved"]
        assert res[0].recipe.recipe.title == "Pastitsio"

    def test_unknown_dish_reports_not_found(self, sample_profile):
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "spanakorizo"}], sample_profile)
        assert res[0].status == "not_found"
        assert "spanakorizo" in res[0].note

    def test_allergy_conflict_blocks_pin(self, sample_profile):
        """sample_profile is allergic to peanuts — the seed must be skipped."""
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "peanut butter toast"}], sample_profile)
        assert res[0].status == "conflict"
        assert "peanuts" in res[0].note
        # Conflicted seeds never reach placement
        assert svc.place_weekly(res) == ({}, [])
        assert svc.place_daily(res) == ({}, [])

    def test_allergy_synonyms_catch_untagged_ingredients(self):
        """"tree nuts" allergy must block an almond dish even with NO allergen
        tags from RecipeWrangler (live incident: broken graph tagging)."""
        almond = _resolved("r-alm", "Almond Crumbed Chicken", ("dinner",),
                           allergens=(), ingredients="chicken, almond meal, apple")
        svc = SeedService.__new__(SeedService)
        svc.client = FakeRecipeClient({"almond crumbed chicken": almond})
        svc.extractor = None
        res = svc.resolve_seeds([{"name": "almond crumbed chicken"}],
                                {"allergies": ["Tree Nuts"]})
        assert res[0].status == "conflict"

    def test_trailing_typo_resolves_via_truncation(self, sample_profile):
        """"bolognesse" (extra letter) still finds Bolognese via prefix retry."""
        bolognese = _resolved("r-bol", "Spaghetti Bolognese", ("dinner",))

        class PrefixClient(FakeRecipeClient):
            def autocomplete(self, name, limit=5):
                hits = [(r.recipe.recipe_id, r.recipe.title)
                        for r in self.recipes.values()
                        if r.recipe.title.lower().startswith(name.lower())]
                return hits

        svc = SeedService.__new__(SeedService)
        svc.client = PrefixClient({"spaghetti bolognese": bolognese})
        svc.extractor = None
        res = svc.resolve_seeds([{"name": "spaghetti bolognesse"}], sample_profile)
        assert res[0].status == "resolved"
        assert res[0].recipe.recipe.title == "Spaghetti Bolognese"


class TestPlacement:
    def test_weekly_hint_honored(self, sample_profile):
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "pastitsio", "meal_type": "dinner", "day": 7}], sample_profile)
        pinned, dropped = svc.place_weekly(res)
        assert pinned == {(7, 2): PASTITSIO.recipe}
        assert dropped == []

    def test_weekly_spread_without_hints(self, sample_profile):
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "pastitsio"}, {"name": "fakes"}], sample_profile)
        pinned, dropped = svc.place_weekly(res)
        # Both placed, on different days, in their dish-type slots
        assert len(pinned) == 2
        days = [day for (day, _idx) in pinned]
        assert len(set(days)) == 2
        assert (1, 2) in pinned  # pastitsio -> first spread day, dinner

    def test_daily_placement_uses_dish_types(self, sample_profile):
        svc = make_seed_service()
        res = svc.resolve_seeds([{"name": "fakes"}], sample_profile)
        pinned, dropped = svc.place_daily(res)
        assert pinned == {"lunch": FAKES.recipe}
        assert dropped == []

    def test_second_dish_for_a_slot_is_dropped_and_owned(self, sample_profile):
        """Phase 0: 'pastitsio and fakes for dinner' can't both pin one slot
        today — the extra is dropped, but the reply must acknowledge it
        instead of the old describe() falsely claiming both were worked in."""
        svc = make_seed_service()
        res = svc.resolve_seeds(
            [{"name": "pastitsio", "meal_type": "dinner"},
             {"name": "fakes", "meal_type": "dinner"}],
            sample_profile,
        )
        pinned, dropped = svc.place_daily(res)
        assert pinned == {"dinner": PASTITSIO.recipe}
        assert [d.recipe.recipe.title for d in dropped] == ["Fakes (Greek lentil soup)"]

        note = svc.describe(res, dropped)
        assert "Pastitsio" in note and "as you asked" in note
        # The dropped dish is acknowledged, NOT claimed as worked in
        assert "You also asked for “Fakes (Greek lentil soup)”" in note
        assert "one dish per meal" in note


class TestPipelinePinning:
    def test_pinned_slot_is_sole_candidate_and_excluded_from_fetch(self, sample_profile, monkeypatch):
        from services import planning_pipeline as pp
        from services.planning_pipeline import PlanningPipeline

        class FakeCandidates:
            def __init__(self):
                self.called_with = None

            def split_cuisines(self, likes):
                """No vocabulary in tests, so nothing is a cuisine.

                Mirrors the real client's behaviour when the vocabulary fetch fails —
                everything stays an ingredient — which is the safe degradation these
                tests should be asserting against anyway.
                """
                return [], list(likes or [])

            def pool(self, **kwargs):
                self.called_with = kwargs
                slot = lambda p: [CandidateRecipe(f"{p}1", p, "i", "d")]
                return {"breakfast": slot("b"), "lunch": slot("l"), "dinner": slot("d")}

        class PassGrader:
            def grade_daily_plans(self, query, candidates, profile, feedback_history=""):
                from models.recipe import ScoredPlan
                return [ScoredPlan(
                    breakfast=candidates["breakfast"][0], lunch=candidates["lunch"][0],
                    dinner=candidates["dinner"][0], score=5, reasoning="ok",
                )]

        fake = FakeCandidates()
        monkeypatch.setattr(pp, "CANDIDATES", fake)
        monkeypatch.setattr(pp, "_fetch_candidate_pool", lambda **kw: fake.pool(**kw))
        pipeline = PlanningPipeline.__new__(PlanningPipeline)
        pipeline.grader = PassGrader()

        plans = pipeline.generate("dinner plan", sample_profile,
                                  pinned={"dinner": PASTITSIO.recipe})
        assert plans[0].dinner is PASTITSIO.recipe
        assert fake.called_with["exclude_recipe_ids"] == ["r-past"]
        # Favourites are forwarded from the profile now, not as a flat kwarg —
        # `plan_meals` takes them as a ranking boost, so a pinned plan still
        # floats a member's favourites in the slots it did not pin.
        assert fake.called_with["profile"].get("favorite_recipe_ids", []) == []


class TestWeeklyPlannerPinning:
    def test_pinned_slots_bypass_candidate_selection(self):
        from services.weekly_planner.environment import WeeklyMealPlanEnv
        from services.weekly_planner.planner import WeeklyPlanner

        class FakeActions:
            def get_candidate_actions(self, meal_type, state):
                return [{
                    "recipe_id": f"free-{state['day']}-{state['meal_idx']}",
                    "recipe_title": "Filler", "recipe_ingredients": "x",
                    "recipe_directions": "y",
                }]

            def mark_selected(self, recipe_id):
                pass

        class FlatReward:
            def calculate_step_reward(self, action, tracker, preferences, user_query=None):
                return 0.0

        env = WeeklyMealPlanEnv(
            user_profile={"preferences": []},
            action_space=FakeActions(),
            reward_calculator=FlatReward(),
        )
        pinned = {(7, 2): {
            "recipe_id": "r-past", "recipe_title": "Pastitsio",
            "recipe_ingredients": "pasta, beef", "recipe_directions": "bake",
        }}
        entries = WeeklyPlanner(env).generate_full_plan(user_query="week", pinned=pinned)

        assert len(entries) == 21
        sunday_dinner = next(e for e in entries if e["day"] == 7 and e["meal_idx"] == 2)
        assert sunday_dinner["recipe"]["recipe_id"] == "r-past"
        assert sunday_dinner["recipe"]["pinned"] is True
        # Every other slot came from the action space
        others = [e for e in entries if not e["recipe"].get("pinned")]
        assert len(others) == 20


class TestFavoritesOffer:
    def _orchestrator(self, session_service, plan_calls: list):
        from services.orchestrator_service import OrchestratorService

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service
        orch.foodscholar_service = None
        orch.weekly_plan_service = None
        orch.memory_service = None        # nudges off in these routing tests

        class FakeChatService:
            def process_plan_request(self, session_id, message, is_refinement=False,
                                     seeds=None, skip_clarification=False):
                plan_calls.append({"message": message, "seeds": seeds})
                return "plan text", False, None

        orch.chat_service = FakeChatService()

        class FakeSeedService:
            client = CLIENT

            def extract_seeds(self, message):
                return []

        orch.seed_service = FakeSeedService()

        class DailyClassifier:
            def classify(self, message, history):
                return {"intent": "daily_plan", "target_plan_type": None}

        orch.orchestrator = DailyClassifier()
        return orch

    def _session_with_favorites(self, session_service, sample_profile):
        profile = {**sample_profile, "favorite_recipe_ids": ["r-past"]}
        return session_service.create_session(f"member-{uuid.uuid4()}", profile)

    def test_no_tollbooth_before_the_first_plan(self, session_service, sample_profile):
        """A plan request generates a plan, favourites or not.

        The offer used to intercept the first plan of EVERY session with a
        yes/no question — a member who plans daily answered it daily and
        called it repetitive. Favourites need no question: they are a soft
        ranking boost that every hard filter still gates, and a member who
        does not want them says so once, held by PlanningState.
        """
        calls = []
        orch = self._orchestrator(session_service, calls)
        session = self._session_with_favorites(session_service, sample_profile)

        turn = orch.process(session.session_id, session.member_id, "plan my day")
        assert turn.intent == "daily_plan"
        assert len(calls) == 1, "the plan generates immediately, no interception"

    def test_decline_generates_without_seeds(self, session_service, sample_profile):
        calls = []
        orch = self._orchestrator(session_service, calls)
        session = self._session_with_favorites(session_service, sample_profile)

        orch.process(session.session_id, session.member_id, "plan my day")
        orch.process(session.session_id, session.member_id, "no thanks")
        assert calls[0]["seeds"] == []

    def test_no_offer_without_favorites(self, session_service, sample_profile):
        calls = []
        orch = self._orchestrator(session_service, calls)
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)

        turn = orch.process(session.session_id, session.member_id, "plan my day")
        assert turn.intent == "daily_plan"
        assert len(calls) == 1

    def test_repeat_plan_requests_never_ask_anything(self, session_service, sample_profile):
        calls = []
        orch = self._orchestrator(session_service, calls)
        session = self._session_with_favorites(session_service, sample_profile)

        orch.process(session.session_id, session.member_id, "plan my day")
        turn = orch.process(session.session_id, session.member_id, "another plan")
        assert turn.intent == "daily_plan"
        assert len(calls) == 2, "every plan request plans; none interrogates"

    def test_explicit_seeds_suppress_offer(self, session_service, sample_profile):
        calls = []
        orch = self._orchestrator(session_service, calls)
        session = self._session_with_favorites(session_service, sample_profile)

        orch.seed_service.extract_seeds = lambda message: [{"name": "fakes"}]
        turn = orch.process(session.session_id, session.member_id, "plan a day with fakes")
        assert turn.intent == "daily_plan"
        assert calls[0]["seeds"] == [{"name": "fakes"}]


class TestPlanQuestion:
    """plan_question turns are answered ABOUT the canvas, never by refining it."""

    def _orchestrator(self, session_service):
        from services.orchestrator_service import OrchestratorService

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service
        orch.weekly_plan_service = None
        orch.chat_service = None
        orch.memory_service = None

        class Classifier:
            def classify(self, message, history):
                return {"intent": "plan_question", "target_plan_type": None}

        orch.orchestrator = Classifier()

        class FakeAnalyst:
            def __init__(self):
                self.calls = []

            def answer(self, question, plan_summary, history=None):
                self.calls.append({"question": question, "summary": plan_summary})
                return "Yes — your plan averages 95g protein/day."

        orch.plan_analyst = FakeAnalyst()

        class FakeFoodScholar:
            def __init__(self):
                self.questions = []

            def process_question(self, session_id, message, question=None, raw_question=None):
                self.questions.append(question or message)

                class T:
                    text = "evidence answer"
                    needs_clarification = False
                    attribution = None

                return T()

        orch.foodscholar_service = FakeFoodScholar()
        return orch

    def test_question_about_plan_is_answered_not_refined(self, session_service, sample_profile):
        from conftest import make_candidates

        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        session_service.add_meal_plan(session.session_id, make_candidates("q"), "r", {})

        orch = self._orchestrator(session_service)
        turn = orch.process(session.session_id, session.member_id,
                            "does my meal plan adhere to that?")

        assert turn.intent == "plan_question"
        assert "95g protein" in turn.content
        call = orch.plan_analyst.calls[0]
        assert "Daily plan" in call["summary"]      # grounded in the canvas
        assert orch.foodscholar_service.questions == []
        # The Q&A is persisted to the conversation
        restored = session_service.get_session(session.session_id)
        assert restored.conversation[-1].content == turn.content

    def test_no_canvas_falls_through_to_foodscholar(self, session_service, sample_profile):
        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        orch = self._orchestrator(session_service)

        turn = orch.process(session.session_id, session.member_id,
                            "does a protein-rich diet help with satiety?")
        assert turn.intent == "nutrition_question"
        assert orch.foodscholar_service.questions != []
        assert orch.plan_analyst.calls == []


class TestSeedDescription:
    """A re-injected pin must not claim the member asked for it this turn."""

    @staticmethod
    def _resolution(title, kept):
        from types import SimpleNamespace
        from services.seed_service import SeedResolution
        recipe = SimpleNamespace(recipe=SimpleNamespace(title=title))
        return SeedResolution(requested_name=title, status="resolved",
                              recipe=recipe, kept=kept)

    def test_freshly_asked_seeds_say_as_you_asked(self):
        from services.seed_service import SeedService
        note = SeedService.describe([self._resolution("Pastitsio", kept=False)])
        assert "worked in “Pastitsio” as you asked" in note
        assert "kept" not in note

    def test_carried_over_pins_say_kept_and_offer_to_change(self):
        from services.seed_service import SeedService
        note = SeedService.describe([self._resolution("Pastitsio", kept=True)])
        assert "I kept “Pastitsio”" in note
        assert "as you asked" not in note
        assert "changed too" in note

    def test_mixed_turn_separates_the_two(self):
        from services.seed_service import SeedService
        note = SeedService.describe([
            self._resolution("Moussaka", kept=False),
            self._resolution("Pastitsio", kept=True),
        ])
        assert "worked in “Moussaka” as you asked" in note
        assert "I kept “Pastitsio”" in note
