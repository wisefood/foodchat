"""M6 — meal classification and per-day weekly plan summaries (LLM-free)."""

import uuid

from models.recipe import CandidateRecipe, RecipeEnrichment
from services.weekly_planner.day_summary import (
    build_day_summaries,
    classify_meal,
    is_meat_meal,
    summarize_day,
)


def _entry(day, meal_idx, meal_type, recipe):
    return {"day": day, "meal_idx": meal_idx, "meal_type": meal_type,
            "recipe": recipe, "reward": 0.0}


class TestClassifyMeal:
    def test_rw_tags_are_authoritative(self):
        # Tag wins even when the title would keyword-match ("chick" ≠ chicken,
        # but make it explicit with a real conflict).
        recipe = {"recipe_title": "Chicken-style Seitan Roast",
                  "recipe_ingredients": "seitan, spices", "tags": ["vegan"]}
        assert classify_meal(recipe) == "vegan"

    def test_vegetarian_tag(self):
        assert classify_meal({"recipe_title": "Halloumi Wrap",
                              "tags": ["vegetarian", "gluten_free"]}) == "vegetarian"

    def test_fish_by_ingredients_without_tags(self):
        assert classify_meal({"recipe_title": "Poke Bowl",
                              "recipe_ingredients": "rice, salmon, soy"}) == "fish"

    def test_red_meat_and_poultry(self):
        assert classify_meal({"recipe_title": "Beef Stew",
                              "recipe_ingredients": "beef, potato"}) == "red meat"
        assert classify_meal({"recipe_title": "Roast Chicken",
                              "recipe_ingredients": "chicken, thyme"}) == "poultry"

    def test_overlay_display_keys_supported(self):
        # After the adapted-recipe overlay, recipes carry title/ingredients.
        assert classify_meal({"title": "Tuna Melt",
                              "ingredients": "tuna, cheese, bread"}) == "fish"

    def test_no_meat_signal_defaults_to_vegetarian(self):
        assert classify_meal({"recipe_title": "Lentil Curry",
                              "recipe_ingredients": "lentils, coconut"}) == "vegetarian"

    def test_word_boundary_matching(self):
        # "meatless" must not substring-match "meat"; "graham" not "ham".
        assert classify_meal({"recipe_title": "Meatless Monday Chili",
                              "recipe_ingredients": "beans, graham cracker crust"}) == "vegetarian"


class TestIsMeatMeal:
    def test_veg_tag_overrides_keywords(self):
        assert not is_meat_meal("Beef-style Tofu", "tofu, soy", tags=["vegetarian"])

    def test_pescatarian_fish_exemption(self):
        assert is_meat_meal("Grilled Salmon", "salmon, lemon", count_fish=True)
        assert not is_meat_meal("Grilled Salmon", "salmon, lemon", count_fish=False)
        # Red meat still counts even when fish doesn't
        assert is_meat_meal("Beef Stew", "beef", count_fish=False)


class TestSummarizeDay:
    def _veg(self, kcal=None):
        recipe = {"recipe_title": "Veggie Bowl", "recipe_ingredients": "rice, beans",
                  "tags": ["vegetarian"]}
        if kcal is not None:
            recipe["nutrition"] = {"kcal": kcal}
        return recipe

    def test_single_standout_names_the_meal(self):
        day = [
            _entry(1, 0, "breakfast", self._veg()),
            _entry(1, 1, "lunch", self._veg()),
            _entry(1, 2, "dinner", {"recipe_title": "Steak Frites",
                                    "recipe_ingredients": "steak, potato"}),
        ]
        assert summarize_day(day) == "dinner with red meat"

    def test_all_vegetarian_day(self):
        day = [_entry(1, i, m, self._veg()) for i, m in
               enumerate(["breakfast", "lunch", "dinner"])]
        assert summarize_day(day) == "vegetarian day"

    def test_light_qualifier_from_kcal(self):
        day = [_entry(1, i, m, self._veg(kcal=350)) for i, m in
               enumerate(["breakfast", "lunch", "dinner"])]
        assert summarize_day(day) == "light vegetarian day"

    def test_no_qualifier_when_nutrition_missing(self):
        day = [
            _entry(1, 0, "breakfast", self._veg(kcal=350)),
            _entry(1, 1, "lunch", self._veg()),
            _entry(1, 2, "dinner", self._veg()),
        ]
        # Only 1 of 3 meals has kcal — not enough signal for a qualifier.
        assert summarize_day(day) == "vegetarian day"

    def test_dominant_category_day(self):
        day = [
            _entry(1, 0, "breakfast", self._veg()),
            _entry(1, 1, "lunch", {"recipe_title": "Tuna Salad",
                                   "recipe_ingredients": "tuna"}),
            _entry(1, 2, "dinner", {"recipe_title": "Baked Cod",
                                    "recipe_ingredients": "cod"}),
        ]
        assert summarize_day(day) == "fish day"

    def test_two_categories_named(self):
        day = [
            _entry(1, 0, "breakfast", self._veg()),
            _entry(1, 1, "lunch", {"recipe_title": "Tuna Salad",
                                   "recipe_ingredients": "tuna"}),
            _entry(1, 2, "dinner", {"recipe_title": "Beef Stew",
                                    "recipe_ingredients": "beef"}),
        ]
        assert summarize_day(day) == "fish and red meat"

    def test_empty_day(self):
        assert summarize_day([]) == ""


class TestBuildDaySummaries:
    def test_covers_all_days(self):
        entries = [
            _entry(d, i, m, {"recipe_title": "Lentil Curry",
                             "recipe_ingredients": "lentils"})
            for d in range(1, 8) for i, m in enumerate(["breakfast", "lunch", "dinner"])
        ]
        summaries = build_day_summaries(entries)
        assert sorted(summaries) == list(range(1, 8))
        assert all(s == "vegetarian day" for s in summaries.values())


class FakeCandidates:
    """RecipeWrangler stand-in: 1 recipe per slot per fetch, enrichment by slot."""

    ENRICHMENT = {
        "breakfast": dict(title="Oat Bowl", kcal=300.0, tags=["vegetarian"]),
        "lunch": dict(title="Grilled Salmon", kcal=450.0, tags=[]),
        "dinner": dict(title="Beef Stew", kcal=700.0, tags=[]),
    }
    INGREDIENTS = {"breakfast": "oats, milk", "lunch": "salmon, lemon",
                   "dinner": "beef, potato"}

    def __init__(self):
        self.calls = 0

    def fetch_candidates(self, **kwargs):
        self.calls += 1
        return {
            slot: [CandidateRecipe(f"{slot}-{self.calls}", spec["title"],
                                   self.INGREDIENTS[slot], "cook")]
            for slot, spec in self.ENRICHMENT.items()
        }

    def fetch_details(self, recipe_ids):
        out = {}
        for rid in recipe_ids:
            slot = str(rid).split("-")[0]
            spec = self.ENRICHMENT.get(slot)
            if spec:
                out[str(rid)] = RecipeEnrichment(
                    recipe_id=str(rid), title=spec["title"], kcal=spec["kcal"],
                    tags=list(spec["tags"]), dish_types=[slot], allergens=[],
                )
        return out


class TestWeeklyServiceDaySummaries:
    """Service-level wiring: enrichment tags reach entries, summaries persist."""

    def _service(self, session_service, monkeypatch):
        # NB: `from services import weekly_plan_service` would resolve to the
        # init-time singleton attribute (None in tests), not the module.
        import importlib

        wps = importlib.import_module("services.weekly_plan_service")
        aa = importlib.import_module("services.weekly_planner.action_adapter")
        from services.weekly_planner.reward_logic import RewardCalculator

        fake = FakeCandidates()
        monkeypatch.setattr(wps, "CANDIDATES", fake)
        monkeypatch.setattr(aa, "CANDIDATES", fake)

        svc = wps.WeeklyPlanService.__new__(wps.WeeklyPlanService)
        svc.session_service = session_service
        svc.reward_calculator = RewardCalculator()

        class NoDiet:
            def extract(self, content):
                return []

        class NoSignals:
            def get_signals(self, member_id):
                class S:
                    downvoted_recipe_ids = []
                return S()

        class EchoWriter:
            def write(self, facts, content, fallback=""):
                self.facts = facts
                return fallback or "ok"

        svc.diet_extractor = NoDiet()
        svc.seed_service = None      # unused: no seeds, no standing seeds
        svc.feedback_service = NoSignals()
        svc.response_writer = EchoWriter()
        return svc

    def test_summaries_generated_persisted_and_exposed(self, session_service, monkeypatch):
        profile = {"diet": [], "allergies": [], "preferences": [],
                   "food_likes": [], "food_dislikes": []}
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)

        svc = self._service(session_service, monkeypatch)
        _, plan = svc.process_message(session.session_id, "plan my week")

        assert len(plan.entries) == 21
        # Enrichment tags/dish_types reach the stored entries
        breakfast = next(e for e in plan.entries if e["meal_type"] == "breakfast")
        assert breakfast["recipe"]["tags"] == ["vegetarian"]
        assert breakfast["recipe"]["dish_types"] == ["breakfast"]
        # Every day gets a headline; salmon lunch + beef dinner => two categories
        assert sorted(plan.day_summaries) == list(range(1, 8))
        assert all(s == "fish and red meat" for s in plan.day_summaries.values())
        # The chat reply facts carry readable day lines
        facts = svc.response_writer.facts
        assert facts["day_summaries"][0] == "Monday: fish and red meat"

        # Survives a DB round-trip on a fresh service (replica restart)
        from services.session_service import SessionService
        restored = SessionService().get_session(session.session_id)
        restored_plan = restored.get_current_weekly_plan()
        assert restored_plan.day_summaries == plan.day_summaries
        assert all(isinstance(k, int) for k in restored_plan.day_summaries)

        # API response model exposes the field additively
        from routers.foodchat_router import WeeklyMealPlanResponse
        response = WeeklyMealPlanResponse.from_weekly_meal_plan(restored_plan)
        assert response.day_summaries == plan.day_summaries
        assert len(response.entries) == 21

    def test_pre_m6_plans_deserialize_with_empty_summaries(self, session_service):
        profile = {"diet": [], "allergies": [], "preferences": []}
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)
        entries = [
            _entry(d, i, m, {"recipe_id": f"w-{d}-{i}", "recipe_title": "Meal",
                             "recipe_ingredients": "x", "recipe_directions": "y"})
            for d in range(1, 8) for i, m in enumerate(["breakfast", "lunch", "dinner"])
        ]
        session_service.add_weekly_meal_plan(session.session_id, entries)

        from services.session_service import SessionService
        restored = SessionService().get_session(session.session_id)
        plan = restored.get_current_weekly_plan()
        assert plan.day_summaries == {}

        from routers.foodchat_router import WeeklyMealPlanResponse
        assert WeeklyMealPlanResponse.from_weekly_meal_plan(plan).day_summaries == {}
