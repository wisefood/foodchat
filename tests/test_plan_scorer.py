"""Plan scorer, steps 1–3 — parsing, grounding, building, and the score_plan turn.

LLM-free: the parser, the recipe lookup and the classifier are fakes; the
line scanner, the grounding rules, the builders and the routing are real.
"""

import uuid

import pytest
from types import SimpleNamespace

from models.pasted_plan import (
    APPROXIMATE,
    AS_WRITTEN,
    CLOSEST_RECIPE,
    FROM_RECIPE,
    MATCHED,
    UNKNOWN,
    UNRESOLVED,
    GroundedMeal,
)
from models.recipe import CandidateRecipe, ProfiledNutrition, RecipeEnrichment, ResolvedRecipe
from services.candidates_client import ProfilingTimeout, profile_nutrition
from services.orchestrator_service import ChatTurn, OrchestratorService
from services.plan_scorer.building import PASTED_ID_PREFIX, build_daily, build_weekly, recipe_dict
from services.plan_scorer.grounding import (
    MAX_ESTIMATED_DISHES,
    MAX_PROFILE_CALLS,
    PROFILE_TIMEOUT_SECONDS,
    DishGrounder,
    borrowable,
    same_dish,
    title_similarity,
)
from services.plan_scorer.parsing import (
    content_tokens,
    days_from_answer,
    dish_heads,
    looks_like_plan_listing,
    needs_shape_question,
    parse_plan_text,
    prepass,
    scan,
    spelling_variants,
    split_into_days,
)
from services.plan_scorer.scoring import ScoreResult
from services.plan_scorer.service import CLARIFICATION_KIND, PlanScorerService, ScoreTurn
from services.seed_service import SeedService
from services.weekly_planner.day_summary import classify_meal
from services.weekly_planner.explainability import (
    REPEAT_BY_AUTHOR,
    REPEAT_KIND,
    build_weekly_explainability,
    guideline_checklist,
    variety_metrics,
)
from test_orchestrator_routing import QueuedClassifier, make_orchestrator


def read(plan):
    return [(day.day, meal.slot, meal.title) for day in plan.days for meal in day.meals]


class NoParser:
    """A parser that is down: every call returns None."""

    def __init__(self):
        self.calls = []

    def parse(self, text, structure_hint=""):
        self.calls.append(text)
        return None


class ForbiddenParser:
    def parse(self, text, structure_hint=""):
        raise AssertionError("the parser must not be called for structured text")


class FixedParser:
    def __init__(self, payload):
        self.payload = payload
        self.hints = []

    def parse(self, text, structure_hint=""):
        self.hints.append(structure_hint)
        return self.payload


# --------------------------------------------------------------------- #
# Step 1 — the line scanner                                              #
# --------------------------------------------------------------------- #

class TestScanner:
    def test_day_headings_in_three_styles(self):
        plan = prepass("Monday\nbreakfast: oats\nDay 2\nlunch: lentil soup\nWed: dinner: grilled salmon")

        assert plan.plan_type == "weekly"
        assert read(plan) == [
            (1, "breakfast", "oats"),
            (2, "lunch", "lentil soup"),
            (3, "dinner", "grilled salmon"),
        ]

    def test_slots_on_one_line_with_no_day_heading_are_one_day(self):
        plan = prepass("breakfast: oats, lunch: lentil soup, dinner: salmon")

        assert plan.plan_type == "daily"
        assert read(plan) == [
            (None, "breakfast", "oats"),
            (None, "lunch", "lentil soup"),
            (None, "dinner", "salmon"),
        ]

    def test_bullets_under_a_slot_heading_and_parentheses(self):
        plan = prepass(
            "Breakfast:\n- porridge (oats, milk)\n- banana\nDinner - chickpea curry (2 servings)"
        )
        meals = plan.meals

        assert [(m.slot, m.title) for m in meals] == [
            ("breakfast", "porridge"), ("breakfast", "banana"), ("dinner", "chickpea curry"),
        ]
        assert meals[0].ingredients == "oats, milk"
        assert meals[2].ingredients is None
        assert meals[2].quantity_note == "2 servings"

    def test_preamble_and_questions_are_ignored_but_stray_lines_are_kept_verbatim(self):
        plan = prepass(
            "Here is my week, how does it look?\nMonday\nbreakfast: oats\n"
            "went to the gym\nlunch: soup\nthoughts?"
        )

        assert read(plan) == [(1, "breakfast", "oats"), (1, "lunch", "soup")]
        assert plan.unparsed == ["went to the gym"]

    def test_sun_dried_tomatoes_are_not_sunday(self):
        plan = prepass("breakfast: toast\nSun-dried tomato pasta for dinner")

        assert plan.plan_type == "daily"
        assert [d.day for d in plan.days] == [None]
        assert ("dinner", "Sun-dried tomato pasta") in [(m.slot, m.title) for m in plan.meals]

    def test_a_dish_named_like_a_verb_keeps_its_name(self):
        plan = prepass("oats for breakfast, haddock for dinner")

        assert [m.title for m in plan.meals] == ["oats", "haddock"]

    def test_the_sentence_introducing_a_listing_is_not_a_dish(self):
        """Live: the first dish was read as "Rate for me a daily plan
        consisting of fried eggs"."""
        plan = prepass(
            "Rate for me a daily plan consisting of fried eggs for breakfast, "
            "pasta with zucchini for lunch and chicken noodle soup for dinner"
        )

        assert [(m.slot, m.title) for m in plan.meals] == [
            ("breakfast", "fried eggs"),
            ("lunch", "pasta with zucchini"),
            ("dinner", "chicken noodle soup"),
        ]

    def test_an_ordinary_dish_keeps_its_words(self):
        plan = prepass("a bowl of porridge for breakfast, beans on toast for dinner")

        assert [m.title for m in plan.meals] == ["a bowl of porridge", "beans on toast"]

    def test_prose_is_read_but_only_as_a_hint(self):
        result = scan("rate this: oats for breakfast, lentil soup for lunch and salmon for dinner")

        assert result.used_prose
        assert [(m.slot, m.title) for m in result.plan.meals] == [
            ("breakfast", "oats"), ("lunch", "lentil soup"), ("dinner", "salmon"),
        ]

    def test_a_heading_listed_twice_is_one_day(self):
        plan = prepass("Monday\nbreakfast: oats\nTuesday\nlunch: soup\nMonday\ndinner: fish")

        assert [(d.day, len(d.meals)) for d in plan.days] == [(1, 2), (2, 1)]

    def test_days_past_seven_are_named_not_scored(self):
        plan = prepass("Day 1\nlunch: soup\nDay 8\nlunch: salad")

        assert read(plan) == [(1, "lunch", "soup")]
        assert any("Day 8" in w for w in plan.warnings)


class TestListingDetection:
    def test_two_slots_is_a_listing(self):
        assert looks_like_plan_listing("breakfast: oats, dinner: salmon")

    def test_a_question_about_dinner_is_not(self):
        assert not looks_like_plan_listing("what should I have for dinner?")

    def test_one_dish_is_not(self):
        assert not looks_like_plan_listing("dinner: pasta")


# --------------------------------------------------------------------- #
# Step 1 — the parser and the checks against the text                   #
# --------------------------------------------------------------------- #

class TestParsePlanText:
    def test_structured_text_costs_no_model_call(self):
        plan = parse_plan_text("Monday\nbreakfast: oats\nlunch: soup", ForbiddenParser())

        assert read(plan) == [(1, "breakfast", "oats"), (1, "lunch", "soup")]

    def test_the_parser_reading_is_checked_against_the_text(self):
        text = "Mon: had chicken curry and a salad\nTue: pasta with pesto\nhow is it?"
        parser = FixedParser({
            "plan_type": "weekly",
            "days": [
                {"day": 1, "label": "Mon", "meals": [
                    {"slot": "dinner", "title": "chicken curry",
                     "ingredients": "chicken, garlic, coconut milk"},
                    {"slot": "dinner", "title": "salad"},
                    {"slot": "lunch", "title": "beef burger"},
                ]},
                {"day": None, "label": "Tue", "meals": [
                    {"slot": "Supper", "title": "pasta with pesto", "ingredients": "pasta, pesto"},
                ]},
            ],
            "unparsed": ["had chicken curry and a salad", "something the model made up"],
        })

        plan = parse_plan_text(text, parser)

        assert parser.hints, "the scanner could not place a line, so the parser runs"
        assert read(plan) == [
            (1, "dinner", "chicken curry"),
            (1, "dinner", "salad"),
            (2, "dinner", "pasta with pesto"),
        ]
        # Garlic and coconut milk are not in the text: the whole list goes.
        assert plan.meals[0].ingredients is None
        assert plan.meals[2].ingredients == "pasta, pesto"
        assert plan.unparsed == ["had chicken curry and a salad"]
        assert any("not in your text" in w for w in plan.warnings)
        assert any("Ingredient lists" in w for w in plan.warnings)

    def test_a_failed_parser_leaves_the_scanner_reading_with_a_warning(self):
        parser = NoParser()
        plan = parse_plan_text("oats for breakfast, soup for lunch", parser)

        assert len(parser.calls) == 1
        assert [m.title for m in plan.meals] == ["oats", "soup"]
        assert any("line by line" in w for w in plan.warnings)

    def test_an_explicit_weekly_type_splits_instead_of_asking(self):
        text = "breakfast: oats\nlunch: soup\nbreakfast: eggs\nlunch: salad"
        plan = parse_plan_text(text, ForbiddenParser(), plan_type="weekly")

        assert plan.plan_type == "weekly"
        assert [len(d.meals) for d in plan.days] == [2, 2]


class TestShape:
    SIX = "breakfast: oats\nlunch: soup\ndinner: fish\nbreakfast: eggs\nlunch: salad\ndinner: tofu"

    def test_several_repeated_meals_with_no_heading_is_a_question(self):
        assert needs_shape_question(prepass(self.SIX))

    def test_two_plates_at_one_meal_is_not(self):
        assert not needs_shape_question(prepass("dinner: pasta\ndinner: salad\nlunch: soup"))

    def test_labelled_days_are_not(self):
        assert not needs_shape_question(prepass("Monday\n" + self.SIX))

    def test_answers(self):
        assert days_from_answer("it's 3 days") == 3
        assert days_from_answer("two days") == 2
        assert days_from_answer("the whole week") == 7
        assert days_from_answer("just one day") == 1
        assert days_from_answer("several days") == 0
        assert days_from_answer("sure") is None

    def test_split_where_a_meal_repeats_and_say_so_when_counts_differ(self):
        plan = split_into_days(prepass(self.SIX), 3)

        assert plan.plan_type == "weekly"
        assert [(d.day, len(d.meals)) for d in plan.days] == [(1, 3), (2, 3)]
        assert any("You said 3 day(s)" in w for w in plan.warnings)

    def test_one_day_keeps_everything_together(self):
        plan = split_into_days(prepass(self.SIX), 1)

        assert plan.plan_type == "daily" and len(plan.days) == 1 and plan.meal_count == 6


# --------------------------------------------------------------------- #
# Step 2 — grounding                                                     #
# --------------------------------------------------------------------- #

def _resolved(rid, title, ingredients, allergens=(), tags=(), nutrition=None):
    return ResolvedRecipe(
        recipe=CandidateRecipe(rid, title, ingredients, "", nutrition=nutrition),
        dish_types=["dinner"], allergens=list(allergens), tags=list(tags),
    )


OATMEAL = _resolved("r-oat", "Berry Oatmeal", "rolled oats, blueberries, milk",
                    tags=("vegetarian",), nutrition={"kcal": 320})
CURRY = _resolved("r-curry", "Thai Green Curry with Chicken", "chicken, coconut milk, curry paste")
NOODLES = _resolved("r-pn", "Peanut Noodles", "noodles, peanut butter, soy sauce",
                    allergens=("peanuts",))


class FakeSeeds:
    """find_dish + fetch_recipe + fetch_details stand-in (word-overlap search)."""

    def __init__(self, recipes, details=None, fail=False, profiles=None):
        self.recipes = recipes
        self.details = details or {}
        self.fail = fail
        self.profiles = profiles
        self.queries = []
        self.detail_calls = []
        self.profile_calls = []
        self.profile_timeouts = []
        self.client = self

    def profile_recipe(self, raw_recipe, region=None, timeout=None):
        """Keyed by the recipe text's first line, the dish title. A plain dict
        in ``profiles`` is a fully covered profile."""
        if self.profiles is None:
            raise AssertionError("profile_recipe called on a client without profiles")
        self.profile_calls.append(raw_recipe)
        self.profile_timeouts.append(timeout)
        found = self.profiles.get(raw_recipe.splitlines()[0].strip().lower())
        if isinstance(found, dict):
            return ProfiledNutrition(nutrition=found, coverage=1.0)
        return found

    def find_dish(self, name, limit=5):
        self.queries.append(name)
        if self.fail:
            raise RuntimeError("RecipeWrangler down")
        words = set(name.lower().split())
        return [r.recipe for r in self.recipes if words & set(r.recipe.title.lower().split())][:limit]

    def fetch_recipe(self, recipe_id):
        return next((r for r in self.recipes if r.recipe.recipe_id == recipe_id), None)

    def fetch_details(self, ids):
        self.detail_calls.append(list(ids))
        return {i: self.details[i] for i in ids if i in self.details}


def ground(text, recipes, profile=None, **fake):
    seeds = FakeSeeds(recipes, **fake)
    grounded = DishGrounder(seeds).ground(prepass(text), profile or {})
    return grounded, seeds


class TestGrounding:
    def test_similarity_is_word_overlap_not_word_order(self):
        assert title_similarity("oatmeal with berries", "Berry Oatmeal") == 1.0
        assert 0.4 <= title_similarity("chicken curry", "Thai Green Curry with Chicken") < 0.75
        # Live miss: "cups" names the catalogue dish; "a bowl of" names nothing.
        assert title_similarity("porridge with banana and honey", "Honey banana cups") < 0.75
        assert title_similarity("a bowl of porridge", "Porridge") == 1.0

    def test_matched_dish_is_scored_as_the_recipe(self):
        (meal,), _ = ground("breakfast: oatmeal with berries", [OATMEAL])

        assert meal.state == MATCHED
        assert meal.recipe_id == "r-oat" and meal.title_matched == "Berry Oatmeal"
        assert meal.ingredients_source == FROM_RECIPE
        assert "blueberries" in meal.ingredients
        assert meal.nutrition == {"kcal": 320}
        assert meal.tags == ["vegetarian"]

    def test_a_close_match_never_lends_its_ingredients(self):
        """Live: "vegetable lasagna" read as a beef lasagna failed a vegetarian
        profile on meat the member never ate."""
        (bare,), _ = ground("dinner: chicken curry", [CURRY])
        (written,), _ = ground("dinner: chicken curry (chicken, spinach)", [CURRY])

        assert bare.state == APPROXIMATE
        assert bare.ingredients == "" and bare.ingredients_source == UNKNOWN
        assert written.state == APPROXIMATE
        assert written.ingredients == "chicken, spinach"
        assert written.ingredients_source == AS_WRITTEN

    def test_a_generic_recipe_lends_its_figures_a_different_dish_does_not(self):
        lasagna = _resolved("r-las", "Lasagna", "beef mince, pasta sheets", tags=("contains_meat",),
                            nutrition={"kcal": 650})
        strudel = _resolved("r-str", "Apple strudel", "apples, pastry, sugar", nutrition={"kcal": 420})
        (veg,), _ = ground("dinner: vegetable lasagna", [lasagna])
        (apple,), _ = ground("snack: an apple", [strudel])

        assert veg.state == APPROXIMATE and veg.borrows_nutrition
        assert veg.nutrition == {"kcal": 650}
        assert veg.ingredients == "" and veg.tags == []
        assert apple.state == APPROXIMATE and not apple.borrows_nutrition
        assert apple.nutrition is None

    def test_similarity_is_measured_on_the_recipe_actually_fetched(self):
        class Mismatched(FakeSeeds):
            def fetch_recipe(self, recipe_id):
                return _resolved(recipe_id, "Honey banana cups", "banana, honey, yoghurt")

        seeds = Mismatched([_resolved("r-por", "Porridge with banana and honey", "oats, banana, honey")])
        (meal,) = DishGrounder(seeds).ground(prepass("breakfast: porridge with banana and honey"), {})

        assert meal.state == APPROXIMATE
        assert meal.title_matched == "Honey banana cups" and not meal.borrows_nutrition

    def test_an_allergen_only_in_a_close_match_is_a_possibility(self):
        satay = _resolved("r-sat", "Chicken satay with peanut sauce", "chicken, peanuts",
                          allergens=("peanuts",))
        (meal,), _ = ground("dinner: chicken skewers with sauce", [satay], {"allergies": ["peanuts"]})

        assert meal.state == APPROXIMATE
        assert meal.allergen_conflicts == [{"allergen": "peanuts", "evidence": CLOSEST_RECIPE}]

    def test_unresolved_dish_knows_only_the_words(self):
        (meal,), _ = ground("dinner: nonna's stew", [OATMEAL, CURRY])

        assert meal.state == UNRESOLVED
        assert meal.recipe_id is None and meal.nutrition is None
        assert meal.ingredients_source == UNKNOWN

    def test_an_allergen_is_grounded_and_reported_not_dropped(self):
        (meal,), _ = ground("lunch: peanut noodles", [NOODLES], {"allergies": ["peanuts"]})

        assert meal.state == MATCHED
        assert [c["allergen"] for c in meal.allergen_conflicts] == ["peanuts"]

    def test_the_recipe_alone_can_carry_the_allergen(self):
        almond = _resolved("r-alm", "Crumbed Chicken", "chicken, almond meal")
        (meal,), _ = ground("dinner: crumbed chicken", [almond], {"allergies": ["tree nuts"]})

        assert meal.allergen_conflicts == [{"allergen": "tree nuts", "evidence": FROM_RECIPE}]

    def test_one_lookup_per_distinct_dish(self):
        grounded, seeds = ground(
            "Monday\nbreakfast: oatmeal with berries\nTuesday\nbreakfast: Oatmeal with berries",
            [OATMEAL],
        )

        assert len(grounded) == 2 and len(seeds.queries) == 1

    def test_missing_nutrition_is_filled_in_one_batch(self):
        bare = _resolved("r-soup", "Lentil Soup", "lentils, carrot")
        rich = RecipeEnrichment(recipe_id="r-soup", title="Lentil Soup", kcal=410.0, protein_g=21.0,
                                tags=["vegan"])
        grounded, seeds = ground("lunch: lentil soup\ndinner: lentil soup", [bare],
                                 details={"r-soup": rich})

        assert seeds.detail_calls == [["r-soup"]]
        assert grounded[0].nutrition["kcal"] == 410.0
        assert grounded[0].tags == ["vegan"]

    def test_a_lookup_failure_leaves_the_dish_unresolved(self):
        (meal,), _ = ground("dinner: chicken curry", [CURRY], fail=True)

        assert meal.state == UNRESOLVED


class TestFindDish:
    def _service(self, suggestions):
        svc = SeedService.__new__(SeedService)
        svc.extractor = None

        class Client:
            def autocomplete(self, name, limit=5):
                return suggestions

        svc.client = Client()
        return svc

    def test_search_is_not_filtered_by_the_member(self, monkeypatch):
        seen = []
        hit = CandidateRecipe("r-pn", "Peanut Noodles", "peanut butter", "")
        monkeypatch.setattr(SeedService, "_search_by_name",
                            lambda self, name, profile: seen.append(profile) or [hit])

        assert self._service([]).find_dish("peanut noodles") == [hit]
        assert seen == [{}]

    def test_autocomplete_is_the_fallback(self, monkeypatch):
        monkeypatch.setattr(SeedService, "_search_by_name", lambda self, name, profile: [])

        found = self._service([("r-1", "Pastitsio")]).find_dish("pastitsio")

        assert [(c.recipe_id, c.title) for c in found] == [("r-1", "Pastitsio")]


# --------------------------------------------------------------------- #
# Step 3 — building                                                      #
# --------------------------------------------------------------------- #

def gm(day, slot, title, state=MATCHED, rid=None, ingredients="", tags=(), nutrition=None):
    return GroundedMeal(
        day=day, slot=slot, title_given=title, state=state, recipe_id=rid,
        title_matched=title if rid else None, ingredients=ingredients,
        ingredients_source=FROM_RECIPE if rid else UNKNOWN,
        tags=list(tags), nutrition=nutrition,
    )


class TestBuildWeekly:
    WEEK = [
        gm(1, "breakfast", "Oats", rid="r-oat", ingredients="oats, milk"),
        gm(1, "dinner", "Salmon", rid="r-sal", ingredients="salmon, lemon"),
        gm(1, "snack", "Apple", state=UNRESOLVED),
        gm(2, "breakfast", "Oats", rid="r-oat", ingredients="oats, milk"),
        gm(2, "lunch", "Lentil soup", state=UNRESOLVED, ingredients="lentils"),
    ]

    def test_entries_have_the_planner_shape_and_snacks_stay_apart(self):
        built = build_weekly(self.WEEK)

        assert [(e["day"], e["meal_idx"], e["meal_type"]) for e in built.entries] == [
            (1, 0, "breakfast"), (1, 2, "dinner"), (2, 0, "breakfast"), (2, 1, "lunch"),
        ]
        assert [e["meal_type"] for e in built.extras] == ["snack"]
        assert built.days == 2
        assert built.entries[0]["recipe"]["title"] == "Oats"
        assert built.entries[0]["recipe"]["pasted"] is True

    def test_a_dish_on_two_days_is_the_members_own_repeat(self):
        built = build_weekly(self.WEEK)
        repeat = built.entries[2]["recipe"]

        assert built.author_repeats == 1
        assert repeat["repeat_of_day"] == 1 and repeat["repeat_source"] == REPEAT_BY_AUTHOR
        metrics = variety_metrics(built.entries)
        assert metrics["planned_repeats"] == 1
        assert metrics["unexplained_repeats"] == 0

    def test_the_weekly_explainability_runs_and_does_not_hold_them_to_the_cooldown(self):
        """Adjacent days would violate the planner's gap. It is not their policy."""
        built = build_weekly(self.WEEK)
        result = build_weekly_explainability(built.entries, {"preferences": []})

        row = next(r for r in result["constraints_applied"] if "repeat" in r["constraint"])
        assert row["constraint"] == "repeats are your own choice"
        assert row["status"] == "satisfied" and row["source"] == "your own plan"
        assert "1 of your own" in row["detail"]
        chip = next(
            r for r in built.entries[2]["recipe"]["match_reasons"] if r["kind"] == REPEAT_KIND
        )
        assert chip["label"] == "the same breakfast as Monday, as you planned it"
        assert "1 your own choice" in result["reasoning"]

    def test_approximate_dishes_are_not_repeats_of_each_other(self):
        built = build_weekly([
            gm(1, "dinner", "Chicken curry", state=APPROXIMATE, rid="r-curry"),
            gm(3, "dinner", "Thai green curry", state=APPROXIMATE, rid="r-curry"),
        ])

        ids = [e["recipe"]["recipe_id"] for e in built.entries]
        assert all(i.startswith(PASTED_ID_PREFIX) for i in ids) and ids[0] != ids[1]
        assert built.author_repeats == 0

    def test_three_days_scale_the_guideline_targets(self):
        meals = [
            gm(day, slot, f"{slot} {day}", state=UNRESOLVED, ingredients="beans")
            for day in (1, 2, 3) for slot in ("breakfast", "lunch", "dinner")
        ]
        variety = variety_metrics(build_weekly(meals).entries)
        rows = guideline_checklist(variety["category_distribution"], variety["total_meals"])

        assert variety["total_meals"] == 9
        assert rows[2]["target"] == "at least 5 of 9 meals"

    def test_an_unresolved_dish_is_still_categorised_by_its_name(self):
        assert classify_meal(recipe_dict(gm(1, "dinner", "Grilled salmon", state=UNRESOLVED))) == "fish"


class TestBuildDaily:
    def test_a_complete_day_is_a_scored_plan(self):
        built = build_daily([
            gm(None, "breakfast", "Oats", rid="r-oat"),
            gm(None, "lunch", "Soup", state=UNRESOLVED),
            gm(None, "dinner", "Salmon", rid="r-sal"),
            gm(None, "snack", "Apple", state=UNRESOLVED),
        ])
        plan = built.as_scored_plan()

        assert plan is not None
        assert [c.title for c in plan.courses] == ["Oats", "Soup", "Salmon"]
        assert plan.lunch.recipe_id.startswith(PASTED_ID_PREFIX)
        assert [c.title for c in built.extras] == ["Apple"]

    def test_a_partial_or_two_plate_day_is_not(self):
        partial = build_daily([gm(None, "breakfast", "Oats"), gm(None, "dinner", "Salmon")])
        two_plates = build_daily([
            gm(None, "breakfast", "Oats"), gm(None, "lunch", "Soup"),
            gm(None, "dinner", "Pasta"), gm(None, "dinner", "Salad"),
        ])

        assert partial.as_scored_plan() is None and partial.missing_slots == ["lunch"]
        assert two_plates.as_scored_plan() is None


# --------------------------------------------------------------------- #
# The turn                                                               #
# --------------------------------------------------------------------- #

class EchoGrounder:
    """Every dish unresolved, as written; records the profile it was given."""

    def __init__(self, conflicts=None):
        self.profiles = []
        self.conflicts = conflicts or {}

    def ground(self, plan, profile):
        self.profiles.append(profile)
        return [
            GroundedMeal(
                day=day.day, slot=meal.slot, title_given=meal.title, state=UNRESOLVED,
                ingredients=meal.ingredients or "",
                ingredients_source=AS_WRITTEN if meal.ingredients else UNKNOWN,
                allergen_conflicts=list(self.conflicts.get(meal.title, [])),
            )
            for day in plan.days for meal in day.meals
        ]


class EmptyScorer:
    """Step-4 stand-in: no metrics and no rows — steps 1–3 are under test here."""

    def __init__(self):
        self.calls = []

    def score(self, plan_type, grounded, built, profile, context=None):
        self.calls.append({"plan_type": plan_type, "profile": profile, "context": context})
        return ScoreResult()


class FallbackWriter:
    """ResponseWriter stand-in that always answers with the fallback."""

    def __init__(self):
        self.facts = []

    def write(self, facts, user_message, fallback):
        self.facts.append(facts)
        return fallback


def scorer_service(session_service, **kwargs):
    kwargs.setdefault("scorer", EmptyScorer())
    kwargs.setdefault("writer", FallbackWriter())
    return PlanScorerService(session_service, **kwargs)


def new_session(session_service, sample_profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)


class TestScoreTurn:
    def test_a_pasted_week_is_read_persisted_and_touches_no_canvas(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        grounder = EchoGrounder()
        scorer = scorer_service(session_service, parser=NoParser(), grounder=grounder)

        turn = scorer.process(
            session.session_id, "Monday\nbreakfast: oats\nTuesday\nbreakfast: oats\nlunch: soup",
        )

        payload = turn.plan_score
        assert not turn.needs_clarification
        assert payload["plan_type"] == "weekly"
        assert (payload["days_scored"], payload["meals_scored"]) == (2, 3)
        assert payload["metrics"] == [] and payload["constraints_applied"] == []
        assert [row["state"] for row in payload["grounding"]] == [UNRESOLVED] * 3
        assert turn.scoring_input.author_repeats == 1
        assert grounder.profiles[0]["allergies"] == ["peanuts"]
        assert payload["scored_plan"]["origin"] == "pasted"
        assert [d["day"] for d in payload["scored_plan"]["days"]] == [1, 2]

        session = session_service.get_session(session.session_id)
        assert session.daily_canvas is None and session.weekly_canvas is None
        assert not session.meal_plans and not session.weekly_meal_plans
        assert [(m.role, m.intent) for m in session.conversation[-2:]] == [
            ("user", None), ("assistant", "score_plan"),
        ]

    def test_an_allergen_is_named_in_the_reply(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        grounder = EchoGrounder({"peanut noodles": [{"allergen": "peanuts", "evidence": AS_WRITTEN}]})
        scorer = scorer_service(session_service, parser=NoParser(), grounder=grounder)

        turn = scorer.process(session.session_id, "lunch: peanut noodles\ndinner: soup")

        assert "Heads up: “peanut noodles” contains peanuts" in turn.text
        assert turn.plan_score["grounding"][0]["allergen_conflicts"] == [
            {"allergen": "peanuts", "evidence": AS_WRITTEN}
        ]

    def test_nothing_readable_asks_once_and_keeps_the_text(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        scorer = scorer_service(session_service, parser=NoParser(), grounder=EchoGrounder())

        turn = scorer.process(session.session_id, "my usual food, nothing special")

        assert turn.needs_clarification
        session = session_service.get_session(session.session_id)
        assert session.state == "clarifying"
        assert session.clarification == {
            "kind": CLARIFICATION_KIND, "reason": "no_meals",
            "pasted_text": "my usual food, nothing special",
            "plan_type": "auto", "context": None,
        }

        answered = scorer.continue_clarification(session.session_id, "breakfast: oats\nlunch: soup")
        assert answered.plan_score["meals_scored"] == 2
        assert session_service.get_session(session.session_id).state == "ready"

    def test_a_reply_that_is_not_an_answer_is_handed_back(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        scorer = scorer_service(session_service, parser=NoParser(), grounder=EchoGrounder())
        scorer.process(session.session_id, "my usual food, nothing special")
        before = len(session_service.get_session(session.session_id).conversation)

        outcome = scorer.continue_clarification(session.session_id, "never mind")

        session = session_service.get_session(session.session_id)
        assert outcome.unresolved
        assert session.state == "ready"
        assert len(session.conversation) == before

    def test_the_shape_question_and_its_answer(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        scorer = scorer_service(session_service, parser=ForbiddenParser(), grounder=EchoGrounder())

        asked = scorer.process(session.session_id, TestShape.SIX)
        assert asked.needs_clarification
        assert "2 breakfasts, 2 lunches and 2 dinners" in asked.text
        assert session_service.get_session(session.session_id).clarification["reason"] == "shape"

        turn = scorer.continue_clarification(session.session_id, "it's 2 days")
        assert turn.plan_score["plan_type"] == "weekly"
        assert turn.plan_score["days_scored"] == 2

    def test_without_asking_nothing_readable_ends_the_turn(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        scorer = scorer_service(session_service, parser=NoParser(), grounder=EchoGrounder())

        turn = scorer.process(session.session_id, "my usual food", may_ask=False)

        assert not turn.needs_clarification and turn.plan_score is None
        assert session_service.get_session(session.session_id).state == "ready"


# --------------------------------------------------------------------- #
# Routing — the intent is recognised apart from every other one          #
# --------------------------------------------------------------------- #

class RecordingScorer:
    def __init__(self):
        self.processed = []

    def process(self, session_id, message, *, plan_type="auto", context=None, may_ask=True):
        self.processed.append((message, may_ask))
        return ScoreTurn(text="Read it.", plan_score={"plan_type": "daily"})


LISTING = "rate this:\nbreakfast: oats\nlunch: lentil soup\ndinner: salmon"


class TestRouting:
    def test_the_intent_exists_everywhere_the_router_checks(self):
        from agents import OrchestratorAgent
        from schemas import OrchestratorSchema

        assert "score_plan" in OrchestratorAgent.VALID_INTENTS
        assert OrchestratorSchema(intent="score_plan", reasoning="pasted").intent == "score_plan"

    def test_a_classified_score_plan_goes_to_the_scorer(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        classifier = QueuedClassifier("score_plan")
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(session.session_id, session.member_id,
                            "today I had oats, then a sandwich, then fish. ok?")

        assert turn.intent == "score_plan" and turn.plan_score == {"plan_type": "daily"}
        assert orch.plan_scorer.processed == [("today I had oats, then a sandwich, then fish. ok?", True)]

    def test_an_explicit_listing_skips_the_classifier(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        classifier = QueuedClassifier()
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(session.session_id, session.member_id, LISTING)

        assert turn.intent == "score_plan"
        assert classifier.calls == []

    def test_an_explicit_listing_supersedes_a_pending_question(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        session_service.set_clarification_state(session.session_id, {"kind": "edit_slot"})
        orch = make_orchestrator(session_service, QueuedClassifier())
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(session.session_id, session.member_id, LISTING)

        assert turn.intent == "score_plan"
        assert session_service.get_session(session.session_id).state == "ready"

    def test_a_scoring_word_without_a_listing_still_asks_the_classifier(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        classifier = QueuedClassifier("chat")
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = RecordingScorer()

        orch.process(session.session_id, session.member_id, "rate my week")

        assert classifier.calls == ["rate my week"]
        assert orch.plan_scorer.processed == []

    def test_other_requests_are_not_mistaken_for_a_listing(self):
        explicit = OrchestratorService.is_explicit_score_request
        assert not explicit("ask food scholar to rate this: breakfast: oats, lunch: soup")
        assert not explicit("plan my day with oats for breakfast and salmon for dinner")
        assert not explicit("how does my plan look?")
        assert explicit("how does this look? breakfast: yogurt, dinner: pasta")

    def test_an_unanswered_score_question_falls_through_and_is_never_asked_twice(
        self, session_service, sample_profile
    ):
        session = new_session(session_service, sample_profile)
        classifier = QueuedClassifier("score_plan", "score_plan")
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = scorer_service(session_service, parser=NoParser(), grounder=EchoGrounder())

        first = orch.process(session.session_id, session.member_id, "my usual food, nothing special")
        assert first.needs_clarification

        second = orch.process(session.session_id, session.member_id, "honestly just the normal stuff")

        assert second.intent == "score_plan"
        assert not second.needs_clarification
        assert "nothing to score" in second.content
        assert len(classifier.calls) == 2
        assert session_service.get_session(session.session_id).state == "ready"

    def test_a_managed_router_prompt_without_the_intent_gets_it_appended(self, monkeypatch):
        import agents

        class StalePrompt:
            def compile(self, **_):
                return "You are the intent router. Intents: daily_plan, chat."

        assert agents.OrchestratorAgent.system_prompt().count("ADDITIONAL INTENT") == 0
        monkeypatch.setattr(agents, "ORCHESTRATOR_SYSTEM", StalePrompt())
        prompt = agents.OrchestratorAgent.system_prompt()
        assert "score_plan" in prompt and "ADDITIONAL INTENT" in prompt

    def test_the_wire_model_carries_the_reading(self):
        from routers.foodchat_router import _chat_turn_response

        turn = ChatTurn(role="assistant", content="Read it.", intent="score_plan", plan_score={
            "plan_type": "daily", "days_scored": 1, "meals_scored": 1,
            "metrics": [], "constraints_applied": [],
            "grounding": [{
                "day": None, "slot": "lunch", "title_given": "soup", "title_matched": None,
                "recipe_id": None, "state": UNRESOLVED, "ingredients_source": UNKNOWN,
                "has_nutrition": False, "allergen_conflicts": [],
            }],
            "unparsed": ["went to the gym"], "warnings": [],
        })

        response = _chat_turn_response(turn)

        assert response.plan_score.grounding[0].state == UNRESOLVED
        assert response.plan_score.unparsed == ["went to the gym"]


# --------------------------------------------------------------------- #
# Step 2 — calories when no recipe matches, and spelling variants        #
# --------------------------------------------------------------------- #

class FakeEstimator:
    """DishIngredientEstimator stand-in, keyed by dish title (lower case)."""

    def __init__(self, servings=None, fail=False):
        self.servings = servings or {}
        self.fail = fail
        self.calls = []

    def estimate(self, dishes):
        self.calls.append([dict(d) for d in dishes])
        if self.fail:
            raise RuntimeError("model unavailable")
        return {
            index: self.servings[dish["title"].lower()]
            for index, dish in enumerate(dishes) if dish["title"].lower() in self.servings
        }


STEW = {"ingredients": [("200 g", "beef"), ("100 g", "carrots"), ("150 g", "potatoes")], "kcal": 520.0}


class TestEstimatedCalories:
    """Dishes no recipe gave calories: typical ingredients, profiled; the
    model's own guess only when the profiler cannot help."""

    def test_typical_ingredients_are_profiled_for_a_dish_with_no_recipe(self):
        seeds = FakeSeeds([], profiles={"nonna's stew": {"kcal": 480.0, "protein_g": 22.0}})
        estimator = FakeEstimator({"nonna's stew": STEW})
        plan = prepass("dinner: Nonna's stew (beef, carrots, potatoes)")

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(plan, {})

        assert meal.state == UNRESOLVED
        assert meal.nutrition == {"kcal": 480.0, "protein_g": 22.0}
        assert meal.nutrition_source == "typical_ingredients"
        assert meal.to_row()["nutrition_source"] == "typical_ingredients"
        # The model is told what the member wrote; the profiler gets one
        # serving with quantities, never the bare dish name.
        assert estimator.calls == [[{
            "title": "Nonna's stew", "ingredients": "beef, carrots, potatoes", "quantity": None,
        }]]
        assert seeds.profile_calls == ["Nonna's stew\nServes 1\n200 g beef\n100 g carrots\n150 g potatoes"]
        assert seeds.profile_timeouts == [PROFILE_TIMEOUT_SECONDS]

    def test_low_coverage_falls_back_to_the_models_calorie_guess(self):
        partial = ProfiledNutrition(nutrition={"kcal": 90.0}, coverage=0.33)
        seeds = FakeSeeds([], profiles={"nonna's stew": partial})

        (meal,) = DishGrounder(seeds, estimator=FakeEstimator({"nonna's stew": STEW})).ground(
            prepass("dinner: Nonna's stew"), {},
        )

        assert meal.nutrition == {"kcal": 520.0}
        assert meal.nutrition_source == "model_estimate"

    def test_a_profiled_figure_double_off_the_guess_is_not_used(self):
        """Probe: pho profiled at 2,025 kcal against a guess of 600."""
        pho = {"ingredients": [("80 g", "rice noodles"), ("200 g", "beef")], "kcal": 600.0}
        seeds = FakeSeeds([], profiles={"pho": {"kcal": 2025.0}, "stew": {"kcal": 1000.0}})
        estimator = FakeEstimator({"pho": pho, "stew": STEW})

        found, stew = DishGrounder(seeds, estimator=estimator).ground(prepass("lunch: pho\ndinner: stew"), {})

        assert found.nutrition == {"kcal": 600.0} and found.nutrition_source == "model_estimate"
        assert found.discarded_profile_kcal == 2025.0
        assert "(2,025 kcal) was more than double off that guess" in found.guess_remarks()[0]
        assert stew.nutrition == {"kcal": 1000.0} and stew.nutrition_source == "typical_ingredients"

    def test_the_profilers_low_coverage_flag_is_believed(self):
        flagged = ProfiledNutrition(nutrition={"kcal": 90.0}, coverage=0.9, low_coverage=True)

        assert not flagged.reliable
        assert ProfiledNutrition(nutrition={"kcal": 90.0}, coverage=0.6).reliable
        assert ProfiledNutrition(nutrition={"kcal": 90.0}).reliable

    def test_an_implausible_guess_is_no_estimate(self):
        seeds = FakeSeeds([], profiles={})
        estimator = FakeEstimator({"nonna's stew": {"ingredients": [], "kcal": 12000.0}})

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(prepass("dinner: Nonna's stew"), {})

        assert meal.nutrition is None and meal.nutrition_source == ""
        assert seeds.profile_calls == [], "no ingredients, nothing to profile"

    def test_a_matched_recipes_own_figures_are_not_replaced(self):
        soup = _resolved("r-soup", "Lentil soup", "lentils, carrot", nutrition={"kcal": 300})
        seeds = FakeSeeds([soup], profiles={"lentil soup": {"kcal": 999.0}})
        estimator = FakeEstimator({"lentil soup": STEW})

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(prepass("lunch: lentil soup"), {})

        assert meal.nutrition == {"kcal": 300} and meal.nutrition_source == "recipe"
        assert estimator.calls == [] and seeds.profile_calls == []

    def test_without_an_estimator_there_is_no_estimate_and_no_call(self):
        seeds = FakeSeeds([], profiles={"nonna's stew": {"kcal": 480.0}})

        (meal,) = DishGrounder(seeds).ground(prepass("dinner: Nonna's stew"), {})

        assert meal.nutrition is None and meal.nutrition_source == ""
        assert seeds.profile_calls == []

    def test_a_deployment_without_the_endpoint_still_gets_the_guess(self):
        class NoProfiler:
            def fetch_details(self, ids):
                return {}

        (meal,) = DishGrounder(
            FakeSeeds([]), profile_client=NoProfiler(), estimator=FakeEstimator({"nonna's stew": STEW}),
        ).ground(prepass("dinner: Nonna's stew"), {})

        assert meal.nutrition == {"kcal": 520.0} and meal.nutrition_source == "model_estimate"

    def test_a_failing_profiler_leaves_the_guess(self):
        class Boom(FakeSeeds):
            def profile_recipe(self, raw_recipe, region=None, timeout=None):
                self.profile_calls.append(raw_recipe)
                raise RuntimeError("profiling pipeline down")

        seeds = Boom([], profiles={})
        (meal,) = DishGrounder(seeds, estimator=FakeEstimator({"nonna's stew": STEW})).ground(
            prepass("dinner: Nonna's stew"), {},
        )

        assert seeds.profile_calls and meal.nutrition_source == "model_estimate"

    def test_a_failing_estimator_leaves_the_dish_without_calories(self):
        seeds = FakeSeeds([], profiles={})
        estimator = FakeEstimator(fail=True)

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(prepass("dinner: Nonna's stew"), {})

        assert estimator.calls and meal.nutrition is None and seeds.profile_calls == []

    def test_profiling_stops_after_the_first_timeout(self):
        """Live, two stalled calls held the turn for two minutes."""

        class Stalled(FakeSeeds):
            def profile_recipe(self, raw_recipe, region=None, timeout=None):
                self.profile_calls.append(raw_recipe)
                raise ProfilingTimeout("stalled")

        seeds = Stalled([], profiles={})
        estimator = FakeEstimator({f"mystery dish {n}": STEW for n in range(3)})
        text = "\n".join(f"snack: mystery dish {n}" for n in range(3))

        grounded = DishGrounder(seeds, estimator=estimator, profile_workers=1).ground(prepass(text), {})

        assert len(seeds.profile_calls) == 1
        assert [m.nutrition_source for m in grounded] == ["model_estimate"] * 3

    def test_one_dish_repeated_is_estimated_once(self):
        seeds = FakeSeeds([], profiles={"nonna's stew": {"kcal": 480.0}})
        estimator = FakeEstimator({"nonna's stew": STEW})

        grounded = DishGrounder(seeds, estimator=estimator).ground(
            prepass("lunch: Nonna's stew\ndinner: Nonna's stew"), {},
        )

        assert len(estimator.calls[0]) == 1 and len(seeds.profile_calls) == 1
        assert [m.nutrition for m in grounded] == [{"kcal": 480.0}, {"kcal": 480.0}]

    def test_the_row_carries_the_calories_the_serving_and_what_is_a_guess(self):
        seeds = FakeSeeds([], profiles={"nonna's stew": {"kcal": 480.4}})

        (meal,) = DishGrounder(seeds, estimator=FakeEstimator({"nonna's stew": STEW})).ground(
            prepass("dinner: Nonna's stew"), {},
        )
        row = meal.to_row()

        assert row["kcal"] == 480
        assert row["typical_ingredients"] == [
            {"name": "beef", "quantity": "200 g"},
            {"name": "carrots", "quantity": "100 g"},
            {"name": "potatoes", "quantity": "150 g"},
        ]
        assert meal.variety_ingredients == "beef, carrots, potatoes"
        remarks = row["guess_remarks"]
        assert len(remarks) == 2
        assert remarks[0].startswith("The calories are an estimate")
        assert remarks[1].startswith("The ingredients are a guess at a typical serving")
        assert "not checked against your allergies" in remarks[1]

    def test_a_calorie_guess_says_it_is_a_rough_one(self):
        (meal,) = DishGrounder(
            FakeSeeds([], profiles={}), estimator=FakeEstimator({"nonna's stew": STEW}),
        ).ground(prepass("dinner: Nonna's stew"), {})

        assert meal.to_row()["kcal"] == 520
        assert meal.guess_remarks()[0].startswith("The calories are a rough guess by a language model")

    def test_the_members_own_ingredients_are_not_replaced_by_the_guess(self):
        seeds = FakeSeeds([], profiles={"nonna's stew": {"kcal": 480.0}})

        (meal,) = DishGrounder(seeds, estimator=FakeEstimator({"nonna's stew": STEW})).ground(
            prepass("dinner: Nonna's stew (beef, carrots)"), {},
        )

        assert meal.ingredients == "beef, carrots" and meal.variety_ingredients == "beef, carrots"
        assert not meal.guessed_ingredients
        assert len(meal.guess_remarks()) == 1, "only the calories are a guess"

    def test_a_dish_with_calories_but_no_ingredients_gets_a_serving_and_no_profiling(self):
        porridge = _resolved("r-por", "Porridge", "", nutrition={"kcal": 250})
        seeds = FakeSeeds([porridge], profiles={"porridge": {"kcal": 999.0}})
        estimator = FakeEstimator({"porridge": {"ingredients": [("50 g", "oats"), ("200 ml", "milk")], "kcal": 300}})

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(prepass("breakfast: porridge"), {})

        assert meal.nutrition == {"kcal": 250} and meal.nutrition_source == "recipe"
        assert meal.variety_ingredients == "oats, milk"
        assert seeds.profile_calls == []
        assert [r[:30] for r in meal.guess_remarks()] == ["The ingredients are a guess at"]

    def test_a_row_without_calories_says_so(self):
        row = GroundedMeal(day=None, slot="lunch", title_given="Mystery", state=UNRESOLVED).to_row()

        assert row["kcal"] is None and row["typical_ingredients"] == [] and row["guess_remarks"] == []

    def test_dishes_missing_calories_come_first_under_the_cap(self):
        known = [
            _resolved(f"r-{n}", f"Known dish {n}", "", nutrition={"kcal": 100})
            for n in range(MAX_ESTIMATED_DISHES)
        ]
        seeds = FakeSeeds(known, profiles={})
        estimator = FakeEstimator()
        text = "\n".join(
            [f"snack: known dish {n}" for n in range(MAX_ESTIMATED_DISHES)] + ["dinner: mystery stew"]
        )

        DishGrounder(seeds, estimator=estimator).ground(prepass(text), {})

        asked = [d["title"] for d in estimator.calls[0]]
        assert len(asked) == MAX_ESTIMATED_DISHES and asked[0] == "mystery stew"

    def test_profiling_is_capped_and_the_rest_keep_the_guess(self):
        """Live: a 21-dish week left its last three dishes with no calories."""
        count = MAX_PROFILE_CALLS + 4
        seeds = FakeSeeds([], profiles={f"mystery dish {n}": {"kcal": 500.0} for n in range(count)})
        estimator = FakeEstimator({f"mystery dish {n}": STEW for n in range(count)})
        text = "\n".join(f"snack: mystery dish {n}" for n in range(count))

        grounded = DishGrounder(seeds, estimator=estimator).ground(prepass(text), {})

        assert len(estimator.calls[0]) == count, "one model call covers every dish"
        assert len(seeds.profile_calls) == MAX_PROFILE_CALLS
        sources = [m.nutrition_source for m in grounded]
        assert sources == ["typical_ingredients"] * MAX_PROFILE_CALLS + ["model_estimate"] * 4

    def test_the_model_call_is_capped_at_a_week(self):
        count = MAX_ESTIMATED_DISHES + 3
        estimator = FakeEstimator()
        text = "\n".join(f"snack: mystery dish {n}" for n in range(count))

        DishGrounder(FakeSeeds([], profiles={}), estimator=estimator).ground(prepass(text), {})

        assert len(estimator.calls[0]) == MAX_ESTIMATED_DISHES


# The shape the demo gateway returned for "pasta with zucchini" (trimmed).
PROFILE_RESPONSE = {
    "success": True,
    "result": {
        "serves": 2,
        "nutrition_source_key": "irish",
        "nutrition_coverage": 1.0,
        "nutrition_low_coverage": False,
        "profiling_totals": {
            "total_energy_kcal_irish": 1012.0,
            "total_energy_kcal_per_serving_irish": 506.0,
            "total_protein_g_irish": 34.0,
            "total_protein_g_per_serving_irish": 17.0,
            "total_carbohydrate_g_per_serving_irish": 80.5,
            "total_fat_g_per_serving_irish": 12.25,
        },
        "full_profile": {"nutrition_summary": {"energy_kcal": 1012.0, "energy_kcal_per_serving": 506.0}},
        "ingredients": [{"name": "pasta", "protein_g": 25.0}],
    },
}


class TestProfileReader:
    def test_per_serving_totals_not_the_whole_recipe(self):
        found = profile_nutrition(PROFILE_RESPONSE)

        assert found.nutrition == {"kcal": 506.0, "protein_g": 17.0, "carbs_g": 80.5, "fat_g": 12.25}
        assert found.coverage == 1.0 and found.reliable

    def test_the_named_composition_table_wins(self):
        totals = {
            "total_energy_kcal_per_serving_usda": 700.0,
            "total_energy_kcal_per_serving_irish": 506.0,
        }
        found = profile_nutrition({"nutrition_source_key": "irish", "profiling_totals": totals})

        assert found.nutrition["kcal"] == 506.0

    def test_the_summary_per_serving_figure_is_the_fallback(self):
        found = profile_nutrition({
            "full_profile": {"nutrition_summary": {"energy_kcal": 900.0, "energy_kcal_per_serving": 450.0}},
            "nutrition_coverage": 0.4,
            "nutrition_low_coverage": True,
        })

        assert found.nutrition == {"kcal": 450.0}
        assert found.coverage == 0.4 and not found.reliable

    def test_no_calories_is_no_profile(self):
        assert profile_nutrition({"result": {"profiling_totals": {"total_energy_kcal_per_serving_irish": 0}}}) is None
        assert profile_nutrition({"result": {}}) is None
        assert profile_nutrition(None) is None


class TestDishIngredientEstimator:
    def estimator(self, content):
        from agents import DishIngredientEstimator

        class Reply:
            def __init__(self):
                self.messages = []

            def invoke(self, messages, config=None):
                self.messages.append(messages)
                if isinstance(content, Exception):
                    raise content
                return SimpleNamespace(content=content)

        agent = DishIngredientEstimator.__new__(DishIngredientEstimator)
        agent.llm = Reply()
        return agent

    def test_ingredients_and_the_guess_are_read_per_dish(self):
        agent = self.estimator(
            '{"dishes": [{"index": 1, "ingredients": [{"name": "zucchini", "quantity": "150 g"},'
            ' {"name": "", "quantity": "1 tbsp"}], "kcal_per_serving": 450},'
            ' {"index": 7, "ingredients": [{"name": "stray"}]},'
            ' {"index": 0, "ingredients": "not a list", "kcal_per_serving": "lots"}]}'
        )

        estimates = agent.estimate([
            {"title": "fried eggs"},
            {"title": "pasta with zucchini", "ingredients": "pasta, zucchini", "quantity": "a big plate"},
        ])

        assert estimates == {
            0: {"ingredients": [], "kcal": None},
            1: {"ingredients": [("150 g", "zucchini")], "kcal": 450.0},
        }

        user_text = agent.llm.messages[0][1].content
        assert "1. pasta with zucchini (the user's ingredients: pasta, zucchini) [amount: a big plate]" in user_text

    def test_an_ingredient_listed_twice_is_kept_once(self):
        agent = self.estimator(
            '{"dishes": [{"index": 0, "ingredients": [{"name": "black pudding", "quantity": "1 slice"},'
            ' {"name": "egg", "quantity": "1"}, {"name": "Black pudding", "quantity": "1 slice"}]}]}'
        )

        assert agent.estimate([{"title": "full english"}])[0]["ingredients"] == [
            ("1 slice", "black pudding"), ("1", "egg"),
        ]

    def test_a_failed_call_is_no_estimates(self):
        assert self.estimator(RuntimeError("429")).estimate([{"title": "stew"}]) == {}
        assert self.estimator("not json").estimate([{"title": "stew"}]) == {}
        assert self.estimator("[]").estimate([{"title": "stew"}]) == {}


class TestSameDish:
    """Live battery: names that are close in words but not the same dish."""

    @pytest.mark.parametrize("title,heads", [
        ("roast chicken with potatoes", ["chicken", "potato"]),
        ("houmous and pitta", ["hummus", "pita"]),
        ("vegetable lasagne", ["lasagna"]),
        ("tuna nicoise salad", ["nicoise"]),
        ("salmon served with rice", ["salmon", "rice"]),
        ("beans on toast", ["bean", "toast"]),
    ])
    def test_the_parts_a_title_names(self, title, heads):
        assert dish_heads(title) == heads

    @pytest.mark.parametrize("given,candidate,same", [
        ("vegetable lasagne", "Roasted vegetable lasagne", True),
        ("chicken wrap", "Smoked chicken wrap", True),
        ("pad thai", "Vegetarian pad Thai", False),
        ("caesar salad", "Vegan caesar salad", False),
        ("caesar salad", "Mexican Caesar salad", False),
        ("Scrambled eggs on rye toast", "Scrambled egg on toast", True),
        ("Tomato lentil soup", "Spicy lentil and tomato soup", True),
        ("roast chicken with potatoes", "Roast potatoes", False),
        ("caesar salad", "Chicken Caesar salad", False),
        ("avocado toast", "Avocado ricotta toast", False),
        ("beef tacos", "Beef and kimchi tacos", False),
        ("Mushroom stroganoff with rice", "Mushroom Stroganoff", False),
    ])
    def test_same_dish(self, given, candidate, same):
        assert same_dish(given, candidate) is same

    def test_a_generic_name_lends_calories_only_when_it_keeps_every_part(self):
        assert borrowable("vegetable lasagna", "Lasagna")
        assert not borrowable("houmous and pitta", "Hummus")
        assert not borrowable("baked cod with roasted vegetables", "Roasted vegetables")

    def test_a_close_name_that_adds_a_food_is_only_approximate(self):
        caesar = _resolved("r-cc", "Chicken Caesar salad", "chicken, lettuce, parmesan cheese",
                           tags=("gluten_free",), nutrition={"kcal": 420})
        seeds = FakeSeeds([caesar])

        (meal,) = DishGrounder(seeds).ground(prepass("lunch: caesar salad"), {"allergies": ["dairy"]})

        assert meal.state == APPROXIMATE
        assert meal.ingredients == "" and meal.tags == [] and meal.nutrition is None
        assert meal.allergen_conflicts == [{"allergen": "dairy", "evidence": CLOSEST_RECIPE}]

    def test_the_same_dish_is_preferred_over_a_closer_neighbour(self):
        potatoes = _resolved("r-rp", "Roast chicken potatoes", "potatoes, chicken fat")
        chicken = _resolved("r-rc", "Roast chicken with potatoes and gravy", "chicken, potatoes, gravy")
        seeds = FakeSeeds([potatoes, chicken])

        (meal,) = DishGrounder(seeds).ground(prepass("dinner: roast chicken potatoes"), {})

        assert meal.title_matched == "Roast chicken potatoes"


class TestWhatTheMemberWrote:
    def test_their_ingredients_outrank_a_matched_recipes(self):
        """Live: banana oat pancakes made with eggs were failed for the milk
        in the catalogue's Banana Pancakes."""
        pancakes = _resolved("r-bp", "Banana Pancakes", "banana, milk, flour, egg",
                             allergens=("dairy",), tags=("vegetarian",), nutrition={"kcal": 350})
        seeds = FakeSeeds([pancakes])
        plan = prepass("breakfast: Banana oat pancakes (2 bananas, 2 eggs, 50 g oat flour)")

        (meal,) = DishGrounder(seeds).ground(plan, {"allergies": ["dairy", "lactose"]})

        assert meal.state == MATCHED
        assert meal.ingredients == "2 bananas, 2 eggs, 50 g oat flour"
        assert meal.ingredients_source == AS_WRITTEN
        assert meal.allergen_conflicts == [] and meal.tags == []
        assert meal.nutrition == {"kcal": 350}, "the recipe's calories are still the best figure"

    def test_a_bracketed_list_with_amounts_is_ingredients(self):
        plan = prepass("breakfast: Greek yogurt with walnuts (150 g yogurt, 20 g walnuts, 1 tsp honey)\n"
                       "lunch: soup (300 g)")

        breakfast, lunch = plan.meals
        assert breakfast.ingredients == "150 g yogurt, 20 g walnuts, 1 tsp honey"
        assert breakfast.quantity_note is None
        assert lunch.quantity_note == "300 g" and lunch.ingredients is None

    def test_plant_milks_and_nut_butters_are_not_dairy(self):
        plan = prepass("lunch: chickpea curry (chickpeas, coconut milk, spinach)\n"
                       "snack: toast (bread, peanut butter)\ndinner: pasta (pasta, butter)")

        curry, toast, pasta = DishGrounder(FakeSeeds([])).ground(plan, {"allergies": ["dairy", "lactose"]})

        assert curry.allergen_conflicts == [] and toast.allergen_conflicts == []
        assert pasta.allergen_conflicts == [{"allergen": "dairy", "evidence": AS_WRITTEN}]

    def test_a_peanut_allergy_still_sees_peanut_butter(self):
        (toast,) = DishGrounder(FakeSeeds([])).ground(
            prepass("snack: toast (bread, peanut butter)"), {"allergies": ["peanuts"]},
        )

        assert toast.allergen_conflicts == [{"allergen": "peanuts", "evidence": AS_WRITTEN}]


class TestImplausibleRecipeCalories:
    def test_a_meal_at_a_dozen_kcal_is_set_aside_and_estimated(self):
        chili = _resolved("r-qc", "Chilli", "beans, beef, tomato", nutrition={"kcal": 12})
        seeds = FakeSeeds([chili], profiles={"chilli": {"kcal": 540.0}})
        estimator = FakeEstimator({"chilli": STEW})

        (meal,) = DishGrounder(seeds, estimator=estimator).ground(prepass("dinner: chilli"), {})

        assert meal.state == MATCHED and meal.ingredients_source == "recipe"
        assert meal.nutrition == {"kcal": 540.0} and meal.nutrition_source == "typical_ingredients"
        assert meal.rejected_recipe_kcal == {"title": "Chilli", "kcal": 12.0}
        assert meal.guess_remarks()[0] == (
            "The catalogue recipe “Chilli” lists 12 kcal a serving, which is too little for this "
            "meal to be right, so that figure was not used."
        )

    def test_the_floor_is_lower_for_a_snack(self):
        apple = _resolved("r-ap", "Apple", "apple", nutrition={"kcal": 52})

        (snack,) = DishGrounder(FakeSeeds([apple])).ground(prepass("snack: apple"), {})
        (lunch,) = DishGrounder(FakeSeeds([apple])).ground(prepass("lunch: apple"), {})

        assert snack.nutrition == {"kcal": 52}
        assert lunch.nutrition is None and lunch.rejected_recipe_kcal["kcal"] == 52.0

    def test_details_figures_get_the_same_check(self):
        soup = _resolved("r-ts", "Tomato soup", "tomatoes, stock")
        rich = SimpleNamespace(nutrition_dict=lambda: {"kcal": 24}, tags=[], image_url=None)
        seeds = FakeSeeds([soup], details={"r-ts": rich})

        (meal,) = DishGrounder(seeds).ground(prepass("lunch: tomato soup"), {})

        assert meal.nutrition is None and meal.rejected_recipe_kcal == {"title": "Tomato soup", "kcal": 24.0}


class TestSpellingVariants:
    def test_one_dish_one_spelling(self):
        assert content_tokens("vegetable lasagne") == content_tokens("vegetable lasagna")
        assert content_tokens("greek yoghurt") == content_tokens("Greek yogurt")
        assert content_tokens("crème fraîche") == ["creme", "fraiche"]

    def test_similarity_sees_through_the_spelling(self):
        assert title_similarity("vegetable lasagne", "Roasted vegetable lasagna") >= 0.75

    def test_the_other_spellings_are_offered_for_search(self):
        assert spelling_variants("vegetable lasagna") == ["vegetable lasagne"]
        assert spelling_variants("chicken soup") == []

    def test_a_hit_in_the_other_spelling_matches_without_a_second_query(self):
        """The search found it anyway, and normalising the spelling is what
        turns it into a match rather than a near miss."""
        lasagne = _resolved("r-las", "Roasted vegetable lasagne", "aubergine, pasta, tomato")
        seeds = FakeSeeds([lasagne])

        (meal,) = DishGrounder(seeds).ground(prepass("dinner: vegetable lasagna"), {})

        assert seeds.queries == ["vegetable lasagna"]
        assert meal.state == MATCHED and meal.title_matched == "Roasted vegetable lasagne"

    def test_a_search_that_needs_the_exact_spelling_is_asked_again(self):
        """Some search backends match words literally. When the first query
        finds nothing close, the other spelling is searched too."""

        class ExactWords(FakeSeeds):
            def find_dish(self, name, limit=5):
                self.queries.append(name)
                words = set(name.lower().split())
                return [
                    r.recipe for r in self.recipes
                    if words <= set(r.recipe.title.lower().split())
                ][:limit]

        lasagne = _resolved("r-las", "vegetable lasagne", "aubergine, pasta, tomato")
        seeds = ExactWords([lasagne])

        (meal,) = DishGrounder(seeds).ground(prepass("dinner: vegetable lasagna"), {})

        assert seeds.queries == ["vegetable lasagna", "vegetable lasagne"]
        assert meal.state == MATCHED and meal.title_matched == "vegetable lasagne"

    def test_a_match_on_the_first_spelling_costs_one_query(self):
        soup = _resolved("r-soup", "Lentil soup", "lentils, carrot")
        seeds = FakeSeeds([soup])

        DishGrounder(seeds).ground(prepass("lunch: lentil soup"), {})

        assert seeds.queries == ["lentil soup"]
