"""Plan scorer steps 4–5, the /score-plan endpoint, and the planner's fences.

LLM-free. Two halves:

- non-regression: the metric functions hoisted out of ChatService return
  exactly what the originals did (verbatim copies below are the oracle), the
  planner's prompts are byte-identical, the planner's own judges send the same
  messages they always did, and ordinary planner messages still reach the
  classifier;
- the scorer: constraint rows re-measured per dish, daily and weekly metrics,
  the single judge call, caps, the summary, persistence and the endpoint.
"""

import hashlib
import importlib
import re
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import prompts
import services
from conftest import make_candidates
from models.pasted_plan import (
    APPROXIMATE,
    AS_WRITTEN,
    CLOSEST_RECIPE,
    FROM_RECIPE,
    MATCHED,
    NUTRITION_MODEL_ESTIMATE,
    NUTRITION_TYPICAL,
    UNKNOWN,
    UNRESOLVED,
    GroundedMeal,
)
from models.recipe import ScoredPlan
from routers import foodchat_router
from routers.foodchat_router import PlanScoreResponse, ScorePlanRequest, _chat_turn_response
from services.orchestrator_service import OrchestratorService
from services.plan_scorer.building import build_daily, build_weekly
from services.plan_scorer.scoring import (
    DAILY_METRIC_KEYS,
    JUDGED_METRIC_KEYS,
    NOT_GRADED,
    UNCHECKED,
    WEEKLY_METRIC_KEYS,
    PastedPlanScorer,
    constraint_rows,
    fit_metric,
)
from services.plan_scorer.service import (
    PlanScorerService,
    allergen_sentences,
    ensure_allergen_warnings,
    ensure_calorie_caveat,
    summary_facts,
)
from services.plan_scoring import guidelines_text, ingredient_names
from services.session_service import SessionService
from test_orchestrator_routing import QueuedClassifier, make_orchestrator
from test_plan_scorer import EchoGrounder, FallbackWriter, ForbiddenParser, NoParser, RecordingScorer

# --------------------------------------------------------------------- #
# Oracle: the ChatService functions exactly as they were before the hoist #
# --------------------------------------------------------------------- #


def _old_extract_ingredient_names(ingredients_text: str) -> list[str]:
    """Normalize a free-text ingredients blob into comparable item names."""
    if not isinstance(ingredients_text, str):
        return []
    cleaned = []
    for part in re.split(r"[\n,;•\-]+", ingredients_text):
        t = part.strip().lower()
        t = re.sub(r"\([^\)]*\)", "", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)
    return cleaned


def _old_food_variety_score(plan) -> tuple[int, str]:
    items: list[str] = []
    for course in plan.courses:
        items.extend(_old_extract_ingredient_names(course.ingredients))
    unique_items = sorted(set(items))
    reasoning = (
        f"Unique food items across meals: {len(unique_items)} "
        f"(e.g., {', '.join(unique_items[:8])}{'...' if len(unique_items) > 8 else ''})"
    )
    return len(unique_items), reasoning


def _old_plan_as_text(plan) -> str:
    return "\n".join(
        f"{name}: {course.title}\nIngredients: {course.ingredients}\nDirections: {course.directions}\n"
        for name, course in (
            ("Breakfast", plan.breakfast), ("Lunch", plan.lunch), ("Dinner", plan.dinner),
        )
    )


class FakeGrader:
    """One of the planner's single-metric judges."""

    def __init__(self, score=4, reasoning="fine"):
        self.result = {"score": score, "reasoning": reasoning}
        self.calls = []

    def score(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return dict(self.result)


class FakeJudge:
    """The scorer's merged judge: three judgements from one call."""

    DEFAULT = {
        "diversity": {"score": 4, "reasoning": "diverse"},
        "guideline_adherence": {"score": 3, "reasoning": "mostly fine"},
        "fit": {"score": 5, "reasoning": "great fit"},
    }

    def __init__(self, payload=None, fail=False):
        self.payload = payload if payload is not None else self.DEFAULT
        self.fail = fail
        self.calls = []

    def judge(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("judge unavailable")
        return {key: dict(value) for key, value in self.payload.items()}


class RecordingClient:
    def __init__(self, content='{"score": 4, "reasoning": "ok"}'):
        self.content = content
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class ScriptedWriter:
    def __init__(self, text):
        self.text = text
        self.facts = []

    def write(self, facts, user_message, fallback):
        self.facts.append(facts)
        return self.text


def make_scorer(judge=None):
    judge = judge if judge is not None else FakeJudge()
    return PastedPlanScorer(judge=judge, guidelines=lambda scope: f"{scope} rules"), judge


def dish(day, slot, title, state=UNRESOLVED, rid=None, ingredients="", source=None,
         tags=(), nutrition=None, conflicts=(), nutrition_source=""):
    if source is None:
        source = FROM_RECIPE if state == MATCHED else (AS_WRITTEN if ingredients else UNKNOWN)
    matched_title = title if state == MATCHED else ("Closest recipe" if state == APPROXIMATE else None)
    return GroundedMeal(
        day=day, slot=slot, title_given=title, state=state, recipe_id=rid,
        title_matched=matched_title, ingredients=ingredients, ingredients_source=source,
        tags=list(tags), nutrition=nutrition, allergen_conflicts=[dict(c) for c in conflicts],
        nutrition_source=nutrition_source,
    )


def new_session(session_service, profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", profile)


def full_service(session_service, **overrides):
    scorer, _judge = make_scorer()
    kwargs = {
        "parser": ForbiddenParser(), "grounder": EchoGrounder(),
        "scorer": scorer, "writer": FallbackWriter(),
    }
    kwargs.update(overrides)
    return PlanScorerService(session_service, **kwargs)


def orchestrator_with(session_service, monkeypatch, classifier=None, **overrides):
    orch = make_orchestrator(session_service, classifier or QueuedClassifier())
    orch.plan_scorer = full_service(session_service, **overrides)
    monkeypatch.setattr(services, "orchestrator_service", orch)
    return orch


def row(rows, constraint):
    return next(r for r in rows if r["constraint"] == constraint)


# ===================================================================== #
# Non-regression: the planner                                            #
# ===================================================================== #

SAMPLES = [
    "oats, milk",
    "2 cups (250 ml) milk, rolled oats; honey\n- banana • berries",
    "Crème fraîche, jalapeños, 1/2 lime",
    "chicken-thigh fillets, salt & pepper",
    "",
    None,
]


class TestHoistedMetricsAreUnchanged:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_ingredient_names(self, text):
        assert ingredient_names(text) == _old_extract_ingredient_names(text)

    def test_one_normalizer_everywhere(self):
        chat_service = importlib.import_module("services.chat_service")
        explainability = importlib.import_module("services.weekly_planner.explainability")

        assert explainability._ingredient_names is ingredient_names
        assert chat_service._extract_ingredient_names(SAMPLES[1]) == _old_extract_ingredient_names(SAMPLES[1])

    def test_variety_count_and_plan_text(self):
        chat_service = importlib.import_module("services.chat_service")
        plan = ScoredPlan(*make_candidates("h"), 4, "because")

        assert chat_service._food_variety_score(plan) == _old_food_variety_score(plan)
        assert chat_service._plan_as_text(plan) == _old_plan_as_text(plan)

    def test_the_daily_metrics_dict_is_unchanged(self):
        chat_service = importlib.import_module("services.chat_service")
        svc = chat_service.ChatService.__new__(chat_service.ChatService)
        svc.diversity_grader = FakeGrader(3, "varied enough")
        svc.guideline_grader = FakeGrader(4, "mostly fine")
        plan = ScoredPlan(*make_candidates("h"), 4, "because")

        metrics = svc._compute_metrics("s-1", plan)

        count, reasoning = _old_food_variety_score(plan)
        assert list(metrics.items()) == [
            ("llm_score", 4), ("llm_reasoning", "because"),
            ("fvs_count", count), ("fvs_reasoning", reasoning),
            ("diversity_llm_score", 3), ("diversity_llm_reasoning", "varied enough"),
            ("guideline_adherence_score", 4), ("guideline_adherence_reasoning", "mostly fine"),
        ]
        text = _old_plan_as_text(plan)
        assert svc.diversity_grader.calls == [((text,), {})]
        assert svc.guideline_grader.calls == [((text, guidelines_text("daily")), {})]


PLANNER_PROMPTS = {
    "GRADER_SYSTEM_INSTRUCTIONS": "a8cb1a6194d1dfbf41b6b26e87a42732665a54369494a6bef95e14464d758f80",
    "GRADER_USER_INSTRUCTIONS": "cc498e52c0fa838f1d107200d79196f061a38b185f216278503c58ed142ab8c9",
    "BATCH_GRADER_USER_INSTRUCTIONS": "0f803dcf4f3014c37983f359b3cbbe51ce52f2ec310b8c2edb063424f0db2cec",
    "MEAL_DIVERSITY_SYSTEM_INSTRUCTIONS": "b5373b2f6fd46ab98f2680c70d5e4a28496004c18a3b1ec3bd3656f6c85e2164",
    "GUIDELINE_ADHERENCE_SYSTEM_INSTRUCTIONS": "8ee8697411adbae6f21ae8006599fd9ba4ec37f65ad372ab5bc772380fc678a6",
    "RESPONSE_WRITER_SYSTEM_INSTRUCTIONS": "58e003b8844de4e2c9f817f755b43214f7a1fbce753c8d7b3339d7e4504daf91",
    "RESPONSE_WRITER_USER_INSTRUCTIONS": "fd6f19679401c6bc0cb8cd061564751d806559937114a3598c2f81cd1dc0cde8",
}


class TestPlannerPromptsAndJudges:
    @pytest.mark.parametrize("name,digest", sorted(PLANNER_PROMPTS.items()))
    def test_planner_prompts_are_byte_identical(self, name, digest):
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == digest

    def test_the_rubric_is_shared_not_copied(self):
        for piece in (prompts.PLAN_SCORING_RUBRIC, prompts.SLOT_PLAUSIBILITY_RULES,
                      prompts.ASSESSOR_STANCE):
            assert piece in prompts.GRADER_SYSTEM_INSTRUCTIONS
            assert piece in prompts.PLAN_JUDGE_DAILY_SYSTEM_INSTRUCTIONS
            assert piece in prompts.PLAN_JUDGE_WEEKLY_SYSTEM_INSTRUCTIONS
        assert prompts.GRADER_SYSTEM.fallback == prompts.GRADER_SYSTEM_INSTRUCTIONS

    def test_the_planners_diversity_judge_is_untouched(self):
        from agents import MealDiversityGrader

        grader = MealDiversityGrader()
        grader.client = RecordingClient()

        assert grader.score("PLAN TEXT") == {"score": 4, "reasoning": "ok"}
        (messages,) = grader.client.calls
        assert [m.content for m in messages] == [prompts.MEAL_DIVERSITY_SYSTEM.compile(), "PLAN TEXT"]

    def test_the_planners_guideline_judge_is_untouched(self):
        from agents import GuidelineAdherenceGrader

        grader = GuidelineAdherenceGrader()
        grader.client = RecordingClient()
        grader.score("PLAN", "RULES")

        (messages,) = grader.client.calls
        assert messages[0].content == prompts.GUIDELINE_ADHERENCE_SYSTEM.compile()
        assert messages[1].content == "GUIDELINES:\nRULES\n\nMEAL PLAN:\nPLAN"

    def test_the_scorer_judge_asks_for_three_scores_in_one_call(self):
        from agents import PlanJudge

        judge = PlanJudge()
        judge.client = RecordingClient(
            '{"diversity": {"reasoning": "d", "score": 4},'
            ' "guideline_adherence": {"reasoning": "g", "score": 3},'
            ' "fit": {"reasoning": "f", "score": 2}}'
        )

        payload = judge.judge(
            weekly=False, plan_text="PLAN-X", plan_shape="SHAPE-X", hard_constraints="HARD-X",
            conflicts="CONFLICT-X", preferences="PREF-X", aim="AIM-X", guidelines="RULES-X",
            facts="FACTS-X",
        )

        assert sorted(payload) == ["diversity", "fit", "guideline_adherence"]
        system, user = judge.client.calls[0]
        assert system.content == prompts.PLAN_JUDGE_DAILY_SYSTEM.compile()
        assert "JSON" in system.content
        for piece in ("PLAN-X", "(SHAPE-X)", "HARD-X", "CONFLICT-X", "PREF-X", "AIM-X",
                      "RULES-X", "FACTS-X"):
            assert piece in user.content

    def test_a_plan_of_several_days_gets_the_weekly_prompt(self):
        from agents import PlanJudge

        judge = PlanJudge()
        judge.client = RecordingClient('{"diversity": {"reasoning": "d", "score": 4},'
                                       ' "guideline_adherence": {"reasoning": "g", "score": 3},'
                                       ' "fit": {"reasoning": "f", "score": 2}}')
        judge.judge(weekly=True, plan_text="P", plan_shape="3 days", hard_constraints="",
                    conflicts="", preferences="", aim="", guidelines="", facts="")

        system = judge.client.calls[0][0]
        assert system.content == prompts.PLAN_JUDGE_WEEKLY_SYSTEM.compile()
        assert prompts.PLAN_JUDGE_WEEKLY_SYSTEM_INSTRUCTIONS != prompts.PLAN_JUDGE_DAILY_SYSTEM_INSTRUCTIONS
        assert "ACROSS the days" in prompts.PLAN_JUDGE_WEEKLY_SYSTEM_INSTRUCTIONS

    def test_new_prompts_are_registered(self):
        names = {p.name for p in prompts.ALL_PROMPTS}
        for name in ("plan_judge_daily_system", "plan_judge_weekly_system", "plan_judge_user",
                     "plan_text_parser_system"):
            assert f"foodchat/{name}" in names


# Messages a planner user sends every day. None of them may skip the
# classifier on the way to the scorer.
PLANNER_MESSAGES = [
    "plan my week",
    "Give me a daily plan with oats for breakfast and salmon for dinner",
    "make me a plan: breakfast: oats, lunch: lentil soup, dinner: salmon — how does it look?",
    "What do you think of adding salmon for dinner and oats for breakfast?",
    "swap Tuesday's dinner for something lighter",
    "change the lunch: something with more protein, and breakfast: no eggs",
    "is my plan healthy?",
    "how does my plan look for protein?",
    "rate the lunch",
    "breakfast: oats, lunch: soup",
    "ask food scholar to rate this: breakfast: oats, lunch: soup",
]


class TestOrdinaryMessagesStillRouteAsBefore:
    @pytest.mark.parametrize("message", PLANNER_MESSAGES)
    def test_not_an_explicit_score_request(self, message):
        assert not OrchestratorService.is_explicit_score_request(message)

    @pytest.mark.parametrize("message", [m for m in PLANNER_MESSAGES if "scholar" not in m])
    def test_the_classifier_still_decides(self, message, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        classifier = QueuedClassifier("chat")
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(session.session_id, session.member_id, message)

        assert classifier.calls == [message]
        assert orch.plan_scorer.processed == []
        assert turn.intent == "chat"

    @pytest.mark.parametrize("message", [
        "Here's my day, how does it look?\nbreakfast: oats\nlunch: soup",
        "Rate my week:\nMonday\n- breakfast: yogurt\n- dinner: pasta",
        # Live: answered as small talk because the bypass wanted "slot:" lines
        # and the classifier was rate limited.
        "Rate for me a daily plan consisting of fried eggs for breakfast, pasta with "
        "zucchini for lunch and chicken noodle soup for dinner",
    ])
    def test_a_real_paste_still_skips_it(self, message):
        assert OrchestratorService.is_explicit_score_request(message)


class FailingClassifier:
    """What OrchestratorAgent.classify returns when every attempt failed."""

    def __init__(self):
        self.calls = []

    def classify(self, message, history):
        self.calls.append(message)
        return {"intent": "chat", "target_plan_type": None, "failed": True}


class TestClassifierOutage:
    def test_the_router_says_when_it_could_not_classify(self):
        from agents import OrchestratorAgent

        agent = OrchestratorAgent()

        class Boom:
            def invoke(self, messages, config=None):
                raise RuntimeError("429 rate limit reached")

        agent.llm = Boom()

        assert agent.classify("plan my day", []) == {
            "intent": "chat", "target_plan_type": None, "failed": True,
        }

    def test_a_listed_plan_is_scored_rather_than_chatted_at(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        classifier = FailingClassifier()
        orch = make_orchestrator(session_service, classifier)
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(
            session.session_id, session.member_id,
            "fried eggs for breakfast, pasta with zucchini for lunch, chicken noodle soup for dinner",
        )

        assert turn.intent == "score_plan"
        assert orch.plan_scorer.processed
        assert classifier.calls, "the classifier is still asked first"

    @pytest.mark.parametrize("message", [
        "What do you think of adding salmon for dinner and oats for breakfast?",
        "hey, what's for dinner tonight?",
        "thanks, that looks great",
    ])
    def test_everything_else_still_falls_back_to_chat(self, message, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        orch = make_orchestrator(session_service, FailingClassifier())
        orch.plan_scorer = RecordingScorer()

        turn = orch.process(session.session_id, session.member_id, message)

        assert turn.intent == "chat"
        assert orch.plan_scorer.processed == []


# ===================================================================== #
# Step 4 — constraint rows                                               #
# ===================================================================== #

class TestConstraintRows:
    def test_an_allergen_found_is_violated_and_named(self):
        rows, violations = constraint_rows(
            [dish(None, "lunch", "Peanut noodles", ingredients="noodles, peanuts",
                  conflicts=[{"allergen": "peanuts", "evidence": AS_WRITTEN}])],
            {"allergies": ["Peanuts"]},
        )

        allergy = row(rows, "no Peanuts")
        assert allergy["status"] == "violated" and allergy["type"] == "hard"
        assert allergy["detail"] == "found in “Peanut noodles”"
        assert violations["allergens"] == [{"allergen": "Peanuts", "dishes": ["Peanut noodles"]}]

    def test_an_allergen_only_in_the_matched_recipe_says_so(self):
        rows, _ = constraint_rows(
            [dish(None, "dinner", "Crumbed chicken", state=MATCHED, rid="r-1",
                  ingredients="chicken, almond meal",
                  conflicts=[{"allergen": "tree nuts", "evidence": FROM_RECIPE}])],
            {"allergies": ["tree nuts"]},
        )

        assert "(in the matched recipe)" in row(rows, "no tree nuts")["detail"]

    def test_an_allergen_only_in_the_closest_recipe_warns_without_a_verdict(self):
        rows, violations = constraint_rows(
            [dish(None, "dinner", "Chicken skewers", state=APPROXIMATE, rid="r-sat",
                  conflicts=[{"allergen": "peanuts", "evidence": CLOSEST_RECIPE}])],
            {"allergies": ["peanuts"]},
        )

        allergy = row(rows, "no peanuts")
        assert allergy["status"] == UNCHECKED
        assert allergy["detail"].startswith("may be in “Chicken skewers”")
        assert violations["allergens"] == []
        assert violations["possible_allergens"] == [{"allergen": "peanuts", "dishes": ["Chicken skewers"]}]

    def test_the_judge_hears_about_a_possible_allergen_without_capping_fit(self):
        """Live, the judge wrote "no allergens" beside a warning that the
        pasta may contain lactose."""
        day = [
            dish(None, "lunch", "Pasta with zucchini", state=APPROXIMATE, rid="r-pz",
                 conflicts=[{"allergen": "lactose", "evidence": CLOSEST_RECIPE}]),
            dish(None, "dinner", "Lentil soup", ingredients="lentils"),
        ]
        scorer, judge = make_scorer()

        result = scorer.score("daily", day, build_daily(day), {"allergies": ["lactose"]})

        conflicts = judge.calls[0]["conflicts"]
        assert "POSSIBLE ONLY: “Pasta with zucchini” might contain lactose, an allergy" in conflicts
        assert "Not a broken hard constraint" in conflicts
        assert next(m for m in result.metrics if m["key"] == "fit")["score"] == 5

    def test_no_allergen_is_satisfied_but_admits_dishes_checked_by_name_only(self):
        rows, _ = constraint_rows([dish(None, "dinner", "Nonna's stew")], {"allergies": ["peanuts"]})

        allergy = row(rows, "no peanuts")
        assert allergy["status"] == "satisfied"
        assert "only their names were checked" in allergy["detail"]

    @pytest.mark.parametrize("diet,title,ingredients,broken", [
        ("vegetarian", "Chicken curry", "", True),
        ("vegetarian", "Lentil soup", "lentils, carrot", False),
        ("vegan", "Porridge", "oats, milk", True),
        ("vegan", "Peanut butter toast", "bread, peanut butter", False),
        ("vegan", "Oat milk porridge", "oats, oat milk", False),
        ("vegan", "Vegan cheese toastie", "bread, vegan cheese", False),
        ("pescatarian", "Grilled salmon", "salmon", False),
        ("pescatarian", "Beef stew", "beef, potato", True),
        ("gluten_free", "Spaghetti bolognese", "", True),
        ("gluten_free", "Gluten-free pasta salad", "gluten-free pasta, tomato", False),
        ("dairy_free", "Cheese omelette", "eggs, cheese", True),
        ("nut_free", "Almond cake", "almonds, sugar", True),
    ])
    def test_diets_are_checked_dish_by_dish(self, diet, title, ingredients, broken):
        rows, violations = constraint_rows(
            [dish(None, "dinner", title, ingredients=ingredients)], {"diet": [diet]},
        )

        assert row(rows, diet)["status"] == ("violated" if broken else "satisfied")
        assert bool(violations["diet"]) is broken

    def test_a_close_matchs_tags_do_not_overrule_the_members_words(self):
        rows, _ = constraint_rows(
            [dish(None, "dinner", "Chicken curry", state=APPROXIMATE, rid="r-veg", tags=["vegetarian"])],
            {"diet": ["vegetarian"]},
        )

        assert row(rows, "vegetarian")["status"] == "violated"

    def test_what_a_dish_cannot_settle_is_unchecked(self):
        rows, _ = constraint_rows(
            [dish(None, "lunch", "Pasta")], {"diet": ["low-carb"], "dietary_goals": ["weight_loss"]},
        )

        assert rows and {r["status"] for r in rows} == {UNCHECKED}

    def test_a_dislike_is_a_soft_violation(self):
        rows, violations = constraint_rows(
            [dish(None, "lunch", "Greek salad", ingredients="tomato, olives, feta")],
            {"food_dislikes": ["olives"]},
        )

        dislike = row(rows, "avoiding olives")
        assert dislike["type"] == "soft" and dislike["status"] == "violated"
        assert violations["dislikes"] == [{"dislike": "olives", "dishes": ["Greek salad"]}]

    def test_household_attribution_survives(self):
        rows, _ = constraint_rows([], {
            "allergies": ["peanuts"], "cooking_for_names": ["Ana", "Tom"],
            "constraint_origins": {"allergies": {"peanuts": ["Tom"]}},
        })

        assert row(rows, "no peanuts")["members"] == ["Tom"]


# ===================================================================== #
# Step 4 — the fit score and its caps                                    #
# ===================================================================== #

ALLERGEN = {"allergens": [{"allergen": "peanuts", "dishes": ["Peanut noodles"]}], "diet": [], "dislikes": []}
DIET = {"allergens": [], "diet": [{"diet": "vegetarian", "dishes": ["Beef stew"]}], "dislikes": []}
NONE = {"allergens": [], "diet": [], "dislikes": []}


class TestFitCaps:
    def test_an_allergen_caps_at_one_and_says_why(self):
        fit = fit_metric((5, "Lovely."), ALLERGEN)

        assert fit["score"] == 1 and fit["detail"]["model_score"] == 5
        assert "Capped at 1: “Peanut noodles” contains peanuts" in fit["reasoning"]

    def test_a_broken_diet_caps_at_two(self):
        fit = fit_metric((4, "Good."), DIET)

        assert fit["score"] == 2 and "breaks your vegetarian diet" in fit["reasoning"]

    def test_a_score_under_the_cap_is_left_alone(self):
        fit = fit_metric((1, "Poor."), DIET)

        assert fit["score"] == 1 and fit["detail"]["cap_applied"] is False

    def test_no_conflicts_no_cap(self):
        assert fit_metric((5, "Great."), NONE)["score"] == 5

    def test_an_allergen_settles_the_score_even_without_the_judge(self):
        fit = fit_metric((None, NOT_GRADED), ALLERGEN)

        assert fit["score"] == 1 and "without the fit judge" in fit["reasoning"]

    def test_no_judge_and_no_conflict_is_not_graded(self):
        fit = fit_metric((None, NOT_GRADED), NONE)

        assert fit["score"] is None and fit["reasoning"] == NOT_GRADED


# ===================================================================== #
# Step 4 — daily and weekly metrics, one judge call                      #
# ===================================================================== #

class TestDailyScoring:
    DAY = [
        dish(None, "breakfast", "Berry oatmeal", state=MATCHED, rid="r-oat",
             ingredients="rolled oats, blueberries, milk", nutrition={"kcal": 400, "protein_g": 12}),
        dish(None, "lunch", "Lentil soup", ingredients="lentils, carrot"),
        dish(None, "dinner", "Nonna's stew"),
    ]
    PROFILE = {"preferences": ["1800 calories target"], "food_likes": ["lentils"]}

    def score(self, judge=None):
        scorer, judge = make_scorer(judge)
        result = scorer.score("daily", self.DAY, build_daily(self.DAY), self.PROFILE, context="less meat")
        return result, judge

    def test_the_metrics_and_their_order(self):
        result, _ = self.score()

        assert [m["key"] for m in result.metrics] == list(DAILY_METRIC_KEYS)
        assert [m["score"] for m in result.metrics if m["kind"] == "likert5"] == [4, 3, 5]

    def test_one_judge_call_carries_everything(self):
        _, judge = self.score()

        assert len(judge.calls) == 1
        call = judge.calls[0]
        assert call["weekly"] is False
        assert call["guidelines"] == "daily rules"
        assert call["facts"] == ""
        assert call["aim"] == "less meat"
        assert call["plan_shape"] == "one day"
        assert "1800 calories target" in call["preferences"]
        assert "Likes: lentils" in call["preferences"]
        assert call["conflicts"] == "None found."
        text = call["plan_text"]
        assert "Ingredients from the catalogue recipe “Berry oatmeal”: rolled oats" in text
        assert "Ingredients as the user wrote them: lentils, carrot" in text
        assert "Ingredients not known." in text
        assert "About 400 kcal per serving, 12 g protein" in text

    def test_food_variety_counts_what_is_known_and_says_what_is_not(self):
        result, _ = self.score()
        fvs = result.metrics[0]

        assert fvs["score"] == 5
        assert fvs["detail"]["dishes_without_ingredients"] == 1
        assert "1 dish(es) have no ingredient list" in fvs["reasoning"]

    def test_guessed_ingredients_count_towards_variety_and_say_so(self):
        eggs = dish(None, "breakfast", "Fried eggs")
        eggs.typical_ingredients = [{"name": "egg", "quantity": "2"}, {"name": "olive oil", "quantity": "1 tsp"}]
        day = [eggs, dish(None, "lunch", "Lentil soup", ingredients="lentils, carrot"), dish(None, "dinner", "Stew")]
        scorer, judge = make_scorer()

        result = scorer.score("daily", day, build_daily(day), {})
        fvs = result.metrics[0]

        assert fvs["score"] == 4
        assert fvs["detail"]["dishes_with_guessed_ingredients"] == 1
        assert fvs["detail"]["dishes_without_ingredients"] == 1
        assert "1 dish(es) (“Fried eggs”) are counted with a typical serving's ingredients — a guess" \
            in fvs["reasoning"]
        assert "1 dish(es) have no ingredient list" in fvs["reasoning"]
        assert "a typical serving might contain (a guess, not the user's words): egg, olive oil" \
            in judge.calls[0]["plan_text"]

    def test_a_guessed_ingredient_is_never_an_allergy_verdict(self):
        noodles = dish(None, "lunch", "Noodles")
        noodles.typical_ingredients = [{"name": "peanuts", "quantity": "20 g"}]

        rows, violations = constraint_rows([noodles], {"allergies": ["peanuts"]})

        assert row(rows, "no peanuts")["status"] == "satisfied"
        assert violations["allergens"] == []

    def test_calories_for_the_day_against_the_members_target(self):
        result, _ = self.score()
        calories = result.metrics[1]

        assert calories["score"] == 22
        assert "About 400 kcal for the day against your target of 1,800 kcal (22%)" in calories["reasoning"]
        assert "counting only the 1 of 3 dishes with nutrition data" in calories["reasoning"]
        assert calories["detail"]["target_is_default"] is False

    def test_estimated_calories_are_labelled_in_the_metric_and_for_the_judge(self):
        day = [
            dish(None, "breakfast", "Nonna's porridge", nutrition={"kcal": 320},
                 nutrition_source=NUTRITION_TYPICAL),
            dish(None, "lunch", "Lentil soup", nutrition={"kcal": 410},
                 nutrition_source=NUTRITION_MODEL_ESTIMATE),
            dish(None, "dinner", "Fish pie", ingredients="fish"),
        ]
        scorer, judge = make_scorer()
        result = scorer.score("daily", day, build_daily(day), {})

        calories = next(m for m in result.metrics if m["key"] == "daily_nutrition")
        assert calories["detail"]["estimated_dishes"] == {NUTRITION_TYPICAL: 1, NUTRITION_MODEL_ESTIMATE: 1}
        assert "1 dish(es) had no recipe, so their calories are estimated from typical ingredients." \
            in calories["reasoning"]
        assert "1 dish(es) use a rough calorie guess" in calories["reasoning"]
        assert "(estimated from typical ingredients, not a recipe)" in judge.calls[0]["plan_text"]
        assert "(a rough guess, not a recipe)" in judge.calls[0]["plan_text"]

    def test_the_plan_card_carries_reason_chips(self):
        result, _ = self.score()
        lunch = result.plan_entries[1]["recipe"]

        assert {"kind": "profile", "label": "you like lentils"} in lunch["match_reasons"]

    def test_a_failed_judge_leaves_the_measured_metrics_standing(self):
        result, judge = self.score(FakeJudge(fail=True))
        by_key = {m["key"]: m for m in result.metrics}

        assert len(judge.calls) == 2, "the call is retried once"
        for key in JUDGED_METRIC_KEYS:
            assert by_key[key]["score"] is None
            assert by_key[key]["reasoning"] == NOT_GRADED
        assert by_key["fvs"]["score"] == 5
        assert by_key["daily_nutrition"]["score"] == 22

    def test_one_missing_section_does_not_sink_the_others(self):
        judge = FakeJudge({"diversity": {"score": 4, "reasoning": "diverse"},
                           "fit": {"score": 5, "reasoning": "fits"}})
        result, _ = self.score(judge)
        by_key = {m["key"]: m for m in result.metrics}

        assert by_key["diversity"]["score"] == 4
        assert by_key["fit"]["score"] == 5
        assert by_key["guideline_adherence"]["score"] is None


class TestWeeklyScoring:
    WEEK = [
        dish(1, "breakfast", "Oats", state=MATCHED, rid="r-oat", ingredients="oats, milk", nutrition={"kcal": 350}),
        dish(1, "dinner", "Grilled salmon", ingredients="salmon, lemon"),
        dish(1, "snack", "Apple", ingredients="apple"),
        dish(2, "breakfast", "Oats", state=MATCHED, rid="r-oat", ingredients="oats, milk", nutrition={"kcal": 350}),
        dish(2, "dinner", "Beef stew", ingredients="beef, potato"),
        dish(3, "lunch", "Lentil soup", ingredients="lentils"),
    ]

    def score(self, week=None, profile=None, judge=None):
        week = week or self.WEEK
        scorer, judge = make_scorer(judge)
        result = scorer.score("weekly", week, build_weekly(week), profile or {"preferences": []})
        return result, judge

    def test_the_metrics_and_their_order(self):
        result, _ = self.score()

        assert [m["key"] for m in result.metrics] == list(WEEKLY_METRIC_KEYS)
        assert [m["score"] for m in result.metrics if m["kind"] == "likert5"] == [4, 3, 5]

    def test_snacks_are_eaten_but_are_not_meals(self):
        result, _ = self.score()
        variety, _checklist, calories = result.metrics[:3]

        assert variety["detail"]["total_meals"] == 5
        assert variety["detail"]["planned_repeats"] == 1
        assert calories["detail"]["coverage"] == {"meals_with_data": 2, "total_meals": 6}

    def test_guessed_ingredients_join_the_weeks_ingredient_count(self):
        stew = dish(2, "lunch", "Mystery stew")
        stew.typical_ingredients = [{"name": "beans", "quantity": "100 g"}, {"name": "salmon", "quantity": ""}]
        without, _ = self.score()
        result, _ = self.score(self.WEEK + [stew])

        before = without.metrics[0]["detail"]["unique_ingredients"]
        variety = result.metrics[0]
        assert variety["detail"]["unique_ingredients"] == before + 1, "salmon is already counted"
        assert variety["detail"]["dishes_with_guessed_ingredients"] == 1
        assert f"; {before + 1} unique ingredients;" in variety["reasoning"]
        assert "(“Mystery stew”) are counted with a typical serving's ingredients — a guess" in variety["reasoning"]
        assert variety["detail"]["category_distribution"] == \
            self.score(self.WEEK + [dish(2, "lunch", "Mystery stew")])[0].metrics[0]["detail"]["category_distribution"]

    def test_calories_are_judged_over_the_days_pasted_not_a_week(self):
        result, _ = self.score()
        calories = result.metrics[2]

        assert calories["detail"]["days"] == 3
        assert calories["detail"]["daily_average_kcal"] == 233.3
        assert calories["detail"]["weekly_targets"]["kcal"] == 6000
        assert "a default target of 2,000 kcal" in calories["reasoning"]

    def test_the_frequency_checklist(self):
        result, _ = self.score()
        checklist = result.metrics[1]

        assert checklist["score"] == 3 and checklist["detail"]["applicable"] == 3
        assert checklist["detail"]["rules"][2]["target"] == "at least 3 of 5 meals"

    def test_a_short_plan_without_fish_has_not_missed_a_weekly_rule(self):
        no_fish = [d for d in self.WEEK if d.title_given != "Grilled salmon"]
        result, _ = self.score(no_fish)
        fish = result.metrics[1]["detail"]["rules"][0]

        assert fish["met"] is None and "covers 3 day(s)" in fish["note"]
        assert result.metrics[1]["detail"]["applicable"] == 2

    def test_measured_rows_speak_of_the_days_pasted(self):
        result, _ = self.score()
        constraints = {r["constraint"]: r for r in result.constraints}

        assert constraints["at most 2 meat meal(s) over these 3 days"]["status"] == "satisfied"
        assert constraints["weekly calorie target"]["source"] == "default target"
        assert constraints["repeats are your own choice"]["status"] == "satisfied"

    def test_the_judge_is_told_it_is_a_week_and_given_the_measured_facts(self):
        _, judge = self.score()
        call = judge.calls[0]

        assert call["weekly"] is True
        assert call["plan_shape"] == "3 days"
        assert call["guidelines"] == "weekly rules"
        assert call["facts"].startswith("The plan covers 3 day(s).")
        assert "eat fish 1–2 times a week: 1 (target 1–2 meals), met" in call["facts"]
        assert "Day 1:" in call["plan_text"] and "Day 3:" in call["plan_text"]

    def test_an_allergen_in_one_day_caps_the_week(self):
        week = list(self.WEEK) + [dish(3, "dinner", "Satay", ingredients="chicken, peanuts",
                                       conflicts=[{"allergen": "peanuts", "evidence": AS_WRITTEN}])]
        result, _ = self.score(week, {"allergies": ["peanuts"], "preferences": []})

        assert result.metrics[-1]["score"] == 1
        assert row(result.constraints, "no peanuts")["status"] == "violated"


class FlakyJudge(FakeJudge):
    """Fails or returns nothing usable once, then answers."""

    def __init__(self, first):
        super().__init__()
        self.first = first

    def judge(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            if isinstance(self.first, Exception):
                raise self.first
            return self.first
        return {key: dict(value) for key, value in self.DEFAULT.items()}


@pytest.mark.parametrize("first", [
    RuntimeError("429 rate limit"),
    {"diversity": {"score": 0, "reasoning": "bad"}, "guideline_adherence": {"score": 0, "reasoning": "bad"},
     "fit": {"score": 0, "reasoning": "bad"}},
])
def test_the_judge_call_gets_one_more_try(first):
    judge = FlakyJudge(first)
    scorer, _ = make_scorer(judge)
    day = TestDailyScoring.DAY

    result = scorer.score("daily", day, build_daily(day), {})

    diversity = next(m for m in result.metrics if m["key"] == "diversity")
    assert diversity["score"] == 4 and len(judge.calls) == 2


# ===================================================================== #
# Step 5 — summary, persistence, endpoint                               #
# ===================================================================== #

class TestSummary:
    def test_the_writer_gets_the_scores_and_an_allergen_is_never_dropped(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        writer = ScriptedWriter("Solid effort overall.")
        grounder = EchoGrounder({"peanut noodles": [{"allergen": "peanuts", "evidence": AS_WRITTEN}]})
        svc = full_service(session_service, parser=NoParser(), grounder=grounder, writer=writer)

        turn = svc.process(session.session_id, "lunch: peanut noodles\ndinner: soup")

        assert turn.text.startswith("Solid effort overall.")
        assert "Heads up: “peanut noodles” contains peanuts" in turn.text
        facts = writer.facts[0]
        assert facts["action"] == "scored_pasted_plan"
        assert {"metric": "Fit to your profile", "score": 1, "out_of": 5} in facts["scores"]
        assert any(c.startswith("no peanuts") for c in facts["constraints_broken"])

    def test_the_fallback_reads_and_scores(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        svc = full_service(session_service)

        turn = svc.process(session.session_id, "breakfast: oats\nlunch: soup\ndinner: lentil stew")

        assert turn.text.startswith("I read this as a one-day plan with 3 dish(es).")
        assert "Scores: nutritional diversity 4 out of 5" in turn.text
        assert "fit to your profile 5 out of 5." in turn.text

    def test_one_dish_with_related_allergens_is_one_warning(self):
        """Live: "may contain lactose" and "may contain dairy" were two
        sentences about the same pasta."""
        meals = [dish(None, "lunch", "Pasta with zucchini", conflicts=[
            {"allergen": "lactose", "evidence": CLOSEST_RECIPE},
            {"allergen": "dairy", "evidence": CLOSEST_RECIPE},
        ])]

        assert allergen_sentences(meals) == [(
            ["lactose", "dairy"],
            "Heads up: “Pasta with zucchini” may contain lactose and dairy, which are on your allergy list.",
        )]
        assert ensure_allergen_warnings("Looks balanced.", meals) == (
            "Looks balanced. Heads up: “Pasta with zucchini” may contain lactose and dairy, "
            "which are on your allergy list."
        )
        assert ensure_allergen_warnings("Watch the lactose and dairy in the pasta.", meals) == \
            "Watch the lactose and dairy in the pasta."

    def test_dishes_sharing_an_allergen_are_one_warning(self):
        """Live: the same tree-nut sentence three times in one reply."""
        nuts = [{"allergen": "tree nuts", "evidence": AS_WRITTEN}]
        meals = [
            dish(1, "breakfast", "Walnut yogurt", conflicts=nuts),
            dish(2, "dinner", "Cashew stir fry", conflicts=nuts),
            dish(3, "breakfast", "Almond porridge", conflicts=nuts),
            dish(3, "lunch", "Muesli", conflicts=[{"allergen": "tree nuts", "evidence": CLOSEST_RECIPE}]),
        ]

        assert [s for _, s in allergen_sentences(meals)] == [
            "Heads up: “Walnut yogurt”, “Cashew stir fry” and “Almond porridge” contain tree nuts, "
            "which is on your allergy list.",
            "Heads up: “Muesli” may contain tree nuts, which is on your allergy list.",
        ]

    def test_a_reply_naming_the_allergen_with_an_odd_hyphen_is_not_repeated(self):
        meals = [dish(None, "snack", "Almonds", conflicts=[{"allergen": "tree nuts", "evidence": AS_WRITTEN}])]

        assert ensure_allergen_warnings("The almonds are a tree\u2011nut risk.", meals) == \
            "The almonds are a tree\u2011nut risk."

    def test_a_calorie_total_without_its_caveat_gets_one(self):
        """Live: "about 1,030 kcal, far below your target" for a day two of
        whose three dishes were estimates."""
        estimated = [
            dish(None, "lunch", "Soup", nutrition={"kcal": 300}, nutrition_source=NUTRITION_TYPICAL),
            dish(None, "dinner", "Stew", nutrition={"kcal": 500}, nutrition_source="recipe"),
        ]
        partial = [dish(None, "lunch", "Soup", nutrition={"kcal": 300}, nutrition_source="recipe"),
                   dish(None, "dinner", "Stew")]
        known = [dish(None, "lunch", "Soup", nutrition={"kcal": 300}, nutrition_source="recipe")]

        assert ensure_calorie_caveat("About 800 kcal, short of your target.", estimated) == (
            "About 800 kcal, short of your target. Calories for 1 of 2 dishes are estimates, "
            "not recipe figures."
        )
        assert ensure_calorie_caveat("About 300 kcal.", partial).endswith(
            "Calories are known for only 1 of 2 dishes, so any total is incomplete."
        )
        assert ensure_calorie_caveat("About 800 kcal, partly estimated.", estimated) == \
            "About 800 kcal, partly estimated."
        assert ensure_calorie_caveat("Good variety.", estimated) == "Good variety."
        assert ensure_calorie_caveat("About 300 kcal.", known) == "About 300 kcal."

    def test_the_writer_may_not_total_calories_it_only_partly_knows(self):
        """Live: one dish of three had calories, and the reply said the day
        fell far short of energy needs."""
        from models.pasted_plan import PastedDay, PastedPlan
        from services.plan_scorer.scoring import ScoreResult

        plan = PastedPlan(plan_type="daily", days=[PastedDay()])
        partial = [dish(None, "lunch", "Soup", nutrition={"kcal": 300}), dish(None, "dinner", "Stew")]
        estimated = [
            dish(None, "lunch", "Soup", nutrition={"kcal": 300}, nutrition_source=NUTRITION_TYPICAL),
            dish(None, "dinner", "Stew", nutrition={"kcal": 500}, nutrition_source="recipe"),
        ]
        known = [dish(None, "lunch", "Soup", nutrition={"kcal": 300}, nutrition_source="recipe")]

        facts = summary_facts(plan, partial, ScoreResult())
        assert facts["calories"]["known_for"] == "1 of 2 dishes"
        assert "Do not state or compare the plan's total calories" in facts["calories"]["instruction"]
        assert summary_facts(plan, estimated, ScoreResult())["calories"]["estimated_for"] == "1 of 2 dishes"
        assert "calories" not in summary_facts(plan, known, ScoreResult())

    def test_a_reply_that_names_the_allergen_is_left_as_written(self):
        meals = [dish(None, "lunch", "Peanut noodles", conflicts=[{"allergen": "peanuts", "evidence": AS_WRITTEN}])]

        assert ensure_allergen_warnings("Careful: the noodles have peanut in them.", meals) == \
            "Careful: the noodles have peanut in them."

    def test_the_reply_tells_a_close_match_from_a_different_dish(self):
        from models.pasted_plan import PastedDay, PastedPlan
        from services.plan_scorer.service import describe_reading

        close = dish(None, "dinner", "vegetable lasagna", state=APPROXIMATE, rid="r-las")
        close.title_matched, close.borrows_nutrition = "Lasagna", True
        loose = dish(None, "snack", "an apple", state=APPROXIMATE, rid="r-str")
        loose.title_matched = "Apple strudel"

        text = describe_reading(PastedPlan(plan_type="daily", days=[PastedDay()]), [close, loose])

        assert "so the calories come from reading “vegetable lasagna” as “Lasagna”" in text
        assert "“an apple” (nearest: “Apple strudel”) are different dishes" in text


class TestPersistence:
    def test_the_payload_is_stored_with_the_reply_and_survives_a_reload(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        turn = full_service(session_service).process(
            session.session_id, "breakfast: oats\nlunch: soup\ndinner: lentil stew",
        )

        reloaded = SessionService().get_session(session.session_id)
        assert reloaded.conversation[-1].intent == "score_plan"
        assert reloaded.conversation[-1].plan_score == turn.plan_score
        assert reloaded.conversation[-2].plan_score is None

        page = session_service.get_messages_page(session.session_id)
        assert page[-1]["plan_score"]["metrics"][0]["key"] == "fvs"
        assert page[-2]["plan_score"] is None

    def test_an_old_messages_table_gains_the_column(self, monkeypatch, tmp_path):
        import sqlalchemy as sa

        import db

        engine = sa.create_engine(f"sqlite:///{tmp_path / 'old.db'}")
        with engine.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, member_id TEXT, user_profile TEXT, "
                "daily_canvas TEXT, weekly_canvas TEXT, clarification_state TEXT, title TEXT)"
            ))
            conn.execute(sa.text(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                "content TEXT, intent TEXT, plan_id TEXT, timestamp TEXT, attribution TEXT)"
            ))
            conn.execute(sa.text(
                "CREATE TABLE meal_plans (id TEXT PRIMARY KEY, session_id TEXT, plan_type TEXT, payload TEXT, "
                "version INTEGER NOT NULL DEFAULT 1, parent_id TEXT, saved INTEGER NOT NULL DEFAULT 0, "
                "saved_title TEXT, saved_at TIMESTAMP)"
            ))
        monkeypatch.setattr(db, "engine", engine)

        db._migrate_existing_db()
        db._migrate_existing_db()  # idempotent

        assert "plan_score" in {c["name"] for c in sa.inspect(engine).get_columns("messages")}


class TestScorePlanEndpoint:
    WEEK = (
        "Monday\nbreakfast: oats\ndinner: lentil curry\n"
        "Tuesday\nbreakfast: oats\nlunch: salmon salad\nsnack: apple"
    )
    SIX = "breakfast: oats\nlunch: soup\ndinner: fish\nbreakfast: eggs\nlunch: salad\ndinner: tofu"

    def test_a_pasted_week_comes_back_as_a_scored_card(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orchestrator_with(session_service, monkeypatch)

        response = foodchat_router.score_plan(session.session_id, ScorePlanRequest(
            member_id=session.member_id, plan_text=self.WEEK, context="more fish",
        ))

        assert response.intent == "score_plan" and not response.needs_clarification
        card = response.plan_score
        assert [m.key for m in card.metrics] == list(WEEKLY_METRIC_KEYS)
        assert card.context == "more fish"
        assert card.scored_plan.origin == "pasted"
        assert [d.day for d in card.scored_plan.days] == [1, 2]
        assert "snack" in {e.meal_type for e in card.scored_plan.entries}
        # sample_profile is vegetarian: the salmon salad breaks it, and says so
        assert row(card.constraints_applied, "vegetarian")["status"] == "violated"
        assert next(m for m in card.metrics if m.key == "fit").score == 2
        assert response.meal_plan is None and response.weekly_meal_plan is None
        assert response.plan_version is None
        stored = session_service.get_session(session.session_id)
        assert stored.daily_canvas is None and stored.weekly_canvas is None

    def test_the_text_box_settles_the_shape_and_supersedes_a_pending_question(
        self, session_service, sample_profile, monkeypatch,
    ):
        session = new_session(session_service, sample_profile)
        session_service.set_clarification_state(session.session_id, {"kind": "edit_slot"})
        orchestrator_with(session_service, monkeypatch)

        response = foodchat_router.score_plan(session.session_id, ScorePlanRequest(
            member_id=session.member_id, plan_text=self.SIX, plan_type="weekly",
        ))

        assert not response.needs_clarification
        assert response.plan_score.days_scored == 2
        assert session_service.get_session(session.session_id).state == "ready"

    def test_another_members_session_is_not_found(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orchestrator_with(session_service, monkeypatch)

        with pytest.raises(HTTPException) as error:
            foodchat_router.score_plan(session.session_id, ScorePlanRequest(
                member_id="someone-else", plan_text="breakfast: oats\nlunch: soup",
            ))
        assert error.value.status_code == 404

    def test_empty_text_is_refused(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orchestrator_with(session_service, monkeypatch)

        with pytest.raises(ValidationError):
            ScorePlanRequest(member_id=session.member_id, plan_text="")
        with pytest.raises(HTTPException) as error:
            foodchat_router.score_plan(session.session_id, ScorePlanRequest(
                member_id=session.member_id, plan_text="   ",
            ))
        assert error.value.status_code == 400

    def test_the_conversation_page_returns_the_card(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orchestrator_with(session_service, monkeypatch)
        foodchat_router.score_plan(session.session_id, ScorePlanRequest(
            member_id=session.member_id, plan_text="breakfast: oats\nlunch: soup\ndinner: lentil stew",
        ))

        page = foodchat_router.get_conversation(
            session.session_id, member_id=session.member_id, before_id=None, limit=20,
        )

        assert page.messages[-1]["plan_score"]["plan_type"] == "daily"
        assert page.messages[-2]["plan_score"] is None

    def test_the_text_box_keeps_every_entry_point_guard(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orch = orchestrator_with(session_service, monkeypatch)

        turn = orch.score_plan(session.session_id, session.member_id, "breakfast: chicken wrap\nlunch: soup")
        assert turn.memory_suggestions == [{"kind": "dislike", "value": "chicken"}]

        stored = session_service.get_session(session.session_id)
        stored.max_messages = len(stored.conversation)
        assert orch.score_plan(session.session_id, session.member_id, "breakfast: oats").at_message_limit

    def test_a_chat_paste_produces_the_same_wire_card(self, session_service, sample_profile, monkeypatch):
        session = new_session(session_service, sample_profile)
        orch = orchestrator_with(session_service, monkeypatch)

        turn = orch.process(
            session.session_id, session.member_id,
            "rate this:\nbreakfast: oats\nlunch: lentil soup\ndinner: bean chilli",
        )
        response = _chat_turn_response(turn)

        assert response.intent == "score_plan"
        assert [m.key for m in response.plan_score.metrics] == list(DAILY_METRIC_KEYS)

    def test_a_partial_two_plate_day_renders_as_plates(self, session_service, sample_profile):
        session = new_session(session_service, sample_profile)
        turn = full_service(session_service).process(
            session.session_id, "breakfast: oats\ndinner: pasta\ndinner: green salad",
        )
        card = PlanScoreResponse(**turn.plan_score)

        meals = card.scored_plan.days[0].meals
        assert [(m.meal_type, [p.title for p in m.plates]) for m in meals] == [
            ("breakfast", ["oats"]), ("dinner", ["pasta", "green salad"]),
        ]
        assert card.scored_plan.entries == []
        assert any("No lunch is listed" in w for w in card.warnings)
