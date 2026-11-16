"""A plan turn says only what the member asked for and what the plan holds.

From production, 16 Nov 2026. A member typed "Plan my day". Four stages, each
one reasonable on its own, turned that into a plan that did not add up and a
reply that described a different one:

1. The query rewriter "added detail" the member never gave: "breakfast,
   lunch, dinner and two snacks, about 2000 calories, each main meal 500-600".
2. The strategist read the 2000 kcal as `low_calorie` and the search returned
   only light dishes: the day came to 1,003 kcal.
3. The grader, told the request had two snacks, marked every three-meal day
   down for missing them ("the maximum possible score is 2"), and that note
   was stored as the plan's reasoning and shown to the member.
4. The reply writer, given the rewrite as the member's message, promised
   "two quick snacks" the plan did not have.

Each test here pins one link of that chain.
"""
from __future__ import annotations

import sys
import uuid
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                              # noqa: E402
from models.recipe import CandidateRecipe, ScoredPlan                # noqa: E402
from services import plan_budget                                     # noqa: E402


def _dish(rid: str, kcal: float | None) -> CandidateRecipe:
    nutrition = None if kcal is None else {"calories": kcal, "protein_g": 20}
    return CandidateRecipe(rid, f"Dish {rid}", "beans, rice", "cook", nutrition=nutrition)


def _day(score: int, *kcals, prefix: str = "d") -> ScoredPlan:
    slots = {
        slot: _dish(f"{prefix}{i}", kcal)
        for i, (slot, kcal) in enumerate(zip(("breakfast", "lunch", "dinner"), kcals))
    }
    return ScoredPlan(score=score, reasoning="", slots=slots)


# ── 2. A budget is not a request for low-calorie dishes ───────────────────

class TestABudgetIsNotALowCalorieRequest:
    VOCAB = {"cuisines": ["greek"], "moods": [], "flavor_profiles": [], "food_groups": ["vegetables"]}

    def test_the_strategists_low_calorie_is_dropped_under_a_full_day_budget(self):
        """The production proposal, verbatim in effect: low_calorie with a
        2000 kcal target."""
        brief = PlanBrief().with_strategy(
            {"claim_tags": ["low_calorie"], "kcal_target": 2000}, self.VOCAB,
        )
        assert "low_calorie" not in brief.claim_tags
        assert brief.kcal_target == 2000

    def test_a_member_who_asked_for_light_food_keeps_it(self):
        """Their own tag stands; the guard only removes the strategist's."""
        brief = PlanBrief(claim_tags=("low_calorie",)).with_strategy(
            {"claim_tags": ["low_calorie"], "kcal_target": 2000}, self.VOCAB,
        )
        assert "low_calorie" in brief.claim_tags

    def test_a_small_budget_may_still_be_low_calorie(self):
        brief = PlanBrief().with_strategy(
            {"claim_tags": ["low_calorie"], "kcal_target": 1300}, self.VOCAB,
        )
        assert "low_calorie" in brief.claim_tags

    def test_with_no_budget_the_strategist_is_unchanged(self):
        brief = PlanBrief().with_strategy({"claim_tags": ["low_calorie"]}, self.VOCAB)
        assert "low_calorie" in brief.claim_tags

    def test_a_reference_counts_as_a_budget(self):
        """A day reported against 2000 kcal must not be searched as light."""
        brief = PlanBrief(kcal_reference=2000.0).with_strategy(
            {"claim_tags": ["low_calorie", "high_protein"]}, self.VOCAB,
        )
        assert brief.claim_tags == ("high_protein",)


# ── The day served is not one that misses the budget by half ──────────────

class TestTheDayServedRespectsTheBudget:
    def test_the_graders_first_is_kept_when_it_is_near(self):
        first, second = _day(4, 500, 700, 700), _day(4, 600, 700, 700, prefix="e")
        assert plan_budget.pick([first, second], 2000) is first

    def test_a_close_runner_up_replaces_a_first_at_half_the_budget(self):
        first = _day(3, 232, 448, 324)                    # 1,004 kcal
        second = _day(2, 450, 700, 750, prefix="e")       # 1,900 kcal
        assert plan_budget.pick([first, second], 2000) is second

    def test_a_much_worse_rated_day_is_not_preferred_for_its_calories(self):
        first = _day(4, 232, 448, 324)
        poor = _day(1, 450, 700, 750, prefix="e")
        assert plan_budget.pick([first, poor], 2000) is first

    def test_no_budget_changes_nothing(self):
        first, second = _day(3, 232, 448, 324), _day(3, 450, 700, 750, prefix="e")
        assert plan_budget.pick([first, second], None) is first

    def test_a_day_with_an_unknown_dish_is_not_judged_by_a_partial_sum(self):
        first = _day(3, 232, None, 324)
        second = _day(3, 450, 700, 750, prefix="e")
        assert plan_budget.pick([first, second], 2000) is first
        assert plan_budget.day_kcal(first) is None

    def test_nothing_to_pick_from(self):
        assert plan_budget.pick([], 2000) is None


# ── 1, 3, 4. The prompts say what the code now relies on ──────────────────

class TestThePromptsForbidInventing:
    def test_the_rewriter_restates_and_never_designs(self):
        from prompts import QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS as text
        assert "never design the plan" in text
        assert "snacks" in text and "per-meal" in text

    def test_the_grader_never_marks_a_day_down_for_a_meal_it_does_not_have(self):
        from prompts import PLAN_GRADER_SYSTEM_INSTRUCTIONS as text
        assert "NEVER mark a day down for a meal or snack it does not contain" in text

    def test_the_writer_describes_only_the_meals_in_the_facts(self):
        from prompts import RESPONSE_WRITER_SYSTEM_INSTRUCTIONS as text
        assert "Describe ONLY the meals listed in the facts" in text

    def test_the_strategist_does_not_read_a_budget_as_low_calorie(self):
        from prompts import PLAN_STRATEGIST_SYSTEM_INSTRUCTIONS as text
        assert "A calorie budget is NOT a request for low-calorie food" in text


# ── The whole daily turn, with the models and the recipe service faked ────

class _Writer:
    def __init__(self):
        self.heard = None

    def write(self, facts, query, fallback):
        self.heard = query
        return f"Here's a day with {facts['meals']['dinner']} for dinner."


class _Pipeline:
    def __init__(self, plans):
        self.plans = plans
        self.query = None

    def generate(self, query, profile, **_kwargs):
        self.query = query
        return list(self.plans)


class _Strategist:
    def __init__(self):
        self.heard = None

    def propose(self, message, brief, vocab):
        self.heard = message
        return {"claim_tags": ["low_calorie"], "kcal_target": 2000}


@pytest.fixture
def turn(session_service, sample_profile, monkeypatch):
    """A ChatService that runs `_generate_and_store` for real, offline."""
    import importlib

    # `services` re-exports a `chat_service` name of its own; the module is
    # what the patches below must reach.
    chat_service = importlib.import_module("services.chat_service")
    ChatService = chat_service.ChatService

    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)

    # The module's client is replaced outright rather than patched. Patching
    # the shared instance leaves the original bound method behind as an
    # instance attribute when undone (which broke test_planning_state_api's
    # class-level patch), and an earlier test's leftover would shadow a
    # class-level patch here. A module attribute is restored cleanly.
    monkeypatch.setattr(chat_service, "CANDIDATES", SimpleNamespace(
        fetch_details=lambda ids: {},
        vocabularies=lambda: {"cuisines": ["greek"]},
    ))
    monkeypatch.setattr(
        chat_service, "household_signals",
        lambda *a, **k: SimpleNamespace(downvoted_recipe_ids=[], history_text=""),
    )
    monkeypatch.setattr(chat_service.guideline_coverage, "ledger_rows", lambda *a, **k: [])
    monkeypatch.setattr(ChatService, "_compute_metrics", lambda self, *a, **k: {
        "llm_score": 2,
        "llm_reasoning": "It omits the two snack slots, so the maximum possible "
                         "score is 2. It earns the highest score among the batch.",
    })

    svc = ChatService.__new__(ChatService)
    svc.session_service = session_service
    svc.feedback_service = None
    svc.strategist = _Strategist()
    svc.response_writer = _Writer()
    light = _day(2, 232, 448, 324)
    svc.pipeline = _Pipeline([light, _day(2, 450, 700, 750, prefix="e")])
    return svc, session.session_id, sample_profile


class TestTheTurnHearsTheMember:
    REWRITE = ("Create a full-day meal plan that includes breakfast, lunch, dinner, "
               "and two snacks, about 2000 calories")

    def _run(self, turn):
        svc, session_id, profile = turn
        text, needs, plan = svc._generate_and_store(
            session_id, self.REWRITE, dict(profile), False, member_words="Plan my day",
        )
        return svc, text, plan

    def test_the_writer_strategist_and_grader_hear_the_members_words(self, turn):
        svc, _text, _plan = self._run(turn)
        assert svc.response_writer.heard == "Plan my day"
        assert svc.strategist.heard == "Plan my day"
        assert svc.pipeline.query == "Plan my day"

    def test_the_plans_reasoning_is_what_the_member_was_told(self, turn):
        _svc, text, plan = self._run(turn)
        assert plan.reasoning == text
        assert "batch" not in plan.reasoning

    def test_the_judges_note_is_still_kept_for_reviewers(self, turn):
        _svc, _text, plan = self._run(turn)
        assert "batch" in (plan.llm_reasoning or "")

    def test_the_day_near_the_budget_is_served(self, turn):
        """The grader tied two days; one is 1,004 kcal, the other 1,900."""
        _svc, _text, plan = self._run(turn)
        assert plan.dinner.recipe_id == "e2"

    def test_without_member_words_the_query_is_used_as_before(self, turn):
        svc, session_id, profile = turn
        svc._generate_and_store(session_id, "plan my day", dict(profile), False)
        assert svc.response_writer.heard == "plan my day"


class TestACalorieTargetIsTheMembersOnlyIfTheySaidOne:
    """A guest typed "plan my day", answered "surprise me", and was told the
    day "falls a bit shy of the 2,000 kcal goal" and checked against "your
    2000 kcal target": the strategist had proposed the number."""

    NO_TARGET = {"diet": [], "allergies": [], "preferences": [], "history": ""}

    def test_an_invented_figure_does_not_become_the_target(self, turn):
        svc, _session_id, _profile = turn
        brief = svc._brief_for("Plan my day — No preference, surprise me", dict(self.NO_TARGET))
        assert brief.kcal_target is None

    def test_a_figure_the_member_named_does(self, turn):
        svc, _session_id, _profile = turn
        svc.strategist.propose = lambda message, brief, vocab: {"kcal_target": 1500}
        brief = svc._brief_for("Plan me a 1500 calorie day", dict(self.NO_TARGET))
        assert brief.kcal_target == 1500

    def test_the_members_own_profile_target_is_untouched(self, turn):
        svc, _session_id, profile = turn  # sample_profile: "2000 calories target"
        brief = svc._brief_for("Plan my day", dict(profile))
        assert brief.kcal_target == 2000
