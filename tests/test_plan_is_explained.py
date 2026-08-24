"""
The reply explains the food, not the rule-check.

FoodChat is not a constraint solver — it is meant to help a household shape
meals that are better for them and say why. The facts handed to the response
writer said otherwise. Of eight keys, five were constraint bookkeeping:

    constraints_honored        constraints_not_honored
    verified_problems          repair                     pantry

and **none** was about health. So every plan was explained as a compliance
result: what was permitted, what was swapped, what fell short. Meanwhile the
system had already measured food variety, guideline adherence, meal diversity,
Nutri-Score and the day's calories — and handed the member a collapsed panel of
scores out of five instead of a sentence.

`plan_value` is the other half. Everything in it is MEASURED — a count, a
total, or a judge's own sentence — because the writer may only phrase what the
facts contain, and "healthy" is not a measurement it is allowed to supply.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.session import DayPlan, Meal, MealCourse, MealPlan   # noqa: E402
from services import transparency                                # noqa: E402


def _plate(rid, title, kcal=None, protein=None, grade=None, role="main"):
    nutrition = None
    if kcal is not None or grade is not None:
        nutrition = {"kcal": kcal, "protein_g": protein, "nutri_score_label": grade}
    return MealCourse(rid, title, "beans, rice", "cook", nutrition=nutrition, role=role)


def _plan(*plates, days=1):
    day_plans = [
        DayPlan(day=n + 1, meals=[Meal("dinner", list(plates))])
        for n in range(days)
    ]
    return MealPlan.from_days(day_plans, reasoning="")


class TestItSaysWhatIsGoodAboutThePlan:
    def test_the_day_adds_up_to_a_number(self):
        plan = _plan(_plate("a", "Stew", kcal=600, protein=30))
        value = transparency.plan_value(plan, {})
        assert value["nutrition"]["kcal_per_day"] == 600
        assert value["nutrition"]["protein_g_per_day"] == 30

    def test_it_is_per_day_not_per_plan(self):
        """A week's total is not a fact about a dinner."""
        plan = _plan(_plate("a", "Stew", kcal=600, protein=30), days=3)
        assert transparency.plan_value(plan, {})["nutrition"]["kcal_per_day"] == 600

    def test_a_target_the_member_set_is_carried(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(plan, {}, kcal_target=2000)
        assert value["nutrition"]["kcal_target"] == 2000

    def test_a_target_nobody_set_is_not_invented(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        assert "kcal_target" not in transparency.plan_value(plan, {})["nutrition"]

    def test_a_partial_sum_says_so(self):
        """A reply must not present the profiled half of a meal as the meal."""
        plan = _plan(_plate("a", "Stew", kcal=600), _plate("b", "Salad", role="side"))
        assert "partial" in transparency.plan_value(plan, {})["nutrition"]

    def test_nothing_measured_means_no_nutrition_claim(self):
        plan = _plan(_plate("a", "Stew"))
        assert "nutrition" not in transparency.plan_value(plan, {})

    def test_variety_is_a_count_of_real_things(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(plan, {}, metrics={"fvs_count": 23})
        assert value["distinct_foods"] == 23

    def test_the_judges_own_sentence_is_used_not_its_score(self):
        """"3 out of 5" is not something to tell someone about their dinner."""
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(plan, {}, metrics={
            "guideline_adherence_score": 3,
            "guideline_adherence_reasoning": "Fish twice this week, red meat once.",
            "diversity_llm_score": 4,
            "diversity_llm_reasoning": "Three different cuisines across the day.",
        })
        assert value["guidance"].startswith("Fish twice")
        assert value["balance"].startswith("Three different")
        assert "3" not in str(value.get("guidance", ""))[:1]

    def test_nutri_score_is_reported_as_a_proportion(self):
        plan = _plan(
            _plate("a", "Stew", kcal=600, grade="A"),
            _plate("b", "Salad", kcal=100, grade="A", role="side"),
            _plate("c", "Cake", kcal=400, grade="D", role="dessert"),
        )
        assert transparency.plan_value(plan, {})["nutri_score"] == \
            "2 of 3 dishes are Nutri-Score A or B"

    def test_ungraded_dishes_produce_no_claim(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        assert "nutri_score" not in transparency.plan_value(plan, {})

    def test_the_household_is_named_when_there_is_one(self):
        """Cooking for four is a different job from cooking for one."""
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(plan, {"cooking_for_names": ["Anne", "Dimitris"]})
        assert value["cooking_for"] == ["Anne", "Dimitris"]

    def test_one_diner_is_not_a_household(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        assert "cooking_for" not in transparency.plan_value(
            plan, {"cooking_for_names": ["Anne"]},
        )

    def test_what_it_uses_up_is_the_sustainability_half(self):
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(
            plan, {}, pantry_facts={"used": ["spinach", "olives"], "unused": []},
        )
        assert value["using_up"] == ["spinach", "olives"]

    def test_an_empty_metric_is_omitted_not_zeroed(self):
        """A reply that says "0 unique foods" because a metric was skipped is
        worse than one that talks about the food."""
        plan = _plan(_plate("a", "Stew", kcal=600))
        value = transparency.plan_value(plan, {}, metrics={
            "fvs_count": 0, "guideline_adherence_reasoning": "  ",
        })
        assert "distinct_foods" not in value
        assert "guidance" not in value


class TestBothPathsHandItToTheWriter:
    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_plan_value_reaches_the_facts(self, method):
        import ast
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method)).lstrip()
        tree = ast.parse(src)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "plan_value"
        ]
        assert calls, f"{method} explains the plan only as a rule-check"

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_it_is_given_the_measurements(self, method):
        import ast
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method)).lstrip()
        tree = ast.parse(src)
        call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "plan_value"
        )
        passed = {kw.arg for kw in call.keywords}
        assert {"metrics", "kcal_target"} <= passed


class TestTheVoiceIsNotAComplianceReport:
    @staticmethod
    def _prompt() -> str:
        from prompts import RESPONSE_WRITER_SYSTEM_INSTRUCTIONS

        return RESPONSE_WRITER_SYSTEM_INSTRUCTIONS.lower()

    def test_it_is_told_to_lead_with_the_food(self):
        assert "lead with" in self._prompt()

    def test_it_is_told_a_diagnosis_is_not_a_headline(self):
        """Someone's coeliac disease is not the headline of their dinner."""
        prompt = self._prompt()
        assert "never lead with" in prompt
        assert "health condition" in prompt

    def test_it_may_not_enumerate_restrictions(self):
        assert "enumerate" in self._prompt()

    def test_it_may_not_supply_the_word_healthy_itself(self):
        """The one claim a writer must never invent, because it is the claim
        the whole product rests on."""
        prompt = self._prompt()
        assert "only call something healthy" in prompt

    def test_a_failure_is_still_allowed_to_lead(self):
        """Decency is not silence: something that could not be honoured is the
        one case where a constraint goes first."""
        assert "could not be honoured" in self._prompt()

    def test_it_is_registered_under_a_new_name(self):
        """`sync_prompts` creates only missing prompts, so editing the text
        under the old name would ship it dead."""
        import prompts

        assert any(
            p.name.endswith("response_writer_system_v2") for p in prompts.ALL_PROMPTS
        )
