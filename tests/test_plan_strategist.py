"""
The reasoning step, and the guarantees around it.

The user asked for an agent that reasons rather than a deterministic chain of
LLM calls. The shape agreed was hybrid: a strategist decides HOW to search, the
existing pipeline executes, a deterministic verifier checks the result.

That only works if the reasoning step cannot make things worse. These tests
pin the three properties that guarantee it:

    it cannot invent    every value is validated against the live vocabulary
    it cannot unsafe    allergens and diet are not in its schema at all
    it cannot break     a failure leaves the deterministic brief untouched

LLM-free: the client is replaced per test, which is how every agent in this
suite is tested.
"""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from agents import PlanStrategist                # noqa: E402
from models.plan_brief import PlanBrief          # noqa: E402

VOCAB = {
    "cuisines": ["thai", "greek"],
    "moods": ["light", "hearty"],
    "flavor_profiles": ["spicy"],
    "food_groups": ["legumes"],
}


class _Reply:
    def __init__(self, payload):
        self.content = payload if isinstance(payload, str) else json.dumps(payload)


class _Client:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return _Reply(self.payload)


def _strategist(payload=None, error=None):
    agent = PlanStrategist.__new__(PlanStrategist)
    agent.llm = _Client(payload, error)
    return agent


class TestItProposes:
    def test_a_proposal_comes_back_as_a_dict(self):
        agent = _strategist({"moods": ["light"], "rationale": "light after the gym"})
        out = agent.propose("something light", PlanBrief(), VOCAB)
        assert out["moods"] == ["light"]
        assert out["rationale"] == "light after the gym"

    def test_the_live_vocabulary_reaches_the_prompt(self):
        """Not decoration: an unlisted value empties the search rather than
        narrowing it, so the model has to be told what exists."""
        agent = _strategist({})
        agent.propose("something light", PlanBrief(), VOCAB)
        sent = " ".join(m.content for m in agent.llm.calls[0])
        assert "thai" in sent and "hearty" in sent

    def test_what_is_already_standing_reaches_the_prompt(self):
        """Otherwise it re-proposes what is already in force, or contradicts it."""
        agent = _strategist({})
        agent.propose("more of the same", PlanBrief(diet=("vegetarian",)), VOCAB)
        sent = " ".join(m.content for m in agent.llm.calls[0])
        assert "vegetarian" in sent

    def test_a_long_message_is_truncated_before_it_is_sent(self):
        agent = _strategist({})
        agent.propose("x" * 5000, PlanBrief(), VOCAB)
        sent = " ".join(m.content for m in agent.llm.calls[0])
        assert "x" * 601 not in sent


class TestItCannotBreakTheTurn:
    def test_no_vocabulary_means_no_call_at_all(self):
        """With no live list there is nothing safe to propose, and spending a
        reasoning-tier call to find that out is waste."""
        agent = _strategist({"moods": ["light"]})
        assert agent.propose("light", PlanBrief(), {}) == {}
        assert agent.llm.calls == []

    def test_a_failed_call_returns_no_adjustment(self):
        agent = _strategist(error=RuntimeError("groq is down"))
        assert agent.propose("light", PlanBrief(), VOCAB) == {}

    def test_unparseable_output_returns_no_adjustment(self):
        assert _strategist("not json at all").propose("x", PlanBrief(), VOCAB) == {}

    def test_a_non_object_payload_is_refused(self):
        assert _strategist([1, 2, 3]).propose("x", PlanBrief(), VOCAB) == {}

    def test_the_deterministic_brief_survives_every_failure(self):
        """The plan that used to be built is the floor, never the casualty."""
        brief = PlanBrief(diet=("vegetarian",), allergens=("peanuts",),
                          moods=("hearty",), kcal_target=1900.0)
        for agent in (_strategist(error=RuntimeError("boom")),
                      _strategist("garbage"),
                      _strategist({})):
            after = brief.with_strategy(agent.propose("x", brief, VOCAB), VOCAB)
            assert after.diet == ("vegetarian",)
            assert after.allergens == ("peanuts",)
            assert after.moods == ("hearty",)
            assert after.kcal_target == 1900.0


class TestItRunsOnTheReasoningTier:
    def test_it_is_not_on_the_fast_tier(self):
        """"Something light after the gym" becoming high protein with a light
        mood is a judgement about food, not a span to pick out of a sentence."""
        import inspect

        src = inspect.getsource(PlanStrategist.__init__)
        assert "DEFAULT_MODEL" in src and "FAST_MODEL" not in src


class TestTheSchemaExcludesSafety:
    def test_there_is_no_allergen_or_diet_field(self):
        """Not validation — absence. A reasoning step may decide how to search
        and may not decide to drop a safety constraint, and the cheapest way to
        guarantee that is to give it no way to say so."""
        from schemas import PlanStrategySchema

        fields = set(PlanStrategySchema.model_fields)
        assert "allergens" not in fields
        assert "diet" not in fields
        assert "exclude_ingredients" not in fields

    def test_it_can_say_the_things_it_is_for(self):
        from schemas import PlanStrategySchema

        fields = set(PlanStrategySchema.model_fields)
        assert {"cuisines", "moods", "flavor_profiles", "food_groups",
                "claim_tags", "relaxation_order", "rationale"} <= fields


class TestEndToEnd:
    def test_a_good_proposal_shapes_the_search(self):
        agent = _strategist({
            "moods": ["light"], "claim_tags": ["high_protein"],
            "rationale": "read 'light after the gym' as protein with a light mood",
        })
        brief = PlanBrief.build({"diet": ["vegetarian"]})
        after = brief.with_strategy(
            agent.propose("something light after the gym", brief, VOCAB), VOCAB)
        assert after.moods == ("light",)
        assert "high_protein" in after.claim_tags
        assert after.diet == ("vegetarian",), "the diet is untouched"
        assert "light after the gym" in after.rationale

    def test_a_hallucinating_proposal_produces_the_plain_brief(self):
        """The failure mode that matters: a plan that is unshaped, not a plan
        that is empty."""
        agent = _strategist({
            "moods": ["energising"], "cuisines": ["atlantean"],
            "claim_tags": ["keto"], "kcal_target": 9000,
        })
        brief = PlanBrief.build({"diet": ["vegetarian"]})
        after = brief.with_strategy(agent.propose("energy boost", brief, VOCAB), VOCAB)
        assert after.moods == () and after.cuisines == ()
        assert after.claim_tags == () and after.kcal_target is None
        assert after.diet == ("vegetarian",)

    @pytest.mark.parametrize("prompt_name", [
        "plan_strategist_system", "plan_strategist_user",
    ])
    def test_the_prompts_are_registered_under_new_names(self, prompt_name):
        """`sync_prompts` creates only missing prompts and never overwrites, so
        editing an existing prompt body ships dead. A new capability needs a new
        name or it never reaches production."""
        import prompts

        # Names are namespaced ("foodchat/plan_strategist_system").
        assert any(p.name.endswith(prompt_name) for p in prompts.ALL_PROMPTS)
