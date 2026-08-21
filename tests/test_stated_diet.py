"""
A diet stated in chat must reach the candidate fetch.

This is the transcript that prompted the work:

    member    "i need something vegetarian"
    assistant "The current plan includes chicken, pork, and meatballs, which
               conflict with your request … adjust the plan to be fully
               vegetarian?"
    member    "yes please"
    assistant "I couldn't find enough recipes for a complete plan with your
               current constraints (diet: omnivore; allergens excluded: nuts,
               peanuts; avoiding: mushrooms)."

Every part of that is a bug. The conflict was detected, the member confirmed,
and the plan was still built from a pool filtered on `profile["diet"]` alone —
`DietaryIntentExtractor` was wired only into the weekly service. Then the
apology blamed `omnivore`, which `normalize_diet_tags` drops before the request
as non-restrictive, and never mentioned the vegetarian filter at all.

These tests are LLM-free: the extractor is faked, and the assertions are about
what the fetch layer receives and what the member is told.
"""

from __future__ import annotations

import pytest

from models.planning_state import PlanningState, PlanningStateDelta
from services import diet_intent
from services.candidates_client import effective_diet, split_diet_intent


class FakeDietExtractor:
    def __init__(self, tags):
        self._tags = tags
        self.calls: list[str] = []

    def extract(self, query):
        self.calls.append(query)
        return list(self._tags)


# ── the request becomes a filter ─────────────────────────────────────────

def test_a_stated_diet_becomes_a_delta():
    delta = diet_intent.extract_diet_delta(
        "i need something vegetarian", extractor=FakeDietExtractor(["vegetarian"])
    )
    assert delta.diet_tags == ("vegetarian",)
    assert not delta.is_empty


def test_the_stated_diet_reaches_the_fetch_over_an_omnivore_profile():
    """The whole point: `omnivore` must not out-vote the member."""
    state = PlanningState().merge(
        diet_intent.extract_diet_delta(
            "i need something vegetarian",
            extractor=FakeDietExtractor(["vegetarian"]),
        )
    )
    profile = {"diet": ["omnivore"], "_diet_tags": list(state.diet_tags)}
    assert effective_diet(profile) == ["vegetarian"]


def test_it_survives_later_turns():
    """Silence is not a retraction — the next turn still plans vegetarian."""
    state = PlanningState().merge(PlanningStateDelta(diet_tags=("vegetarian",)))
    state = state.merge(PlanningStateDelta(pantry_add=("zucchini",)))
    assert state.diet_tags == ("vegetarian",)


def test_an_explicit_retraction_clears_it():
    """Answering "no, follow my profile" to the conflict question."""
    state = PlanningState().merge(PlanningStateDelta(diet_tags=("vegetarian",)))
    assert state.merge(PlanningStateDelta(diet_clear=True)).diet_tags == ()


def test_a_stated_diet_does_not_erase_a_profile_restriction():
    """A vegetarian request from a coeliac member must satisfy both."""
    assert set(effective_diet({
        "diet": ["gluten_free"], "_diet_tags": ["vegetarian"],
    })) == {"gluten_free", "vegetarian"}


def test_extraction_failure_changes_nothing():
    class Boom:
        def extract(self, q):
            raise RuntimeError("groq down")

    assert diet_intent.extract_diet_delta("vegetarian please", extractor=Boom()).is_empty


# ── nutrition claims are not diet filters ────────────────────────────────

def test_a_nutrition_claim_never_becomes_a_diet_filter():
    """low-carb is on zero recipes as a diet tag — as a filter it empties
    every slot, so "I want a low-carb week" became an outage."""
    delta = diet_intent.extract_diet_delta(
        "something low-carb", extractor=FakeDietExtractor(["low-carb"])
    )
    assert delta.diet_tags == ()
    # And it is not swallowed either. This assertion used to check `notes`,
    # which turned out to be write-only — so the claim was saved from emptying
    # the plan and then dropped. It now reaches RecipeWrangler's `tags` field.
    assert delta.claim_tags, "a stated claim must reach the request"
    assert delta.notes == (), "notes is write-only; nothing may be routed there"


def test_a_mixed_statement_routes_both_ways():
    delta = diet_intent.extract_diet_delta(
        "vegetarian and low-carb",
        extractor=FakeDietExtractor(["vegetarian", "low-carb"]),
    )
    assert delta.diet_tags == ("vegetarian",)
    assert delta.claim_tags, "the claim half must reach the tags field"


@pytest.mark.parametrize("claim", ["low-carb", "low_fat", "high-protein"])
def test_every_claim_tag_is_routed_not_filtered(claim):
    filterable, claims = split_diet_intent([claim])
    assert filterable == []
    assert claims


# ── the failure message tells the truth ──────────────────────────────────

class TestFailureMessage:
    def _msg(self, **profile):
        from services.chat_service import no_plan_message

        return no_plan_message(profile)

    def test_it_names_the_diet_that_was_applied(self):
        msg = self._msg(diet=["omnivore"], _diet_tags=["vegetarian"],
                        allergies=["nuts", "peanuts"], food_dislikes=["mushrooms"])
        assert "vegetarian" in msg

    def test_it_does_not_present_omnivore_as_the_blocker(self):
        """The original message read "diet: omnivore" as a cause. It was not
        even sent — normalize_diet_tags drops it."""
        msg = self._msg(diet=["omnivore"], _diet_tags=["vegetarian"],
                        allergies=["nuts"])
        assert "diet: omnivore" not in msg
        assert "isn't a restriction" in msg

    def test_omnivore_alone_is_not_mentioned_at_all(self):
        """With no stated diet there is no diet clause to qualify, so the
        note would be an accusation against a setting that did nothing."""
        msg = self._msg(diet=["omnivore"], allergies=["nuts"])
        assert "omnivore" not in msg
        assert "allergens excluded: nuts" in msg

    def test_a_real_stored_restriction_is_still_named(self):
        msg = self._msg(diet=["gluten_free"], _diet_tags=["vegetarian"])
        assert "gluten_free" in msg and "vegetarian" in msg


# ── the consent nudge ────────────────────────────────────────────────────

class TestDietMemoryNudge:
    def test_it_offers_to_remember_a_stated_diet(self):
        nudge = diet_intent.suggest_diet_memory(
            ("vegetarian",), "i need something vegetarian", {"diet": ["omnivore"]}
        )
        assert nudge["kind"] == "diet"
        assert nudge["value"] == "vegetarian"

    def test_the_evidence_is_the_members_own_words(self):
        """It answers "why am I seeing this?" — so it must be true."""
        nudge = diet_intent.suggest_diet_memory(
            ("vegetarian",), "i need something vegetarian", {"diet": []}
        )
        assert nudge["evidence"] == "i need something vegetarian"

    def test_no_nudge_when_the_profile_already_says_so(self):
        assert diet_intent.suggest_diet_memory(
            ("vegetarian",), "vegetarian please", {"diet": ["vegetarian"]}
        ) is None

    def test_no_nudge_without_a_stated_diet(self):
        assert diet_intent.suggest_diet_memory((), "hello", {"diet": []}) is None

    def test_the_kind_is_writable(self):
        from services.memory_service import VALID_KINDS

        assert "diet" in VALID_KINDS


class TestDurableWrite:
    """Accepting the nudge sets the profile — the user's explicit decision."""

    def test_the_session_mirror_replaces_omnivore(self):
        from services.memory_service import MemoryService

        class S:
            user_profile = {"diet": ["omnivore"]}
            session_id = "s1"

        session = S()
        # Construct without __init__ so no gateway/DB client is needed, and
        # stub persistence — the assertion is about the in-memory session copy.
        svc = MemoryService.__new__(MemoryService)
        svc.session_service = type("X", (), {"persist_profile": lambda *a, **k: None})()
        svc._apply_to_session_profile(session, "diet", "vegetarian")
        assert session.user_profile["diet"] == ["vegetarian"]

    def test_the_gateway_vocabulary_is_enforced(self):
        """dietary_groups is a Postgres enum array — an off-list value is
        rejected at the API boundary, so it must be rejected here first."""
        from services.profile_service import GATEWAY_DIET_GROUPS

        assert "vegetarian" in GATEWAY_DIET_GROUPS
        # Not in the gateway enum — writing it would 422.
        assert "keto_carnivore" not in GATEWAY_DIET_GROUPS
        assert "lactose_free" not in GATEWAY_DIET_GROUPS

    def test_the_diets_we_can_filter_on_are_the_diets_we_can_store(self):
        """This assertion previously said the opposite, which encoded a real
        bug: the write vocabulary was limited to the UI picker's five values,
        so the three diets FoodChat can actually FILTER on were the three it
        refused to persist. "Remember I'm gluten-free" was offered, accepted,
        and then rejected at the write with applied=false."""
        from services.candidates_client import VALID_RW_DIET_TAGS
        from services.profile_service import GATEWAY_DIET_GROUPS

        for filterable in ("gluten_free", "dairy_free", "nut_free",
                           "vegetarian", "vegan", "pescatarian"):
            assert filterable in VALID_RW_DIET_TAGS, filterable
            assert filterable in GATEWAY_DIET_GROUPS, (
                f"{filterable} can be filtered but not stored"
            )

    def test_the_write_vocabulary_matches_the_gateway_enum_exactly(self):
        """Drift in either direction is a bug: a missing value silently refuses
        a legitimate memory, an extra one 422s at the boundary."""
        import pathlib
        import re

        from services.profile_service import GATEWAY_DIET_GROUPS

        gw = pathlib.Path(
            "/mnt/workspaces/wisefood/wisefood-api/src/schemas.py"
        )
        if not gw.exists():  # gateway not checked out beside us
            pytest.skip("wisefood-api not available")
        block = gw.read_text()
        block = block[block.find("class DietaryGroupEnum"):]
        block = block[:block.find("\n\n\n")]
        enum = set(re.findall(r'=\s*"([a-z_0-9]+)"', block))
        assert GATEWAY_DIET_GROUPS == enum, (
            f"missing: {sorted(enum - GATEWAY_DIET_GROUPS)} "
            f"extra: {sorted(GATEWAY_DIET_GROUPS - enum)}"
        )
