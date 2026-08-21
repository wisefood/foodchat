"""
Recipe qualities stated in chat must reach the candidate fetch.

"Energy boost meal plan for today" matched nothing, for three stacked reasons:
`plan_meals` declares `moods`/`flavor_profiles`/`food_groups` and RecipeWrangler
accepts all three, but **no caller ever passed any of them**; there is no cuisine
extractor either, so "something Thai tonight" never became a filter; and "energy"
is in no vocabulary at all — it was a slider goal that only ever became prose.

The hard rule these tests protect: **never send a value the corpus does not
carry.** RecipeWrangler ANDs facet values and does not relax an unlisted one to
nothing — it matches no recipe, so an invented mood does not soften the search,
it empties it, and the member is told no meals exist.
"""

from __future__ import annotations

from models.planning_state import PlanningState, PlanningStateDelta
from services import intent_facets as F

VOCAB = {
    "cuisines": ["thai", "greek", "italian"],
    "moods": ["comfort", "light", "hearty", "fresh"],
    "flavor_profiles": ["spicy", "creamy"],
    "food_groups": ["vegetables", "fish", "legumes"],
}


class FakeIntentExtractor:
    def __init__(self, payload):
        self._payload = payload
        self.saw_vocab = None

    def extract(self, message, vocabularies):
        self.saw_vocab = vocabularies
        return self._payload


# ── extraction ───────────────────────────────────────────────────────────

def test_a_stated_mood_becomes_a_delta():
    delta = F.extract_facet_delta(
        "something comforting tonight",
        extractor=FakeIntentExtractor({"moods": ["comfort"]}),
        vocabularies=VOCAB,
    )
    assert delta.moods == ("comfort",)
    assert not delta.is_empty


def test_a_stated_cuisine_becomes_a_delta():
    """There was no cuisine extractor at all — the only cuisines filter ever
    sent came from the stored profile."""
    delta = F.extract_facet_delta(
        "thai tonight",
        extractor=FakeIntentExtractor({"cuisines": ["thai"]}),
        vocabularies=VOCAB,
    )
    assert delta.cuisines == ("thai",)


def test_the_live_vocabulary_is_handed_to_the_extractor():
    fake = FakeIntentExtractor({})
    F.extract_facet_delta("anything", extractor=fake, vocabularies=VOCAB)
    assert fake.saw_vocab == VOCAB


def test_no_vocabulary_means_no_facets():
    """With no live list there is no value that is safe to send."""
    delta = F.extract_facet_delta(
        "comforting", extractor=FakeIntentExtractor({"moods": ["comfort"]}),
        vocabularies={},
    )
    assert delta.is_empty


def test_extraction_failure_changes_nothing():
    class Boom:
        def extract(self, m, v):
            raise RuntimeError("groq down")

    assert F.extract_facet_delta("x", extractor=Boom(), vocabularies=VOCAB).is_empty


# ── the guard that matters ───────────────────────────────────────────────

class TestNeverSendAnUnlistedValue:
    def test_the_agent_drops_a_value_not_in_the_vocabulary(self):
        """A hallucinated facet is not a narrower search — it is an empty one."""
        from agents import PlanIntentExtractor

        class Stub(PlanIntentExtractor):
            def __init__(self):
                pass  # no client needed; only post-validation is under test

        stub = Stub()
        stub.llm = type("L", (), {
            "invoke": lambda self, *a, **k: type("R", (), {
                "content": '{"moods": ["energising", "comfort"], "cuisines": [],'
                           ' "flavor_profiles": [], "food_groups": []}'
            })()
        })()
        got = stub.extract("energising please", VOCAB)
        assert got["moods"] == ["comfort"], "an unlisted mood must be dropped"

    def test_no_vocabulary_returns_all_empty(self):
        from agents import PlanIntentExtractor

        class Stub(PlanIntentExtractor):
            def __init__(self):
                pass

        got = Stub().extract("anything", {})
        assert got == {"cuisines": [], "moods": [], "flavor_profiles": [],
                       "food_groups": []}


# ── standing state ───────────────────────────────────────────────────────

class TestStandingState:
    def test_facets_survive_later_turns(self):
        state = PlanningState().merge(PlanningStateDelta(moods=("comfort",)))
        state = state.merge(PlanningStateDelta(pantry_add=("rice",)))
        assert state.facets() == {"moods": ["comfort"]}

    def test_facets_accumulate_across_families(self):
        state = PlanningState().merge(PlanningStateDelta(moods=("comfort",)))
        state = state.merge(PlanningStateDelta(cuisines=("thai",)))
        assert state.facets() == {"cuisines": ["thai"], "moods": ["comfort"]}

    def test_a_chip_can_be_removed(self):
        """The UI's removable chips need this; so does "actually not spicy"."""
        state = PlanningState().merge(
            PlanningStateDelta(moods=("comfort",), flavor_profiles=("spicy",))
        )
        state = state.merge(PlanningStateDelta(facets_remove=("spicy",)))
        assert state.facets() == {"moods": ["comfort"]}

    def test_removal_does_not_disturb_other_families(self):
        state = PlanningState().merge(
            PlanningStateDelta(cuisines=("thai",), moods=("comfort",))
        )
        state = state.merge(PlanningStateDelta(facets_remove=("thai",)))
        assert state.facets() == {"moods": ["comfort"]}

    def test_reset_clears_them(self):
        state = PlanningState().merge(PlanningStateDelta(moods=("comfort",)))
        assert state.merge(PlanningStateDelta(reset=True)).facets() == {}

    def test_they_round_trip_through_persistence(self):
        import json

        state = PlanningState().merge(
            PlanningStateDelta(moods=("hearty",), food_groups=("vegetables",))
        )
        back = PlanningState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert back.facets() == state.facets()

    def test_they_are_described_for_the_member(self):
        state = PlanningState().merge(PlanningStateDelta(moods=("comfort",)))
        assert "comfort" in state.describe()


# ── "energy boost" resolves to something searchable ──────────────────────

class TestGoalMapping:
    def test_energy_becomes_real_vocabulary(self):
        """There is no "energising" mood in the corpus. Read as sustaining food
        — protein and fibre — rather than invented as a facet."""
        assert F.facets_for_goal("energy") == {"moods": ["hearty"]}
        assert F.claim_tags_for("energy") == ["high_protein", "high_fibre"]

    def test_weight_loss_becomes_real_vocabulary(self):
        assert F.facets_for_goal("weight_loss") == {"moods": ["light"]}
        assert F.claim_tags_for("weight_loss") == ["low_calorie"]

    def test_an_unknown_goal_invents_nothing(self):
        assert F.facets_for_goal("turbo") == {}
        assert F.claim_tags_for("turbo") == []
        assert F.facets_for_goal(None) == {}

    def test_a_stated_claim_becomes_a_claim_tag(self):
        """low-carb has no diet tag and no corpus tag; low_calorie is the
        closest honest proxy, and that judgement is written down."""
        assert F.claim_tags_for(None, ["low-carb"]) == ["low_calorie"]
        assert F.claim_tags_for(None, ["high-protein"]) == ["high_protein"]

    def test_claim_tags_do_not_duplicate(self):
        assert F.claim_tags_for("energy", ["high-protein"]) == [
            "high_protein", "high_fibre"
        ]


# ── what actually reaches the request ────────────────────────────────────

class TestFetchArguments:
    def _cands(self):
        from services.candidates_client import CANDIDATES

        CANDIDATES.__class__._vocab_cache = dict(VOCAB)
        return CANDIDATES

    def test_facet_kwargs_always_carries_all_four_families(self):
        self._cands()
        got = F.facet_kwargs({"food_likes": []})
        assert {"cuisines", "moods", "flavor_profiles", "food_groups", "tags"} == set(got)

    def test_stated_facets_reach_the_request(self):
        self._cands()
        got = F.facet_kwargs({"_facets": {"moods": ["hearty"]}, "food_likes": []})
        assert got["moods"] == ["hearty"]

    def test_a_profile_word_lands_in_its_own_family(self):
        """A stored "comfort" used to be searched for as an INGREDIENT."""
        self._cands()
        got = F.facet_kwargs({"food_likes": ["greek", "comfort", "chickpeas"]})
        assert got["cuisines"] == ["greek"]
        assert got["moods"] == ["comfort"]

    def test_the_callers_cuisines_are_merged_not_replaced(self):
        self._cands()
        got = F.facet_kwargs({"_facets": {"cuisines": ["thai"]}, "food_likes": []},
                             ["greek"])
        assert set(got["cuisines"]) == {"thai", "greek"}

    def test_an_unrecognised_like_stays_an_ingredient(self):
        self._cands()
        split = self._cands().split_preferences(["chickpeas"])
        assert split["ingredients"] == ["chickpeas"]
        assert not any(split[f] for f in ("cuisines", "moods", "flavor_profiles",
                                          "food_groups"))

    def test_no_vocabulary_sends_no_facets(self):
        from services.candidates_client import CANDIDATES

        CANDIDATES.__class__._vocab_cache = {}
        try:
            got = F.facet_kwargs({"food_likes": ["greek", "comfort"]})
            assert all(v == [] for v in got.values())
        finally:
            CANDIDATES.__class__._vocab_cache = dict(VOCAB)


# ── claim tags: the fix for a dead end ───────────────────────────────────

class TestClaimTags:
    """Nutrition claims stopped being empty diet filters yesterday and were
    routed to `notes` — which turned out to be write-only, read only by
    `describe()`, which is only logged. So the claim was saved from breaking
    the plan and then dropped on the floor. They are RecipeWrangler claim tags,
    a different request field entirely."""

    def test_a_stated_claim_becomes_a_claim_tag_not_a_note(self):
        from services import diet_intent

        class Fake:
            def extract(self, q):
                return ["vegetarian", "high-protein"]

        delta = diet_intent.extract_diet_delta("veggie and high protein",
                                               extractor=Fake())
        assert delta.diet_tags == ("vegetarian",)
        assert delta.claim_tags == ("high_protein",)
        assert delta.notes == (), "notes is write-only; nothing may be routed there"

    def test_claim_tags_are_standing_state(self):
        state = PlanningState().merge(PlanningStateDelta(claim_tags=("high_protein",)))
        state = state.merge(PlanningStateDelta(pantry_add=("rice",)))
        assert state.claim_tags == ("high_protein",)

    def test_they_round_trip(self):
        import json

        state = PlanningState().merge(PlanningStateDelta(claim_tags=("high_fibre",)))
        back = PlanningState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert back.claim_tags == ("high_fibre",)

    def test_a_diet_retraction_clears_them_too(self):
        """They arrive on the same turn as the diet they accompany."""
        state = PlanningState().merge(
            PlanningStateDelta(diet_tags=("vegetarian",), claim_tags=("high_protein",))
        )
        cleared = state.merge(PlanningStateDelta(diet_clear=True))
        assert cleared.claim_tags == () and cleared.diet_tags == ()

    def test_the_goal_slider_contributes_tags_and_facets(self):
        from services.candidates_client import CANDIDATES

        CANDIDATES.__class__._vocab_cache = {**VOCAB, "tags": ["high_protein",
                                                              "high_fibre"]}
        got = F.facet_kwargs({"plan_parameters": {"goal": "energy"},
                              "food_likes": []})
        assert got["tags"] == ["high_protein", "high_fibre"]
        assert got["moods"] == ["hearty"]


class TestTheCapabilityGate:
    """RecipeWrangler's request model is extra="forbid", so sending `tags` to a
    deployment that predates the parameter is a 422, not a shrug. The published
    vocabulary is the capability flag."""

    def _capture(self):
        import services.plan_client as pc

        captured: dict = {}

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"days": []}

        class Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None, **k):
                captured.clear()
                captured.update(json or {})
                return Resp()

        pc.httpx.Client = Client
        return pc, captured

    def test_tags_are_sent_when_advertised(self):
        from services.candidates_client import CANDIDATES

        pc, captured = self._capture()
        CANDIDATES.__class__._vocab_cache = {"tags": ["high_protein"]}
        pc.PLANNER.plan_meals(days=1, tags=["high_protein"])
        assert captured.get("tags") == ["high_protein"]

    def test_tags_are_withheld_when_not_advertised(self):
        from services.candidates_client import CANDIDATES

        pc, captured = self._capture()
        CANDIDATES.__class__._vocab_cache = {"cuisines": ["greek"]}
        pc.PLANNER.plan_meals(days=1, tags=["high_protein"])
        assert "tags" not in captured, "would 422 against an older RecipeWrangler"

    def test_the_rest_of_the_request_is_unaffected(self):
        from services.candidates_client import CANDIDATES

        pc, captured = self._capture()
        CANDIDATES.__class__._vocab_cache = {"cuisines": ["greek"]}
        pc.PLANNER.plan_meals(days=1, tags=["high_protein"], cuisines=["greek"])
        assert captured.get("cuisines") == ["greek"]
