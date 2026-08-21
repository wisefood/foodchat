"""
The standing planning state, over HTTP.

Everything the member has said that outlives a turn lives in `PlanningState`,
and it was reachable only by saying it again. So:

* the pantry was invisible — a member who said "I have zucchini" had no way to
  see whether it was heard, or to correct it if it wasn't;
* a facet inferred from a sentence ("something light") could not be taken back
  except by arguing with the assistant;
* a reload showed a plan whose constraints had no explanation on screen.

These are the endpoints the pantry panel and the removable facet chips call.
Ownership is the load-bearing part: every one of them is session-scoped, and a
member who does not own the session must get a 404 — not a 403, which would
confirm the session exists.
"""

from __future__ import annotations

import sys
import uuid

import pytest
from fastapi import HTTPException

sys.path.insert(0, "src")

import services                                                      # noqa: E402
from models.planning_state import PlanningState, PlanningStateDelta   # noqa: E402
from routers import foodchat_router as api                            # noqa: E402


@pytest.fixture
def owned():
    """A session with a standing state worth reading back."""
    member = f"member-{uuid.uuid4()}"
    session = services.session_service.create_session(member, {"diet": ["vegetarian"]})
    services.session_service.set_planning_state(
        session.session_id,
        PlanningState().merge(PlanningStateDelta(
            pantry_add=("zucchini", "spinach"),
            moods=("light",),
            cuisines=("thai",),
            diet_tags=("vegetarian",),
            claim_tags=("high_protein",),
        )),
    )
    return session.session_id, member


def _state(session_id):
    return services.session_service.get_planning_state(session_id)


# ── reading ──────────────────────────────────────────────────────────────

class TestRead:
    def test_it_returns_what_is_standing(self, owned):
        session_id, member = owned
        out = api.get_planning_state(session_id, member_id=member)
        assert out.pantry == ["zucchini", "spinach"]
        assert out.facets["moods"] == ["light"]
        assert out.facets["cuisines"] == ["thai"]
        assert out.diet_tags == ["vegetarian"]
        assert out.claim_tags == ["high_protein"]

    def test_every_facet_family_is_present_even_when_empty(self, owned):
        """A client that has to check whether a key exists is a client that
        will forget to."""
        session_id, member = owned
        out = api.get_planning_state(session_id, member_id=member)
        assert set(out.facets) == {"cuisines", "moods", "flavor_profiles", "food_groups"}

    def test_the_favourites_answer_keeps_its_three_states(self, owned):
        session_id, member = owned
        assert api.get_planning_state(session_id, member_id=member).use_favorites is None
        services.session_service.set_planning_state(
            session_id, _state(session_id).merge(PlanningStateDelta(use_favorites=False)),
        )
        assert api.get_planning_state(session_id, member_id=member).use_favorites is False

    def test_it_says_whether_the_plan_shape_is_the_default(self, owned):
        session_id, member = owned
        out = api.get_planning_state(session_id, member_id=member)
        assert out.plan_shape_is_default is True
        assert isinstance(out.plan_shape, dict)

    def test_it_carries_the_query_a_regeneration_would_run(self, owned):
        session_id, member = owned
        out = api.get_planning_state(session_id, member_id=member)
        assert "vegetarian" in out.query and "zucchini" in out.query

    def test_an_untouched_session_reads_empty_not_missing(self):
        member = f"member-{uuid.uuid4()}"
        session = services.session_service.create_session(member, {})
        out = api.get_planning_state(session.session_id, member_id=member)
        assert out.pantry == [] and out.diet_tags == []
        assert out.facets == {"cuisines": [], "moods": [], "flavor_profiles": [],
                              "food_groups": []}


# ── the pantry panel ─────────────────────────────────────────────────────

class TestPantryWrites:
    def test_a_replace_can_empty_the_pantry(self, owned):
        """The panel's save sends the whole list. An additive-only write could
        never express "I cleared the last item"."""
        session_id, member = owned
        out = api.set_pantry(session_id, api.PantryRequest(member_id=member, items=[]))
        assert out.pantry == []
        assert _state(session_id).pantry == ()

    def test_a_replace_adds_and_removes_in_one_call(self, owned):
        session_id, member = owned
        out = api.set_pantry(
            session_id,
            api.PantryRequest(member_id=member, items=["spinach", "Feta", "olives"]),
        )
        assert out.pantry == ["spinach", "feta", "olives"], "zucchini should be gone"

    def test_items_are_normalised_the_same_way_chat_normalises_them(self, owned):
        session_id, member = owned
        out = api.set_pantry(
            session_id,
            api.PantryRequest(member_id=member, items=["  Zucchini ", "TOMATOES", ""]),
        )
        assert out.pantry == ["zucchini", "tomatoes"]

    def test_an_add_leaves_the_rest_alone(self, owned):
        session_id, member = owned
        out = api.add_pantry_items(
            session_id, api.PantryRequest(member_id=member, items=["feta"]),
        )
        assert out.pantry == ["zucchini", "spinach", "feta"]

    def test_an_add_of_nothing_usable_is_a_400_not_a_silent_success(self, owned):
        session_id, member = owned
        with pytest.raises(HTTPException) as e:
            api.add_pantry_items(
                session_id, api.PantryRequest(member_id=member, items=["", "  "]),
            )
        assert e.value.status_code == 400

    def test_one_item_can_be_taken_out(self, owned):
        session_id, member = owned
        out = api.remove_pantry_item(session_id, "Zucchini", member_id=member)
        assert out.pantry == ["spinach"]

    def test_removing_something_absent_is_not_an_error(self, owned):
        """Two clients racing on the same tick-off must not produce a failure."""
        session_id, member = owned
        out = api.remove_pantry_item(session_id, "quinoa", member_id=member)
        assert out.pantry == ["zucchini", "spinach"]

    def test_a_write_returns_the_new_state_not_an_acknowledgement(self, owned):
        """A client that has to re-fetch to see what its own write did is a
        client that will render a stale chip."""
        session_id, member = owned
        out = api.add_pantry_items(
            session_id, api.PantryRequest(member_id=member, items=["feta"]),
        )
        assert isinstance(out, api.PlanningStateResponse)
        assert "feta" in out.pantry


# ── the removable facet chips ────────────────────────────────────────────

class TestFacetRemoval:
    def test_a_mood_can_be_taken_back(self, owned):
        session_id, member = owned
        out = api.remove_facet(session_id, "light", member_id=member)
        assert out.facets["moods"] == []
        assert out.facets["cuisines"] == ["thai"], "unrelated facets must survive"

    def test_the_client_does_not_have_to_know_which_family_it_was(self, owned):
        """A member removing "light" does not know whether it was read as a
        mood or a flavour, and requiring the client to know would make the
        chip's own rendering the source of truth for what it deletes."""
        session_id, member = owned
        out = api.remove_facet(session_id, "thai", member_id=member)
        assert out.facets["cuisines"] == []

    def test_it_is_case_insensitive(self, owned):
        session_id, member = owned
        out = api.remove_facet(session_id, "LIGHT", member_id=member)
        assert out.facets["moods"] == []

    def test_an_empty_value_is_a_400(self, owned):
        session_id, member = owned
        with pytest.raises(HTTPException) as e:
            api.remove_facet(session_id, "   ", member_id=member)
        assert e.value.status_code == 400

    def test_the_removal_survives_the_next_read(self, owned):
        session_id, member = owned
        api.remove_facet(session_id, "light", member_id=member)
        assert api.get_planning_state(session_id, member_id=member).facets["moods"] == []

    def test_a_removed_facet_does_not_come_back_when_restated_by_the_same_delta(self, owned):
        """`facets_remove` wins within one merge — otherwise a re-extraction in
        the same turn would resurrect what the member just deleted."""
        session_id, member = owned
        api.remove_facet(session_id, "light", member_id=member)
        state = _state(session_id).merge(
            PlanningStateDelta(moods=("light",), facets_remove=("light",))
        )
        assert state.moods == ()


# ── ownership: the reason these endpoints are session-scoped ─────────────

class TestOwnership:
    def _calls(self, session_id, member):
        """Every new endpoint, with a member that must be checked."""
        return [
            ("get_planning_state",
             lambda: api.get_planning_state(session_id, member_id=member)),
            ("set_pantry",
             lambda: api.set_pantry(
                 session_id, api.PantryRequest(member_id=member, items=["x"]))),
            ("add_pantry_items",
             lambda: api.add_pantry_items(
                 session_id, api.PantryRequest(member_id=member, items=["x"]))),
            ("remove_pantry_item",
             lambda: api.remove_pantry_item(session_id, "x", member_id=member)),
            ("remove_facet",
             lambda: api.remove_facet(session_id, "light", member_id=member)),
        ]

    def test_another_member_gets_a_404_from_every_endpoint(self, owned):
        session_id, _ = owned
        intruder = f"member-{uuid.uuid4()}"
        for name, call in self._calls(session_id, intruder):
            with pytest.raises(HTTPException) as e:
                call()
            assert e.value.status_code == 404, name

    def test_a_404_never_confirms_the_session_exists(self, owned):
        """403 would leak that the id is real. The message must be the same as
        for a session that does not exist at all."""
        session_id, _ = owned
        intruder = f"member-{uuid.uuid4()}"
        with pytest.raises(HTTPException) as real:
            api.get_planning_state(session_id, member_id=intruder)
        with pytest.raises(HTTPException) as fake:
            api.get_planning_state("no-such-session", member_id=intruder)
        assert real.value.status_code == fake.value.status_code == 404
        assert real.value.detail == fake.value.detail

    def test_an_intruder_changes_nothing(self, owned):
        session_id, _ = owned
        intruder = f"member-{uuid.uuid4()}"
        with pytest.raises(HTTPException):
            api.set_pantry(
                session_id, api.PantryRequest(member_id=intruder, items=[]),
            )
        assert _state(session_id).pantry == ("zucchini", "spinach")


# ── the query a regeneration runs ────────────────────────────────────────

class TestRegenerationQuery:
    def test_it_describes_what_is_wanted_not_what_was_removed(self):
        """Describing the edit ("without the spicy flavour") would search for
        the thing being taken away."""
        state = PlanningState().merge(PlanningStateDelta(
            diet_tags=("vegetarian",), claim_tags=("high_protein",),
            moods=("light",), pantry_add=("zucchini",),
        ))
        query = state.as_query()
        assert "vegetarian" in query and "high protein" in query and "light" in query
        assert "zucchini" in query
        assert "without" not in query and "remove" not in query

    def test_an_empty_state_still_asks_for_a_plan(self):
        assert PlanningState().as_query() == "a meal plan"

    def test_slugs_become_words(self):
        state = PlanningState().merge(
            PlanningStateDelta(claim_tags=("30_minutes_or_less",))
        )
        assert "30 minutes or less" in state.as_query()

    def test_it_does_not_repeat_a_value_stated_two_ways(self):
        state = PlanningState().merge(PlanningStateDelta(
            moods=("light",), flavor_profiles=("light",),
        ))
        assert state.as_query().count("light") == 1


class TestReplanNeedsTheOrchestrator:
    def test_it_is_a_503_when_the_orchestrator_never_started(self, owned):
        """The guard exists because a 500 here would look like a planning
        failure rather than a service that is not up."""
        session_id, member = owned
        saved = services.orchestrator_service
        services.orchestrator_service = None
        try:
            with pytest.raises(HTTPException) as e:
                api.replan(session_id, api.RegenerateRequest(member_id=member))
            assert e.value.status_code == 503
        finally:
            services.orchestrator_service = saved
