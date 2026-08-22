"""
What the product can do, and what the agent can reach.

The gap was not in the API. Saving a plan has had an endpoint since saved plans
were built, and "save this" was small talk — the capability existed, the
conversation could not get to it. That is the same defect as a missing feature
from where the member is sitting, and it is invisible from either side on its
own: the route table looks complete, and the tool registry looks complete.

These tests are written from the member's sentence inward. Each one names
something a person says out loud and asserts it reaches the thing that does it.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

import tools                                                        # noqa: E402
from services.orchestrator_service import OrchestratorService       # noqa: E402


def _entry(day, idx, slot, rid, title, ingredients):
    return {
        "day": day, "meal_idx": idx, "meal_type": slot,
        "recipe": {
            "recipe_id": rid, "recipe_title": title,
            "recipe_ingredients": ingredients, "recipe_directions": "cook",
            "nutrition": {"calories": 500, "protein_g": 20},
        },
    }


@pytest.fixture
def weekly(session_service, sample_profile):
    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    entries = []
    for day in range(1, 4):
        entries.append(_entry(day, 0, "breakfast", f"b{day}", f"Porridge {day}",
                              "oats, milk, honey"))
        entries.append(_entry(day, 1, "lunch", f"l{day}", f"Soup {day}",
                              "lentils, carrot, onion"))
        entries.append(_entry(day, 2, "dinner", f"d{day}", f"Curry {day}",
                              "chickpeas, onion, rice"))
    session_service.add_weekly_meal_plan(session.session_id, entries, day_summaries={})
    return session_service.get_session(session.session_id)


@pytest.fixture
def daily(session_service, sample_profile):
    from conftest import make_candidates

    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    session_service.add_meal_plan(
        session.session_id, make_candidates("sl"), "r", {},
    )
    return session_service.get_session(session.session_id)


# ── "save this" ──────────────────────────────────────────────────────────

class TestSavingThePlanOnScreen:
    def test_it_saves_the_active_canvas(self, session_service, weekly):
        result = tools.invoke("save_plan", {"session_id": weekly.session_id})
        assert result["saved"] is True
        assert result["plan_type"] == "weekly"
        saved = session_service.get_member_saved_plans(weekly.member_id)
        assert [row["plan_id"] for row in saved] == [result["plan_id"]]

    def test_a_name_the_member_gave_it_is_kept(self, session_service, weekly):
        tools.invoke("save_plan", {
            "session_id": weekly.session_id, "title": "Meatless Monday",
        })
        saved = session_service.get_member_saved_plans(weekly.member_id)
        assert saved[0]["saved_title"] == "Meatless Monday"

    def test_it_can_be_taken_back_off_the_list(self, session_service, weekly):
        tools.invoke("save_plan", {"session_id": weekly.session_id})
        result = tools.invoke("save_plan", {
            "session_id": weekly.session_id, "saved": "false",
        })
        assert result["saved"] is False
        assert session_service.get_member_saved_plans(weekly.member_id) == []

    def test_with_no_plan_it_says_so_rather_than_saving_nothing(self, session_service,
                                                               sample_profile):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        with pytest.raises(tools.ToolError) as exc:
            tools.invoke("save_plan", {"session_id": session.session_id})
        assert "no plan" in str(exc.value).lower()

    def test_it_does_not_touch_the_plan(self, session_service, weekly):
        """Saving is about whether a plan outlives the conversation, not about
        its contents — so the canvas must not be reloaded as if it changed."""
        assert tools.get("save_plan").mutates is False
        before = session_service.get_session(weekly.session_id).get_current_weekly_plan()
        tools.invoke("save_plan", {"session_id": weekly.session_id})
        after = session_service.get_session(weekly.session_id).get_current_weekly_plan()
        assert [e["recipe"]["recipe_id"] for e in after.entries] == \
               [e["recipe"]["recipe_id"] for e in before.entries]


# ── "what do I need to buy?" ─────────────────────────────────────────────

class TestTheShoppingList:
    def test_it_gathers_every_dish_on_the_week(self, weekly):
        result = tools.invoke("shopping_list", {"session_id": weekly.session_id})
        assert result["dishes"] == 9
        items = {row["item"] for row in result["items"]}
        assert {"oats", "milk", "honey", "lentils", "carrot", "onion",
                "chickpeas", "rice"} <= items

    def test_a_repeated_item_is_one_line_naming_its_meals(self, weekly):
        """Onion is in six dishes. A week's plan should be one list, not 21."""
        result = tools.invoke("shopping_list", {"session_id": weekly.session_id})
        onion = next(r for r in result["items"] if r["item"] == "onion")
        assert len(onion["for"]) == 6
        assert "Monday lunch" in onion["for"]

    def test_the_busiest_items_come_first(self, weekly):
        result = tools.invoke("shopping_list", {"session_id": weekly.session_id})
        counts = [len(row["for"]) for row in result["items"]]
        assert counts == sorted(counts, reverse=True)

    def test_it_says_it_has_no_quantities(self, weekly):
        """The corpus stores ingredients as free text. "6 tbsp" would be a
        number with nothing behind it, so the absence is stated rather than
        left for a reply to imply."""
        result = tools.invoke("shopping_list", {"session_id": weekly.session_id})
        assert "not available" in result["quantities"]
        assert not any("quantity" in row for row in result["items"])

    def test_it_uses_the_same_normalizer_as_the_variety_score(self, weekly):
        """Two answers to "what is in this plan" is one too many."""
        import inspect

        from services import plan_quality

        source = inspect.getsource(sys.modules["tools.plan_tools"])
        assert "extract_ingredient_names" in source
        assert callable(plan_quality.extract_ingredient_names)

    def test_it_reads_the_daily_canvas_too(self, daily):
        result = tools.invoke("shopping_list", {"session_id": daily.session_id})
        assert result["plan_type"] == "daily" and result["items"]

    def test_it_is_llm_free(self):
        assert tools.get("shopping_list").uses_model is False
        assert tools.get("shopping_list").mutates is False


# ── the selector can actually pass what these need ───────────────────────

class _Selector:
    def __init__(self, choice):
        self.choice = choice

    def choose(self, message, **_k):
        return dict(self.choice)


def _orch(session_service, choice):
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    OrchestratorService._tool_selector = _Selector(choice)

    class _Analyst:
        def answer(self, *_a, **_k):
            return ""            # force the canned sentence

    orch.plan_analyst = _Analyst()
    return orch


class TestTheSelectorCanReachThem:
    def test_a_title_is_threaded_through(self, session_service, weekly):
        orch = _orch(session_service, {
            "tool": "save_plan", "title": "Meatless Monday",
        })
        turn = orch._maybe_use_tool(weekly, weekly.session_id,
                                    "save this as Meatless Monday")
        assert turn is not None and "Meatless Monday" in turn.content
        saved = session_service.get_member_saved_plans(weekly.member_id)
        assert saved[0]["saved_title"] == "Meatless Monday"

    def test_unsaving_is_reachable(self, session_service, weekly):
        tools.invoke("save_plan", {"session_id": weekly.session_id})
        orch = _orch(session_service, {"tool": "save_plan", "saved": False})
        turn = orch._maybe_use_tool(weekly, weekly.session_id,
                                    "actually don't keep that one")
        assert turn is not None
        assert session_service.get_member_saved_plans(weekly.member_id) == []

    def test_an_argument_the_tool_does_not_take_is_dropped(self, session_service, weekly):
        """One schema answers for every tool, so the selector can name a title
        for a tool that has none. The registry would reject the key as a
        member-facing error — a complaint about a field nobody mentioned."""
        orch = _orch(session_service, {
            "tool": "summarize_week", "title": "not a thing here",
        })
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "how's my week?")
        assert turn is not None
        assert "does not take" not in turn.content

    def test_the_shopping_list_reply_never_implies_amounts(self, session_service, weekly):
        orch = _orch(session_service, {"tool": "shopping_list"})
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "what do I need to buy?")
        assert turn is not None
        assert "No quantities" in turn.content


# ── "actually not spicy" ─────────────────────────────────────────────────

class TestTakingAFacetBackInWords:
    """The UI has always been able to: a chip has an × and it calls
    `DELETE /facets/{value}`. Chat could not — the extractor answers with what
    a member ASKED for, and nothing read a retraction. So the affordance
    existed on one channel and a member who said it out loud watched the chip
    stay put.

    Deliberately deterministic and deliberately narrow: it can only remove a
    value that is ALREADY standing, so the failure mode is missing a retraction
    rather than inventing one.
    """

    @staticmethod
    def _standing():
        from models.planning_state import PlanningState, PlanningStateDelta

        return PlanningState().merge(PlanningStateDelta(
            cuisines=("thai",), flavor_profiles=("spicy",), moods=("comfort_food",),
        ))

    @pytest.mark.parametrize("text,gone", [
        ("actually not spicy", "spicy"),
        ("no more thai", "thai"),
        ("drop the comfort food", "comfort_food"),
        ("not too spicy", "spicy"),
        ("without the thai food", "thai"),
        ("forget that comfort-food thing", "comfort_food"),
        ("less spicy please", "spicy"),
    ])
    def test_a_retraction_is_read(self, text, gone):
        from services import intent_facets

        removed = intent_facets.extract_facet_removals(text, self._standing())
        assert removed.facets_remove == (gone,)

    @pytest.mark.parametrize("text", [
        "make it thai",                  # a request, not a retraction
        "no chicken",                    # never a facet of this session
        "spicy is great",
        "",
    ])
    def test_it_removes_nothing_it_was_not_told_to(self, text):
        from services import intent_facets

        assert intent_facets.extract_facet_removals(text, self._standing()).is_empty

    def test_a_session_with_no_facets_asks_nothing(self):
        from models.planning_state import PlanningState
        from services import intent_facets

        assert intent_facets.extract_facet_removals(
            "not spicy", PlanningState(),
        ).is_empty

    def test_the_chip_actually_goes(self, session_service, sample_profile, monkeypatch):
        """Through intake, which is what a real turn runs."""
        from models.planning_state import PlanningStateDelta
        from services import turn_intake

        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(cuisines=("thai",), flavor_profiles=("spicy",)),
            ),
        )
        turn_intake.forget()

        state = turn_intake.intake(
            session.session_id, "actually not spicy",
            session_service=session_service,
        )
        assert state.flavor_profiles == ()
        assert state.cuisines == ("thai",), "it should take back only what was named"

    def test_it_repairs_an_extractor_that_read_a_negation_as_a_request(self,
                                                                      session_service,
                                                                      sample_profile,
                                                                      monkeypatch):
        """Removal runs last and against the merged state, so a facet extractor
        that answers "spicy" to "not spicy" cannot ship the opposite of what was
        said."""
        from models.planning_state import PlanningStateDelta
        from services import turn_intake

        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())
        monkeypatch.setattr(
            intent_facets, "extract_facet_delta",
            lambda *a, **k: PlanningStateDelta(flavor_profiles=("spicy",)),
        )

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        state = turn_intake.intake(
            session.session_id, "not spicy", session_service=session_service,
        )
        assert state.flavor_profiles == ()
