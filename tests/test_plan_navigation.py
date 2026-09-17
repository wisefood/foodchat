"""Going back to a version, and being told when we cannot reorder.

Both of these arrived as ordinary sentences and reached the edit path, which
can only replace the dish on a slot:

    > go back to the first version
    < [a brand new plan, v5, one meal]

    > hmmm better before lunch, i will have lunch later today
    < Done — I swapped the breakfast: "Tofu Scramble" → "Sunday Lunch" · 82 → 1907 kcal

The first is now a pointer move on a lineage that has always existed. The
second cannot be served — eating order is derived from the meal's NAME, in two
independent sorters — so the turn says that instead of changing food nobody
asked about.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe                          # noqa: E402
from services import plan_navigation as nav                        # noqa: E402
from services import turn_intake                                   # noqa: E402
from services.orchestrator_service import ChatTurn, OrchestratorService  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_turn():
    turn_intake.forget()
    yield
    turn_intake.forget()


def _cand(tag):
    return [CandidateRecipe(f"{tag}{i}", f"{tag.upper()} dish {i}", "i", "d")
            for i in range(3)]


@pytest.fixture
def planned(session_service, sample_profile):
    """A session with three versions on the daily canvas."""
    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    session_service.add_meal_plan(session.session_id, _cand("v1"), "first", {})
    for tag in ("v2", "v3"):
        session_service.refine_meal_plan(
            session.session_id, _cand(tag), reasoning=tag, metrics={},
        )
    return session.session_id


@pytest.fixture
def orch(session_service):
    svc = OrchestratorService.__new__(OrchestratorService)
    svc.session_service = session_service
    return svc


def _navigate(orch, session_service, session_id, message):
    session = session_service.get_session(session_id)
    return orch._maybe_navigate(session, session_id, message)


# ── reading the request ──────────────────────────────────────────────────

class TestReadingTheRequest:
    @pytest.mark.parametrize("message,expected", [
        ("go back to the first version", 1),
        ("go back to the original", 1),
        ("restore v2", 2),
        ("back to version 3", 3),
        ("revert to the second one", 2),
    ])
    def test_a_named_version(self, message, expected):
        assert nav.restore_request(message) == expected

    @pytest.mark.parametrize("message", [
        "undo", "undo that", "revert", "go back to the previous one",
    ])
    def test_one_step_back_is_not_a_named_version(self, message):
        """"Undo" means the step just taken; "the first version" names one.
        Guessing either way hands somebody the wrong plan."""
        assert nav.restore_request(message) == "previous"

    @pytest.mark.parametrize("message", [
        "something lighter for lunch", "add a snack", "plan my week",
        "i want to go back to eating fish",
    ])
    def test_it_leaves_ordinary_messages_alone(self, message):
        assert nav.restore_request(message) is None


# ── restoring ────────────────────────────────────────────────────────────

class TestRestoring:
    def test_it_moves_the_pointer_instead_of_planning(self, orch, session_service, planned):
        turn = _navigate(orch, session_service, planned, "go back to the first version")

        assert turn is not None
        assert turn.meal_plan is not None
        assert turn.meal_plan.version == 1
        assert "version 1" in turn.content

    def test_the_canvas_actually_moves(self, orch, session_service, planned):
        _navigate(orch, session_service, planned, "go back to the first version")
        current = session_service.get_session(planned).get_current_daily_plan()

        assert current.version == 1

    def test_nothing_is_thrown_away(self, orch, session_service, planned):
        """Restoring v1 must not lose v2 and v3 — a member who changes their
        mind again would otherwise find the versions gone."""
        _navigate(orch, session_service, planned, "go back to the first version")

        assert [v for v, _i, _c in session_service.plan_versions(planned)] == [1, 2, 3]
        again = _navigate(orch, session_service, planned, "restore v3")
        assert again.meal_plan.version == 3

    def test_undo_steps_back_one(self, orch, session_service, planned):
        turn = _navigate(orch, session_service, planned, "undo")
        assert turn.meal_plan.version == 2

    def test_a_version_that_does_not_exist_says_what_does(self, orch, session_service,
                                                          planned):
        turn = _navigate(orch, session_service, planned, "go back to version 9")

        assert turn.meal_plan is None
        assert "1, 2, 3" in turn.content

    def test_already_there_says_so(self, orch, session_service, planned):
        turn = _navigate(orch, session_service, planned, "restore v3")
        assert turn.meal_plan is None or turn.meal_plan.version == 3

    def test_with_no_plan_at_all_it_routes_on(self, orch, session_service, sample_profile):
        """"Go back" with nothing on the canvas is not a restore — let the
        classifier have it rather than answering about versions that do not
        exist."""
        session = session_service.create_session(f"m-{uuid.uuid4()}", sample_profile)
        assert _navigate(orch, session_service, session.session_id, "undo") is None


# ── the order we cannot give ─────────────────────────────────────────────

class TestReorder:
    @pytest.mark.parametrize("message", [
        "hmmm better before lunch, i will have lunch later today",
        "move the snack",
        "put the snack before lunch",
        "change the order of the meals",
    ])
    def test_it_is_recognised(self, message):
        assert nav.asks_to_reorder(message) is True

    @pytest.mark.parametrize("message", [
        "something lighter for lunch",
        "add a snack",
        "swap the chicken for fish",
    ])
    def test_ordinary_edits_are_not_reorders(self, message):
        assert nav.asks_to_reorder(message) is False

    def test_it_declines_instead_of_swapping_a_dish(self, orch, session_service, planned):
        turn = _navigate(orch, session_service, planned,
                         "hmmm better before lunch, i will have lunch later today")

        assert isinstance(turn, ChatTurn)
        assert turn.meal_plan is None, "it changed the plan anyway"
        assert "can't move meals around" in turn.content
        assert "add one, take one out" in turn.content, "it should say what it CAN do"

    def test_the_plan_is_left_exactly_as_it_was(self, orch, session_service, planned):
        before = session_service.get_session(planned).get_current_daily_plan()
        _navigate(orch, session_service, planned, "move the snack before lunch")
        after = session_service.get_session(planned).get_current_daily_plan()

        assert after.id == before.id and after.version == before.version

    def test_an_addition_that_mentions_order_still_plans(self, orch, session_service,
                                                         planned, monkeypatch):
        """"Add a salad before lunch" is an addition that happens to mention an
        order. Declining it would refuse a request we can serve."""
        monkeypatch.setattr(turn_intake, "added_shape", lambda: ["added snack"])

        assert _navigate(orch, session_service, planned,
                         "add a snack before lunch") is None
