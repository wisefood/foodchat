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
def four_meals(session_service, sample_profile):
    """A four-meal day — the shape the reorder reports came from."""
    from models.plan_spec import PlanSpec
    from models.planning_state import PlanningStateDelta
    from models.session import DayPlan, Meal, MealCourse, MealPlan

    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    session_service.set_planning_state(
        session.session_id,
        session_service.get_planning_state(session.session_id).merge(
            PlanningStateDelta(spec=PlanSpec(
                meals=("breakfast", "lunch", "snack", "dinner"),
            )),
        ),
    )
    session_service.add_prepared_meal_plan(session.session_id, MealPlan.from_days([
        DayPlan(day=1, meals=[
            Meal(slot, [MealCourse(f"{slot}-r", f"{slot.title()} dish", "i", "d")])
            for slot in ("breakfast", "lunch", "snack", "dinner")
        ]),
    ], "four meals"))
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

    def test_half_a_request_is_asked_about_rather_than_guessed(self, orch,
                                                               session_service, four_meals):
        """The member's own words name one meal and a direction. Which meal
        MOVES is the part that matters, and guessing it is how somebody's
        breakfast gets moved when they meant their snack."""
        turn = _navigate(orch, session_service, four_meals,
                         "hmmm better before lunch, i will have lunch later today")

        assert isinstance(turn, ChatTurn)
        assert turn.meal_plan is None, "it changed the plan on half a request"
        assert "before the lunch" in turn.content
        assert "?" in turn.content

    def test_a_whole_request_is_carried_out(self, orch, session_service, four_meals):
        turn = _navigate(orch, session_service, four_meals, "put the snack before lunch")

        assert turn.meal_plan is not None
        order = [m.meal_type for m in turn.meal_plan.day_plans[0].meals]
        assert order.index("snack") < order.index("lunch")

    def test_it_rearranges_rather_than_regenerates(self, orch, session_service, four_meals):
        """The member likes the food and wants it at another time of day.
        Re-planning would answer a question they did not ask."""
        before = session_service.get_session(four_meals).get_current_daily_plan()
        titles_before = {
            p.title for m in before.day_plans[0].meals for p in m.plates
        }
        turn = _navigate(orch, session_service, four_meals, "put the snack before lunch")
        titles_after = {
            p.title for m in turn.meal_plan.day_plans[0].meals for p in m.plates
        }

        assert titles_after == titles_before, "it changed the food"
        assert turn.meal_plan.version > before.version, "it should be a new version"

    def test_the_arrangement_is_standing(self, orch, session_service, four_meals):
        """The NEXT plan keeps it too — otherwise the member re-orders their
        day after every single request."""
        _navigate(orch, session_service, four_meals, "put the snack before lunch")
        spec = session_service.get_planning_state(four_meals).spec

        assert spec.meals.index("snack") < spec.meals.index("lunch")

    def test_a_move_the_plan_cannot_make_says_what_it_has(self, orch, session_service,
                                                          four_meals):
        turn = _navigate(orch, session_service, four_meals, "put the brunch before lunch")

        assert turn.meal_plan is None
        assert "brunch" in turn.content

    def test_an_addition_that_mentions_order_still_plans(self, orch, session_service,
                                                         four_meals, monkeypatch):
        """"Add a salad before lunch" is an addition that happens to mention an
        order. Declining it would refuse a request we can serve."""
        monkeypatch.setattr(turn_intake, "added_shape", lambda: ["added snack"])

        assert _navigate(orch, session_service, four_meals,
                         "add a snack before lunch") is None


class TestAPastedPlanIsNotNavigation:
    """This bypass runs ahead of the scorer a paste is meant for.

    "Snack before lunch" is a line in somebody's own plan far more often than
    it is an instruction about ours — and without a guard, a member pasting
    their week to be scored would have had OUR plan rearranged instead, around
    a meal the regex picked out of their text.
    """

    PASTE = (
        "Here's my plan, score it:\n"
        "Breakfast: porridge\n"
        "Snack before lunch: apple\n"
        "Lunch: soup\n"
        "Dinner: salmon"
    )

    def test_the_reader_alone_cannot_tell(self):
        """Stated so the guard below is not mistaken for belt and braces."""
        assert nav.reorder_request(self.PASTE) is not None

    def test_the_turn_routes_on_to_the_scorer(self, orch, session_service, four_meals):
        assert _navigate(orch, session_service, four_meals, self.PASTE) is None

    def test_the_plan_is_untouched(self, orch, session_service, four_meals):
        before = session_service.get_session(four_meals).get_current_daily_plan()
        _navigate(orch, session_service, four_meals, self.PASTE)
        after = session_service.get_session(four_meals).get_current_daily_plan()

        assert after.id == before.id

    def test_a_real_request_still_gets_through(self, orch, session_service, four_meals):
        turn = _navigate(orch, session_service, four_meals, "put the snack before lunch")
        assert turn is not None and turn.meal_plan is not None


class TestVersionNumbersStayUnique:
    """Reported: "it began counting again from the beginning and now i have two v2s."

    `parent.version + 1` was right while a canvas was a straight line. It stopped
    being one the moment a member could go BACK: restoring v1 and editing made a
    second "version 2", and then "go back to v2" had no single answer.
    """

    def test_editing_after_a_restore_does_not_reuse_a_number(self, orch,
                                                             session_service, planned):
        _navigate(orch, session_service, planned, "go back to the first version")
        session_service.refine_meal_plan(
            planned, _cand("v4"), reasoning="after the restore", metrics={},
        )
        versions = [v for v, _id, _cur in session_service.plan_versions(planned)]

        assert versions == [1, 2, 3, 4]
        assert len(versions) == len(set(versions)), "two plans wear the same number"

    def test_the_branch_still_records_where_it_grew_from(self, orch, session_service,
                                                         planned):
        """Unique numbering is not a flattening — the parent link is what makes
        the lineage a tree, and going back is the whole reason it is one."""
        _navigate(orch, session_service, planned, "go back to the first version")
        restored = session_service.get_session(planned).get_current_daily_plan()
        session_service.refine_meal_plan(
            planned, _cand("v4"), reasoning="branch", metrics={},
        )
        newest = session_service.get_session(planned).get_current_daily_plan()

        assert newest.version == 4
        assert newest.parent_id == restored.id

    def test_a_named_version_still_resolves_after_branching(self, orch, session_service,
                                                            planned):
        _navigate(orch, session_service, planned, "go back to the first version")
        session_service.refine_meal_plan(planned, _cand("v4"), reasoning="b", metrics={})
        turn = _navigate(orch, session_service, planned, "to v3 now")

        assert turn.meal_plan is not None and turn.meal_plan.version == 3


class TestABareVersionIsARestore:
    """"To v5 now" is the follow-up to "go back to the first version" — exactly
    when a member stops repeating the verb. It reached the planner instead and
    produced a brand new v5, which is the opposite of the request."""

    @pytest.mark.parametrize("message,expected", [
        ("to v5 now", 5), ("v2", 2), ("version 3 please", 3), ("to version 1", 1),
    ])
    def test_it_is_read_as_one(self, message, expected):
        assert nav.restore_request(message) == expected

    @pytest.mark.parametrize("message", [
        "add v8 protein powder", "give me 5 dinners",
        "i want version 2 of the plan and also a snack",
    ])
    def test_a_version_inside_a_sentence_is_not(self, message):
        """Anchored at both ends: a bare number mid-sentence is far more often
        a quantity than a version."""
        assert nav.restore_request(message) is None

    def test_it_restores_rather_than_plans(self, orch, session_service, planned):
        turn = _navigate(orch, session_service, planned, "to v1 now")
        assert turn is not None and turn.meal_plan.version == 1
