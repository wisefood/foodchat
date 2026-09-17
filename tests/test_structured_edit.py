"""
A slot swap on a multi-day plan must not destroy the plan.

`_edit_daily` rebuilt the plan from three scalar courses — `refine_meal_plan`
→ `MealPlan.from_courses` → `days=None`. On a plan `plan_structured` had built,
that meant:

    a 4-day plan came back as 1 day
    every side, dessert and drink was gone
    the swap could hand back a recipe already on another day
    "change day 3's dinner" edited day 1

All four are the same root cause: three scalars are a *compatibility
projection* of day 1's mains, not the plan. A plan that has `days` is now
patched in place instead — deep-copied, one `MealCourse` replaced.

No LLM: the replacement search is the seam, and it is stubbed.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe, RecipeEnrichment          # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan       # noqa: E402
from services.edit_service import DirectivePredicate, EditService, _named_day  # noqa: E402
from services.session_service import SessionService                  # noqa: E402


def _plate(rid, title, role="main", kcal=600):
    return MealCourse(
        recipe_id=rid, title=title, ingredients=f"{title} ingredients",
        directions="cook", role=role,
        nutrition={"calories": kcal}, image_url=f"http://img/{rid}",
        match_reasons=[{"kind": "diet", "label": "vegetarian"}],
    )


def _four_days() -> MealPlan:
    """Four days; day 2's dinner has a side, day 3 has a dessert."""
    days = [
        DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("d1b", "Porridge")]),
            Meal("lunch", [_plate("d1l", "Soup")]),
            Meal("dinner", [_plate("d1d", "Risotto")]),
        ]),
        DayPlan(day=2, meals=[
            Meal("breakfast", [_plate("d2b", "Toast")]),
            Meal("lunch", [_plate("d2l", "Salad bowl")]),
            Meal("dinner", [_plate("d2d", "Moussaka"),
                            _plate("d2s", "Greek salad", role="side")]),
        ]),
        DayPlan(day=3, meals=[
            Meal("breakfast", [_plate("d3b", "Yoghurt")]),
            Meal("dinner", [_plate("d3d", "Curry"),
                            _plate("d3x", "Baklava", role="dessert")]),
        ]),
        DayPlan(day=4, meals=[
            Meal("dinner", [_plate("d4d", "Pasta")]),
        ]),
    ]
    plan = MealPlan.from_days(days, "four days", {"llm_score": 7})
    plan.constraints_applied = [{"constraint": "vegetarian", "status": "satisfied"}]
    plan.personalization_summary = {"applied": 3}
    return plan


class _Extractor:
    """Stands in for the LLM command extractor."""

    def __init__(self, command=None):
        self.command = command
        self.calls = []

    def extract(self, message, plan_type):
        self.calls.append((message, plan_type))
        return dict(self.command) if self.command else None


def _service(svc, replacement=("new1", "Lentil bake", 400)):
    """An EditService whose replacement search is deterministic."""
    es = EditService(svc, extractor=_Extractor())
    rid, title, kcal = replacement
    choice = CandidateRecipe(
        recipe_id=rid, title=title, ingredients=f"{title} ingredients",
        directions="cook",
    )
    rich = RecipeEnrichment(recipe_id=rid, title=title, kcal=kcal, protein_g=30,
                            image_url=f"http://img/{rid}")
    es._asked = []

    def _find(session, meal_type, predicate, old_recipe_id, exclude_ids):
        es._asked.append({"meal_type": meal_type, "old": old_recipe_id,
                          "exclude": list(exclude_ids)})
        return choice, None, rich, {}

    es._find_replacement = _find
    return es


@pytest.fixture
def planned(session_service, sample_profile):
    session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
    session_service.add_prepared_meal_plan(session.session_id, _four_days())
    return session.session_id


def _plan_of(svc, session_id):
    return svc.get_session(session_id).get_current_daily_plan()


# ── the plan survives ────────────────────────────────────────────────────

class TestTheDaysSurvive:
    def test_all_four_days_are_still_there(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        assert out.meal_plan is not None
        assert len(out.meal_plan.days) == 4, "the other three days were rebuilt away"
        assert [d.day for d in out.meal_plan.days] == [1, 2, 3, 4]

    def test_the_side_plate_of_the_edited_meal_survives(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        dinner = out.meal_plan.days[1].meals[2]
        assert [p.role for p in dinner.plates] == ["main", "side"]
        assert dinner.plates[1].title == "Greek salad"

    def test_plates_on_other_days_are_untouched(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        day3 = out.meal_plan.days[2]
        assert [p.title for p in day3.meals[1].plates] == ["Curry", "Baklava"]
        # and their enrichment came through, not a blank rebuild
        assert day3.meals[1].plates[0].nutrition == {"calories": 600}
        assert day3.meals[1].plates[0].match_reasons

    def test_only_the_target_plate_changed(self, session_service, planned):
        es = _service(session_service)
        before = {
            (d.day, m.meal_type, p.role): p.recipe_id
            for d in _plan_of(session_service, planned).day_plans
            for m in d.meals for p in m.plates
        }
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        after = {
            (d.day, m.meal_type, p.role): p.recipe_id
            for d in out.meal_plan.day_plans for m in d.meals for p in m.plates
        }
        assert set(before) == set(after), "the plan's shape changed"
        differing = {k for k in before if before[k] != after[k]}
        assert differing == {(2, "dinner", "main")}

    def test_the_replaced_plate_keeps_its_role(self, session_service, planned):
        """A side swapped for a side, not promoted to the main course."""
        es = _service(session_service)
        session = es._get_session(planned)
        plan = session.get_current_daily_plan()
        # target the dessert directly through the coordinates helper path
        plan.days[2].meals[1].plates[0].role = "side"   # make the first plate a side
        out = es._edit_daily(
            session, "dinner", DirectivePredicate("different"), "swap it", day=3,
        )
        # No main on that meal any more → Meal.main falls back to plates[0],
        # and the swap must not silently relabel it.
        assert out.meal_plan.days[2].meals[1].plates[0].role == "side"

    def test_the_new_plate_carries_its_nutrition_and_chip(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        new = out.meal_plan.days[1].meals[2].plates[0]
        assert new.title == "Lentil bake"
        assert new.nutrition and new.nutrition.get("kcal") == 400
        assert any(r["kind"] == "pinned" for r in new.match_reasons)

    def test_the_ledger_and_summary_carry_forward(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        assert out.meal_plan.constraints_applied
        assert out.meal_plan.personalization_summary == {"applied": 3}

    def test_the_scores_are_carried_not_reset(self, session_service, planned):
        """A slot swap doesn't re-grade the plan, so the score must not drop to 0."""
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        assert out.meal_plan.llm_score == 7


class TestLineageAndPersistence:
    def test_it_is_version_two_of_the_same_canvas(self, session_service, planned):
        es = _service(session_service)
        original = _plan_of(session_service, planned)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        assert out.meal_plan.version == 2
        assert out.meal_plan.parent_id == original.id

    def test_the_stored_plan_is_the_edited_one(self, session_service, planned):
        es = _service(session_service)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        stored = _plan_of(session_service, planned)
        assert stored.days is not None and len(stored.days) == 4
        assert stored.days[1].meals[2].plates[0].title == "Lentil bake"

    def test_the_ledger_reached_the_database_not_just_the_response(self, session_service, planned):
        """It is attached before the store, so a reload still has it."""
        es = _service(session_service)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        fresh = SessionService().get_session(planned).get_current_daily_plan()
        assert fresh.constraints_applied, "the ledger was lost on reload"
        assert len(fresh.days) == 4

    def test_the_previous_version_is_not_rewritten(self, session_service, planned):
        """The parent is a live object in the session — patching in place would
        rewrite the history the member can scroll back to."""
        es = _service(session_service)
        original = _plan_of(session_service, planned)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        assert original.days[1].meals[2].plates[0].title == "Moussaka"


class TestExclusions:
    def test_every_recipe_in_the_plan_is_excluded_not_just_three(self, session_service, planned):
        """Three scalars are day 1's mains; a swap on day 2 could otherwise
        return day 4's dinner."""
        es = _service(session_service)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=2,
        )
        excluded = set(es._asked[0]["exclude"])
        assert {"d1b", "d1l", "d1d", "d2b", "d2l", "d2d", "d2s",
                "d3b", "d3d", "d3x", "d4d"} <= excluded

    def test_the_old_plate_of_the_target_slot_is_the_one_being_replaced(
        self, session_service, planned
    ):
        es = _service(session_service)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=3,
        )
        assert es._asked[0]["old"] == "d3d"


class TestTargetingMisses:
    def test_a_day_the_plan_does_not_have_says_what_it_covers(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=7,
        )
        assert out.meal_plan is None
        assert "day 1" in out.text and "day 4" in out.text

    def test_a_meal_the_day_does_not_have_names_the_ones_it_does(self, session_service, planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "lunch",
            DirectivePredicate("lighter"), "x", day=4,
        )
        assert out.meal_plan is None
        assert "dinner" in out.text

    def test_nothing_is_stored_on_a_miss(self, session_service, planned):
        es = _service(session_service)
        es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "x", day=7,
        )
        assert _plan_of(session_service, planned).version == 1

    def test_a_single_day_structured_plan_needs_no_day(self, session_service, sample_profile):
        """One day of multi-plate meals: the day is unambiguous."""
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        plan = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("breakfast", [_plate("b", "Porridge")]),
            Meal("dinner", [_plate("d", "Ragu"), _plate("s", "Slaw", role="side")]),
        ])], "one day")
        session_service.add_prepared_meal_plan(session.session_id, plan)
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(session.session_id), "dinner",
            DirectivePredicate("lighter"), "make dinner lighter", day=None,
        )
        assert out.meal_plan is not None
        assert len(out.meal_plan.days[0].meals[1].plates) == 2


class TestLegacyPlansAreUnaffected:
    def test_a_three_course_plan_still_goes_the_old_way(self, session_service, sample_profile):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        courses = [
            CandidateRecipe(recipe_id=r, title=r.upper(), ingredients="i", directions="d")
            for r in ("b", "l", "d")
        ]
        session_service.add_meal_plan(session.session_id, courses, "legacy", {})
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(session.session_id), "lunch",
            DirectivePredicate("lighter"), "lighter lunch",
        )
        assert out.meal_plan is not None
        assert out.meal_plan.days is None, "a legacy plan must not grow days"
        assert out.meal_plan.lunch.title == "Lentil bake"
        assert out.meal_plan.breakfast.title == "B"


# ── the day the member named ─────────────────────────────────────────────

class TestNamedDay:
    @pytest.mark.parametrize("message,expected", [
        ("swap day 3's dinner", 3),
        ("change the dinner on day 1", 1),
        ("make the second day lighter", 2),
        ("the fourth day's lunch please", 4),
        ("DAY 5 breakfast", 5),
    ])
    def test_a_day_stated_in_words_is_read(self, message, expected):
        assert _named_day(message, weekdays=False) == expected

    @pytest.mark.parametrize("message", [
        "make the dinner lighter", "something else please", "", "day 9",
    ])
    def test_no_day_stated_is_none(self, message):
        assert _named_day(message, weekdays=False) is None

    def test_weekday_names_resolve_for_a_weekly_plan(self):
        assert _named_day("swap Thursday's dinner", weekdays=True) == 4
        assert _named_day("monday lunch", weekdays=True) == 1

    def test_weekday_names_do_not_resolve_for_a_multi_day_daily_plan(self):
        """Such a plan has no calendar anchoring — its day 1 is "the first
        day", not Monday. Mapping Thursday onto index 4 would be a guess
        presented as a fact."""
        assert _named_day("swap Thursday's dinner", weekdays=False) is None

    def test_an_explicit_day_number_wins_over_a_weekday(self):
        assert _named_day("day 2, the Thursday one", weekdays=True) == 2


class TestClarificationGate:
    """A multi-day plan on the daily canvas must ask for the day, exactly as
    weekly does — editing day 1 silently is the bug."""

    def test_it_asks_when_the_day_is_missing(self, session_service, planned):
        es = _service(session_service)
        es.extractor.command = {"meal_type": "dinner", "day": None,
                                "directive": "lighter"}
        out = es.process(planned, "make the dinner lighter")
        assert out.needs_clarification
        assert "day" in out.text.lower()

    def test_it_does_not_ask_when_the_words_name_the_day(self, session_service, planned):
        es = _service(session_service)
        es.extractor.command = {"meal_type": "dinner", "day": None,
                                "directive": "lighter"}
        out = es.process(planned, "make day 3's dinner lighter")
        assert not out.needs_clarification
        assert out.meal_plan is not None
        assert out.meal_plan.days[2].meals[1].plates[0].title == "Lentil bake"

    def test_a_single_day_plan_is_never_asked_for_a_day(self, session_service, sample_profile):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        courses = [
            CandidateRecipe(recipe_id=r, title=r, ingredients="i", directions="d")
            for r in ("b", "l", "d")
        ]
        session_service.add_meal_plan(session.session_id, courses, "legacy", {})
        es = _service(session_service)
        es.extractor.command = {"meal_type": "lunch", "day": None,
                                "directive": "lighter"}
        out = es.process(session.session_id, "make the lunch lighter")
        assert not out.needs_clarification

    def test_the_clarification_state_records_the_rule_it_applied(
        self, session_service, planned
    ):
        es = _service(session_service)
        es.extractor.command = {"meal_type": None, "day": None, "directive": "lighter"}
        es.process(planned, "change something")
        state = session_service.get_session(planned).clarification
        assert state["needs_day"] is True

    def test_the_second_pass_honours_the_recorded_rule(self, session_service, planned):
        es = _service(session_service)
        es.extractor.command = {"meal_type": None, "day": None, "directive": "lighter"}
        es.process(planned, "change something")
        # The reply names both the slot and the day.
        es.extractor.command = {"meal_type": "dinner", "day": None,
                                "directive": "lighter"}
        out = es.continue_clarification(planned, "day 2's dinner")
        assert not out.unresolved
        assert out.meal_plan is not None
        assert len(out.meal_plan.days) == 4

    def test_a_reply_with_no_day_stays_unresolved(self, session_service, planned):
        es = _service(session_service)
        es.extractor.command = {"meal_type": None, "day": None, "directive": "lighter"}
        es.process(planned, "change something")
        es.extractor.command = {"meal_type": "dinner", "day": None,
                                "directive": "lighter"}
        out = es.continue_clarification(planned, "the dinner")
        assert out.unresolved


# ── the reply names the day the plan actually has ────────────────────────
#
# A multi-day plan on the daily canvas carries no calendar: its day 1 is the
# first day the member cooks, not Monday. `_named_day(weekdays=False)` has
# always refused to READ weekdays here; the reply was still WRITING them, so a
# swap on day 3 was reported as "Wednesday's dinner" on a plan with no
# Wednesday in it — and the member has no way to check which day was changed.

class TestTheReplyNamesTheRightDay:
    def test_a_swap_on_a_multi_day_daily_plan_says_day_n(self, session_service,
                                                         planned):
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 2 dinner lighter", day=2,
        )
        assert "day 2's dinner" in out.text
        assert "Tuesday" not in out.text

    def test_a_failure_on_a_multi_day_daily_plan_still_names_the_day(
        self, session_service, planned
    ):
        """It named no day at all, so "I looked for a replacement for the
        dinner" was the answer to a question about one day of four."""
        es = _service(session_service)
        es._find_replacement = lambda *a, **k: (None, None, None, {})
        out = es._edit_daily(
            es._get_session(planned), "dinner",
            DirectivePredicate("lighter"), "day 3 dinner lighter", day=3,
        )
        assert "day 3's dinner" in out.text
        assert "Wednesday" not in out.text

    def test_a_single_day_plan_names_no_day(self, session_service, sample_profile):
        """Nothing to disambiguate, so the sentence stays as it was."""
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_prepared_meal_plan(
            session.session_id,
            MealPlan.from_days([DayPlan(day=1, meals=[
                Meal("dinner", [MealCourse(
                    recipe_id="only", title="Stew", ingredients="x",
                    directions="cook", nutrition={"calories": 700},
                )]),
            ])], "one day"),
        )
        es = _service(session_service)
        out = es._edit_daily(
            es._get_session(session.session_id), "dinner",
            DirectivePredicate("lighter"), "dinner lighter", day=1,
        )
        assert "the dinner" in out.text
        assert "day 1" not in out.text and "Monday" not in out.text

    def test_the_weekly_canvas_still_says_the_weekday(self):
        """A weekly plan IS calendar-anchored — day 3 there really is Wednesday."""
        phrase = EditService._slot_phrase("dinner", 3, weekdays=True)
        assert phrase == "Wednesday's dinner"

    def test_the_phrase_helper_covers_both_and_neither(self):
        assert EditService._slot_phrase("lunch", None, weekdays=True) == "the lunch"
        assert EditService._slot_phrase("lunch", 2, weekdays=False) == "day 2's lunch"
        # Beyond a week there is no weekday to name even on a weekly plan.
        assert EditService._slot_phrase("lunch", 9, weekdays=True) == "day 9's lunch"
