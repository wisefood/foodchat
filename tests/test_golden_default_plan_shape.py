"""Golden-default regression fence for DYNAMIC_MEALS_PLAN.md.

Pins the EXACT daily/weekly plan shape produced with no dynamic options —
DB payload keys, model round-trip, and wire response — so the plates-as-list
model rewrite (Phase 1) cannot silently change the default. When a phase
intentionally changes the default (it must not), these are the alarms that
should force the conversation.

No LLM calls.
"""

from conftest import make_candidates
from models.session import MealCourse, MealPlan, WeeklyMealPlan
from routers.foodchat_router import MealPlanResponse, WeeklyMealPlanResponse
from services.session_service import (
    _serialize_meal_plan,
    _deserialize_meal_plan,
    _serialize_weekly_plan,
    _deserialize_weekly_plan,
)

# The frozen contract: exactly these keys in a default daily payload.
DEFAULT_DAILY_KEYS = {
    "id", "created_at", "version", "parent_id",
    "reasoning", "llm_score", "llm_reasoning", "fvs_count", "fvs_reasoning",
    "diversity_llm_score", "diversity_llm_reasoning",
    "guideline_adherence_score", "guideline_adherence_reasoning",
    "constraints_applied", "personalization_summary",
    "breakfast", "lunch", "dinner",
}
COURSE_KEYS = {"recipe_id", "title", "ingredients", "directions",
               "nutrition", "image_url", "match_reasons"}


def _default_plan() -> MealPlan:
    return MealPlan.from_courses(make_candidates("g"), "a fine day", version=1)


class TestDailyGolden:
    def test_from_courses_still_requires_exactly_three(self):
        import pytest
        with pytest.raises(ValueError):
            MealPlan.from_courses(make_candidates("g")[:2], "too few")

    def test_default_payload_keys_are_frozen(self):
        payload = _serialize_meal_plan(_default_plan())
        assert set(payload) == DEFAULT_DAILY_KEYS
        for slot in ("breakfast", "lunch", "dinner"):
            assert set(payload[slot]) == COURSE_KEYS

    def test_round_trip_is_lossless(self):
        plan = _default_plan()
        restored = _deserialize_meal_plan(plan.id, _serialize_meal_plan(plan))
        for slot in ("breakfast", "lunch", "dinner"):
            a, b = getattr(plan, slot), getattr(restored, slot)
            assert (a.recipe_id, a.title, a.ingredients, a.directions) == \
                   (b.recipe_id, b.title, b.ingredients, b.directions)
        assert restored.version == 1 and restored.parent_id is None

    def test_wire_response_exposes_three_scalar_courses(self):
        resp = MealPlanResponse.from_meal_plan(_default_plan())
        assert resp.breakfast.recipe_id and resp.lunch.recipe_id and resp.dinner.recipe_id
        # No 'days' field leaking before Phase 1 adds it additively
        assert not hasattr(resp, "days") or resp.__dict__.get("days") is None


class TestPlatesAsListAdditive:
    """Phase 1: the new day/plate view exists WITHOUT changing the default."""

    def test_scalar_plan_synthesizes_a_single_day_view(self):
        plan = _default_plan()
        assert plan.days is None and not plan.is_multiplate
        view = plan.day_plans
        assert len(view) == 1 and view[0].day == 1
        assert [m.meal_type for m in view[0].meals] == ["breakfast", "lunch", "dinner"]
        assert all(len(m.plates) == 1 for m in view[0].meals)
        assert view[0].meals[0].main is plan.breakfast

    def test_from_days_builds_multiplate_and_mirrors_scalars(self):
        from models.session import DayPlan, Meal, MealCourse
        pasta = MealCourse("r-pasta", "Pasta", "i", "d")
        salad = MealCourse("r-salad", "Salad", "i", "d", role="side")
        day = DayPlan(day=1, meals=[
            Meal("breakfast", [MealCourse("r-b", "Oats", "i", "d")]),
            Meal("lunch", [pasta, salad]),
            Meal("dinner", [MealCourse("r-d", "Stew", "i", "d")]),
        ])
        plan = MealPlan.from_days([day], "multi-plate day")
        assert plan.is_multiplate
        # Scalars mirror day-1 first main plate for legacy/back-compat readers
        assert plan.lunch.recipe_id == "r-pasta"
        # The lunch meal carries both plates
        assert [p.title for p in plan.day_plans[0].meals[1].plates] == ["Pasta", "Salad"]

    def test_multiplate_round_trip_through_db_serialization(self):
        from models.session import DayPlan, Meal, MealCourse
        day = DayPlan(day=1, meals=[
            Meal("breakfast", [MealCourse("r-b", "Oats", "i", "d")]),
            Meal("lunch", [MealCourse("r-pasta", "Pasta", "i", "d"),
                           MealCourse("r-salad", "Salad", "i", "d", role="side")]),
            Meal("dinner", [MealCourse("r-d", "Stew", "i", "d")]),
        ])
        plan = MealPlan.from_days([day], "multi-plate")
        payload = _serialize_meal_plan(plan)
        assert "days" in payload                       # only present for multi-plate
        assert payload["days"][0]["meals"][1]["plates"][1]["role"] == "side"

        restored = _deserialize_meal_plan(plan.id, payload)
        assert restored.is_multiplate
        lunch_plates = restored.day_plans[0].meals[1].plates
        assert [p.title for p in lunch_plates] == ["Pasta", "Salad"]
        assert lunch_plates[1].role == "side"

    def test_wire_response_carries_days_only_for_multiplate(self):
        from models.session import DayPlan, Meal, MealCourse
        # Default plan → days is None
        assert MealPlanResponse.from_meal_plan(_default_plan()).days is None
        # Multi-plate plan → days populated with the plates
        day = DayPlan(day=1, meals=[
            Meal("breakfast", [MealCourse("r-b", "Oats", "i", "d")]),
            Meal("lunch", [MealCourse("r-p", "Pasta", "i", "d"),
                           MealCourse("r-s", "Salad", "i", "d", role="side")]),
            Meal("dinner", [MealCourse("r-d", "Stew", "i", "d")]),
        ])
        resp = MealPlanResponse.from_meal_plan(MealPlan.from_days([day], "x"))
        assert resp.days is not None
        assert [p.role for p in resp.days[0].meals[1].plates] == ["main", "side"]
        # Legacy scalar fields still populated (rollout back-compat)
        assert resp.lunch.recipe_id == "r-p"


class TestWeeklyGolden:
    def _weekly(self) -> WeeklyMealPlan:
        entries = [
            {"day": d, "meal_idx": i, "meal_type": mt,
             "recipe": {"recipe_id": f"r{d}{i}", "title": f"{mt} {d}"}, "reward": 0.0}
            for d in range(1, 8)
            for i, mt in enumerate(("breakfast", "lunch", "dinner"))
        ]
        return WeeklyMealPlan(id="wk", created_at=_default_plan().created_at, entries=entries)

    def test_weekly_has_21_flat_entries(self):
        wp = self._weekly()
        assert len(wp.entries) == 21
        assert {e["meal_idx"] for e in wp.entries} == {0, 1, 2}
        assert {e["day"] for e in wp.entries} == set(range(1, 8))

    def test_weekly_round_trip_preserves_entries(self):
        wp = self._weekly()
        restored = _deserialize_weekly_plan(wp.id, _serialize_weekly_plan(wp))
        assert restored.entries == wp.entries

    def test_weekly_wire_response_entries_flat(self):
        resp = WeeklyMealPlanResponse.from_weekly_meal_plan(self._weekly())
        assert len(resp.entries) == 21
        assert resp.entries[0].meal_type == "breakfast"
