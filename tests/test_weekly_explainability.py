"""M7 — weekly plan explainability (measured ledger, metrics, reasons).

All deterministic — no LLM, no network, per tests/conftest.py.
"""

import uuid

from models.session import MealCourse
from services.weekly_planner.explainability import (
    attach_match_reasons,
    build_weekly_explainability,
    calorie_budget_status,
    day_breakdown,
    guideline_checklist,
    nutrition_metrics,
    variety_metrics,
    weekly_constraints_ledger,
)
from services.weekly_planner.reward_logic import apply_hard_constraints
from services.weekly_planner.state_tracking import WeeklyNutritionalTracker


def _entry(day, meal_idx, meal_type, recipe):
    return {"day": day, "meal_idx": meal_idx, "meal_type": meal_type,
            "recipe": recipe, "reward": 0.0}


def _veg(rid="v1", kcal=None, **extra):
    recipe = {"recipe_id": rid, "recipe_title": "Lentil Bowl",
              "recipe_ingredients": "lentils, rice", "recipe_directions": "d",
              "tags": ["vegetarian"], **extra}
    if kcal is not None:
        recipe["nutrition"] = {"kcal": kcal}
    return recipe


def _meat(rid="m1", kcal=None, **extra):
    recipe = {"recipe_id": rid, "recipe_title": "Beef Stew",
              "recipe_ingredients": "beef, potato", "recipe_directions": "d",
              **extra}
    if kcal is not None:
        recipe["nutrition"] = {"kcal": kcal}
    return recipe


def _week(recipe_fn):
    """21 entries, distinct ids, one recipe archetype."""
    return [
        _entry(d, i, m, recipe_fn(rid=f"r-{d}-{i}"))
        for d in range(1, 8) for i, m in enumerate(["breakfast", "lunch", "dinner"])
    ]


class TestAttachMatchReasons:
    def test_pinned_favorite_like_and_adapted_chips(self):
        profile = {"favorite_recipe_ids": ["fav-1"], "food_likes": ["lentils"],
                   "memory_log": []}
        entries = [
            _entry(1, 0, "breakfast", _veg(rid="p-1", pinned=True)),
            _entry(1, 1, "lunch", _veg(rid="fav-1")),
            _entry(1, 2, "dinner", _meat(rid="a-1", adapted=True)),
        ]
        attach_match_reasons(entries, profile)

        pinned_kinds = [r["kind"] for r in entries[0]["recipe"]["match_reasons"]]
        assert "pinned" in pinned_kinds

        fav_kinds = [r["kind"] for r in entries[1]["recipe"]["match_reasons"]]
        assert "favorite" in fav_kinds
        assert "profile" in fav_kinds  # "you like lentils"

        adapted = entries[2]["recipe"]["match_reasons"]
        assert adapted == [{"kind": "adapted", "label": "Your adapted version"}]

    def test_adapted_overlay_display_keys_used_for_likes(self):
        # After the overlay, adapted ingredients live under "ingredients".
        profile = {"food_likes": ["tofu"], "memory_log": []}
        entries = [_entry(1, 0, "breakfast", {
            "recipe_id": "x", "recipe_title": "Beef Stew",
            "recipe_ingredients": "beef", "ingredients": "tofu, soy",
            "adapted": True,
        })]
        attach_match_reasons(entries, profile)
        labels = [r["label"] for r in entries[0]["recipe"]["match_reasons"]]
        assert "you like tofu" in labels


class TestVarietyMetrics:
    def test_distinct_ingredients_and_categories(self):
        entries = _week(_veg)
        entries[2]["recipe"] = _meat(rid="r-1-2")  # Monday dinner
        metrics = variety_metrics(entries)
        assert metrics["distinct_recipes"] == 21
        assert metrics["total_meals"] == 21
        assert metrics["category_distribution"] == {"vegetarian": 20, "red meat": 1}
        # lentils, rice, beef, potato
        assert metrics["unique_ingredients"] == 4
        assert "All 21 meals are distinct recipes" in metrics["reasoning"]


class TestGuidelineChecklist:
    def test_frequency_rules_from_category_counts(self):
        rows = guideline_checklist({"fish": 2, "red meat": 1, "vegetarian": 18}, 21)
        by_rule = {r["rule"]: r for r in rows}
        assert by_rule["eat fish 1–2 times a week"]["met"] is True
        assert by_rule["limit red meat"]["met"] is True
        assert by_rule["make most meals plant-based"]["met"] is True
        assert by_rule["make most meals plant-based"]["actual"] == 18

    def test_missing_fish_not_met(self):
        rows = guideline_checklist({"vegetarian": 21}, 21)
        fish = next(r for r in rows if "fish" in r["rule"])
        assert fish["actual"] == 0 and fish["met"] is False


class TestNutritionMetrics:
    def test_totals_targets_and_full_coverage(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=600.0))
        targets = WeeklyNutritionalTracker(
            {"preferences": ["2000 calories target"]}
        ).targets
        nutrition = nutrition_metrics(entries, targets)
        assert nutrition["weekly_totals"]["kcal"] == 12600.0
        assert nutrition["daily_average_kcal"] == 1800.0
        assert nutrition["weekly_targets"]["kcal"] == 14000.0
        assert nutrition["budget_used_pct"] == 90
        assert nutrition["coverage"] == {"meals_with_data": 21, "total_meals": 21}
        assert nutrition["note"] == ""

    def test_partial_coverage_gets_a_note(self):
        entries = _week(_veg)  # no nutrition anywhere
        entries[0]["recipe"]["nutrition"] = {"kcal": 500.0}
        nutrition = nutrition_metrics(entries, {"calories": 14000.0})
        assert nutrition["coverage"]["meals_with_data"] == 1
        assert "1 of 21 meals" in nutrition["note"]

    def test_no_data_no_percentage(self):
        nutrition = nutrition_metrics(_week(_veg), {"calories": 14000.0})
        assert nutrition["budget_used_pct"] is None
        assert nutrition["daily_average_kcal"] is None


class TestAReportedZeroIsMissingData:
    """RecipeWrangler returns ``kcal: 0`` for recipes it has no composition
    data for, and every caller read that as a measurement. On a real week two
    such recipes made the plan claim `meals_with_data: 21 of 21`, suppress the
    "based on N meals" note, and count the unknown calories as if the member
    would eat nothing."""

    def test_a_zero_does_not_count_as_coverage(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=600.0))
        entries[3]["recipe"]["nutrition"]["kcal"] = 0
        entries[7]["recipe"]["nutrition"]["kcal"] = 0

        nutrition = nutrition_metrics(entries, {"calories": 14000.0})

        assert nutrition["coverage"] == {"meals_with_data": 19, "total_meals": 21}
        assert "19 of 21 meals" in nutrition["note"]

    def test_a_week_of_zeroes_knows_it_measured_nothing(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=0.0))

        nutrition = nutrition_metrics(entries, {"calories": 14000.0})

        assert nutrition["coverage"]["meals_with_data"] == 0
        assert nutrition["budget_used_pct"] is None
        assert nutrition["daily_average_kcal"] is None

    def test_a_zero_is_left_out_of_the_day_qualifier(self):
        from services.weekly_planner.day_summary import recipe_kcal, summarize_day

        assert recipe_kcal({"nutrition": {"kcal": 0}}) is None
        assert recipe_kcal({"nutrition": {"kcal": 600.0}}) == 600.0

        day = [
            _entry(1, 0, "breakfast", _veg(rid="a", kcal=0.0)),
            _entry(1, 1, "lunch", _veg(rid="b", kcal=900.0)),
            _entry(1, 2, "dinner", _veg(rid="c", kcal=900.0)),
        ]
        # 900+900 over two known meals scaled to three is 2700 — a hearty day.
        # Counting the zero as known averages it down to 1800 and calls it
        # light, on the strength of a number nobody measured.
        assert summarize_day(day) == "hearty vegetarian day"


class TestTheCalorieBudgetHasTwoSides:
    """A ceiling on its own is not a budget check. A real week planned at 47%
    of target — 933 kcal a day — passed as satisfied, and `split_ledger` then
    handed it to the reply as an honoured request."""

    def test_a_week_well_under_target_is_not_satisfied(self):
        assert calorie_budget_status(6531.0, 14000.0, 21, 21) == "under"

    def test_a_week_on_target_is(self):
        assert calorie_budget_status(13450.0, 14000.0, 21, 21) == "on_track"

    def test_a_week_over_target_still_is_not(self):
        assert calorie_budget_status(16000.0, 14000.0, 21, 21) == "over"

    def test_the_floor_is_measured_against_the_meals_we_know_about(self):
        """The property that keeps this from becoming a coverage complaint:
        a week whose 11 known meals are on target is on target, even though
        its total is barely half the weekly figure."""
        on_target_for_11 = 14000.0 * 11 / 21

        assert calorie_budget_status(on_target_for_11, 14000.0, 11, 21) == "on_track"
        assert calorie_budget_status(on_target_for_11 * 0.5, 14000.0, 11, 21) == "under"

    def test_nothing_measured_says_nothing(self):
        assert calorie_budget_status(0.0, 14000.0, 0, 21) is None
        assert calorie_budget_status(6531.0, 0.0, 21, 21) is None

    def test_the_ledger_row_reports_the_shortfall(self):
        nutrition = {"weekly_totals": {"kcal": 6531.0},
                     "coverage": {"meals_with_data": 21, "total_meals": 21},
                     "budget_status": "under", "note": ""}
        ledger = weekly_constraints_ledger(
            {"preferences": []}, meat_count=0,
            targets={"calories": 14000.0, "meat_limit": 0},
            selection_events=[], downvoted_count=0, nutrition=nutrition,
        )
        row = next(r for r in ledger if "calorie" in r["constraint"])

        assert row["status"] == "violated"
        assert "short of your target" in row["detail"]

    def test_a_payload_with_no_status_is_not_read_as_a_violation(self):
        """Pre-M10 stored plans and hand-built dicts carry no `budget_status`.
        Deriving it is right; treating its absence as a failure would be the
        same mistake in the other direction."""
        nutrition = {"weekly_totals": {"kcal": 13450.0},
                     "coverage": {"meals_with_data": 21, "total_meals": 21},
                     "note": ""}
        ledger = weekly_constraints_ledger(
            {"preferences": []}, meat_count=0,
            targets={"calories": 14000.0, "meat_limit": 0},
            selection_events=[], downvoted_count=0, nutrition=nutrition,
        )
        row = next(r for r in ledger if "calorie" in r["constraint"])

        assert row["status"] == "satisfied"

    def test_the_justification_says_it_too(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=200.0))
        result = build_weekly_explainability(
            entries, {"preferences": ["2000 calories target"]},
        )

        assert "noticeably short of what you asked for" in result["reasoning"]

    def test_an_on_target_week_says_nothing_of_the_sort(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=600.0))
        result = build_weekly_explainability(
            entries, {"preferences": ["2000 calories target"]},
        )

        assert "short of what you asked for" not in result["reasoning"]


class TestWeeklyConstraintsLedger:
    def _targets(self, prefs):
        return WeeklyNutritionalTracker({"preferences": prefs}).targets

    def test_meat_within_limit_is_satisfied_with_measured_detail(self):
        ledger = weekly_constraints_ledger(
            {"preferences": []}, meat_count=2, targets=self._targets([]),
            selection_events=[], downvoted_count=0, nutrition={},
        )
        row = next(r for r in ledger if "meat" in r["constraint"])
        assert row["status"] == "satisfied"
        assert "2 of 3 meat meal(s) planned" in row["detail"]

    def test_relax_event_reported_honestly(self):
        events = [{"type": "meat_pool_pruned", "dropped": 4, "day": 4, "meal_type": "lunch"},
                  {"type": "meat_limit_relaxed", "day": 5, "meal_type": "dinner"}]
        ledger = weekly_constraints_ledger(
            {"preferences": ["meat limit 1"]}, meat_count=2,
            targets=self._targets(["meat limit 1"]),
            selection_events=events, downvoted_count=0, nutrition={},
        )
        row = next(r for r in ledger if "meat" in r["constraint"])
        assert row["status"] == "relaxed"
        assert "Friday dinner" in row["detail"]
        assert "Thursday lunch" in row["detail"]  # first prune slot

    def test_exceeding_without_relaxation_is_violated(self):
        # e.g. pinned dishes bypass constraints
        ledger = weekly_constraints_ledger(
            {"preferences": ["meat limit 1"]}, meat_count=3,
            targets=self._targets(["meat limit 1"]),
            selection_events=[], downvoted_count=0, nutrition={},
        )
        row = next(r for r in ledger if "meat" in r["constraint"])
        assert row["status"] == "violated"

    def test_calorie_budget_row_measured(self):
        nutrition = {"weekly_totals": {"kcal": 13450.0},
                     "coverage": {"meals_with_data": 21, "total_meals": 21},
                     "note": ""}
        ledger = weekly_constraints_ledger(
            {"preferences": []}, meat_count=0, targets={"calories": 14000.0, "meat_limit": 0},
            selection_events=[], downvoted_count=0, nutrition=nutrition,
        )
        row = next(r for r in ledger if "calorie" in r["constraint"])
        assert row["status"] == "satisfied" and row["type"] == "soft"
        assert "13,450 of 14,000 kcal planned (96%)" in row["detail"]

    def test_calorie_overshoot_is_violated(self):
        nutrition = {"weekly_totals": {"kcal": 16000.0},
                     "coverage": {"meals_with_data": 21, "total_meals": 21},
                     "note": ""}
        ledger = weekly_constraints_ledger(
            {"preferences": []}, meat_count=0, targets={"calories": 14000.0, "meat_limit": 0},
            selection_events=[], downvoted_count=0, nutrition=nutrition,
        )
        row = next(r for r in ledger if "calorie" in r["constraint"])
        assert row["status"] == "violated"

    def test_profile_rows_still_present(self):
        ledger = weekly_constraints_ledger(
            {"allergies": ["peanuts"], "preferences": []}, meat_count=0,
            targets={"calories": 0.0, "meat_limit": 3},
            selection_events=[], downvoted_count=2, nutrition={},
        )
        constraints = [r["constraint"] for r in ledger]
        assert "no peanuts" in constraints
        assert "excluding 2 recipe(s) you disliked" in constraints


class TestSelectionEventRecording:
    def test_prune_and_relax_events_recorded_at_decision_time(self):
        tracker = WeeklyNutritionalTracker({"preferences": ["meat limit 1"]})
        tracker.update_tracker(MealCourse("r", "Beef Stew", "beef", "d"))
        events = []

        mixed = [_meat("m"), _veg("v")]
        filtered = apply_hard_constraints(
            mixed, tracker, events=events, slot={"day": 2, "meal_type": "lunch"})
        assert [c["recipe_id"] for c in filtered] == ["v"]
        assert events == [{"type": "meat_pool_pruned", "dropped": 1,
                           "day": 2, "meal_type": "lunch"}]

        all_meat = [_meat("m2"), _meat("m3")]
        assert apply_hard_constraints(
            all_meat, tracker, events=events, slot={"day": 2, "meal_type": "dinner"},
        ) == all_meat
        assert events[-1] == {"type": "meat_limit_relaxed", "day": 2, "meal_type": "dinner"}

    def test_no_events_when_limit_not_hit(self):
        tracker = WeeklyNutritionalTracker({"preferences": []})
        events = []
        pool = [_meat("m"), _veg("v")]
        assert apply_hard_constraints(pool, tracker, events=events, slot={}) == pool
        assert events == []


class TestDayBreakdown:
    def test_per_day_rows_with_kcal_and_highlights(self):
        entries = [
            _entry(1, 0, "breakfast", _veg(rid="p", kcal=400.0, pinned=True)),
            _entry(1, 1, "lunch", _veg(rid="v2", kcal=500.0)),
            _entry(2, 0, "breakfast", _veg(rid="v3")),
        ]
        attach_match_reasons(entries, {"food_likes": [], "memory_log": []})
        days = day_breakdown(entries, {1: "vegetarian day"})
        assert days[0]["name"] == "Monday"
        assert days[0]["summary"] == "vegetarian day"
        assert days[0]["kcal"] == 900.0
        assert days[0]["meals_with_data"] == 2
        assert days[0]["highlights"] == ["Lentil Bowl: requested by you"]
        assert days[1]["kcal"] is None


class TestBuildWeeklyExplainability:
    def test_full_payload_shape_and_reasoning(self):
        entries = _week(lambda rid: _veg(rid=rid, kcal=600.0))
        entries[1]["recipe"] = _meat(rid="r-1-1", kcal=600.0)  # one meat meal
        profile = {"preferences": ["2000 calories target"], "diet": [],
                   "allergies": [], "food_likes": [], "memory_log": []}

        result = build_weekly_explainability(
            entries, profile, selection_events=[],
            day_summaries={1: "lunch with red meat"},
            downvoted_count=1, feedback_lines=2,
        )

        meat_row = next(r for r in result["constraints_applied"] if "meat" in r["constraint"])
        assert meat_row["status"] == "satisfied"
        assert result["personalization_summary"]["feedback_signals"] == 2

        metrics = result["metrics"]
        assert metrics["variety"]["category_distribution"] == {"vegetarian": 20, "red meat": 1}
        assert metrics["nutrition"]["budget_used_pct"] == 90
        assert len(metrics["days"]) == 7
        assert metrics["days"][0]["summary"] == "lunch with red meat"
        assert metrics["selection_events"] == []

        # Match reasons were attached in place
        assert all("match_reasons" in e["recipe"] for e in entries)

        reasoning = result["reasoning"]
        assert "All 21 meals this week are distinct recipes" in reasoning
        assert "Stayed within your weekly meat limit (1 of 3 meat meals)" in reasoning
        assert "12,600 of your 14,000 kcal weekly budget (90%)" in reasoning

    def test_pescatarian_fish_not_counted_as_meat(self):
        entries = _week(lambda rid: {
            "recipe_id": rid, "recipe_title": "Grilled Salmon",
            "recipe_ingredients": "salmon, lemon", "recipe_directions": "d",
        })
        result = build_weekly_explainability(
            entries, {"diet": ["pescatarian"], "preferences": []},
        )
        meat_row = next(r for r in result["constraints_applied"] if "meat" in r["constraint"])
        assert "0 of 3 meat meal(s) planned" in meat_row["detail"]


class TestServiceWiring:
    """Explainability populated, persisted, and exposed on the response."""

    def test_generated_plan_carries_explainability(self, session_service, monkeypatch):
        from test_day_summary import TestWeeklyServiceDaySummaries

        profile = {"diet": [], "allergies": [], "preferences": [],
                   "food_likes": [], "food_dislikes": []}
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)

        svc = TestWeeklyServiceDaySummaries()._service(session_service, monkeypatch)
        _, plan = svc.process_message(session.session_id, "plan my week")

        # Fake pools serve salmon lunch + beef dinner daily with no
        # alternative — the meat limit (3) must show as relaxed, honestly.
        meat_row = next(r for r in plan.constraints_applied if "meat" in r["constraint"])
        assert meat_row["status"] == "relaxed"
        assert plan.metrics["variety"]["distinct_recipes"] == 21
        assert plan.metrics["nutrition"]["coverage"]["meals_with_data"] == 21
        assert any(e["type"] == "meat_limit_relaxed"
                   for e in plan.metrics["selection_events"])
        assert plan.reasoning
        assert plan.personalization_summary is not None
        assert all("match_reasons" in e["recipe"] for e in plan.entries)

        # The chat reply facts carry the measured summary
        facts = svc.response_writer.facts
        assert facts["week_summary"] == plan.reasoning
        assert facts["constraints_honored"]

        # Survives a DB round-trip on a fresh service (replica restart)
        from services.session_service import SessionService
        restored = SessionService().get_session(session.session_id).get_current_weekly_plan()
        assert restored.constraints_applied == plan.constraints_applied
        assert restored.metrics == plan.metrics
        assert restored.reasoning == plan.reasoning

        # API response model exposes everything additively
        from routers.foodchat_router import WeeklyMealPlanResponse
        response = WeeklyMealPlanResponse.from_weekly_meal_plan(restored)
        assert response.constraints_applied == plan.constraints_applied
        assert response.metrics == plan.metrics
        assert response.reasoning == plan.reasoning
        assert "match_reasons" in response.entries[0].recipe

    def test_pre_m7_plans_deserialize_with_empty_explainability(self, session_service):
        profile = {"diet": [], "allergies": [], "preferences": []}
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)
        entries = [
            _entry(d, i, m, {"recipe_id": f"w-{d}-{i}", "recipe_title": "Meal",
                             "recipe_ingredients": "x", "recipe_directions": "y"})
            for d in range(1, 8) for i, m in enumerate(["breakfast", "lunch", "dinner"])
        ]
        session_service.add_weekly_meal_plan(session.session_id, entries)

        from services.session_service import SessionService
        plan = SessionService().get_session(session.session_id).get_current_weekly_plan()
        assert plan.constraints_applied == []
        assert plan.personalization_summary is None
        assert plan.metrics == {}
        assert plan.reasoning == ""

        from routers.foodchat_router import WeeklyMealPlanResponse
        response = WeeklyMealPlanResponse.from_weekly_meal_plan(plan)
        assert response.constraints_applied == [] and response.metrics == {}
