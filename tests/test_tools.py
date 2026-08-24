"""
The local tool surface.

The agent was a fixed chain: one classification picked one handler, and
anything that handler could not do was unreachable. "Summarise my week" and
"redo Thursday" had no path at all — the closest available action was a full
refinement, which regenerates all 21 slots and silently discards a slot edit
the member had already approved.

These tests cover the registry contract and the two things that carry real
risk: that `replace_day` leaves the other days byte-identical, and that the
totals are honest about the meals they could not see.

LLM-free. `replace_day` is exercised against a fake action space so the
pinning behaviour is tested without touching RecipeWrangler.
"""

from __future__ import annotations

import pytest

import tools


# ── registry contract ────────────────────────────────────────────────────

class TestRegistry:
    def test_the_manifest_is_generated_not_written(self):
        """Every registered tool is listed, so the manifest cannot drift."""
        names = {t["name"] for t in tools.manifest()}
        assert {"summarize_week", "summarize_day", "replace_day",
                "plan_totals", "swap_meal"} <= names

    def test_every_tool_declares_a_usable_schema(self):
        for t in tools.all_tools():
            params = t.parameters
            assert params.get("type") == "object", t.name
            assert params.get("properties"), t.name
            # A model has to know which arguments it cannot omit.
            for req in params.get("required", []):
                assert req in params["properties"], f"{t.name}: {req}"

    def test_every_tool_says_whether_it_changes_anything(self):
        """A caller deciding whether it can retry needs this to be explicit."""
        readers = {"summarize_week", "summarize_day", "plan_totals"}
        for t in tools.all_tools():
            if t.name in readers:
                assert t.mutates is False, t.name
            assert isinstance(t.uses_model, bool)

    def test_an_unknown_tool_names_the_alternatives(self):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("reticulate_splines", {})
        assert "summarize_week" in str(e.value)

    def test_a_missing_required_argument_is_caught_before_the_handler(self):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("summarize_day", {"session_id": "s1"})
        assert "day" in str(e.value)

    def test_an_unknown_argument_is_rejected_with_the_real_ones(self):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("summarize_week", {"session_id": "s1", "dya": 3})
        assert "dya" in str(e.value) and "session_id" in str(e.value)

    def test_out_of_range_days_are_caught_here_not_in_the_planner(self):
        for bad in (0, 8, 99):
            with pytest.raises(tools.ToolError):
                tools.invoke("summarize_day", {"session_id": "s1", "day": bad})

    def test_http_strings_are_coerced_to_integers(self):
        """An HTTP caller sends "3"; the planner needs 3."""
        t = tools.get("summarize_day")
        from tools import _validate

        assert _validate(t, {"session_id": "s1", "day": "3"})["day"] == 3

    def test_a_non_numeric_day_says_so_plainly(self):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("summarize_day", {"session_id": "s1", "day": "Thursday"})
        assert "whole number" in str(e.value)

    def test_describe_tools_is_prose_a_prompt_can_carry(self):
        text = tools.describe_tools()
        assert "replace_day" in text
        assert "changes the plan" in text  # mutation is visible to the model


# ── totals ───────────────────────────────────────────────────────────────

class TestTotals:
    def _sum(self, pairs):
        from tools.plan_tools import _sum_nutrition

        return _sum_nutrition(pairs)

    def test_it_adds_up(self):
        got = self._sum([
            ("a", {"calories": 400, "protein_g": 20}),
            ("b", {"calories": 650, "protein_g": 35}),
        ])
        assert got["calories"] == 1050.0
        assert got["protein_g"] == 55.0
        assert got["complete"] is True

    def test_it_reports_what_it_could_not_see(self):
        """A bare total would understate the plan and imply completeness."""
        got = self._sum([
            ("a", {"calories": 400}),
            ("b", {}),
            ("c", None),
        ])
        assert got["calories"] == 400.0
        assert (got["meals_counted"], got["meals_total"]) == (1, 3)
        assert got["complete"] is False

    def test_kcal_is_accepted_as_an_alias_for_calories(self):
        # The corpus uses both spellings depending on the enrichment path.
        assert self._sum([("a", {"kcal": 500})])["calories"] == 500.0

    def test_an_empty_plan_is_not_complete(self):
        got = self._sum([])
        assert got["calories"] == 0.0 and got["complete"] is False


# ── replace_day ──────────────────────────────────────────────────────────

def _entry(day: int, idx: int, meal: str, rid: str, title: str, kcal: int = 500):
    return {
        "day": day, "meal_idx": idx, "meal_type": meal,
        "recipe": {
            "recipe_id": rid, "recipe_title": title,
            "recipe_ingredients": f"{title} ingredients",
            "recipe_directions": "cook it",
            "nutrition": {"calories": kcal, "protein_g": 20},
        },
    }


def _week():
    """A full 21-slot week, one recipe id per slot."""
    out = []
    for day in range(1, 8):
        for idx, meal in enumerate(["breakfast", "lunch", "dinner"]):
            out.append(_entry(day, idx, meal, f"r{day}{idx}", f"Dish {day}{idx}"))
    return out


class TestReplaceDayPinning:
    """The whole value of the tool is that it does NOT touch the other days."""

    def test_it_pins_every_slot_except_the_target_day(self):
        entries = _week()
        day = 4
        kept = [e for e in entries if e["day"] != day]
        pinned = {
            (int(e["day"]), int(e["meal_idx"])): dict(e["recipe"])
            for e in kept
        }
        # 21 slots, minus the 3 being replaced.
        assert len(pinned) == 18
        assert (4, 0) not in pinned and (4, 1) not in pinned and (4, 2) not in pinned
        assert (3, 0) in pinned and (5, 2) in pinned

    def test_a_pinned_slot_bypasses_selection_entirely(self):
        """This is the mechanism the tool relies on. If the planner ever stops
        honouring `pinned`, replace_day silently becomes a full regeneration
        and starts eating approved edits — so assert it directly."""
        import inspect

        from services.weekly_planner import planner

        src = inspect.getsource(planner.WeeklyPlanner.generate_full_plan)
        assert "slot_key in pinned" in src
        assert "bypass candidate selection" in src.lower()

    def test_the_day_being_replaced_is_not_offered_back(self):
        """Its recipes are marked selected, so the "new" day is actually new."""
        import inspect

        from tools import plan_tools

        src = inspect.getsource(plan_tools.replace_day)
        # every exclusion source the tool must apply
        assert "for entry in kept" in src            # rest of the week
        assert "for entry in replaced" in src        # the day being replaced
        assert "downvoted_recipe_ids" in src         # rejected recipes
        assert "excluded_recipe_ids" in src          # "not that one"


class TestReplaceDayGuards:
    def test_it_refuses_a_day_the_plan_does_not_have(self, session_service, sample_profile):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        # A three-day plan: asking for day 7 must say what it does cover.
        entries = [e for e in _week() if e["day"] <= 3]
        session_service.add_weekly_meal_plan(
            session.session_id, entries, day_summaries={},
        )
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("replace_day", {"session_id": session.session_id, "day": 7})
        msg = str(e.value)
        assert "Sunday" in msg
        assert "Monday" in msg  # it names what IS there

    def test_it_refuses_when_there_is_no_weekly_plan(self, session_service, sample_profile):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("replace_day", {"session_id": session.session_id, "day": 1})
        assert "weekly plan" in str(e.value).lower()

    def test_a_missing_session_is_a_tool_error_not_a_crash(self):
        with pytest.raises(tools.ToolError):
            tools.invoke("summarize_week", {"session_id": "nope-does-not-exist"})


# ── read tools against a stored plan ─────────────────────────────────────

class TestReadTools:
    @pytest.fixture
    def planned(self, session_service, sample_profile):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_weekly_meal_plan(
            session.session_id, _week(),
            day_summaries={1: "fish for dinner"},
        )
        return session.session_id

    def test_summarize_week_covers_every_day(self, planned):
        out = tools.invoke("summarize_week", {"session_id": planned})
        assert len(out["days"]) == 7
        assert out["days"][0]["name"] == "Monday"
        assert out["days"][6]["name"] == "Sunday"
        # 21 meals at 500 kcal
        assert out["week_totals"]["calories"] == 10500.0
        assert out["daily_average_kcal"] == 1500.0

    def test_summarize_week_reports_each_day_in_meal_order(self, planned):
        out = tools.invoke("summarize_week", {"session_id": planned})
        assert [m["meal_type"] for m in out["days"][0]["meals"]] == [
            "breakfast", "lunch", "dinner"
        ]

    def test_summarize_day_names_the_weekday(self, planned):
        out = tools.invoke("summarize_day", {"session_id": planned, "day": 4})
        assert out["name"] == "Thursday"
        assert len(out["meals"]) == 3
        assert out["totals"]["calories"] == 1500.0

    def test_summarize_day_carries_the_reasons(self, planned):
        out = tools.invoke("summarize_day", {"session_id": planned, "day": 1})
        # No chips on this fabricated plan, but the key must exist so a caller
        # can render "why this dish" without special-casing.
        assert all("why" in meal for meal in out["meals"])

    def test_plan_totals_weekly_breaks_down_per_day(self, planned):
        out = tools.invoke(
            "plan_totals", {"session_id": planned, "plan_type": "weekly"}
        )
        assert len(out["per_day"]) == 7
        assert out["total"]["calories"] == 10500.0
        assert all(d["calories"] == 1500.0 for d in out["per_day"])

    def test_plan_totals_defaults_to_daily(self, planned):
        """There is no daily canvas here, so it must say so rather than
        silently answering about the week."""
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("plan_totals", {"session_id": planned})
        assert "daily plan" in str(e.value).lower()

    def test_plan_type_is_constrained(self, planned):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke(
                "plan_totals", {"session_id": planned, "plan_type": "monthly"}
            )
        assert "daily" in str(e.value) and "weekly" in str(e.value)


# ── the day readers on the DAILY canvas ──────────────────────────────────
#
# `summarize_week` and `summarize_day` read `plan.entries` and nothing else, so
# a member looking at a three-day plan asked for a summary and was told
# "there's no weekly plan in this conversation yet". Multi-day plans live on the
# daily canvas — that is what `plan_structured` produces — and these two exist
# to summarise days.

def _daily_plate(rid, title, role="main", kcal=500):
    from models.session import MealCourse

    return MealCourse(
        recipe_id=rid, title=title, ingredients=f"{title} ingredients",
        directions="cook", role=role, nutrition={"calories": kcal},
        match_reasons=[{"kind": "diet", "label": "vegetarian"}],
    )


def _three_days():
    """Three days; day 2's dinner is two plates, so a reader can drop one."""
    from models.session import DayPlan, Meal, MealPlan

    days = [
        DayPlan(day=1, meals=[
            Meal("breakfast", [_daily_plate("d1b", "Porridge")]),
            Meal("dinner", [_daily_plate("d1d", "Bean stew")]),
        ]),
        DayPlan(day=2, meals=[
            Meal("breakfast", [_daily_plate("d2b", "Toast")]),
            Meal("dinner", [_daily_plate("d2d", "Moussaka"),
                            _daily_plate("d2s", "Greek salad", role="side")]),
        ]),
        DayPlan(day=3, meals=[
            Meal("dinner", [_daily_plate("d3d", "Curry")]),
        ]),
    ]
    plan = MealPlan.from_days(days, "three days")
    plan.constraints_applied = [{"constraint": "vegetarian", "status": "satisfied"}]
    return plan


class TestDayReadersOnTheDailyCanvas:
    @pytest.fixture
    def multiday(self, session_service, sample_profile):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_prepared_meal_plan(session.session_id, _three_days())
        return session.session_id

    def test_summarize_week_reads_a_multi_day_daily_plan(self, multiday):
        """The reported failure: this used to raise "no weekly plan"."""
        out = tools.invoke("summarize_week", {"session_id": multiday})
        assert out["plan_type"] == "daily"
        assert [d["day"] for d in out["days"]] == [1, 2, 3]

    def test_it_does_not_invent_weekdays(self, multiday):
        """A daily plan starts when the member cooks it, not on Monday."""
        out = tools.invoke("summarize_week", {"session_id": multiday})
        names = [d["name"] for d in out["days"]]
        assert names == ["Day 1", "Day 2", "Day 3"]
        assert "Monday" not in str(out)

    def test_every_plate_is_counted_and_named(self, multiday):
        out = tools.invoke("summarize_week", {"session_id": multiday})
        # 6 plates at 500 kcal — the side on day 2 included.
        assert out["total"]["calories"] == 3000.0
        assert out["daily_average_kcal"] == 1000.0
        dinner = next(m for m in out["days"][1]["meals"] if m["meal_type"] == "dinner")
        assert dinner["title"] == "Moussaka + Greek salad"
        assert [p["role"] for p in dinner["plates"]] == ["main", "side"]

    def test_it_carries_the_ledger_rather_than_dropping_it(self, multiday):
        out = tools.invoke("summarize_week", {"session_id": multiday})
        assert "vegetarian" in out["constraints_honored"]

    def test_summarize_day_reads_one_day_of_it(self, multiday):
        out = tools.invoke("summarize_day", {"session_id": multiday, "day": 2})
        assert out["name"] == "Day 2"
        assert [m["title"] for m in out["meals"]] == [
            "Toast", "Moussaka", "Greek salad",
        ]
        assert out["totals"]["calories"] == 1500.0
        assert all("why" in m for m in out["meals"])

    def test_summarize_day_names_what_the_plan_does_cover(self, multiday):
        with pytest.raises(tools.ToolError) as e:
            tools.invoke("summarize_day", {"session_id": multiday, "day": 6})
        msg = str(e.value)
        assert "Day 6" in msg and "Day 3" in msg
        assert "Saturday" not in msg

    def test_asking_for_weekly_explicitly_still_says_there_is_none(self, multiday):
        """The canvas the member is on decides; an explicit ask is honoured."""
        with pytest.raises(tools.ToolError) as e:
            tools.invoke(
                "summarize_week", {"session_id": multiday, "plan_type": "weekly"}
            )
        assert "weekly plan" in str(e.value).lower()

    def test_the_canvas_the_menu_was_opened_on_wins(self, session_service,
                                                    sample_profile):
        """A session with both canvases reads the one it is told to read."""
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_weekly_meal_plan(
            session.session_id, _week(), day_summaries={},
        )
        session_service.add_prepared_meal_plan(session.session_id, _three_days())

        daily = tools.invoke(
            "summarize_week", {"session_id": session.session_id, "plan_type": "daily"}
        )
        weekly = tools.invoke(
            "summarize_week", {"session_id": session.session_id, "plan_type": "weekly"}
        )
        assert len(daily["days"]) == 3 and daily["plan_type"] == "daily"
        assert len(weekly["days"]) == 7

    def test_plan_totals_splits_a_multi_day_plan_per_day(self, multiday):
        """One number for three days reads as one enormous day."""
        out = tools.invoke("plan_totals", {"session_id": multiday})
        assert [d["day"] for d in out["per_day"]] == [1, 2, 3]
        assert out["per_day"][1]["calories"] == 1500.0   # two plates at dinner
        assert out["daily_average_kcal"] == 1000.0

    def test_plan_totals_says_nothing_per_day_for_a_single_day(
        self, session_service, sample_profile
    ):
        """A one-day plan has no per-day breakdown to give, so it claims none."""
        import uuid

        from models.session import DayPlan, Meal, MealPlan

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_prepared_meal_plan(
            session.session_id,
            MealPlan.from_days([DayPlan(day=1, meals=[
                Meal("dinner", [_daily_plate("x", "Soup")]),
            ])], "one day"),
        )
        out = tools.invoke("plan_totals", {"session_id": session.session_id})
        assert "per_day" not in out
        assert out["total"]["calories"] == 500.0


class TestWhichCanvasAToolRunsOn:
    """A tool offered where it cannot run is the same broken promise as one
    that does not exist. The registry declares it so the menu can ask."""

    def test_replace_day_is_weekly_only(self):
        assert tools.get("replace_day").canvases == ("weekly",)

    def test_the_readers_work_on_either(self):
        for name in ("summarize_week", "summarize_day", "plan_totals",
                     "shopping_list", "swap_meal", "save_plan"):
            assert set(tools.get(name).canvases) == {"daily", "weekly"}, name

    def test_the_manifest_carries_it_so_the_ui_can_filter(self):
        by_name = {t["name"]: t for t in tools.manifest()}
        assert by_name["replace_day"]["canvases"] == ["weekly"]
        assert by_name["summarize_week"]["canvases"] == ["daily", "weekly"]

    def test_for_canvas_narrows_what_a_model_may_choose(self):
        daily = {t.name for t in tools.for_canvas("daily")}
        assert "replace_day" not in daily
        assert "summarize_week" in daily
        assert "replace_day" in {t.name for t in tools.for_canvas("weekly")}
        # No canvas known → no filtering, rather than an empty tool surface.
        assert tools.for_canvas(None) == tools.all_tools()

    def test_the_prose_says_when_a_tool_is_restricted(self):
        assert "weekly plans only" in tools.describe_tools("weekly")
        assert "replace_day" not in tools.describe_tools("daily")


class TestSwapMealNamesTheDayTheReaderCanRead:
    """`swap_meal` built its message with a weekday name whatever the canvas.
    On a multi-day daily plan `edit_service._named_day(weekdays=False)` reads
    nothing but "day N", so the edit came back asking which day it was."""

    def _capture(self, monkeypatch):
        sent = {}

        class _FakeEdit:
            def __init__(self, _svc):
                pass

            def process(self, session_id, message):
                sent["message"] = message
                return type("O", (), {"text": "done", "changed_slots": [],
                                      "unresolved": False})()

        import services.edit_service as es

        monkeypatch.setattr(es, "EditService", _FakeEdit)
        return sent

    def test_a_daily_canvas_gets_day_n(self, monkeypatch, session_service,
                                       sample_profile):
        import uuid

        sent = self._capture(monkeypatch)
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_prepared_meal_plan(session.session_id, _three_days())
        tools.invoke("swap_meal", {
            "session_id": session.session_id, "meal_type": "dinner",
            "directive": "lighter", "day": 2,
        })
        assert "day 2" in sent["message"].lower()
        assert "tuesday" not in sent["message"].lower()

        from services.edit_service import _named_day
        assert _named_day(sent["message"], weekdays=False) == 2

    def test_a_weekly_canvas_still_gets_the_weekday(self, monkeypatch,
                                                    session_service,
                                                    sample_profile):
        import uuid

        sent = self._capture(monkeypatch)
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        session_service.add_weekly_meal_plan(
            session.session_id, _week(), day_summaries={},
        )
        tools.invoke("swap_meal", {
            "session_id": session.session_id, "meal_type": "dinner",
            "directive": "lighter", "day": 2,
        })
        assert "Tuesday" in sent["message"]
