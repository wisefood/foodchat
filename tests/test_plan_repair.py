"""
Fix what the verifier named, once.

The verifier reported which plates failed and why, and `report.offenders` had
no consumer anywhere in the codebase. A plan that failed its own vegetarian
check was rendered with a red chip and handed over — the assistant checked its
work and then filed a complaint about it.

The bound matters as much as the fix. One pass, hard checks only, at most four
plates, and a re-verification afterwards so a repair that did not work cannot
be announced as one.

No network: the candidate client is stubbed.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                            # noqa: E402
from models.recipe import CandidateRecipe, RecipeEnrichment        # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan     # noqa: E402
from services import plan_repair, plan_verifier                    # noqa: E402


def _plate(rid, title, ingredients="", role="main"):
    return MealCourse(recipe_id=rid, title=title,
                      ingredients=ingredients or f"{title.lower()}, salt",
                      directions="cook", role=role)


def _plan(*meals):
    return MealPlan.from_days(
        [DayPlan(day=1, meals=[Meal(slot, [p]) for slot, p in meals])], "test",
    )


class _Client:
    """Offers replacements per slot and records what it was asked for."""

    def __init__(self, by_slot=None, details=None):
        self.by_slot = by_slot if by_slot is not None else {}
        self.details = details or {}
        self.asked: list[dict] = []

    def slot_candidates(self, profile, meal_type, exclude_ids, limit=8):
        self.asked.append({"slot": meal_type, "exclude": list(exclude_ids)})
        return list(self.by_slot.get(meal_type, []))

    def fetch_details(self, recipe_ids):
        return {r: self.details[r] for r in recipe_ids if r in self.details}


def _cand(rid, title, ingredients="lentils, carrot"):
    return CandidateRecipe(recipe_id=rid, title=title,
                           ingredients=ingredients, directions="cook")


# ── what it repairs ───────────────────────────────────────────────────────

class TestItFixesHardFailures:
    def test_an_allergen_plate_is_replaced(self):
        plan = _plan(("breakfast", _plate("ok", "Porridge")),
                     ("dinner", _plate("bad", "Satay", "peanuts, chicken")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        assert report.blocking

        client = _Client({"dinner": [_cand("safe", "Lentil bake")]})
        out = plan_repair.repair(plan, brief, report, {}, client=client)

        assert out.changed
        assert plan.day_plans[0].meals[1].plates[0].recipe_id == "safe"
        assert out.repaired[0]["was"] == "Satay"
        assert out.repaired[0]["now"] == "Lentil bake"

    def test_a_diet_failure_is_replaced(self):
        plan = _plan(("dinner", _plate("beef", "Beef stew")))
        brief = PlanBrief(diet=("vegetarian",))
        rich = {"beef": RecipeEnrichment(recipe_id="beef", title="Beef stew",
                                         diet_tags=["omnivore"])}
        report = plan_verifier.verify(plan, brief.to_requested(), rich)
        client = _Client({"dinner": [_cand("veg", "Chickpea stew")]},
                         details={"veg": RecipeEnrichment(
                             recipe_id="veg", title="Chickpea stew",
                             diet_tags=["vegetarian"])})
        out = plan_repair.repair(plan, brief, report, {}, client=client)
        assert out.changed and plan.day_plans[0].meals[0].plates[0].title == "Chickpea stew"

    def test_untouched_plates_are_byte_identical(self):
        """The plan is already enriched and already carries the member's own
        adapted recipes and chips on every other plate."""
        keeper = _plate("ok", "Porridge")
        keeper.nutrition = {"kcal": 300}
        keeper.match_reasons = [{"kind": "favorite", "label": "a favourite"}]
        plan = _plan(("breakfast", keeper),
                     ("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        plan_repair.repair(plan, brief, report, {},
                           client=_Client({"dinner": [_cand("safe", "Lentil bake")]}))
        after = plan.day_plans[0].meals[0].plates[0]
        assert after is keeper
        assert after.nutrition == {"kcal": 300}
        assert after.match_reasons[0]["kind"] == "favorite"

    def test_the_replacement_keeps_the_plate_role(self):
        plan = MealPlan.from_days([DayPlan(day=1, meals=[Meal("dinner", [
            _plate("main", "Ragu"), _plate("bad", "Satay", "peanuts", "side"),
        ])])], "test")
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        plan_repair.repair(plan, brief, report, {},
                           client=_Client({"dinner": [_cand("slaw", "Slaw")]}))
        assert plan.day_plans[0].meals[0].plates[1].role == "side"

    def test_the_replacement_says_why_it_is_there(self):
        plan = _plan(("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        plan_repair.repair(plan, brief, report, {},
                           client=_Client({"dinner": [_cand("safe", "Lentil bake")]}))
        chips = plan.day_plans[0].meals[0].plates[0].match_reasons
        assert chips and "allergens" in chips[0]["label"]

    def test_it_cannot_reintroduce_a_dish_already_on_the_plan(self):
        plan = _plan(("breakfast", _plate("keep", "Porridge")),
                     ("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        client = _Client({"dinner": [_cand("safe", "Lentil bake")]})
        plan_repair.repair(plan, brief, report, {}, client=client)
        excluded = client.asked[0]["exclude"]
        assert "keep" in excluded and "bad" in excluded


# ── what it deliberately does NOT repair ─────────────────────────────────

class TestItLeavesSoftFailuresAlone:
    @pytest.mark.parametrize("requested,rich", [
        ({"kcal_target": 2000}, {"a": RecipeEnrichment(recipe_id="a", title="x", kcal=9000)}),
        ({"max_minutes": 20}, {"a": RecipeEnrichment(recipe_id="a", title="x", duration=200)}),
        ({"min_nutri_score": "b"}, {"a": RecipeEnrichment(recipe_id="a", title="x",
                                                          nutri_score_label="E")}),
        ({"pantry": ["zucchini"]}, {}),
    ])
    def test_a_soft_miss_is_reported_not_swapped(self, requested, rich):
        """Swapping a dish someone may be looking forward to because the day
        came in 12% over target is worse than the honest sentence."""
        plan = _plan(("dinner", _plate("a", "Ragu")))
        report = plan_verifier.verify(plan, requested, rich)
        client = _Client({"dinner": [_cand("other", "Something else")]})
        out = plan_repair.repair(plan, PlanBrief(), report, {}, client=client)
        assert not out.changed
        assert client.asked == [], "it should not even have fetched"

    def test_a_clean_plan_does_nothing(self):
        plan = _plan(("dinner", _plate("a", "Ragu")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        client = _Client()
        out = plan_repair.repair(plan, brief, report, {}, client=client)
        assert not out.changed and client.asked == []


# ── the bounds ────────────────────────────────────────────────────────────

class TestItIsBounded:
    def test_it_refuses_a_plan_where_almost_everything_fails(self):
        """That is a pool that never satisfied the constraint, not a repair
        job — and a partial fix reads as a whole one."""
        meals = [(f"slot{i}", _plate(f"bad{i}", f"Satay {i}", "peanuts"))
                 for i in range(6)]
        plan = _plan(*meals)
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        client = _Client({f"slot{i}": [_cand(f"ok{i}", f"Fine {i}")] for i in range(6)})
        out = plan_repair.repair(plan, brief, report, {}, client=client)
        assert not out.changed
        assert client.asked == []
        assert len(out.unresolved) == 6

    def test_the_cap_is_small(self):
        assert plan_repair.MAX_REPAIRS <= 5

    def test_it_does_not_loop(self):
        """One pass. A second failure is reported, not chased."""
        import inspect

        src = inspect.getsource(plan_repair.repair)
        assert "while " not in src
        assert "still not right after one attempt" in src

    def test_a_second_failure_is_reported(self):
        plan = _plan(("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        # The replacement is ALSO unsafe.
        client = _Client({"dinner": [_cand("worse", "Peanut curry", "peanuts, rice")]})
        out = plan_repair.repair(plan, brief, report, {}, client=client)
        assert out.changed, "it did swap"
        assert out.unresolved, "and it says the swap did not fix it"
        assert any("still not right" in u["reason"] for u in out.unresolved)


# ── honesty ───────────────────────────────────────────────────────────────

class TestItReVerifies:
    def test_the_report_after_a_repair_describes_the_repaired_plan(self):
        plan = _plan(("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        assert report.get("allergens").status == plan_verifier.FAILED
        out = plan_repair.repair(plan, brief, report, {},
                                 client=_Client({"dinner": [_cand("safe", "Lentil bake")]}))
        assert out.report.get("allergens").status == plan_verifier.PASSED

    def test_nothing_available_is_said_rather_than_hidden(self):
        plan = _plan(("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        out = plan_repair.repair(plan, brief, report, {}, client=_Client({"dinner": []}))
        assert not out.changed
        assert out.unresolved[0]["reason"] == "no other dish in the collection fits"

    def test_a_fetch_failure_leaves_the_plan_alone(self):
        class _Boom(_Client):
            def slot_candidates(self, *_a, **_k):
                raise RuntimeError("recipewrangler is down")

        plan = _plan(("dinner", _plate("bad", "Satay", "peanuts")))
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(plan, brief.to_requested(), {})
        out = plan_repair.repair(plan, brief, report, {}, client=_Boom())
        assert not out.changed
        assert plan.day_plans[0].meals[0].plates[0].recipe_id == "bad"

    def test_describe_says_both_halves(self):
        """A repair announced without its failures lets the member believe a
        plan is clean when one dish on it is not."""
        out = plan_repair.RepairOutcome(
            plan=None,
            repaired=[{"was": "Satay", "now": "Lentil bake", "slot": "dinner",
                       "check": "allergens"}],
            unresolved=[{"recipe_id": "x", "check": "diet", "reason": "nope"}],
        )
        text = plan_repair.describe(out)
        assert "Satay" in text and "Lentil bake" in text
        assert "still fall short" in text

    def test_describe_is_none_when_nothing_happened(self):
        assert plan_repair.describe(plan_repair.RepairOutcome(plan=None)) is None


class TestItIsWiredIn:
    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_both_daily_paths_repair(self, method):
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert "plan_repair.repair(" in src
        assert "report.blocking" in src, "it must only run on a hard failure"

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_the_repair_replaces_the_stale_measured_rows(self, method):
        """The ledger must describe the plan the member is given, not the one
        that existed before the swap."""
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert 'row.get("source") != "measured on the plan"' in src

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_it_is_sheddable_under_a_thin_budget(self, method):
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert 'turn_budget.skip("plan repair"' in src

    @pytest.mark.parametrize("method", ["_generate_and_store", "_generate_structured"])
    def test_the_reply_is_told(self, method):
        import inspect

        from services.chat_service import ChatService

        src = inspect.getsource(getattr(ChatService, method))
        assert 'facts["repair"] = repair_note' in src
