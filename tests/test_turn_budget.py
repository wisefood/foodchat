"""
A turn that runs late gives things up, in a deliberate order.

The timeout ladder was upside down. The UI waited 180 seconds, the gateway gave
up at 90, FoodChat had no limit of its own, and a single Groq call had none
either. So a slow planning turn produced the worst outcome available: the
gateway cut the connection, the member was told it failed, and FoodChat carried
on, finished the plan and stored it. The plan existed. They found it on the
next reload.

The budget makes the innermost layer decide, and makes it shed rather than die:

    grading            → the plan is unranked, not absent
    quality metrics    → no scores on the card
    the response writer → a plain sentence instead of a written one
    fetching, storing  → never skipped; that IS the plan

Everything above the line makes a plan better. Nothing above the line makes it
exist. These tests pin that ordering, because getting it backwards would mean a
beautifully graded plan the member never sees.
"""

from __future__ import annotations

import sys
import time

import pytest

sys.path.insert(0, "src")

from services import turn_budget                      # noqa: E402


class TestTheBudgetItself:
    def test_no_budget_means_no_limit(self):
        """Outside a turn — a tool call, a test, a background job — there is no
        deadline, and everything is affordable. A budget that defaulted to
        expired would disable grading everywhere it was not explicitly started."""
        assert turn_budget.remaining() is None
        assert turn_budget.can_afford(9999) is True
        assert turn_budget.expired() is False

    def test_inside_a_turn_time_is_finite(self):
        with turn_budget.start(30):
            left = turn_budget.remaining()
            assert left is not None and 29 < left <= 30

    def test_it_closes_when_the_turn_does(self):
        with turn_budget.start(30):
            pass
        assert turn_budget.remaining() is None

    def test_a_generous_budget_affords_everything(self):
        with turn_budget.start(120):
            for cost in (turn_budget.COST_GRADING, turn_budget.COST_METRICS,
                         turn_budget.COST_WRITER):
                assert turn_budget.can_afford(cost)

    def test_a_thin_budget_affords_nothing_expensive(self):
        with turn_budget.start(1):
            assert not turn_budget.can_afford(turn_budget.COST_GRADING)
            assert not turn_budget.can_afford(turn_budget.COST_METRICS)

    def test_an_exhausted_budget_reports_expired(self):
        with turn_budget.start(-1):
            assert turn_budget.expired()

    def test_nesting_keeps_the_outermost_deadline(self):
        """`apply_plan_parameters` and `regenerate` route into the same handlers
        a chat turn uses. A nested budget would hand the inner stage a fresh
        full allowance, which is the runaway this exists to prevent."""
        with turn_budget.start(2):
            with turn_budget.start(600):
                left = turn_budget.remaining()
                assert left is not None and left <= 2

    def test_it_unwinds_cleanly_after_an_error(self):
        with pytest.raises(RuntimeError):
            with turn_budget.start(30):
                raise RuntimeError("boom")
        assert turn_budget.remaining() is None

    def test_time_actually_passing_reduces_it(self):
        with turn_budget.start(30):
            first = turn_budget.remaining()
            time.sleep(0.05)
            assert turn_budget.remaining() < first


class TestSkip:
    def test_it_says_no_when_there_is_room(self):
        with turn_budget.start(120):
            assert turn_budget.skip("grading", turn_budget.COST_GRADING) is False

    def test_it_says_yes_when_there_is_not(self):
        with turn_budget.start(1):
            assert turn_budget.skip("grading", turn_budget.COST_GRADING) is True

    def test_it_never_skips_outside_a_turn(self):
        assert turn_budget.skip("grading", 9999) is False

    def test_skipping_is_logged_not_silent(self, caplog):
        """An unranked plan because the turn ran late looks identical to a
        broken grader, and the difference is the first thing anyone debugging
        will want."""
        import logging

        with caplog.at_level(logging.WARNING), turn_budget.start(1):
            turn_budget.skip("plan grading", turn_budget.COST_GRADING)
        assert "plan grading" in caplog.text
        assert "budget" in caplog.text.lower()


class TestItSheddsInTheRightOrder:
    """The writer is the cheapest thing to lose and grading the dearest, so as
    time runs out they must go in that order — not the reverse."""

    def test_a_middling_budget_keeps_the_writer_and_drops_grading(self):
        with turn_budget.start(turn_budget.COST_WRITER + 1):
            assert turn_budget.can_afford(turn_budget.COST_WRITER)
            assert not turn_budget.can_afford(turn_budget.COST_GRADING)

    def test_the_writer_is_the_cheapest_optional_stage(self):
        assert turn_budget.COST_WRITER < turn_budget.COST_METRICS
        assert turn_budget.COST_WRITER < turn_budget.COST_GRADING


class TestItFitsUnderTheGateway:
    def test_the_budget_is_shorter_than_the_gateway_timeout(self):
        """The whole point: FoodChat decides, rather than having the connection
        cut mid-plan and finishing work nobody will see. The gateway allows 90
        seconds for a planning call."""
        assert turn_budget.TURN_BUDGET_SECONDS < 90

    def test_one_model_call_cannot_consume_the_whole_budget(self):
        """A single hung Groq call used to hold a worker indefinitely; now it is
        bounded, and the bound has to leave room for the rest of the turn."""
        from backend.groq import GROQ_MAX_RETRIES, GROQ_TIMEOUT_SECONDS

        worst_case_one_call = GROQ_TIMEOUT_SECONDS * (GROQ_MAX_RETRIES + 1)
        assert worst_case_one_call <= turn_budget.TURN_BUDGET_SECONDS + 30

    def test_the_recipe_fetch_fits_too(self):
        from services.candidates_client import REQUEST_TIMEOUT_SECONDS

        assert REQUEST_TIMEOUT_SECONDS <= turn_budget.TURN_BUDGET_SECONDS


class TestTheStagesAreActuallyWired:
    """A budget nothing consults is a constant."""

    @pytest.mark.parametrize("module,stage", [
        ("services.planning_pipeline", "plan grading"),
        ("services.chat_service", "quality metrics"),
        ("services.chat_service", "response writer"),
    ])
    def test_the_stage_asks_the_budget(self, module, stage):
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module))
        assert f'turn_budget.skip("{stage}"' in src

    def test_every_turn_entry_point_opens_one(self):
        """Four places a turn begins. One that forgets runs unbounded, which is
        the state this replaced."""
        import inspect

        # `services.orchestrator_service` is the SINGLETON, not the module —
        # the package rebinds the name.
        src = inspect.getsource(sys.modules["services.orchestrator_service"])
        assert src.count("turn_budget.start()") == 4
        assert src.count("trace_context(session_id=session_id, user_id=member_id)") == 4

    def test_shedding_grading_still_returns_a_plan(self):
        """The load-bearing property. Skipping must degrade the plan, never
        remove it."""
        import inspect

        from services.planning_pipeline import PlanningPipeline

        src = inspect.getsource(PlanningPipeline.generate)
        skip_at = src.index('turn_budget.skip("plan grading"')
        tail = src[skip_at:skip_at + 400]
        assert "_assemble_from_pool" in tail, "the skip path must still build a plan"
        assert "return None" not in tail


class TestSheddingExecuted:
    """Executed, not grepped. A test that greps for `turn_budget.skip` passes
    for code that calls it and ignores the answer."""

    def _pipeline(self, graded_called):
        from models.recipe import CandidateRecipe
        from services.planning_pipeline import PlanningPipeline

        pipeline = PlanningPipeline.__new__(PlanningPipeline)

        class _Grader:
            def grade_daily_plans(self, *_a, **_k):
                graded_called.append(True)
                return []

        pipeline.grader = _Grader()
        candidates = {
            slot: [CandidateRecipe(recipe_id=f"{slot}-1", title=slot.title(),
                                   ingredients="x", directions="y")]
            for slot in ("breakfast", "lunch", "dinner")
        }
        return pipeline, candidates

    def test_a_thin_budget_skips_the_grader_and_still_returns_a_plan(self):
        called: list = []
        pipeline, candidates = self._pipeline(called)
        with turn_budget.start(1):
            plans = pipeline._assemble_from_pool(candidates, "not ranked")
        assert plans, "the pool must still produce a plan"
        assert not called

    def test_the_unranked_plan_says_it_is_unranked(self):
        """A plan that quietly lost its ranking looks like a plan that was
        ranked badly."""
        called: list = []
        pipeline, candidates = self._pipeline(called)
        plans = pipeline._assemble_from_pool(
            candidates, "not ranked — the plan was taking too long"
        )
        assert "not ranked" in plans[0].reasoning.lower()

    def test_metrics_are_skipped_wholesale_not_partially(self, monkeypatch):
        """Four model calls behind one flag: a partial skip would leave some
        scores populated and others zero, which reads as a bad plan."""
        from services.chat_service import ChatService

        service = ChatService.__new__(ChatService)
        called: list = []
        monkeypatch.setattr(
            ChatService, "_compute_metrics",
            lambda self, sid, plan: called.append(True) or {"llm_score": 9},
        )
        with turn_budget.start(1):
            metrics = (
                {} if turn_budget.skip("quality metrics", turn_budget.COST_METRICS)
                else service._compute_metrics("s", None)
            )
        assert metrics == {} and not called

    def test_the_writer_falls_back_to_the_canned_sentence(self):
        called: list = []

        class _Writer:
            def write(self, facts, query, fallback):
                called.append(True)
                return "written prose"

        writer = _Writer()
        canned = "Here's your plan for today."
        with turn_budget.start(1):
            out = (
                canned if turn_budget.skip("response writer", turn_budget.COST_WRITER)
                else writer.write({}, "q", fallback=canned)
            )
        assert out == canned and not called

    def test_with_time_to_spare_nothing_is_shed(self):
        called: list = []

        class _Writer:
            def write(self, facts, query, fallback):
                called.append(True)
                return "written prose"

        with turn_budget.start(300):
            out = (
                "canned" if turn_budget.skip("response writer", turn_budget.COST_WRITER)
                else _Writer().write({}, "q", fallback="canned")
            )
        assert out == "written prose" and called
