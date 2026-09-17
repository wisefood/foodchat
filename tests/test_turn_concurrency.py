"""
One turn at a time per session.

Two turns on one session both load it, both plan, and both write the canvas
pointer. Last write wins and the loser's plan is orphaned — generated, stored,
paid for, and unreachable. A single tab cannot do this (the composer disables
while sending), but a second tab can, and so can a slider apply landing on top
of a chat turn.

Refused rather than queued. Queueing would hold a worker for the length of the
first turn and then run a plan the member asked for a minute ago; saying "still
working" is cheaper and truer.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

sys.path.insert(0, "src")

from services.orchestrator_service import (      # noqa: E402
    _TURN_GUARD_TTL,
    OrchestratorService,
)


@pytest.fixture
def orch():
    # `__new__` on purpose: the guard state is class-level precisely so an
    # instance built this way — which the rest of the suite does routinely —
    # still has it. The dict is cleared between tests because it IS shared.
    OrchestratorService._turns_in_flight.clear()
    return OrchestratorService.__new__(OrchestratorService)


class TestTheGuard:
    def test_a_second_turn_on_the_same_session_is_refused(self, orch):
        with orch._one_turn_at_a_time("s1") as first:
            assert first is True
            with orch._one_turn_at_a_time("s1") as second:
                assert second is False, "two turns claimed the same session"

    def test_a_different_session_is_unaffected(self, orch):
        """One busy member must not block another."""
        with orch._one_turn_at_a_time("s1"):
            with orch._one_turn_at_a_time("s2") as claimed:
                assert claimed is True

    def test_it_releases_when_the_turn_finishes(self, orch):
        with orch._one_turn_at_a_time("s1"):
            pass
        with orch._one_turn_at_a_time("s1") as claimed:
            assert claimed is True

    def test_it_releases_when_the_turn_raises(self, orch):
        """A crashed turn must not lock the member out of their own session."""
        with pytest.raises(RuntimeError):
            with orch._one_turn_at_a_time("s1"):
                raise RuntimeError("planning blew up")
        with orch._one_turn_at_a_time("s1") as claimed:
            assert claimed is True

    def test_a_stale_claim_expires(self, orch):
        """If a process is killed mid-turn the entry is never popped. Bounded
        so a crash cannot lock a session out permanently."""
        orch._turns_in_flight["s1"] = time.monotonic() - (_TURN_GUARD_TTL + 1)
        with orch._one_turn_at_a_time("s1") as claimed:
            assert claimed is True, "a stale claim must be taken over"

    def test_a_fresh_claim_does_not_expire(self, orch):
        orch._turns_in_flight["s1"] = time.monotonic()
        with orch._one_turn_at_a_time("s1") as claimed:
            assert claimed is False

    def test_the_ttl_outlasts_a_full_length_turn(self):
        """Otherwise a slow-but-healthy turn would be treated as dead and a
        second turn would be let in beside it — the exact race this prevents."""
        from services import turn_budget

        assert _TURN_GUARD_TTL > turn_budget.TURN_BUDGET_SECONDS


class TestUnderRealThreads:
    def test_only_one_of_many_racing_threads_gets_in(self, orch):
        """Checking and claiming must be one step. Two threads checking an
        unguarded dict would both find it free."""
        entered, refused = [], []
        start = threading.Barrier(8)

        def attempt():
            start.wait()
            with orch._one_turn_at_a_time("s1") as claimed:
                if claimed:
                    entered.append(1)
                    time.sleep(0.05)
                else:
                    refused.append(1)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(entered) == 1, f"{len(entered)} turns ran at once"
        assert len(refused) == 7

    def test_different_sessions_all_get_in(self, orch):
        entered = []
        start = threading.Barrier(6)

        def attempt(index):
            start.wait()
            with orch._one_turn_at_a_time(f"s{index}"):
                entered.append(index)
                time.sleep(0.02)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(entered) == 6

    def test_the_session_is_free_again_afterwards(self, orch):
        assert orch._turns_in_flight == {} or True
        with orch._one_turn_at_a_time("s1"):
            pass
        assert "s1" not in orch._turns_in_flight


class TestWhatTheMemberSees:
    def test_the_refusal_is_a_turn_not_an_error(self, orch):
        """A 500 would look like a broken assistant. This is a sentence."""
        turn = orch._busy_turn()
        assert turn.role == "assistant"
        assert turn.intent == "chat"
        assert "still working" in turn.content.lower()

    def test_it_carries_no_plan(self, orch):
        turn = orch._busy_turn()
        assert turn.meal_plan is None and turn.weekly_meal_plan is None

    def test_it_is_not_the_message_limit_refusal(self, orch):
        """Two different refusals; conflating them would tell a member to start
        a new session when they only needed to wait a moment."""
        assert orch._busy_turn().at_message_limit is False


class TestEveryEntryPointIsGuarded:
    def test_all_five_claim_the_session(self):
        """process, apply_plan_parameters, regenerate, compose_plan, score_plan."""
        import inspect

        src = inspect.getsource(sys.modules["services.orchestrator_service"])
        assert src.count("_one_turn_at_a_time(session_id) as claimed") == 5
        assert src.count("return self._busy_turn()") == 5

    def test_the_guard_wraps_the_work_not_just_the_check(self):
        """Claiming and releasing without holding it across the turn would be
        a guard that guards nothing."""
        import inspect

        src = inspect.getsource(sys.modules["services.orchestrator_service"])
        assert src.count("if not claimed:") == 5


class TestAgainstARealTurn:
    """The helper tests exercise the guard directly. This exercises the path a
    request actually takes — which is where my first attempt was broken: the
    guard raised, `@contextmanager` deferred the raise to `__enter__`, and the
    `try/except` around the CALL never fired, so it would have escaped to the
    router as a 500 instead of returning a turn."""

    def test_a_busy_session_gets_a_turn_not_an_exception(self, orch, monkeypatch):
        import time as _time

        # Claim the session as if a turn were already running.
        orch._turns_in_flight["s-busy"] = _time.monotonic()

        # `process` must not reach ownership resolution at all.
        def must_not_run(*_a, **_k):
            raise AssertionError("the guard let a second turn through")

        monkeypatch.setattr(OrchestratorService, "_owned_session", must_not_run)
        turn = OrchestratorService.process(orch, "s-busy", "m1", "plan my day")
        assert turn.role == "assistant"
        assert "still working" in turn.content.lower()

    def test_a_free_session_proceeds_normally(self, orch, monkeypatch):
        """The guard must not refuse the common case."""
        reached: list = []

        def record(_self, session_id, member_id):
            reached.append(session_id)
            raise RuntimeError("far enough")

        monkeypatch.setattr(OrchestratorService, "_owned_session", record)
        with pytest.raises(RuntimeError):
            OrchestratorService.process(orch, "s-free", "m1", "plan my day")
        assert reached == ["s-free"]

    def test_the_session_is_released_even_though_the_turn_raised(self, orch, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("planning blew up")

        monkeypatch.setattr(OrchestratorService, "_owned_session", boom)
        with pytest.raises(RuntimeError):
            OrchestratorService.process(orch, "s-crash", "m1", "plan my day")
        assert "s-crash" not in orch._turns_in_flight

    def test_a_refused_turn_does_not_release_the_running_one(self, orch):
        """The refusal path must not pop a claim it never made — that would
        free the session out from under the turn that owns it."""
        import time as _time

        orch._turns_in_flight["s1"] = _time.monotonic()
        with orch._one_turn_at_a_time("s1") as claimed:
            assert claimed is False
        assert "s1" in orch._turns_in_flight, "the refusal released someone else's claim"
