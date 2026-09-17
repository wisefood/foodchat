"""
Recipes come back in pages, and FoodChat only ever asked for page one.

`plan_meals` ranks deterministically — planning tier, then Nutri-Score, then
curated source, then `recipe_id` as a total tiebreak — over-fetches a window,
and picks a diverse subset of it. Every part of that is deliberate and worth
keeping: it is what stops a regenerated plan looking random.

What was missing is that the window always started at zero. So the pool was
the same pool, the diverse subset of it was the same subset, and a member
asking for a second plan received the first one again. Exclusion narrows that
window; an offset MOVES it, which is the difference between a pool that stays
full and one that shrinks toward empty as a session goes on.

The offset is derived, not random. The same request must produce the same plan,
or a member cannot tell a regeneration from a bug.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from services import plan_history                              # noqa: E402


class _Session:
    def __init__(self, daily=0, weekly=0):
        self.meal_plans = [object()] * daily
        self.weekly_meal_plans = [object()] * weekly


class TestTheWindowWalks:
    def test_the_first_plan_starts_at_the_best_matches(self):
        assert plan_history.window_offset(_Session()) == 0

    def test_each_plan_moves_further_in(self):
        offsets = [plan_history.window_offset(_Session(daily=n)) for n in range(4)]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == len(offsets)

    def test_both_canvases_count(self):
        """A member who made a week and then asks for a day has been served
        21 recipes; the day should not open on the same page as their first
        request did."""
        assert plan_history.window_offset(_Session(weekly=1)) > 0
        assert plan_history.window_offset(_Session(daily=1, weekly=1)) \
            > plan_history.window_offset(_Session(daily=1))

    def test_it_is_derived_not_random(self):
        """The same request must produce the same plan, or a member cannot tell
        a regeneration from a bug."""
        session = _Session(daily=3)
        assert len({plan_history.window_offset(session) for _ in range(5)}) == 1

    def test_it_wraps_rather_than_paging_into_nothing(self):
        """Past a few hundred a narrow diet has nothing left, and starting
        again from the best matches beats an empty slot."""
        deep = plan_history.window_offset(_Session(daily=500))
        assert 0 <= deep <= plan_history.MAX_OFFSET

    def test_no_session_is_page_one(self):
        assert plan_history.window_offset(None) == 0

    def test_the_step_clears_the_window_it_just_used(self):
        """`plan_meals` over-fetches and picks a diverse subset, so the pool a
        slot draws from is several times the count asked for — stepping by the
        count would land inside the window it just used."""
        from services.planning_pipeline import CANDIDATE_LIMIT

        assert plan_history.WINDOW_STEP >= CANDIDATE_LIMIT


class TestItIsGatedOnTheService:
    """`plan_meals` rejects unknown fields with a 422 rather than ignoring
    them, so sending an offset to a RecipeWrangler that predates it would break
    every plan rather than degrade."""

    @staticmethod
    def _client(manifest):
        from services.plan_client import PlanClient

        client = PlanClient()
        PlanClient._manifest_cache = manifest
        return client

    def teardown_method(self):
        from services.plan_client import PlanClient

        PlanClient._manifest_cache = None

    def test_a_service_that_advertises_it_receives_it(self, monkeypatch):
        sent = self._send(monkeypatch, manifest={
            "tools": [{"name": "plan_meals", "accepts": ["slots", "offset"]}],
        }, offset=16)
        assert sent["offset"] == 16

    def test_a_service_that_does_not_is_not_sent_it(self, monkeypatch):
        sent = self._send(monkeypatch, manifest={
            "tools": [{"name": "plan_meals", "accepts": ["slots", "days"]}],
        }, offset=16)
        assert "offset" not in sent

    def test_an_older_manifest_with_no_accepts_is_a_no(self, monkeypatch):
        """Guessing yes would turn every plan into a 422 the moment the two
        services drifted."""
        sent = self._send(monkeypatch, manifest={
            "tools": [{"name": "plan_meals"}],
        }, offset=16)
        assert "offset" not in sent

    def test_no_manifest_at_all_is_a_no(self, monkeypatch):
        sent = self._send(monkeypatch, manifest={}, offset=16)
        assert "offset" not in sent

    def test_page_one_is_never_sent(self, monkeypatch):
        """Zero is the default on the far side; sending it is noise."""
        sent = self._send(monkeypatch, manifest={
            "tools": [{"name": "plan_meals", "accepts": ["offset"]}],
        }, offset=0)
        assert "offset" not in sent

    def _send(self, monkeypatch, *, manifest, offset):
        import httpx

        sent: dict = {}

        class _Response:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"days": []}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                sent.update(json or {})
                return _Response()

        monkeypatch.setattr(httpx, "Client", _Client)
        self._client(manifest).plan_meals(offset=offset)
        return sent


class TestItReachesEveryFetch:
    @pytest.mark.parametrize("target,name", [
        ("generate", "window_offset"),
        ("plan_structured", "window_offset"),
    ])
    def test_the_pipeline_takes_it(self, target, name):
        import inspect

        from services.planning_pipeline import PlanningPipeline

        assert name in inspect.signature(getattr(PlanningPipeline, target)).parameters

    def test_the_caller_supplies_it(self):
        import ast
        import inspect

        from services.chat_service import ChatService

        for method, call in (("_generate_and_store", "generate"),
                             ("_generate_structured", "plan_structured")):
            tree = ast.parse(inspect.getsource(getattr(ChatService, method)).lstrip())
            node = next(
                n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == call
            )
            assert "window_offset" in {kw.arg for kw in node.keywords}, method

    def test_the_composer_passes_it_to_the_pool_fetch(self):
        import inspect

        from services import meal_composer

        assert "offset" in inspect.signature(meal_composer.role_pools).parameters
        assert "offset=offset" in inspect.getsource(meal_composer.role_pools)
