"""
Ask for a daily plan twice and you got the same plan. Identical.

Three things lined up to guarantee it, and none of them is a bug on its own:

* RecipeWrangler returns a **deterministic** order — planning tier, then
  Nutri-Score, then curated source. Same filters, same eight candidates, same
  order.
* The grader runs at **temperature 0.0**. Same pool and same prompt produce the
  same ranking, by design and worth keeping.
* Nothing carried forward. `exclude_recipe_ids` held downvotes and dishes the
  member had explicitly rejected, and **not one recipe they had just been
  served.**

So "plan my day" was asked and answered as though no day had ever been planned,
and the member got courgette omelette every morning.

The fix is exclusion, not randomness. A random pick makes two plans differ and
makes neither explainable, and it throws away a pool order that encodes
planning tier and Nutri-Score. Excluding what was just served keeps every
ranking intact and moves the window along: the second plan is the next best
plan, not a shuffle of the first.

What these tests are most careful about is the degradation. A member on a
narrow diet must not be told "no meals exist" for the crime of asking twice.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                             # noqa: E402
from models.recipe import CandidateRecipe                         # noqa: E402
from services import plan_history                                 # noqa: E402
from services.planning_pipeline import PlanningPipeline           # noqa: E402


def _cand(rid):
    return CandidateRecipe(rid, f"Dish {rid}", "beans, rice", "cook")


class _Pool:
    """A corpus of `size` dishes per slot, honouring exclusions exactly the way
    `plan_meals` does — which is what makes "nothing new left" reachable."""

    def __init__(self, size=6, slots=("breakfast", "lunch", "dinner")):
        self.size = size
        self.slots = slots
        self.calls: list[list[str]] = []

    def fetch(self, **kwargs):
        excluded = set(kwargs.get("exclude_recipe_ids") or [])
        self.calls.append(sorted(excluded))
        return {
            slot: [
                _cand(f"{slot}-{n}") for n in range(self.size)
                if f"{slot}-{n}" not in excluded
            ]
            for slot in self.slots
        }


@pytest.fixture
def pipeline(monkeypatch):
    import services.planning_pipeline as module

    pool = _Pool()
    monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: pool.fetch(**k))

    class _Cand:
        def split_cuisines(self, likes):
            return [], list(likes or [])

    monkeypatch.setattr(module, "CANDIDATES", _Cand())

    p = PlanningPipeline.__new__(PlanningPipeline)

    class _Grader:
        """Ranks nothing — takes the pool's own order, like the real fallback.
        Deterministic on purpose: if two plans differ here it is because the
        POOL differed, which is the only thing under test."""
        def grade_daily_plans(self, *a, **k):
            return []

    p.grader = _Grader()
    p.pool = pool
    return p


def _titles(plans):
    return [c.title for c in plans[0].courses]


class TestAskingTwiceUsedToGiveTheSamePlan:
    def test_with_nothing_avoided_it_still_does(self, pipeline):
        """The baseline, stated rather than assumed: the determinism is real,
        it is worth keeping, and it is not what changed."""
        a = pipeline.generate("plan my day", {"allergies": []})
        b = pipeline.generate("plan my day", {"allergies": []})
        assert _titles(a) == _titles(b)

    def test_avoiding_what_was_served_gives_a_different_plan(self, pipeline):
        first = pipeline.generate("plan my day", {"allergies": []})
        served = [c.recipe_id for c in first[0].courses]

        second = pipeline.generate(
            "plan my day", {"allergies": []}, avoid_recent=served,
        )
        assert not set(c.recipe_id for c in second[0].courses) & set(served)

    def test_the_exclusion_reaches_the_fetch(self, pipeline):
        pipeline.generate(
            "plan my day", {"allergies": []}, avoid_recent=["breakfast-0"],
        )
        assert "breakfast-0" in pipeline.pool.calls[0]

    def test_a_hard_exclusion_is_not_duplicated_by_a_soft_one(self, pipeline):
        pipeline.generate(
            "plan my day", {"allergies": []},
            exclude_recipe_ids=["breakfast-0"], avoid_recent=["breakfast-0"],
        )
        assert pipeline.pool.calls[0].count("breakfast-0") == 1


class TestItNeverCostsThePlan:
    """The whole risk of this change. Excluding history narrows a pool that is
    already narrowed by diet, allergens and the time ceiling."""

    def test_a_slot_the_history_empties_is_refetched(self, monkeypatch):
        import services.planning_pipeline as module

        pool = _Pool(size=2)
        monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: pool.fetch(**k))
        monkeypatch.setattr(module, "CANDIDATES", type("C", (), {
            "split_cuisines": lambda self, likes: ([], list(likes or [])),
        })())

        p = PlanningPipeline.__new__(PlanningPipeline)
        p.grader = type("G", (), {"grade_daily_plans": lambda *a, **k: []})()

        # Every breakfast in the corpus was served recently.
        plans = p.generate(
            "plan my day", {"allergies": []},
            avoid_recent=["breakfast-0", "breakfast-1"],
        )
        assert plans, "an apology is worse than a repeat"
        assert plans[0].slots["breakfast"] is not None

    def test_only_the_emptied_slot_repeats(self, monkeypatch):
        """Asking twice should cost the repeat of one meal, not of the day."""
        import services.planning_pipeline as module

        pool = _Pool(size=2)
        monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: pool.fetch(**k))
        monkeypatch.setattr(module, "CANDIDATES", type("C", (), {
            "split_cuisines": lambda self, likes: ([], list(likes or [])),
        })())
        p = PlanningPipeline.__new__(PlanningPipeline)
        p.grader = type("G", (), {"grade_daily_plans": lambda *a, **k: []})()

        plans = p.generate(
            "plan my day", {"allergies": []},
            avoid_recent=["breakfast-0", "breakfast-1", "lunch-0"],
        )
        # breakfast had to reuse; lunch still had lunch-1 to offer.
        assert plans[0].slots["lunch"].recipe_id == "lunch-1"

    def test_the_repeat_is_stated_not_hidden(self, monkeypatch):
        """A member who asked for something new and got yesterday's dinner
        should be told, not left to recognise the photo."""
        import services.planning_pipeline as module

        pool = _Pool(size=1)
        monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: pool.fetch(**k))
        monkeypatch.setattr(module, "CANDIDATES", type("C", (), {
            "split_cuisines": lambda self, likes: ([], list(likes or [])),
        })())
        p = PlanningPipeline.__new__(PlanningPipeline)
        p.grader = type("G", (), {"grade_daily_plans": lambda *a, **k: []})()

        plans = p.generate(
            "plan my day", {"allergies": []},
            avoid_recent=["breakfast-0", "lunch-0", "dinner-0"],
        )
        assert plans
        assert "nothing new left" in plans[0].reasoning.lower()

    def test_a_hard_exclusion_is_still_never_relaxed(self, monkeypatch):
        """The distinction the two lists exist for: a downvote is a decision,
        recency is a preference."""
        import services.planning_pipeline as module

        pool = _Pool(size=2)
        monkeypatch.setattr(module, "_fetch_candidate_pool", lambda **k: pool.fetch(**k))
        monkeypatch.setattr(module, "CANDIDATES", type("C", (), {
            "split_cuisines": lambda self, likes: ([], list(likes or [])),
        })())
        p = PlanningPipeline.__new__(PlanningPipeline)
        p.grader = type("G", (), {"grade_daily_plans": lambda *a, **k: []})()

        p.generate(
            "plan my day", {"allergies": []},
            exclude_recipe_ids=["breakfast-0"],
            avoid_recent=["breakfast-1"],
        )
        # Both fetches carry the downvote; only the first carries the history.
        assert all("breakfast-0" in call for call in pool.calls)
        assert len(pool.calls) == 2
        assert "breakfast-1" not in pool.calls[1]


class TestReadingTheHistory:
    def test_it_walks_every_day_and_every_plate(self, session_service, sample_profile):
        """The orchestrator's own helper reads `plan.breakfast/lunch/dinner`, so
        it misses every extra day and every second plate — a multi-plate plan's
        sides would keep coming back."""
        from models.session import DayPlan, Meal, MealCourse, MealPlan

        plan = MealPlan.from_days([
            DayPlan(day=1, meals=[Meal("dinner", [
                MealCourse("m1", "Main", "x", "y", role="main"),
                MealCourse("s1", "Side", "x", "y", role="side"),
            ])]),
            DayPlan(day=2, meals=[Meal("dinner", [
                MealCourse("m2", "Main 2", "x", "y", role="main"),
            ])]),
        ], reasoning="")
        assert plan_history.plan_recipe_ids(plan) == ["m1", "s1", "m2"]

    def test_a_weekly_plan_is_read_too(self):
        """A member who had salmon in yesterday's week does not want it as
        today's dinner either."""
        class _Weekly:
            entries = [
                {"day": 1, "meal_type": "dinner", "recipe": {"recipe_id": "w1"}},
                {"day": 1, "meal_type": "dinner", "recipe": {"recipe_id": "w2"}},
            ]
        assert plan_history.plan_recipe_ids(_Weekly()) == ["w1", "w2"]

    def test_only_the_last_few_plans_count(self, session_service, sample_profile):
        """The member should stop seeing this week's repeats, not be barred
        from a dish they liked a month ago."""
        from conftest import make_candidates

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        for n in range(5):
            session_service.add_meal_plan(
                session.session_id, make_candidates(f"p{n}"), "r", {},
            )
        recent = plan_history.recently_served(
            session_service.get_session(session.session_id), plans=2,
        )
        assert any("p4" in r for r in recent)
        assert not any("p0" in r for r in recent)

    def test_it_is_capped(self, session_service, sample_profile):
        from conftest import make_candidates

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        for n in range(3):
            session_service.add_meal_plan(
                session.session_id, make_candidates(f"c{n}"), "r", {},
            )
        assert len(plan_history.recently_served(
            session_service.get_session(session.session_id), cap=4,
        )) == 4

    def test_no_session_is_no_history(self):
        assert plan_history.recently_served(None) == []


class TestARefinementAvoidsThePlanItIsRefining:
    """This started out the other way round, on the reasoning that "keeping its
    unchanged slots is the whole point". That is true of a single-slot swap —
    which `edit_service` handles — and false of this path, which regenerates
    every slot. So "make it lighter" refetched the same pool, took the same
    head of it, and handed back the same three dishes."""

    @staticmethod
    def _avoid_kwarg(method: str, call_name: str) -> str:
        import ast
        import inspect

        from services.chat_service import ChatService

        tree = ast.parse(inspect.getsource(getattr(ChatService, method)).lstrip())
        call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == call_name
        )
        return ast.unparse(
            next(kw for kw in call.keywords if kw.arg == "avoid_recent").value
        )

    def test_both_daily_paths_go_through_one_decision(self):
        """Two call sites deciding this separately is how they drift."""
        assert self._avoid_kwarg("_generate_and_store", "generate") == \
            self._avoid_kwarg("_generate_structured", "plan_structured")

    def test_the_decision_takes_the_refinement_flag(self):
        assert "is_refinement" in self._avoid_kwarg("_generate_and_store", "generate")

    def test_a_refinement_avoids_the_plan_on_screen(self, session_service, sample_profile):
        from conftest import make_candidates
        from services.chat_service import ChatService

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.add_meal_plan(
            session.session_id, make_candidates("cur"), "r", {},
        )
        svc = ChatService.__new__(ChatService)
        svc.session_service = session_service

        avoided = svc._avoid_for(session.session_id, is_refinement=True)
        assert set(avoided) == {"cur-b", "cur-l", "cur-d"}

    def test_a_fresh_plan_avoids_the_session_history(self, session_service, sample_profile):
        from conftest import make_candidates
        from services.chat_service import ChatService

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        for n in range(2):
            session_service.add_meal_plan(
                session.session_id, make_candidates(f"p{n}"), "r", {},
            )
        svc = ChatService.__new__(ChatService)
        svc.session_service = session_service

        avoided = svc._avoid_for(session.session_id, is_refinement=False)
        assert any("p0" in r for r in avoided) and any("p1" in r for r in avoided)

    def test_no_plan_yet_means_nothing_to_avoid(self, session_service, sample_profile):
        from services.chat_service import ChatService

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        svc = ChatService.__new__(ChatService)
        svc.session_service = session_service
        assert svc._avoid_for(session.session_id, is_refinement=True) == []

    def test_weekly_still_only_avoids_on_a_fresh_week(self):
        """A weekly refinement re-runs a 21-slot walk that already excludes
        within itself; avoiding the previous week as well would empty pools on
        a narrow diet before the week was half planned."""
        import inspect

        from services.weekly_plan_service import WeeklyPlanService

        src = inspect.getsource(WeeklyPlanService.process_message)
        assert "avoid_recent" in src and "is_refinement" in src


class TestOneThinSlotDoesNotRepeatTheWholeDay:
    """Reported: "between different plans it seems recipes are getting
    repetitive".

    The structured path gave up the recently-served history for the WHOLE plan
    the moment any one plate came back empty — and a shaped plan runs out of new
    dishes sooner precisely because it has more plates. So one thin slot (a
    snack, or a narrow diet) made every other plate repeat too. The classic path
    has always refetched only the slot that emptied.
    """

    def _spec(self):
        return PlanSpec(meals=("breakfast", "lunch", "snack", "dinner"))

    def test_only_the_empty_plates_are_refetched(self):
        from services.planning_pipeline import _spec_for_plates, _unfilled_plates

        spec = self._spec()
        pools = {1: {
            ("breakfast", "main"): ["b"],
            ("lunch", "main"): ["l"],
            ("snack", "main"): [],          # nothing new left here
            ("dinner", "main"): ["d"],
        }}
        unfilled = _unfilled_plates(pools, spec)

        assert unfilled == {("snack", "main")}
        assert _spec_for_plates(spec, unfilled).meals == ("snack",)

    def test_a_plate_with_roles_keeps_them(self):
        from services.planning_pipeline import _spec_for_plates

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "salad")})
        narrowed = _spec_for_plates(spec, {("dinner", "salad")})

        assert narrowed.meals == ("dinner",)
        assert narrowed.roles_for("dinner") == ("salad",)

    def test_a_full_pool_needs_no_second_fetch(self):
        from services.planning_pipeline import _every_plate_has_a_candidate

        spec = self._spec()
        pools = {1: {(slot, "main"): ["x"] for slot in spec.meals}}

        assert _every_plate_has_a_candidate(pools, spec) is True

    def test_nothing_at_all_counts_every_plate_as_unfilled(self):
        from services.planning_pipeline import _unfilled_plates

        spec = self._spec()
        assert len(_unfilled_plates({}, spec)) == 4
