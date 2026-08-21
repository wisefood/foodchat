"""
The grader can rank a day of any shape.

It was hardcoded to breakfast × lunch × dinner, and `ScoredPlan` had three
named fields. That was not a simplification — it was the reason the structured
planning path shipped with **no grading and no quality metrics at all**: a
four-meal day, or a dinner served as a main and a side, could not be expressed
as a `ScoredPlan`, so it could not be scored, so the path served whatever order
RecipeWrangler happened to return.

Two things have to hold at once here: the three-slot behaviour must be
unchanged for the path that has always used it, and an arbitrary set of slots
must now work. And the combination space has to stay bounded — `product` over
three slots of eight is 512, over seven slots it is two million, so a
generalisation that kept materialising the product would hang the turn on
exactly the plans it exists to support.

LLM-free: the grader's client is replaced.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from agents import DocumentGrader, _COMBO_SCAN_LIMIT     # noqa: E402
from models.recipe import CandidateRecipe          # noqa: E402


def _c(rid: str) -> CandidateRecipe:
    return CandidateRecipe(recipe_id=rid, title=rid.replace("-", " ").title(),
                           ingredients=f"{rid} ingredients", directions="cook")


def _pool(slot: str, n: int) -> list[CandidateRecipe]:
    return [_c(f"{slot}-{i}") for i in range(n)]


class _Client:
    """Grades every plan it is shown, best score to PLAN 0."""

    def __init__(self, grades=None):
        self.grades = grades
        self.prompts: list[str] = []

    def invoke(self, messages, config=None):
        text = "\n".join(m.content for m in messages)
        self.prompts.append(text)
        count = text.count("PLAN ")
        grades = self.grades if self.grades is not None else [
            {"plan_index": i, "score": 5 - min(i, 4), "reasoning": f"plan {i}"}
            for i in range(count)
        ]

        class _R:
            content = json.dumps({"grades": grades})
        return _R()


def _grader(client=None, max_plans=10) -> DocumentGrader:
    g = DocumentGrader.__new__(DocumentGrader)
    g.grader = client or _Client()
    g.max_plans_to_score = max_plans
    return g


# ── the shape that always worked, still works ────────────────────────────

class TestThreeSlotsUnchanged:
    def test_it_returns_scored_plans_best_first(self):
        g = _grader()
        out = g.grade_daily_plans("plan my day", {
            "breakfast": _pool("b", 2),
            "lunch": _pool("l", 2),
            "dinner": _pool("d", 2),
        }, {})
        assert out and out[0].score >= out[-1].score
        assert len(out) <= 3

    def test_the_named_properties_still_resolve(self):
        g = _grader()
        best = g.grade_daily_plans("q", {
            "breakfast": _pool("b", 2),
            "lunch": _pool("l", 2),
            "dinner": _pool("d", 2),
        }, {})[0]
        assert best.breakfast.recipe_id.startswith("b-")
        assert best.lunch.recipe_id.startswith("l-")
        assert best.dinner.recipe_id.startswith("d-")
        assert best.is_classic

    def test_the_top_of_ranking_day_is_always_graded(self):
        """Each slot arrives best-first. The strongest combination must be in
        the batch rather than left to chance."""
        g = _grader()
        g.grade_daily_plans("q", {
            "breakfast": _pool("b", 6),
            "lunch": _pool("l", 6),
            "dinner": _pool("d", 6),
        }, {})
        first_plan = g.grader.prompts[0].split("PLAN 1")[0]
        assert "b-0" in first_plan and "l-0" in first_plan and "d-0" in first_plan

    def test_an_empty_slot_fails_closed(self):
        """The three-slot caller stores through `MealPlan.from_courses`, which
        requires exactly three — so quietly grading a two-slot day because
        lunch came back empty would turn a warning into a 500 two frames
        later. `[]` is the answer; the caller degrades to the unranked pool."""
        g = _grader()
        assert g.grade_daily_plans("q", {
            "breakfast": _pool("b", 2), "lunch": [], "dinner": _pool("d", 2),
        }, {}) == []

    def test_a_grader_failure_returns_empty_for_the_caller_to_degrade(self):
        class _Boom:
            def invoke(self, *_a, **_k):
                raise RuntimeError("groq is down")

        assert _grader(_Boom()).grade_daily_plans("q", {
            "breakfast": _pool("b", 1), "lunch": _pool("l", 1), "dinner": _pool("d", 1),
        }, {}) == []


# ── the shapes that could not be graded at all ───────────────────────────

class TestArbitrarySlots:
    def test_a_four_meal_day_is_graded(self):
        g = _grader()
        out = g.grade_plans("q", {
            "breakfast": _pool("b", 2), "lunch": _pool("l", 2),
            "snack": _pool("s", 2), "dinner": _pool("d", 2),
        }, {})
        assert out
        assert out[0].slot_names == ["breakfast", "lunch", "snack", "dinner"]

    def test_a_two_meal_day_is_graded(self):
        g = _grader()
        out = g.grade_plans("q", {"lunch": _pool("l", 2), "dinner": _pool("d", 2)}, {})
        assert out and out[0].slot_names == ["lunch", "dinner"]
        assert out[0].breakfast is None

    def test_a_multi_plate_meal_is_graded(self):
        """A dinner served as a main and a side — the shape `PlanSpec` exists
        for and the old grader could not express."""
        g = _grader()
        out = g.grade_plans("q", {
            "dinner": _pool("main", 2), "side": _pool("side", 2),
        }, {})
        assert out and set(out[0].slots) == {"dinner", "side"}

    def test_slots_are_inferred_from_the_pool_when_not_given(self):
        g = _grader()
        out = g.grade_plans("q", {"dinner": _pool("d", 1), "breakfast": _pool("b", 1)}, {})
        assert out[0].slot_names == ["breakfast", "dinner"], "eating order, not dict order"

    def test_a_slot_with_no_candidates_is_dropped_not_fatal(self):
        g = _grader()
        out = g.grade_plans("q", {
            "breakfast": _pool("b", 2), "snack": [], "dinner": _pool("d", 2),
        }, {})
        assert out and out[0].slot_names == ["breakfast", "dinner"]

    def test_every_slot_reaches_the_prompt_labelled(self):
        g = _grader()
        g.grade_plans("q", {
            "breakfast": _pool("b", 1), "snack": _pool("s", 1), "dinner": _pool("d", 1),
        }, {})
        prompt = g.grader.prompts[0]
        for slot in ("breakfast:", "snack:", "dinner:"):
            assert slot in prompt


# ── the combinatorics stay bounded ───────────────────────────────────────

class TestItCannotExplode:
    def test_a_seven_slot_day_does_not_enumerate_two_million_combinations(self):
        """The whole risk of generalising. `list(product(*pools))` over seven
        slots of eight is 2,097,152 tuples built before anything is sampled."""
        g = _grader()
        pools = {f"slot{i}": _pool(f"s{i}", 8) for i in range(7)}
        out = g.grade_plans("q", pools, {})
        assert out, "it must still produce a ranking"

    def test_the_batch_never_exceeds_the_cap(self):
        g = _grader(max_plans=5)
        g.grade_plans("q", {f"slot{i}": _pool(f"s{i}", 6) for i in range(5)}, {})
        assert g.grader.prompts[0].count("\nPLAN ") <= 5

    def test_the_scan_limit_is_bounded(self):
        assert _COMBO_SCAN_LIMIT <= 4096

    def test_a_small_pool_is_covered_exhaustively(self):
        """Two slots of two is four combinations; all four should be graded
        rather than three sampled ones — the old behaviour for short slots."""
        g = _grader(max_plans=10)
        g.grade_plans("q", {"lunch": _pool("l", 2), "dinner": _pool("d", 2)}, {})
        assert g.grader.prompts[0].count("\nPLAN ") == 4

    def test_the_batch_has_no_duplicate_days(self):
        """A batch of ten identical days teaches the judge nothing and costs
        the same as ten different ones."""
        g = _grader(max_plans=8)
        g.grade_plans("q", {"lunch": _pool("l", 5), "dinner": _pool("d", 5)}, {})
        prompt = g.grader.prompts[0]
        blocks = [b.strip() for b in prompt.split("\nPLAN ")[1:]]
        bodies = ["\n".join(b.splitlines()[1:]) for b in blocks]
        assert len(bodies) == len(set(bodies))


class TestResultMapping:
    def test_an_out_of_range_index_is_skipped_not_fatal(self):
        g = _grader(_Client(grades=[
            {"plan_index": 99, "score": 5, "reasoning": "nonexistent"},
            {"plan_index": 0, "score": 4, "reasoning": "real"},
        ]))
        out = g.grade_plans("q", {"lunch": _pool("l", 1), "dinner": _pool("d", 1)}, {})
        assert len(out) == 1 and out[0].reasoning == "real"

    def test_the_mapping_puts_each_recipe_in_its_own_slot(self):
        """Off-by-one here would serve a dessert as breakfast."""
        g = _grader(_Client(grades=[{"plan_index": 0, "score": 5, "reasoning": "r"}]))
        out = g.grade_plans("q", {
            "breakfast": [_c("porridge")], "snack": [_c("apple")],
            "dinner": [_c("ragu")],
        }, {})
        assert out[0].slots["breakfast"].recipe_id == "porridge"
        assert out[0].slots["snack"].recipe_id == "apple"
        assert out[0].slots["dinner"].recipe_id == "ragu"

    def test_courses_come_back_in_eating_order(self):
        g = _grader(_Client(grades=[{"plan_index": 0, "score": 5, "reasoning": "r"}]))
        out = g.grade_plans("q", {
            "dinner": [_c("ragu")], "breakfast": [_c("porridge")],
            "snack": [_c("apple")],
        }, {})
        assert [c.recipe_id for c in out[0].courses] == ["porridge", "apple", "ragu"]


class TestThePromptIsNewNotEdited:
    def test_it_uses_the_new_registered_names(self):
        """`sync_prompts` creates only missing prompts and never overwrites, so
        editing `grader_system` in place would work locally and ship dead."""
        import prompts

        names = {p.name for p in prompts.ALL_PROMPTS}
        assert any(n.endswith("plan_grader_system") for n in names)
        assert any(n.endswith("plan_grader_user") for n in names)

    def test_the_old_pair_is_still_registered(self):
        """A Langfuse copy someone edited by hand is not ours to delete."""
        import prompts

        names = {p.name for p in prompts.ALL_PROMPTS}
        assert any(n.endswith("grader_system") for n in names)
        assert any(n.endswith("batch_grader_user") for n in names)

    def test_the_grader_calls_the_new_one(self):
        import inspect

        src = inspect.getsource(DocumentGrader.grade_plans)
        assert "PLAN_GRADER_SYSTEM" in src and "PLAN_GRADER_USER" in src
        assert "BATCH_GRADER_USER" not in src
