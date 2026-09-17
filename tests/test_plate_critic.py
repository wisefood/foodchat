"""
Something reasons about the dish before the plan is delivered.

A member was served this:

    Breakfast   Easter breakfast baskets                     226 kcal
    Lunch       1 recipe = 4 dinners: Herby onion rice        899 kcal
    Dinner      Southwestern Stuffed Potatoes                859 kcal
                "Assembled directly from your constraints
                 (not ranked — grader unavailable)."

Every constraint honoured, day total 1,984 against a 2,000 target, every check
in the system green. And an 11%-of-the-day breakfast, a batch-cooking article
served as lunch, and 45% of the calories in one meal.

The last line is the cause. With no grader the pipeline took the FIRST
candidate per slot — RecipeWrangler's deterministic order, which is planning
tier, then Nutri-Score, then curated source. A good tiebreak; not a decision.
And it has a second consequence: a fixed order plus "take the first" is a
function with one output, so asking twice or refining once returns the same
three dishes.

The critic runs on every plan, grader or no grader, and costs nothing — every
signal is already on the candidate. It reorders; it never drops. A pool it
emptied would turn a quality opinion into "no meals exist".
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe                    # noqa: E402
from services import plate_critic                            # noqa: E402


def _c(rid, title, kcal=None, grade=None):
    return CandidateRecipe(
        rid, title, "ingredients", "cook",
        nutrition={"kcal": kcal} if kcal is not None else None,
        nutri_score=grade,
    )


# The pool that produced the plan above, with a sane alternative behind each
# bad head — which is what makes the ordering the whole test.
REAL_POOL = {
    "breakfast": [
        _c("b1", "Easter breakfast baskets", kcal=226),
        _c("b2", "Oat pancakes with berries", kcal=480, grade="A"),
    ],
    "lunch": [
        _c("l1", "1 recipe = 4 dinners: Herby onion rice", kcal=899),
        _c("l2", "Lentil and squash soup", kcal=640, grade="A"),
    ],
    "dinner": [
        _c("d1", "Southwestern Stuffed Potatoes", kcal=859),
        _c("d2", "Grilled salmon with greens", kcal=720, grade="B"),
    ],
}


class TestTheDayThatWasServed:
    def test_every_slot_changes_its_pick(self):
        ranked, _ = plate_critic.rank_pool(REAL_POOL, kcal_target=2000)
        assert [ranked[s][0].recipe_id for s in ("breakfast", "lunch", "dinner")] == \
            ["b2", "l2", "d2"]

    def test_the_reasons_are_measured_not_asserted(self):
        _, findings = plate_critic.rank_pool(REAL_POOL, kcal_target=2000)
        joined = " ".join(findings)
        assert "226 kcal" in joined and "under" in joined
        assert "899 kcal" in joined and "over" in joined

    def test_nothing_is_dropped(self):
        """A pool the critic emptied would turn an opinion into "no meals
        exist"."""
        ranked, _ = plate_critic.rank_pool(REAL_POOL, kcal_target=2000)
        for slot, pool in REAL_POOL.items():
            assert len(ranked[slot]) == len(pool)
            assert {c.recipe_id for c in ranked[slot]} == {c.recipe_id for c in pool}


class TestTheCaloriesOfOneMeal:
    def test_a_slot_gets_its_share_of_the_day(self):
        """Breakfast is not a third of dinner by accident."""
        slots = ("breakfast", "lunch", "dinner")
        shares = {s: plate_critic.slot_share(s, slots) for s in slots}
        assert shares["dinner"] > shares["lunch"] > shares["breakfast"]
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_skipping_a_meal_still_adds_to_one_day(self):
        """A member who skips breakfast is not on a 75% diet."""
        shares = [plate_critic.slot_share(s, ("lunch", "dinner")) for s in ("lunch", "dinner")]
        assert sum(shares) == pytest.approx(1.0)

    def test_a_meal_near_its_share_is_not_criticised(self):
        verdict = plate_critic.critique("lunch", _c("x", "Soup", kcal=700), kcal_share=700)
        assert verdict.findings == [] and verdict.score >= 0

    def test_small_misses_are_tolerated(self):
        """Portions vary and stored figures are estimates. Reordering a pool
        over 8% is pretending to a precision we do not have."""
        verdict = plate_critic.critique("lunch", _c("x", "Soup", kcal=756), kcal_share=700)
        assert verdict.findings == []

    def test_no_target_means_no_calorie_opinion(self):
        """A dish marked down against a budget nobody chose is being marked
        down for someone else's number."""
        verdict = plate_critic.critique(
            "breakfast", _c("x", "Tiny thing", kcal=90), kcal_share=None,
        )
        assert verdict.findings == [] and verdict.score == 0.0

    def test_a_dish_with_no_macros_is_not_penalised_for_it(self):
        verdict = plate_critic.critique("lunch", _c("x", "Soup"), kcal_share=700)
        assert verdict.findings == []


class TestATitleThatIsAnArticle:
    @pytest.mark.parametrize("title", [
        "1 recipe = 4 dinners: Herby onion rice",
        "5 ways to use up a cabbage",
        "3 recipes from one roast chicken",
        "Batch cook: beef chilli",
        "Meal prep chicken bowls",
        "How to make the perfect omelette",
    ])
    def test_it_is_demoted(self, title):
        assert plate_critic.critique("lunch", _c("x", title)).score < 0

    @pytest.mark.parametrize("title", [
        "Herby onion rice",
        "Lentil and squash soup",
        "Southwestern Stuffed Potatoes",
        "Chicken with 40 cloves of garlic",
        "Three-cheese lasagne",
    ])
    def test_a_real_dish_name_is_not(self, title):
        assert plate_critic.critique("lunch", _c("x", title)).score == 0.0

    def test_it_is_a_penalty_not_a_rejection(self):
        """The recipe behind the headline is usually real. This is a workaround
        for scraped titles, and the proper fix is upstream."""
        pool = {"lunch": [_c("l1", "1 recipe = 4 dinners: Herby onion rice", kcal=700)]}
        ranked, _ = plate_critic.rank_pool(pool, kcal_target=2000)
        assert [c.recipe_id for c in ranked["lunch"]] == ["l1"]


class TestNutriScoreOnlySortsWhatIsAlreadyAllowed:
    def test_a_better_grade_wins_a_tie(self):
        pool = {"lunch": [_c("l1", "Stew", kcal=700, grade="D"),
                          _c("l2", "Stew two", kcal=700, grade="A")]}
        ranked, _ = plate_critic.rank_pool(pool, kcal_target=2000)
        assert ranked["lunch"][0].recipe_id == "l2"

    def test_it_does_not_outweigh_a_bad_portion(self):
        """One axis of one scoring system must not decide a plan on its own."""
        pool = {"lunch": [_c("l1", "Tiny A-grade thing", kcal=120, grade="A"),
                          _c("l2", "Proper lunch", kcal=700, grade="C")]}
        ranked, _ = plate_critic.rank_pool(pool, kcal_target=2000)
        assert ranked["lunch"][0].recipe_id == "l2"

    def test_an_ungraded_dish_is_neither_rewarded_nor_punished(self):
        assert plate_critic.critique("lunch", _c("x", "Soup", kcal=700), kcal_share=700).score == 0.0


class TestTiesKeepRecipeWranglersOrder:
    def test_a_stable_sort(self):
        """Planning tier, then Nutri-Score, then curated source is a good
        tiebreak — it is only a bad decision."""
        pool = {"lunch": [_c(f"l{n}", f"Dish {n}", kcal=700) for n in range(6)]}
        ranked, _ = plate_critic.rank_pool(pool, kcal_target=2000)
        assert [c.recipe_id for c in ranked["lunch"]] == [f"l{n}" for n in range(6)]


class TestItRunsBeforeAnythingSelects:
    def test_the_pipeline_ranks_the_pool_before_grading(self):
        """Reordering beats overriding: the grader then ranks a pool whose head
        already fits the slot, and the fallback's "first" is a reasoned first."""
        import inspect

        from services.planning_pipeline import PlanningPipeline

        src = inspect.getsource(PlanningPipeline.generate)
        critic_at = src.find("plate_critic.rank_pool")
        grade_at = src.find("grade_daily_plans")
        assert critic_at != -1, "the pool is never critiqued"
        assert critic_at < grade_at, "the grader ranks the raw order"

    def test_the_arithmetic_is_logged_and_never_rendered(self):
        """"226 kcal is 54% under what breakfast should carry" is real, and it
        is internal. It belongs where someone debugging a pick will look for
        it, not on a member's plan — the member is owed the fact that this plan
        was not ranked, which `note` already carries."""
        import inspect

        from services.planning_pipeline import PlanningPipeline

        generate = inspect.getsource(PlanningPipeline.generate)
        assert 'logger.info("Plate critic' in generate

        assemble = inspect.getsource(PlanningPipeline._assemble_from_pool)
        assert "critic_findings" not in assemble
        assert "reasoning +=" in assemble  # the note still reaches the member
