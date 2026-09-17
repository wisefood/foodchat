"""Cooking from what you have: coverage is a property of the COMBINATION.

The pantry reached the daily pool (merged coverage-first) and then the grader in
prose — "prefer combinations that together use as many of them as possible". But
the combinations the grader scores are SAMPLED, one candidate per slot at
random, so whether the member's aubergine was actually used came down to the
draw. Three dishes that each use the tomatoes cover one item; three that use
tomatoes, feta and basil cover three, and nothing measured the difference.

`best_covering_combo` computes a day that does, so the batch always contains
one. It does not overrule the grader — it gives the instruction something to
act on.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe                      # noqa: E402
from services.pantry_service import best_covering_combo        # noqa: E402


def _c(rid, ingredients, title=None):
    return CandidateRecipe(
        recipe_id=rid, title=title or rid, ingredients=ingredients,
        directions="cook",
    )


PANTRY = ("tomatoes", "feta", "basil")


class TestItPicksForCoverage:
    def test_it_spreads_across_the_pantry_rather_than_repeating_one_item(self):
        """Every candidate in slot 1 uses tomatoes; the choice that matters is
        whether slots 2 and 3 bring the feta and the basil."""
        pools = [
            [_c("a1", "tomatoes, pasta"), _c("a2", "tomatoes, rice")],
            [_c("b1", "tomatoes, bread"), _c("b2", "feta, cucumber")],
            [_c("c1", "tomatoes, onion"), _c("c2", "basil, potato")],
        ]
        combo = best_covering_combo(pools, PANTRY)

        assert [c.recipe_id for c in combo] == ["a1", "b2", "c2"]

    def test_a_tie_falls_to_the_better_ranked_recipe(self):
        """Pools arrive in RecipeWrangler's order — planning tier, then
        Nutri-Score. Coverage breaks ties; it does not invent its own ranking."""
        pools = [[_c("first", "feta, olives"), _c("second", "feta, bread")]]

        assert best_covering_combo(pools, PANTRY)[0].recipe_id == "first"

    def test_it_counts_distinct_items_not_mentions(self):
        """A dish listing tomatoes twice covers one item."""
        pools = [
            [_c("many", "tomatoes, cherry tomatoes, tomato puree"),
             _c("two", "tomatoes, feta")],
        ]

        assert best_covering_combo(pools, PANTRY)[0].recipe_id == "two"

    def test_it_matches_the_way_the_rest_of_the_pantry_does(self):
        """Singular/plural is the matcher's job and it is shared, so "tomato"
        in a recipe still covers a pantry of "tomatoes"."""
        pools = [[_c("plural", "beans"), _c("singular", "one ripe tomato")]]

        assert best_covering_combo(pools, PANTRY)[0].recipe_id == "singular"


class TestItRefusesToGuess:
    def test_no_pantry_means_no_opinion(self):
        pools = [[_c("a", "tomatoes")]]
        assert best_covering_combo(pools, ()) is None

    def test_an_empty_slot_means_no_combination_exists(self):
        """One slot with nothing in it cannot produce a day, and returning a
        short combination would hand the grader a malformed one."""
        assert best_covering_combo([[_c("a", "tomatoes")], []], PANTRY) is None

    def test_nothing_matching_still_returns_a_whole_day(self):
        """Zero coverage is an answer. The day is still complete — the pantry
        is a preference, and a member who has feta does not stop eating."""
        pools = [[_c("a", "rice")], [_c("b", "oats")]]
        combo = best_covering_combo(pools, PANTRY)

        assert [c.recipe_id for c in combo] == ["a", "b"]


class TestTheGraderIsShownIt:
    def _grader(self):
        from agents import DocumentGrader

        return DocumentGrader.__new__(DocumentGrader)

    def test_the_covering_day_joins_the_batch(self, monkeypatch):
        """Asserted on the batch the model is handed, not on the reply."""
        from agents import DocumentGrader

        grader = DocumentGrader.__new__(DocumentGrader)
        grader.max_plans_to_score = 2
        seen_batches = []

        def fake_invoke(messages, config=None):
            seen_batches.append(messages)
            raise RuntimeError("stop after the batch is built")

        grader.grader = type("C", (), {"invoke": staticmethod(fake_invoke)})()

        candidates = {
            "breakfast": [_c("b-plain", "oats"), _c("b-feta", "feta, eggs")],
            "lunch": [_c("l-plain", "rice"), _c("l-basil", "basil, pasta")],
            "dinner": [_c("d-plain", "bread"), _c("d-tom", "tomatoes, beans")],
        }
        try:
            grader.grade_plans(
                "plan my day", candidates, {}, slots=("breakfast", "lunch", "dinner"),
                prefer_items=PANTRY,
            )
        except Exception:
            pass

        assert seen_batches, "the grader was never called"
        sent = str(seen_batches[0])
        for rid in ("b-feta", "l-basil", "d-tom"):
            assert rid in sent, f"{rid} — the covering day never reached the batch"
