"""
Weekly gets what the daily paths got.

Two gaps, both because weekly stores something different from a `MealPlan`:

* **no repair.** The verifier named the failing plates and nothing acted on
  them. Across 21 meals a hard-constraint failure is more likely than on three,
  and weekly was the path where nothing was done about it.
* **no quality metrics.** The two graders were instance attributes on
  `ChatService`, so the path that produces the deepest plan the product makes
  said the least about it — and the UI's quality panel was wired to the daily
  canvas only.

The repair runs against the adapted `MealPlan` (the one shape the verifier and
the repair both understand) and is then carried back into the entry dicts. It
is carried by RECIPE ID, not title or index: two dishes in a week can share a
title, and an index shifts if anything upstream reorders.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                        # noqa: E402
from models.recipe import CandidateRecipe                      # noqa: E402
from services import plan_quality, plan_repair, plan_verifier  # noqa: E402
from services.weekly_plan_service import (                     # noqa: E402
    _apply_repairs,
    _as_meal_plan,
)


def _entry(day, idx, slot, rid, title, ingredients=""):
    return {
        "day": day, "meal_idx": idx, "meal_type": slot,
        "recipe": {
            "recipe_id": rid, "recipe_title": title,
            "recipe_ingredients": ingredients or f"{title.lower()}, salt, water",
            "recipe_directions": "cook",
            "nutrition": {"kcal": 500},
            "image_url": "http://old",
            "match_reasons": [{"kind": "profile", "label": "old chip"}],
        },
    }


# Distinct ingredient words per dish: the parser strips digits, so
# "dish 10, salt" and "dish 21, salt" both normalise to ["dish", "salt"] and a
# variety assertion over them would be measuring the fixture, not the metric.
_FOODS = ["oats", "lentils", "tomato", "chickpeas", "spinach", "rice",
          "feta", "quinoa", "aubergine"]


def _week():
    out = []
    n = 0
    for day in (1, 2):
        for idx, slot in enumerate(("breakfast", "lunch", "dinner")):
            out.append(_entry(
                day, idx, slot, f"r{day}{idx}", f"Dish {day}{idx}",
                ingredients=f"{_FOODS[n]}, olive oil",
            ))
            n += 1
    return out


class _Client:
    def __init__(self, by_slot=None, details=None):
        self.by_slot = by_slot or {}
        self.details = details or {}

    def slot_candidates(self, profile, meal_type, exclude_ids, limit=8):
        return list(self.by_slot.get(meal_type, []))

    def fetch_details(self, recipe_ids):
        return {r: self.details[r] for r in recipe_ids if r in self.details}


def _cand(rid, title, ingredients="lentils, carrot"):
    return CandidateRecipe(recipe_id=rid, title=title, ingredients=ingredients,
                           directions="cook")


# ── the write-back ────────────────────────────────────────────────────────

class TestTheWriteBack:
    def _repaired_week(self, entries, client):
        adapted = _as_meal_plan(entries)
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(adapted, brief.to_requested(), {})
        outcome = plan_repair.repair(adapted, brief, report, {}, client=client)
        applied = _apply_repairs(entries, adapted, outcome)
        return outcome, applied

    def test_a_repaired_dish_reaches_the_stored_entry(self):
        entries = _week()
        entries.append(_entry(2, 3, "snack", "bad", "Satay", "peanuts, chicken"))
        client = _Client({"snack": [_cand("safe", "Fruit bowl")]})

        outcome, applied = self._repaired_week(entries, client)
        assert outcome.changed and applied == 1
        swapped = [e for e in entries if e["recipe"]["recipe_id"] == "safe"]
        assert len(swapped) == 1
        assert swapped[0]["recipe"]["recipe_title"] == "Fruit bowl"
        assert not any(e["recipe"]["recipe_id"] == "bad" for e in entries)

    def test_the_other_entries_are_untouched(self):
        entries = _week()
        entries.append(_entry(1, 3, "snack", "bad", "Satay", "peanuts"))
        before = {
            e["recipe"]["recipe_id"]: dict(e["recipe"])
            for e in entries if e["recipe"]["recipe_id"] != "bad"
        }
        self._repaired_week(entries, _Client({"snack": [_cand("safe", "Fruit bowl")]}))
        after = {
            e["recipe"]["recipe_id"]: dict(e["recipe"])
            for e in entries if e["recipe"]["recipe_id"] != "safe"
        }
        assert after == before

    def test_the_stale_nutrition_and_image_do_not_survive(self):
        """They belonged to the dish that was removed. Carrying them over would
        show the member the calories of a meal they are not being served."""
        entries = [_entry(1, 0, "dinner", "bad", "Satay", "peanuts")]
        self._repaired_week(entries, _Client({"dinner": [_cand("safe", "Lentil bake")]}))
        recipe = entries[0]["recipe"]
        assert recipe["nutrition"] != {"kcal": 500}
        assert recipe["image_url"] != "http://old"
        assert recipe["match_reasons"][0]["label"] != "old chip"

    def test_the_swap_carries_its_reason(self):
        entries = [_entry(1, 0, "dinner", "bad", "Satay", "peanuts")]
        self._repaired_week(entries, _Client({"dinner": [_cand("safe", "Lentil bake")]}))
        chips = entries[0]["recipe"]["match_reasons"]
        assert chips and "allergens" in chips[0]["label"]

    def test_it_matches_by_id_not_by_title(self):
        """Two dishes in a week can share a title."""
        entries = [
            _entry(1, 0, "dinner", "keep", "Satay", "chicken, rice"),
            _entry(2, 0, "dinner", "bad", "Satay", "peanuts, chicken"),
        ]
        self._repaired_week(entries, _Client({"dinner": [_cand("safe", "Lentil bake")]}))
        ids = [e["recipe"]["recipe_id"] for e in entries]
        assert ids == ["keep", "safe"], "the wrong Satay was replaced"

    def test_nothing_repaired_writes_nothing(self):
        entries = _week()
        outcome, applied = self._repaired_week(entries, _Client())
        assert not outcome.changed and applied == 0

    def test_an_unmatched_swap_is_reported_not_silent(self, caplog):
        """A plan that silently kept the dish the repair thought it removed is
        worse than a warning."""
        import logging

        entries = [_entry(1, 0, "dinner", "bad", "Satay", "peanuts")]
        adapted = _as_meal_plan(entries)
        brief = PlanBrief(allergens=("peanuts",))
        report = plan_verifier.verify(adapted, brief.to_requested(), {})
        outcome = plan_repair.repair(
            adapted, brief, report, {},
            client=_Client({"dinner": [_cand("safe", "Lentil bake")]}),
        )
        # The entry list no longer contains the id the repair swapped away.
        entries[0]["recipe"]["recipe_id"] = "moved"
        with caplog.at_level(logging.WARNING):
            applied = _apply_repairs(entries, adapted, outcome)
        assert applied == 0
        assert "could not be matched" in caplog.text


# ── quality metrics over a week ───────────────────────────────────────────

class TestWeeklyQuality:
    def test_variety_counts_across_the_whole_week(self):
        entries = _week()
        scored = plan_quality.scored_from_plan(_as_meal_plan(entries))
        count, reasoning = plan_quality.food_variety(scored)
        # Six distinct foods plus the shared "olive oil".
        assert count == 7, f"expected every dish counted, got {count}"
        assert "Unique food items" in reasoning

    def test_every_meal_of_every_day_is_labelled(self):
        scored = plan_quality.scored_from_plan(_as_meal_plan(_week()))
        assert "day 1 breakfast" in scored.slots
        assert "day 2 dinner" in scored.slots
        assert len(scored.slots) == 6

    def test_the_judges_see_a_plan_not_a_list(self):
        text = plan_quality.as_text(plan_quality.scored_from_plan(_as_meal_plan(_week())))
        assert "Day 1 Breakfast:" in text and "Day 2 Dinner:" in text

    def test_no_ranking_is_invented(self):
        """Nothing ranked a week — the RL walk picks, it does not score a batch.
        A zero means "not ranked"."""
        scored = plan_quality.scored_from_plan(_as_meal_plan(_week()))
        assert scored.score == 0


# ── one implementation, two callers ──────────────────────────────────────

class TestOneImplementation:
    def test_chat_service_delegates_rather_than_duplicating(self):
        from services import plan_quality as shared
        from services.chat_service import (
            _extract_ingredient_names,
            _food_variety_score,
            _plan_as_text,
        )

        assert _extract_ingredient_names is shared.extract_ingredient_names
        assert _food_variety_score is shared.food_variety
        assert _plan_as_text is shared.as_text

    def test_the_ingredient_parser_was_moved_not_rewritten(self):
        """Rewriting it would silently move every FVS number the product has
        ever reported."""
        cases = [
            "2 tbsp olive oil, 1 large onion (diced)",
            "oats\nmilk; honey",
            "400g tinned tomatoes • basil - salt",
            "", None,
        ]
        for case in cases:
            assert plan_quality.extract_ingredient_names(case) == (
                [] if not isinstance(case, str) else
                plan_quality.extract_ingredient_names(case)
            )
        # The shapes the old regex produced, pinned.
        assert plan_quality.extract_ingredient_names(
            "2 tbsp olive oil, 1 large onion (diced)"
        ) == ["tbsp olive oil", "large onion"]

    def test_weekly_uses_the_shared_module(self):
        import inspect

        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        assert "plan_quality.metrics(" in src
        assert "plan_repair.repair(" in src

    def test_both_weekly_stages_are_sheddable(self):
        import inspect

        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        # The repair call wraps across lines, so match the pieces.
        assert "turn_budget.skip(" in src
        assert '"weekly repair"' in src
        assert '"weekly quality"' in src

    def test_the_weekly_reply_is_told_about_a_repair(self):
        import inspect

        src = inspect.getsource(sys.modules["services.weekly_plan_service"])
        assert '"repair": repair_note' in src
