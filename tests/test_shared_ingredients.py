"""Two different reasons a meal is on the plan, told apart.

A member who said *"I already have tomatoes"* and a plan that bought one
cabbage for two meals are not the same story, and a member cannot act on
them the same way. The first is theirs — they can check the fridge. The
second is the planner's, about ingredients nobody mentioned.

So they get different chips and different ledger rows, and neither is
allowed to claim the other's ground: an item the member named never
carries the cross-day chip, and cross-day reuse is never attributed to
"your pantry".
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from services.pantry_service import annotate_weekly_entries  # noqa: E402
from services.weekly_planner.explainability import (  # noqa: E402
    SHARED_INGREDIENT_KIND,
    annotate_shared_ingredients,
    shared_ingredient_facts,
)


def _entry(day, meal_type, title, ingredients):
    return {
        "day": day, "meal_idx": 0, "meal_type": meal_type,
        "recipe": {
            "recipe_id": f"{day}-{meal_type}", "recipe_title": title,
            "recipe_ingredients": ingredients,
        },
    }


def _kinds(entry):
    return [r.get("kind") for r in (entry["recipe"].get("match_reasons") or [])]


def _labels(entry, kind):
    return [
        r["label"] for r in (entry["recipe"].get("match_reasons") or [])
        if r.get("kind") == kind
    ]


class TestMeasurement:
    def test_a_later_meal_is_credited_against_the_earlier_day(self):
        entries = [
            _entry(1, "dinner", "Braise", "half a cabbage, walnuts"),
            _entry(3, "lunch", "Slaw", "cabbage, apple"),
        ]

        facts = shared_ingredient_facts(entries)

        assert facts["meals"] == 1
        assert facts["entries"][1] == [("cabbage", 1)]
        assert 0 not in facts["entries"], "the first appearance has nothing to point back to"

    def test_the_same_day_is_not_cross_day_reuse(self):
        """Lunch and dinner on one day share a shopping trip, not a saving."""
        entries = [
            _entry(2, "lunch", "Soup", "cabbage, leeks"),
            _entry(2, "dinner", "Braise", "cabbage, walnuts"),
        ]

        assert shared_ingredient_facts(entries)["meals"] == 0

    def test_singular_and_plural_are_the_same_ingredient(self):
        entries = [
            _entry(1, "dinner", "Roast", "2 tomatoes, thyme"),
            _entry(4, "lunch", "Tart", "one tomato, olives"),
        ]

        assert shared_ingredient_facts(entries)["entries"][1] == [("tomatoes", 1)]

    def test_staples_are_not_a_saving(self):
        """Two meals sharing salt and oil have saved nobody anything."""
        entries = [
            _entry(1, "lunch", "A", "olive oil, salt, pepper, rice"),
            _entry(3, "dinner", "B", "olive oil, salt, pasta"),
        ]

        assert shared_ingredient_facts(entries)["meals"] == 0

    def test_a_share_it_cannot_name_is_not_counted(self):
        """The tokeniser splits on whitespace, so "green beans, bay leaf,
        balsamic vinegar" yields "green", "leaf" and "balsamic". Harmless
        when ranking; unusable in a chip. Observed on a real plan: *"also
        uses Monday's green"*, *"Wednesday's brown"*, *"Monday's leaf"*.
        """
        entries = [
            _entry(1, "lunch", "A", "green salad, bay leaf, balsamic vinegar"),
            _entry(3, "dinner", "B", "green olives, bay leaf, balsamic glaze"),
        ]

        assert shared_ingredient_facts(entries)["meals"] == 0

    def test_a_nameable_share_alongside_noise_names_only_the_real_one(self):
        entries = [
            _entry(1, "lunch", "A", "green capsicum, bay leaf"),
            _entry(3, "dinner", "B", "green capsicum, bay leaf"),
        ]

        assert shared_ingredient_facts(entries)["entries"][1] == [("capsicum", 1)]

    def test_an_ingredient_nobody_repeats_is_not_reported(self):
        entries = [
            _entry(1, "lunch", "A", "lentils, spinach"),
            _entry(3, "dinner", "B", "chickpeas, courgette"),
        ]

        assert shared_ingredient_facts(entries) == {"entries": {}, "meals": 0, "items": []}


class TestTheTwoChipsStayApart:
    """The distinction, stated as the thing that must never blur."""

    ENTRIES = [
        _entry(1, "dinner", "Roast", "2 tomatoes, cabbage, thyme"),
        _entry(3, "lunch", "Slaw", "cabbage, apple"),
        _entry(5, "dinner", "Tart", "tomato, olives"),
    ]

    @staticmethod
    def _annotated(pantry=("tomatoes",)):
        entries = [
            _entry(e["day"], e["meal_type"], e["recipe"]["recipe_title"],
                   e["recipe"]["recipe_ingredients"])
            for e in TestTheTwoChipsStayApart.ENTRIES
        ]
        explainability = {"constraints_applied": []}
        annotate_weekly_entries(entries, pantry, explainability=explainability)
        annotate_shared_ingredients(entries, pantry, explainability=explainability)
        return entries, explainability

    def test_a_member_stated_item_gets_the_pantry_chip(self):
        entries, _ = self._annotated()

        assert "pantry" in _kinds(entries[0])
        assert _labels(entries[0], "pantry") == [
            "uses your tomatoes — reducing food waste"
        ]

    def test_a_member_stated_item_never_gets_the_cross_day_chip(self):
        """Day 5's tomato repeats day 1's — but the member named tomatoes,
        so it stays their story, not the planner's."""
        entries, _ = self._annotated()

        assert "pantry" in _kinds(entries[2])
        assert SHARED_INGREDIENT_KIND not in _kinds(entries[2])

    def test_an_ingredient_nobody_mentioned_gets_the_cross_day_chip(self):
        entries, _ = self._annotated()

        assert _labels(entries[1], SHARED_INGREDIENT_KIND) == [
            "also uses Monday's cabbage — reducing food waste"
        ]

    def test_the_cross_day_chip_names_the_day_it_came_from(self):
        entries, _ = self._annotated()
        label = _labels(entries[1], SHARED_INGREDIENT_KIND)[0]

        assert "Monday" in label and "Wednesday" not in label

    def test_the_two_ledger_rows_are_attributed_differently(self):
        _, explainability = self._annotated()
        rows = {r["source"]: r for r in explainability["constraints_applied"]}

        assert "your pantry" in rows, "the member's own items"
        assert "the plan" in rows, "what the planner introduced"
        assert "reuse an ingredient from an earlier day" in rows["the plan"]["constraint"]
        assert "on-hand" in rows["your pantry"]["constraint"]

    def test_the_plan_row_reports_how_many_meals_it_measured(self):
        _, explainability = self._annotated()
        row = next(
            r for r in explainability["constraints_applied"] if r["source"] == "the plan"
        )

        assert row["constraint"].startswith("1 meal(s)")
        assert "cabbage" in row["detail"]

    def test_cross_day_reuse_is_reported_with_no_pantry_at_all(self):
        """It is the plan's doing, so it does not wait on the member."""
        entries, explainability = self._annotated(pantry=())

        assert _labels(entries[1], SHARED_INGREDIENT_KIND)
        sources = [r["source"] for r in explainability["constraints_applied"]]
        assert "the plan" in sources and "your pantry" not in sources


class TestAnnotationIsAdditive:
    def test_existing_chips_survive(self):
        entries = [
            _entry(1, "dinner", "Braise", "cabbage, walnuts"),
            _entry(3, "lunch", "Slaw", "cabbage, apple"),
        ]
        entries[1]["recipe"]["match_reasons"] = [{"kind": "profile", "label": "keep me"}]

        annotate_shared_ingredients(entries)

        assert _kinds(entries[1]) == ["profile", SHARED_INGREDIENT_KIND]

    def test_a_plan_with_no_reuse_adds_no_ledger_row(self):
        entries = [_entry(1, "lunch", "A", "lentils"), _entry(3, "lunch", "B", "quinoa")]
        explainability = {"constraints_applied": []}

        annotate_shared_ingredients(entries, (), explainability=explainability)

        assert explainability["constraints_applied"] == []
