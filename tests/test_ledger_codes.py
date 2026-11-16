"""Every row under a plan can be said in the member's language.

The rows' sentences are English ("1 of 1 days miss your 2000 kcal target —
day 1 is 997 kcal under"), and FoodChat's living labs read them in Hungarian
and Slovenian. Each row now also carries a code and the numbers in it, so the
UI can phrase it; the English stays as the fallback for anything uncoded.

`code`/`params` describe the row's sentence; `name_code`/`name_params` its
name, where the name is more than a check's fixed label.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from models.recipe import RecipeEnrichment                       # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan    # noqa: E402
from services import plan_verifier as pv                          # noqa: E402
from services import transparency                                 # noqa: E402


def _plan(*titles, kcal=None):
    meals = [
        Meal(slot, [MealCourse(f"r{i}", title, f"{title} beans", "cook",
                               nutrition={"kcal": kcal} if kcal else None)])
        for i, (slot, title) in enumerate(zip(("breakfast", "lunch", "dinner"), titles))
    ]
    return MealPlan.from_days([DayPlan(day=1, meals=meals)], "test")


REQUESTED = {
    "allergens": ["peanuts"], "diet": ["vegetarian"], "tags": ["high_protein"],
    "kcal_target": 2000, "max_minutes": 30, "min_nutri_score": "b",
    "pantry": ["broccoli", "spinach"], "anchors": {"dinner": "r9"},
    "cuisines": ["greek"],
}


class TestEveryCheckCarriesACode:
    def test_each_check_says_what_it_measured_as_a_code(self):
        plan = _plan("Smoothie", "Broccoli pasta", "Chili", kcal=330)
        rich = {
            "r0": RecipeEnrichment(recipe_id="r0", title="Smoothie", tags=["high_protein"],
                                   diet_tags=["vegetarian"], duration=10),
            "r1": RecipeEnrichment(recipe_id="r1", title="Broccoli pasta", tags=[],
                                   diet_tags=["vegetarian"], duration=45),
        }
        report = pv.verify(plan, REQUESTED, rich)
        names = {check.name for check in report.checks}
        assert {"allergens", "diet", "claims", "calories", "cooking time",
                "nutri-score", "pantry", "your picks", "taste preferences"} <= names
        for check in report.checks:
            assert check.code, f"{check.name} has no code"
            assert check.code.split(".")[0] in {
                "allergens", "diet", "claims", "kcal", "time", "nutri", "pantry",
                "picks", "tastes",
            }

    def test_the_numbers_in_the_sentence_are_in_the_params(self):
        """The production row: a day 997 kcal under the member's 2000."""
        plan = _plan("Smoothie", "Pasta", "Chili", kcal=334.33)
        check = pv.verify(plan, {"kcal_target": 2000}, {}).get("calories")
        assert check.code == "kcal.miss"
        assert check.params["direction"] == "under"
        assert check.params["target"] == 2000
        assert check.params["own"] is True
        assert 990 <= check.params["delta"] <= 1000
        assert check.params["day"] == 1

    def test_claims_carry_the_tags_not_their_english(self):
        plan = _plan("A", "B", "C")
        rich = {f"r{i}": RecipeEnrichment(recipe_id=f"r{i}", title="x", tags=[]) for i in range(3)}
        check = pv.verify(plan, {"tags": ["high_protein"]}, rich).get("claims")
        assert check.code == "claims.none"
        assert check.params == {"observed": 3, "tags": ["high_protein"]}

    def test_rows_carry_code_and_params_and_stay_json(self):
        plan = _plan("Smoothie", "Pasta", "Chili", kcal=330)
        rows = pv.verify(plan, REQUESTED, {}).as_ledger_rows()
        assert all(row.get("code") for row in rows)
        json.dumps(rows)  # what the API sends
        # The English sentence is still there for anything that reads it.
        assert all(row["detail"] for row in rows)


class TestProfileRulesCarryANameCode:
    def test_allergies_dislikes_diets_and_goals(self):
        rows = transparency.constraints_ledger({
            "allergies": ["peanuts"], "food_dislikes": ["mushrooms"],
            "diet": ["omnivore"], "dietary_goals": ["weight_loss"],
        }, downvoted_count=2)
        by_code = {row.get("name_code"): row for row in rows}
        assert by_code["allergy"]["name_params"] == {"allergen": "peanuts"}
        assert by_code["dislike"]["name_params"] == {"item": "mushrooms"}
        assert by_code["diet"]["name_params"] == {"diet": "omnivore"}
        assert by_code["goal"]["name_params"] == {"goal": "weight_loss"}
        assert by_code["downvoted"]["name_params"] == {"count": 2}
        # "omnivore" filters nothing, and its sentence says so — coded too.
        assert by_code["diet"]["code"] == "diet.notAFilter"
        json.dumps(rows)
