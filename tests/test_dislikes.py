"""Dislikes in the words recipes use, and measured on the plan.

On 2026-11-17 a member who listed "fish" among eighteen dislikes was served
Tuna filo pies: RecipeWrangler phrase-matches `exclude_ingredients` on
ingredient names, and "fish" is not in "tuna". The plan's rule list restated
the first five dislikes, each "satisfied", without looking. These tests pin
the expansion, the measurement, and the end of the unmeasured rows.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                              # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan       # noqa: E402
from services import plan_verifier as pv                             # noqa: E402
from services.dislikes import dislike_hits, expand_dislikes          # noqa: E402


def _plan(*dishes):
    meals = [
        Meal(slot, [MealCourse(recipe_id=f"r{i}", title=title, ingredients=ingredients,
                               directions="cook", role="main")])
        for i, (slot, (title, ingredients)) in enumerate(zip(("breakfast", "lunch", "dinner"), dishes))
    ]
    return MealPlan.from_days([DayPlan(day=1, meals=meals)], "test")


class TestExpansion:
    def test_fish_reaches_the_fish_people_cook(self):
        words = expand_dislikes(["fish"])
        for species in ("tuna", "salmon", "cod", "sardine", "mackerel"):
            assert species in words

    def test_the_dislike_itself_is_always_kept_first(self):
        assert expand_dislikes(["fish"])[0] == "fish"
        assert expand_dislikes(["spinach"]) == ["spinach"]

    def test_hyphenated_and_plural_forms_are_both_sent(self):
        assert expand_dislikes(["almond-milk"])[:2] == ["almond-milk", "almond milk"]
        assert "chickpea" in expand_dislikes(["chickpeas"])

    def test_categories_that_are_not_allergens(self):
        assert "chicken" in expand_dislikes(["poultry"])
        assert "beef" in expand_dislikes(["red meat"]) and "chicken" not in expand_dislikes(["red meat"])
        assert {"prawn", "tuna"} <= set(expand_dislikes(["seafood"]))

    def test_the_allergen_groups_are_shared_not_copied(self):
        from services.candidates_client import ALLERGEN_SYNONYMS

        assert set(ALLERGEN_SYNONYMS["shellfish"]) <= set(expand_dislikes(["shellfish"]))

    def test_nothing_in_nothing_out(self):
        assert expand_dislikes(None) == [] and expand_dislikes(["", "  "]) == []

    def test_no_duplicates(self):
        words = expand_dislikes(["fish", "tuna", "Fish"])
        assert len(words) == len(set(words))


class TestMatchingADish:
    def test_tuna_filo_pies_has_fish(self):
        assert dislike_hits("tuna filo pies tuna, filo pastry, egg", ["spinach", "fish"]) == ["fish"]

    def test_whole_words_only(self):
        assert dislike_hits("hamburger with mincemeat tart", ["ham", "mince"]) == []
        assert dislike_hits("ham and cheese toastie", ["ham"]) == ["ham"]

    def test_plurals_match(self):
        assert dislike_hits("roasted mushrooms on toast", ["mushroom"]) == ["mushroom"]
        assert dislike_hits("a mushroom risotto", ["mushrooms"]) == ["mushrooms"]

    def test_the_member_s_own_word_is_reported(self):
        assert dislike_hits("Salmon fillet", ["Fish"]) == ["Fish"]


class TestTheVerifierMeasuresDislikes:
    def test_a_disliked_dish_is_named(self):
        plan = _plan(("Porridge", "oats, milk"), ("Salad", "lettuce"), ("Tuna filo pies", "tuna, filo pastry"))
        check = pv.verify(plan, {"dislikes": ["fish", "spinach"]}).get("dislikes")
        assert check.status == pv.FAILED and not check.is_blocking
        assert check.params["hits"] == [{"dish": "Tuna filo pies", "dislike": "fish"}]
        assert "Tuna filo pies has fish" in check.detail
        assert check.offenders == ("r2",)

    def test_a_clean_plan_says_how_much_was_checked(self):
        plan = _plan(("Toast", "bread"), ("Salad", "lettuce"), ("Goulash", "beef, paprika"))
        check = pv.verify(plan, {"dislikes": ["fish", "spinach"]}).get("dislikes")
        assert check.status == pv.PASSED
        assert check.params == {"count": 2, "dishes": 3}

    def test_no_dislikes_no_row(self):
        plan = _plan(("Toast", "bread"))
        assert pv.verify(plan, {"dislikes": []}).get("dislikes") is None

    def test_it_is_a_soft_row_measured_on_the_plan(self):
        plan = _plan(("Tuna melt", "tuna"))
        rows = pv.verify(plan, {"dislikes": ["fish"]}).as_ledger_rows()
        row = next(r for r in rows if r["constraint"] == "dislikes")
        assert (row["type"], row["status"], row["source"]) == ("soft", "violated", "measured on the plan")


class TestTheBriefCarriesThem:
    def test_dislikes_reach_what_the_verifier_checks(self):
        brief = PlanBrief.build({"food_dislikes": ["fish", "oats"]})
        assert brief.to_requested()["dislikes"] == ["fish", "oats"]


class TestTheSearchIsAskedProperly:
    def test_plan_meals_sends_the_expanded_list(self, monkeypatch):
        import httpx

        from services.plan_client import PLANNER

        sent = {}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                sent["payload"] = json
                return httpx.Response(200, json={"slots": {}}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "Client", FakeClient)
        try:
            PLANNER.plan_meals(days=1, slots=("dinner",), count_per_slot=3, exclude_ingredients=["fish"])
        except Exception:
            pass  # the empty envelope may not parse; the payload is what is tested
        assert "tuna" in sent["payload"]["exclude_ingredients"]
        assert sent["payload"]["exclude_ingredients"][0] == "fish"


class TestTheRuleListTellsTheTruth:
    """The profile's "avoiding X" rows stay (attribution, a name code) but no
    longer claim "satisfied" for a dish that has X."""

    def _rows(self, dislikes):
        from services.transparency import constraints_ledger

        return constraints_ledger({"food_dislikes": dislikes})

    def test_a_shown_dislike_that_slipped_through_says_so(self):
        plan = _plan(("Porridge", "oats, milk"), ("Salad", "lettuce"), ("Tuna filo pies", "tuna, filo"))
        report = pv.verify(plan, {"dislikes": ["fish", "spinach"]})
        rows = pv.merge(self._rows(["fish", "spinach"]), report)
        fish = next(r for r in rows if r.get("name_params", {}).get("item") == "fish")
        assert fish["status"] == "violated" and fish["params"] == {"dishes": ["Tuna filo pies"]}
        spinach = next(r for r in rows if r.get("name_params", {}).get("item") == "spinach")
        assert spinach["status"] == "satisfied"
        # One finding, said once: no summary row beside the per-item one.
        assert not [r for r in rows if r.get("constraint") == "dislikes"]

    def test_a_dislike_past_the_first_five_still_gets_a_row_when_broken(self):
        dislikes = ["spinach", "oats", "tofu", "seitan", "barley", "fish"]
        plan = _plan(("Toast", "bread"), ("Salad", "lettuce"), ("Salmon bake", "salmon"))
        rows = pv.merge(self._rows(dislikes), pv.verify(plan, {"dislikes": dislikes}))
        fish = [r for r in rows if r.get("name_params", {}).get("item") == "fish"]
        assert len(fish) == 1 and fish[0]["status"] == "violated"

    def test_a_clean_plan_keeps_its_rows_and_the_summary(self):
        plan = _plan(("Toast", "bread"), ("Salad", "lettuce"), ("Goulash", "beef"))
        rows = pv.merge(self._rows(["fish"]), pv.verify(plan, {"dislikes": ["fish"]}))
        assert next(r for r in rows if r.get("name_code") == "dislike")["status"] == "satisfied"
        assert next(r for r in rows if r.get("constraint") == "dislikes")["status"] == "satisfied"

    def test_merging_again_after_a_repair_starts_over(self):
        dislikes = ["spinach", "oats", "tofu", "seitan", "barley", "fish"]
        broken = _plan(("Toast", "bread"), ("Salad", "lettuce"), ("Salmon bake", "salmon"))
        rows = pv.merge(self._rows(dislikes), pv.verify(broken, {"dislikes": dislikes}))
        repaired = _plan(("Toast", "bread"), ("Salad", "lettuce"), ("Goulash", "beef"))
        kept = [r for r in rows if r.get("source") != "measured on the plan"]
        again = pv.merge(kept, pv.verify(repaired, {"dislikes": dislikes}))
        assert not [r for r in again if r.get("status") == "violated"]
        assert not [r for r in again if r.get("name_params", {}).get("item") == "fish"]

    def test_without_a_dislike_check_nothing_changes(self):
        plan = _plan(("Toast", "bread"))
        rows = self._rows(["fish"])
        assert pv.merge(rows, pv.verify(plan, {})) == rows
