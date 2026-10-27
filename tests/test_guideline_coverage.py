"""
Guidelines that shape the plan, not only grade it.

The judge, on a real plan:

    The plan lacks wholemeal cereals/breads at each meal (violates G8) and
    provides far fewer than the 3-5 daily servings of whole grains required
    (G57). No dairy servings are included, breaching G5/G11. … the use of
    vegetable oil and olive oil appears excessive and is not limited to the
    1 tsp per person guideline (G14, G40, G60).

All true, and none of it reachable. **Every guideline call site ran after a
plan existed** — `chat_service`, `weekly_plan_service`,
`explainability.guideline_checklist`, `plan_scorer` — so the planner was
marked down for rules nothing had told it.

The pieces were on both sides and nothing joined them: RecipeWrangler filters
on 13 food groups and returns each recipe's own on every card, `PlanSpec`
carries a per-slot food-group request, and a catalogue rule carries
`food_groups` of its own. The two vocabularies meet on five values — and they
are exactly the families the judge cited.

Where they do not meet is the point of half these tests. A per-person oil
allowance cannot be measured from a corpus that stores ingredient text and no
portions, so it is reported as unverifiable rather than searched for, scored,
or quietly dropped.
"""

from __future__ import annotations

import sys
from unittest import mock



sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                              # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan     # noqa: E402
from services import guideline_coverage as gc                      # noqa: E402


def rule(urn, group, text, kind="food_based"):
    return {"id": urn, "guideline_type": kind, "food_groups": [group],
            "rule_text": text, "topic": []}


G8 = rule("G8", "grains", "Include wholemeal cereals and breads at each meal.")
G57 = rule("G57", "grains", "Aim for 3-5 servings of wholegrains per day.")
G5 = rule("G5", "dairy", "Have 3 servings of milk, yogurt and cheese a day.")
G56 = rule("G56", "vegetables", "Eat at least 5 servings of vegetables, salad and fruit a day.")
G6 = rule("G6", "protein_foods", "Choose 2 servings a day from meat, poultry, fish, eggs and beans.")
G14 = rule("G14", "fats_and_oils", "Limit to 1 teaspoon of oil per person.")
G59 = rule("G59", "beverages", "Drink at least 8 cups of fluid a day.")
ALL = [G8, G57, G5, G56, G6, G14, G59]


def course(recipe_id, title, groups):
    return MealCourse(recipe_id=recipe_id, title=title, ingredients="x",
                      directions="y", food_groups=tuple(groups))


def day(*meals) -> MealPlan:
    return MealPlan.from_days(
        [DayPlan(day=1, meals=[Meal(slot, [course(f"r-{slot}", title, groups)])
                               for slot, title, groups in meals])],
        reasoning="",
    )


THE_REPORTED_PLAN = day(
    ("breakfast", "Spiced lentil fritters", ["legumes", "vegetables"]),
    ("lunch", "Barley salad with feta", ["grains", "vegetables", "dairy"]),
    ("dinner", "Feta-crusted salmon", ["fish", "dairy"]),
)


def by_urn(rows: list[dict]) -> dict:
    return {r["urn"]: r for r in rows}


# ── which rules the corpus can act on ────────────────────────────────────

class TestWhatIsReachable:
    def test_the_families_the_judge_cited_become_targets(self):
        urns = {t.urn for t in gc.from_rules(ALL)}
        assert urns == {"G8", "G57", "G5", "G56", "G6"}

    def test_oil_and_fluids_do_not(self):
        """Not because they were missed — because nothing can measure them."""
        assert {"G14", "G59"} & {t.urn for t in gc.from_rules(ALL)} == set()

    def test_and_they_say_why_instead_of_disappearing(self):
        rows = by_urn(gc.unverifiable(ALL))
        assert "portions" in rows["G14"]["detail"]
        assert "no drinks" in rows["G59"]["detail"]
        assert rows["G14"]["met"] is None and rows["G59"]["met"] is None

    def test_protein_is_a_family_not_one_group(self):
        """"two servings from meat, poultry, fish, eggs or beans" is met by
        any of them; reading it as a single group is why the older counter
        dismissed this field as too coarse to use."""
        protein = next(t for t in gc.from_rules(ALL) if t.urn == "G6")
        assert {"meat", "poultry", "fish", "legumes", "eggs"} <= set(protein.groups)

    def test_a_nutrient_rule_is_not_a_food_group_target(self):
        nutrient = rule("G99", "grains", "Keep saturated fat under 10% of energy",
                        kind="nutrient_based")
        assert gc.from_rules([nutrient]) == []

    def test_a_rule_naming_two_groups_is_left_alone(self):
        """It is describing a meal, not setting two demands — splitting it
        would invent a requirement it never made."""
        both = {"id": "G0", "guideline_type": "food_based",
                "food_groups": ["dairy", "grains"],
                "rule_text": "Have cereal with milk at breakfast.", "topic": []}
        assert gc.from_rules([both]) == []


# ── shaping the fetch ────────────────────────────────────────────────────

class TestShapingTheSearch:
    def test_an_each_meal_rule_reaches_every_slot(self):
        spec, notes = gc.shape(PlanSpec(), gc.from_rules([G8]))
        assert set(spec.slot_food_groups) == set(PlanSpec().meals)
        assert all(g == ("grains",) for g in spec.slot_food_groups.values())
        assert notes

    def test_and_therefore_the_request(self):
        spec, _ = gc.shape(PlanSpec(), gc.from_rules([G8]))
        assert all(
            entry.get("food_groups") == ["grains"]
            for entry in spec.to_request_slots()
        )

    def test_a_counted_rule_does_not_become_a_filter(self):
        """"3-5 servings a day" as a per-slot filter demands grains at five
        meals to satisfy a floor of three, and crowds out what the member
        asked for. It is measured afterwards instead."""
        spec, notes = gc.shape(PlanSpec(), gc.from_rules([G57, G5, G56]))
        assert spec.slot_food_groups == {}
        assert notes == []

    def test_the_members_own_taste_is_kept(self):
        """`with_food_groups` merges. A guideline adds to what they asked for
        and never replaces it — the member outranks the guideline."""
        asked = PlanSpec().with_food_groups("lunch", ["fruit"])
        spec, _ = gc.shape(asked, gc.from_rules([G8]))
        assert set(spec.slot_food_groups["lunch"]) == {"fruit", "grains"}

    def test_the_facet_funnel_carries_it(self):
        """One seam, so all five fetch sites get it without five edits."""
        with mock.patch("services.guidelines_service.fetch", return_value=[G8, G57]):
            assert gc.per_meal_groups({}) == ["grains"]

    def test_a_catalogue_that_cannot_answer_shapes_nothing(self):
        with mock.patch("services.guidelines_service.fetch", side_effect=RuntimeError("down")):
            assert gc.per_meal_groups({}) == []


class TestItReachesTheRealFetch:
    """`per_meal_groups` passing its own tests proves nothing if no fetch
    ever asks for it."""

    def test_effective_facets_includes_the_guideline_group(self):
        from services import intent_facets

        with mock.patch("services.guidelines_service.fetch", return_value=[G8]):
            facets = intent_facets.effective_facets({})
        assert "grains" in (facets.get("food_groups") or [])

    def test_behind_the_members_own_words(self):
        """Ahead of it in the list is where the member's own choices go, and
        RecipeWrangler drops `food_groups` before `cuisines` when a slot
        cannot be filled — so a guideline gives way, not the member."""
        from services import intent_facets

        with mock.patch("services.guidelines_service.fetch", return_value=[G8]):
            facets = intent_facets.effective_facets({"_facets": {"food_groups": ["fruit"]}})
        assert facets["food_groups"] == ["fruit", "grains"]


# ── measuring what came back ─────────────────────────────────────────────

class TestMeasuringThePlan:
    def test_the_reported_plan_is_short_on_grains(self):
        rows = by_urn(gc.measure(THE_REPORTED_PLAN, gc.from_rules([G8])))
        assert rows["G8"]["met"] is False
        assert rows["G8"]["actual"] == 1

    def test_and_it_counts_meals_not_servings(self):
        """The corpus has recipes, not portions. Calling a meal a serving
        would be a measurement wearing the wrong name."""
        rows = by_urn(gc.measure(THE_REPORTED_PLAN, gc.from_rules([G57])))
        assert "meals" in rows["G57"]["unit"]

    def test_grains_says_it_cannot_see_wholemeal(self):
        rows = by_urn(gc.measure(THE_REPORTED_PLAN, gc.from_rules([G8])))
        assert "does not distinguish wholemeal" in rows["G8"]["detail"]

    def test_a_bare_count_is_reported_and_not_scored(self):
        """"Choose 2 servings a day" does not say three is wrong — the same
        line `guidelines_service.split` takes. The count still shows."""
        rows = by_urn(gc.measure(THE_REPORTED_PLAN, gc.from_rules([G6])))
        assert rows["G6"]["met"] is None
        # Breakfast (legumes) and dinner (fish); the barley salad carries none
        # of the protein family.
        assert rows["G6"]["actual"] == 2
        assert "not a bound" in rows["G6"]["detail"]

    def test_a_floor_is_scored(self):
        rows = by_urn(gc.measure(THE_REPORTED_PLAN, gc.from_rules([G56])))
        assert rows["G56"]["met"] is False and rows["G56"]["actual"] == 2

    def test_a_plan_meeting_it_passes(self):
        veg = day(*[(slot, f"Veg {slot}", ["vegetables"]) for slot in
                    ("breakfast", "lunch", "dinner", "snack", "dessert")])
        rows = by_urn(gc.measure(veg, gc.from_rules([G56])))
        assert rows["G56"]["met"] is True

    def test_a_plan_with_no_food_group_data_is_not_a_failure(self):
        """The bug this codebase keeps finding, in its other direction: a plan
        marked down for missing data is as wrong as one marked satisfied."""
        bare = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("lunch", [MealCourse("r", "Mystery", "", "")]),
        ])], reasoning="")
        rows = by_urn(gc.measure(bare, gc.from_rules([G8, G56])))
        assert all(r["met"] is None for r in rows.values())
        assert "no dish" in rows["G56"]["detail"]

    def test_the_multi_day_floor_scales_with_the_days(self):
        two = MealPlan.from_days([
            DayPlan(day=n, meals=[Meal("lunch", [course("l", "Salad", ["vegetables"])])])
            for n in (1, 2)
        ], reasoning="")
        rows = by_urn(gc.measure(two, gc.from_rules([G56])))
        # 2 vegetable meals against 5 a day over 2 days is still short.
        assert rows["G56"]["met"] is False


# ── what the member reads ────────────────────────────────────────────────

class TestTheLedger:
    @staticmethod
    def _rows(plan=THE_REPORTED_PLAN, rules=None):
        with mock.patch("services.guidelines_service.fetch", return_value=rules or ALL):
            return gc.ledger_rows(plan, {})

    def test_a_missed_guideline_is_relaxed_not_violated(self):
        """A guideline is not a rule the member set. A red chip for it is the
        weekly meat limit's mistake — a house standard reported as their own
        broken promise."""
        rows = self._rows()
        statuses = {r["status"] for r in rows}
        assert "relaxed" in statuses and "violated" not in statuses

    def test_every_row_is_soft(self):
        assert all(r["type"] == "soft" for r in self._rows())

    def test_the_unverifiable_ones_are_present_and_grey(self):
        oil = next(r for r in self._rows() if "teaspoon" in r["constraint"])
        assert oil["status"] == "unsupported"
        assert "portions" in oil["detail"]

    def test_the_detail_carries_the_count(self):
        grains = next(r for r in self._rows() if "wholemeal" in r["constraint"])
        assert "1 meal," in grains["detail"]
        assert "against every meal" in grains["detail"]

    def test_no_catalogue_means_no_rows_rather_than_empty_claims(self):
        with mock.patch("services.guidelines_service.fetch", return_value=[]):
            assert gc.ledger_rows(THE_REPORTED_PLAN, {}) == []

    def test_a_catalogue_outage_is_not_a_failed_turn(self):
        with mock.patch("services.guidelines_service.fetch", side_effect=RuntimeError("down")):
            assert gc.ledger_rows(THE_REPORTED_PLAN, {}) == []


class TestTheDailyPathAttachesThem:
    """Measuring passing its own tests proves nothing if no plan carries it."""

    def test_the_call_site_exists(self):
        from pathlib import Path

        source = Path("src/services/chat_service.py").read_text()
        assert "guideline_coverage.ledger_rows(meal_plan, profile)" in source


# ── the plumbing the measurement rests on ────────────────────────────────

class TestFoodGroupsReachThePlate:
    def test_a_candidate_carries_them(self):
        from services.plan_client import PlanClient

        spec = PlanSpec()
        envelope = {"days": [{"day": 1, "slots": [
            {"slot": slot, "recipes": [{
                "recipe_id": f"r-{slot}", "title": slot.title(),
                "food_groups": ["grains", "Dairy", "grains", ""],
            }]}
            for slot in spec.meals
        ]}]}
        pools = PlanClient.to_role_pools(envelope, spec)
        candidate = pools[1][("breakfast", "main")][0]
        # Deduplicated and lower-cased; blanks dropped.
        assert candidate.food_groups == ("grains", "dairy")

    def test_a_plate_keeps_them(self):
        from models.recipe import CandidateRecipe

        plate = MealCourse.from_candidate(
            CandidateRecipe("r", "Porridge", "oats", "cook", food_groups=("grains",))
        )
        assert plate.food_groups == ("grains",)

    def test_and_survives_being_stored(self):
        """A plan reloaded from the database must still be measurable — the
        alternative is refetching every plate for data the card already gave
        us."""
        from services.session_service import (
            _deserialize_meal_plan,
            _serialize_meal_plan,
        )

        stored = _serialize_meal_plan(THE_REPORTED_PLAN)
        back = _deserialize_meal_plan(THE_REPORTED_PLAN.id, stored)
        assert back.days[0].meals[1].plates[0].food_groups == (
            "grains", "vegetables", "dairy",
        )

    def test_a_plate_with_none_serializes_as_it_always_did(self):
        """Additive, like `role`: a legacy course's payload is unchanged."""
        from services.session_service import _serialize_course

        assert "food_groups" not in _serialize_course(
            MealCourse("r", "Toast", "bread", "toast it")
        )

    def test_a_recipe_without_them_is_not_invented(self):
        from services.plan_client import _food_groups

        assert _food_groups({}) == ()
        assert _food_groups({"food_groups": None}) == ()

    def test_the_weekly_path_attaches_them_too(self):
        """Weekly produces 21 meals and had nothing measuring food-group
        coverage across them. No overlap with the checklist already in
        `metrics`: that counts fish/red meat/poultry, this counts the corpus's
        food groups."""
        from pathlib import Path

        source = Path("src/services/weekly_plan_service.py").read_text()
        assert "guideline_coverage.ledger_rows(" in source


# ── whom a rule is for ───────────────────────────────────────────────────
#
# Live, an Irish member's daily plan carried every audience's version of the
# same rule as its own chip — active women, active men, inactive women,
# inactive men, active adults, inactive adults — beside two wordings of the
# dairy rule from two guides. These are those rules, verbatim.

from services import guideline_coverage as gc  # noqa: E402

AUDIENCE_RULES = [
    "Active women aged 19-50 should have 4-5 servings a day of wholemeal cereals and breads, potatoes, pasta and rice.",
    "Active men aged 19-50 should have 5-7 servings a day of wholemeal cereals and breads, potatoes, pasta and rice.",
    "Inactive women aged 19-50 should have 3-4 servings a day of wholemeal cereals and breads, potatoes, pasta and rice.",
    "Inactive men aged 19-50 should have 4-6 servings a day of wholemeal cereals and breads, potatoes, pasta and rice.",
    "Active adults aged 19–50 should have 4–5 servings a day of wholemeal cereals and breads, potatoes, pasta, and rice, increasing to 5–7 servings a day for men.",
    "Inactive adults aged 19–50 should have 3–4 servings a day of wholemeal cereals and breads, potatoes, pasta, and rice, increasing to 4–6 servings a day for men.",
    "Have 3 servings a day of milk, yogurt and cheese.",
    "Include wholemeal cereals and breads, potatoes, pasta, or rice at each meal.",
]


def _texts(rules):
    return [r["rule_text"] for r in rules]


class TestWhomARuleIsFor:
    def rules(self):
        return [{"rule_text": text} for text in AUDIENCE_RULES]

    def test_nobody_gets_an_activity_rule_while_activity_is_unknown(self):
        kept = _texts(gc.applicable(self.rules(), {}))
        assert kept == [
            "Have 3 servings a day of milk, yogurt and cheese.",
            "Include wholemeal cereals and breads, potatoes, pasta, or rice at each meal.",
        ]

    def test_a_stated_activity_and_sex_select_their_own_rule(self):
        profile = {"sex": "female", "nutritional_preferences": {"activity_level": "active"}}
        kept = _texts(gc.applicable(self.rules(), profile))
        assert AUDIENCE_RULES[0] in kept  # active women
        assert AUDIENCE_RULES[4] in kept  # active adults
        for other in (AUDIENCE_RULES[1], AUDIENCE_RULES[2], AUDIENCE_RULES[3], AUDIENCE_RULES[5]):
            assert other not in kept

    def test_a_rule_for_one_sex_needs_the_profile_to_say_it(self):
        rule = {"rule_text": "Women should have 2 servings of oily fish a week."}
        assert not gc.applies_to(rule, {})
        assert gc.applies_to(rule, {"sex": "female"})
        assert not gc.applies_to(rule, {"sex": "male"})

    def test_a_rule_that_names_nobody_always_applies(self):
        assert gc.applies_to({"rule_text": "Use fats, spreads and oils in very small amounts."}, {})
        assert gc.applies_to({"rule_text": "Adults should eat fish twice a week."}, {})


class TestOneTargetPerThingMeasured:
    def target(self, rule, source="guide A"):
        return gc.CoverageTarget(
            rule=rule, groups=("dairy",), direction="about", low=3, high=3,
            period="day", per_meal=False, source=source, urn="", guideline_group="dairy",
        )

    def test_two_guides_saying_the_same_thing_show_once(self):
        first = self.target("Have 3 servings a day of milk, yogurt and cheese.")
        second = self.target("Eat 3 servings a day of milk, yogurt, and cheese.", "guide B")
        assert gc._distinct([first, second]) == [first]

    def test_a_different_amount_is_a_different_target(self):
        three = self.target("three a day")
        two = gc.CoverageTarget(**{**three.__dict__, "low": 2, "high": 2, "rule": "two a day"})
        assert gc._distinct([three, two]) == [three, two]
