"""Cook once, eat twice — and say only what the plan can actually see.

Two changes are tested here, because they are one mechanism:

- **`repeat_meals`** (M10) makes the SCOPE of M9's cooldown the member's to
  set. "How often may a dinner recur before a week reads as lazy rather than
  familiar" is a household question, so it is asked rather than guessed. The
  gap and the cap stay in the planner: they are what stops a thin candidate
  pool from cashing the member's setting in for monotony.
- **Leftovers** — day N's dinner served as day N+1's lunch — are that same
  cooldown with a slot transition, one day instead of two, and a cap of their
  own. Not a new mechanism, and deliberately not a new entry kind: the entry
  holds the whole recipe, because the member really does eat that dish.

The thing this file guards hardest is the boundary of the claim. Nothing in
this service records quantities, purchase dates or cooking times, so the plan
may say "Monday's dinner again" and may not say "the rest of Monday's dinner",
"a double portion", or anything else that implies it knows what is in the
fridge. Every wording assertion below is that boundary.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.recipe import CandidateRecipe  # noqa: E402
from services import plan_parameters  # noqa: E402
from services.weekly_planner import action_adapter  # noqa: E402
from services.weekly_planner.action_adapter import (  # noqa: E402
    LEFTOVER_GAP_DAYS,
    MAX_APPEARANCES,
    MAX_LEFTOVER_MEALS,
    REPEAT_LEFTOVER,
    REPEAT_MIN_GAP_DAYS,
    RecipeActionSpace,
)
from services.weekly_planner.environment import WeeklyMealPlanEnv  # noqa: E402
from services.weekly_planner.explainability import (  # noqa: E402
    REPEAT_BY_LEFTOVER,
    REPEAT_KIND,
    build_weekly_explainability,
    repeat_facts,
    shared_ingredient_facts,
    variety_metrics,
)
from services.weekly_planner.planner import (  # noqa: E402
    IngredientBasket,
    WeeklyPlanner,
    build_preference_scorer,
)
from services.weekly_planner.reward_logic import RewardCalculator  # noqa: E402

BREAKFASTS = [(f"b{i}", f"Breakfast{i}") for i in range(9)]
LUNCHES = [(f"l{i}", f"Lunch{i}") for i in range(9)]
DINNERS = [(f"d{i}", f"Dinner{i}") for i in range(9)]

LEFTOVERS_ON = {"repeat_meals": "leftovers"}


def _cand(pair, ingredients="carrot, spinach"):
    return CandidateRecipe(
        recipe_id=pair[0], title=pair[1], ingredients=ingredients, directions="cook"
    )


@pytest.fixture
def offline(monkeypatch):
    """The real action space, with only its network edges stubbed."""
    calls = {"fetches": 0, "exclusions": []}

    def fake_pool(*, profile, allergens, diet, cuisines, exclude_recipe_ids,
                  limit_per_slot):
        excluded = set(exclude_recipe_ids or [])
        calls["fetches"] += 1
        calls["exclusions"].append(sorted(excluded))
        return {
            "breakfast": [_cand(p) for p in BREAKFASTS if p[0] not in excluded],
            "lunch": [_cand(p) for p in LUNCHES if p[0] not in excluded],
            "dinner": [_cand(p) for p in DINNERS if p[0] not in excluded],
        }

    monkeypatch.setattr(action_adapter, "_fetch_candidate_pool", fake_pool)
    monkeypatch.setattr(action_adapter.CANDIDATES, "fetch_details", lambda ids: {})
    monkeypatch.setattr(
        action_adapter.CANDIDATES, "split_cuisines", lambda likes: ([], list(likes or []))
    )
    from services import pantry_service

    monkeypatch.setattr(
        pantry_service, "fetch_pantry_candidates", lambda profile, pantry, **kw: {}
    )
    return calls


def _profile(**overrides) -> dict:
    profile = {"diet": [], "preferences": [], "plan_parameters": {}}
    profile.update(overrides)
    return profile


def _space(**profile) -> RecipeActionSpace:
    return RecipeActionSpace(_profile(**profile), additional_diet=[], pantry=())


def _serve(space, day, meal_type, pair, **extra):
    """Commit a recipe the way the environment does — payload included."""
    action = {
        "recipe_id": pair[0],
        "recipe_title": pair[1],
        "recipe_ingredients": "carrot, spinach",
        "recipe_directions": "cook",
        **extra,
    }
    space.mark_committed(pair[0], day, meal_type, action=action)
    return action


def _actions(space, day, meal_type):
    return space.get_candidate_actions(meal_type, {"day": day})


def _leftover(space, day, meal_type="lunch"):
    return next(
        (a for a in _actions(space, day, meal_type) if a.get("leftover_of")), None
    )


# --------------------------------------------------------------------- #
# The control                                                             #
# --------------------------------------------------------------------- #


class TestTheRepeatControl:
    """Which slots may repeat is the member's to say — and only that."""

    def test_the_default_is_exactly_what_the_planner_already_did(self):
        assert plan_parameters.repeat_mode({}) == "breakfast"
        assert plan_parameters.repeats_allowed({}, "breakfast")
        assert not plan_parameters.repeats_allowed({}, "dinner")

    def test_off_restores_the_rule_that_every_meal_is_different(self):
        values = {"repeat_meals": "off"}
        for slot in ("breakfast", "lunch", "dinner"):
            assert not plan_parameters.repeats_allowed(values, slot)

    def test_all_opens_lunch_and_dinner_too(self):
        values = {"repeat_meals": "all"}
        for slot in ("breakfast", "lunch", "dinner"):
            assert plan_parameters.repeats_allowed(values, slot)

    def test_each_stop_is_a_superset_of_the_one_before_it(self):
        """The scale only means something if it is monotone — otherwise
        "more repeats" is a different feature at every stop, not a dial."""
        allowed = [
            {
                slot for slot in ("breakfast", "lunch", "dinner")
                if plan_parameters.repeats_allowed({"repeat_meals": mode}, slot)
            }
            for mode in plan_parameters.REPEAT_MODES
        ]
        for narrower, wider in zip(allowed, allowed[1:]):
            assert narrower <= wider

    def test_leftovers_are_the_last_stop_and_nothing_below_it(self):
        for mode in ("off", "breakfast", "all"):
            assert not plan_parameters.leftovers_allowed({"repeat_meals": mode})
        assert plan_parameters.leftovers_allowed({"repeat_meals": "leftovers"})

    def test_an_unreadable_stored_value_falls_back_to_the_default(self):
        """A profile written by a future release, or a hand-edited one, must
        not silently turn the whole policy off — that is a change to the
        member's plan made by a typo."""
        assert plan_parameters.repeat_mode({"repeat_meals": "sometimes"}) == "breakfast"
        assert plan_parameters.repeat_mode({"repeat_meals": None}) == "breakfast"


class TestTheControlReachesTheActionSpace:
    def test_off_keeps_every_committed_recipe_out(self, offline):
        space = _space(plan_parameters={"repeat_meals": "off"})
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert "b0" not in [a["recipe_id"] for a in _actions(space, 4, "breakfast")]

    def test_all_lets_a_dinner_come_back_after_the_same_gap(self, offline):
        space = _space(plan_parameters={"repeat_meals": "all"})
        _serve(space, 1, "dinner", DINNERS[0])

        assert "d0" not in [a["recipe_id"] for a in _actions(space, 2, "dinner")]
        back = next(
            a for a in _actions(space, 1 + REPEAT_MIN_GAP_DAYS, "dinner")
            if a["recipe_id"] == "d0"
        )
        assert back["repeat_of_day"] == 1

    def test_all_still_does_not_move_a_dinner_to_breakfast(self, offline):
        """Widening WHICH slots may repeat never widened the claim that a
        repeat is the same meal again. The one cross-slot move this planner
        makes is the leftover, and it has its own rule."""
        space = _space(plan_parameters={"repeat_meals": "all"})
        _serve(space, 1, "dinner", DINNERS[0])

        assert "d0" not in [a["recipe_id"] for a in _actions(space, 4, "breakfast")]


# --------------------------------------------------------------------- #
# The leftover itself                                                     #
# --------------------------------------------------------------------- #


class TestWhenALeftoverIsOffered:
    def test_yesterdays_dinner_is_offered_as_todays_lunch(self, offline):
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])

        leftover = _leftover(space, 2)

        assert leftover["recipe_id"] == "d0"
        assert leftover["repeat_of_day"] == 1
        assert leftover["repeat_source"] == REPEAT_LEFTOVER
        assert leftover["leftover_of"] == {"day": 1, "meal_type": "dinner"}

    def test_it_is_the_dish_that_was_actually_served(self, offline):
        """Rebuilt from the commitment, not re-fetched: a leftover that
        merely resembles last night's dinner is a false sentence on a card."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0], nutrition={"kcal": 640.0},
               tags=["vegetarian"])

        leftover = _leftover(space, 2)

        assert leftover["recipe_title"] == "Dinner0"
        assert leftover["nutrition"] == {"kcal": 640.0}
        assert leftover["tags"] == ["vegetarian"]

    def test_nothing_is_offered_on_the_first_day(self, offline):
        space = _space(plan_parameters=LEFTOVERS_ON)

        assert _leftover(space, 1) is None

    def test_the_day_before_yesterday_is_not_a_leftover(self, offline):
        """Two days later is a repeat, and has different wording and a
        different gap. Anything longer would be a claim about a fridge, and
        nothing here records when a dish was cooked or how long it keeps."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])

        assert _leftover(space, 1 + LEFTOVER_GAP_DAYS) is not None
        assert _leftover(space, 1 + LEFTOVER_GAP_DAYS + 1) is None

    def test_lunch_never_becomes_the_next_days_dinner(self, offline):
        """The transition is stated in one direction only. Nobody saves half
        a sandwich for tomorrow's dinner."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "lunch", LUNCHES[0])

        assert _leftover(space, 2, "dinner") is None
        assert _leftover(space, 2, "breakfast") is None

    def test_the_setting_is_what_switches_it_on(self, offline):
        for mode in ("off", "breakfast", "all"):
            space = _space(plan_parameters={"repeat_meals": mode})
            _serve(space, 1, "dinner", DINNERS[0])
            assert _leftover(space, 2) is None, mode

    def test_a_pinned_or_downvoted_dish_still_has_no_way_back(self, offline):
        """`mark_selected` means "never again, anywhere" — a member's anchor
        turning up twice is the thing that call exists to prevent, and a
        leftover is a way back like any other."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        space.mark_selected("d0")
        _serve(space, 1, "dinner", DINNERS[0])

        assert _leftover(space, 2) is None

    def test_a_dish_that_has_had_its_two_servings_is_not_offered_again(self, offline):
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])
        _serve(space, 1 + REPEAT_MIN_GAP_DAYS, "dinner", DINNERS[0])

        assert len(space._commitments["d0"]) == MAX_APPEARANCES
        assert _leftover(space, 1 + REPEAT_MIN_GAP_DAYS + 1) is None

    def test_leftovers_are_capped_for_the_week(self, offline):
        """Six leftover lunches is a week where lunch is never cooked, which
        is a different product from the one the member asked for."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        taken = 0
        for day in range(1, 8):
            _serve(space, day, "dinner", DINNERS[day - 1])
            leftover = _leftover(space, day + 1)
            if leftover is not None:
                _serve(space, day + 1, "lunch", DINNERS[day - 1],
                       leftover_of=leftover["leftover_of"])
                taken += 1

        assert taken == MAX_LEFTOVER_MEALS

    def test_a_leftover_is_never_itself_reheated_a_third_time(self, offline):
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])
        leftover = _leftover(space, 2)
        _serve(space, 2, "lunch", DINNERS[0], leftover_of=leftover["leftover_of"])

        assert _leftover(space, 3) is None


class TestTheLunchIsStillPlanned:
    def test_the_leftover_is_added_to_the_pool_not_substituted_for_it(self, offline):
        """The member asked that leftovers be POSSIBLE, not that lunch stop
        being planned. If the leftover loses on score, nothing is lost."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])

        actions = _actions(space, 2, "lunch")

        assert len(actions) == len(LUNCHES) + 1
        assert {a["recipe_id"] for a in actions} >= {p[0] for p in LUNCHES}

    def test_it_costs_no_extra_request(self, offline):
        """Built from what was committed, so RecipeWrangler is never asked
        about a dish the plan already has in hand."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])
        before = offline["fetches"]
        _actions(space, 2, "lunch")
        _actions(space, 2, "lunch")

        assert offline["fetches"] - before == 1

    def test_the_offer_is_recorded_whether_or_not_it_is_taken(self, offline):
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0])
        _actions(space, 2, "lunch")

        assert space.selection_events == [{
            "type": "repeat_offered", "day": 2, "meal_type": "lunch",
            "count": 1, "recipe_ids": ["d0"],
        }]

    def test_the_source_labels_do_not_ride_along(self, offline):
        """A leftover of a dish the member pinned is still the plan's doing:
        they anchored a dinner, not the lunch after it."""
        space = _space(plan_parameters=LEFTOVERS_ON)
        _serve(space, 1, "dinner", DINNERS[0], pinned=True,
               repeat_of_day=None, match_reasons=[{"kind": "pinned", "label": "x"}])

        leftover = _leftover(space, 2)

        assert "pinned" not in leftover
        assert "match_reasons" not in leftover


# --------------------------------------------------------------------- #
# What it is worth, and what it buys                                      #
# --------------------------------------------------------------------- #


class TestTheScorer:
    FRESH = {"recipe_id": "x", "recipe_title": "Bean Stew",
             "recipe_ingredients": "beans, kale"}
    LEFTOVER = {
        "recipe_id": "d0", "recipe_title": "Bean Stew",
        "recipe_ingredients": "beans, kale", "repeat_of_day": 1,
        "repeat_source": REPEAT_LEFTOVER,
        "leftover_of": {"day": 1, "meal_type": "dinner"},
    }

    def test_a_member_who_asked_for_it_gets_it(self):
        """A bare profile leaves nearly every candidate at exactly 0.0, so an
        unweighted leftover would never actually be picked — the same lesson
        that took the repeat penalty to zero, read the other way."""
        scorer = build_preference_scorer(_profile(plan_parameters=LEFTOVERS_ON))

        assert scorer(self.LEFTOVER, ["Bean Stew"], IngredientBasket(), 2) > scorer(
            self.FRESH, ["Bean Stew"], IngredientBasket(), 2
        )

    def test_it_is_worth_nothing_to_a_member_who_did_not(self):
        scorer = build_preference_scorer(_profile())

        assert scorer(self.LEFTOVER, [], IngredientBasket(), 2) == 0.0

    def test_a_favourite_still_wins_its_slot(self):
        """The control asks for less cooking, not for the member's own
        starred dishes to lose."""
        profile = _profile(
            plan_parameters=LEFTOVERS_ON, favorite_recipe_ids=["x"]
        )
        scorer = build_preference_scorer(profile)

        assert scorer(self.FRESH, [], IngredientBasket(), 2) > scorer(
            self.LEFTOVER, [], IngredientBasket(), 2
        )

    def test_a_stated_pantry_still_wins_its_slot(self):
        """"Use up my aubergines" is a request about this week's fridge;
        "cook once, eat twice" is a standing preference. The specific one
        outranks the standing one, the way it outranks a favourite."""
        profile = _profile(plan_parameters=LEFTOVERS_ON)
        scorer = build_preference_scorer(profile, pantry=("aubergine",))
        uses_pantry = {**self.FRESH, "recipe_ingredients": "aubergine, kale"}

        assert scorer(uses_pantry, [], IngredientBasket(), 2) > scorer(
            self.LEFTOVER, [], IngredientBasket(), 2
        )

    def test_it_still_pays_in_full_for_resembling_a_different_dish(self):
        scorer = build_preference_scorer(_profile(plan_parameters=LEFTOVERS_ON))
        alone = scorer(self.LEFTOVER, ["Bean Stew"], IngredientBasket(), 2)
        beside_a_lookalike = scorer(
            self.LEFTOVER, ["Bean Stew", "Bean Chilli"], IngredientBasket(), 2
        )

        assert beside_a_lookalike < alone


class TestALeftoverBuysNothing:
    """The basket is the shopping list, not the menu.

    This is the one place the plan's own model has to distinguish "eaten
    twice" from "bought twice" — the ripple `IDEAS.md` predicted, landing in
    a single branch of the planning loop.
    """

    @staticmethod
    def _run(offline, monkeypatch, profile):
        added: list = []
        from services.weekly_planner import planner as planner_module

        original = planner_module.IngredientBasket.add

        def spy(self, ingredients_text, day):
            added.append((day, str(ingredients_text)))
            return original(self, ingredients_text, day)

        monkeypatch.setattr(planner_module.IngredientBasket, "add", spy)
        space = RecipeActionSpace(profile, additional_diet=[], pantry=())
        env = WeeklyMealPlanEnv(profile, space, RewardCalculator())
        entries = WeeklyPlanner(env).generate_full_plan(
            user_query="a week", scorer=build_preference_scorer(profile),
        )
        return entries, added

    def test_the_shopping_list_never_hears_about_the_second_serving(
        self, offline, monkeypatch
    ):
        profile = _profile(plan_parameters=LEFTOVERS_ON)
        entries, added = self._run(offline, monkeypatch, profile)

        leftovers = [e for e in entries if e["recipe"].get("leftover_of")]

        assert leftovers, "the fixture should produce leftovers to test"
        assert len(added) == 21 - len(leftovers)

    def test_every_other_meal_still_goes_in(self, offline, monkeypatch):
        profile = _profile()
        _entries, added = self._run(offline, monkeypatch, profile)

        assert len(added) == 21


# --------------------------------------------------------------------- #
# What the member is told                                                 #
# --------------------------------------------------------------------- #


def _entry(day, meal_type, recipe_id, title, **recipe):
    return {
        "day": day,
        "meal_idx": {"breakfast": 0, "lunch": 1, "dinner": 2}[meal_type],
        "meal_type": meal_type,
        "recipe": {
            "recipe_id": recipe_id, "recipe_title": title,
            "recipe_ingredients": "aubergine, red lentils", **recipe,
        },
        "reward": 0.0,
    }


def _week_with_a_leftover():
    return [
        _entry(1, "dinner", "d0", "Aubergine Bake"),
        _entry(2, "lunch", "d0", "Aubergine Bake",
               repeat_of_day=1, repeat_source=REPEAT_LEFTOVER,
               leftover_of={"day": 1, "meal_type": "dinner"}),
        _entry(2, "dinner", "d1", "Bean Chilli"),
    ]


class TestTheChip:
    def test_it_names_the_slot_the_dish_was_cooked_in(self):
        """"The same lunch as Monday" about a dish Monday ate for dinner is a
        small lie the member can catch in one glance at their own plan."""
        entries = _week_with_a_leftover()
        build_weekly_explainability(entries, {"preferences": []})

        chip = next(
            r for r in entries[1]["recipe"]["match_reasons"]
            if r["kind"] == REPEAT_KIND
        )

        assert chip["source"] == REPEAT_BY_LEFTOVER
        assert "Monday's dinner again" in chip["label"]

    def test_it_promises_nothing_about_portions(self):
        """Nothing in this service records quantities, so the chip may say the
        dish is served twice and may not say anything was saved or stretched."""
        entries = _week_with_a_leftover()
        build_weekly_explainability(entries, {"preferences": []})

        label = next(
            r for r in entries[1]["recipe"]["match_reasons"]
            if r["kind"] == REPEAT_KIND
        )["label"].lower()

        for forbidden in ("rest of", "double", "portion", "half", "batch"):
            assert forbidden not in label


class TestTheMeasurement:
    def test_it_is_counted_as_a_repeat_and_broken_out_as_a_leftover(self):
        facts = repeat_facts(_week_with_a_leftover())

        assert facts["count"] == 1
        assert facts["leftovers"] == 1
        assert facts["by_source"] == {REPEAT_BY_LEFTOVER: 1}

    def test_it_does_not_drag_the_measured_repeat_gap_below_the_policy(self):
        """A leftover is one day after its source by definition. Averaging it
        into `min_gap_days` would report the week's repeats as tighter than
        the cooldown allows — and the ledger would call that a violation."""
        entries = _week_with_a_leftover()
        facts = repeat_facts(entries)

        assert facts["min_gap_days"] is None

        rows = build_weekly_explainability(entries, {"preferences": []})
        repeat_row = next(
            r for r in rows["constraints_applied"]
            if r["constraint"] == "repeat meals stay spaced and capped"
        )
        assert repeat_row["status"] == "satisfied"

    def test_an_ordinary_repeat_still_sets_the_gap(self):
        entries = _week_with_a_leftover() + [
            _entry(4, "dinner", "d1", "Bean Chilli",
                   repeat_of_day=2, repeat_source="plan"),
        ]
        facts = repeat_facts(entries)

        assert facts["min_gap_days"] == 2
        assert facts["leftovers"] == 1
        assert facts["count"] == 2

    def test_variety_reports_it_without_calling_the_week_monotonous(self):
        metrics = variety_metrics(_week_with_a_leftover())

        assert metrics["leftover_meals"] == 1
        assert metrics["planned_repeats"] == 1
        assert metrics["unexplained_repeats"] == 0

    def test_it_is_not_also_billed_as_ingredient_reuse(self):
        """"Monday's dinner again" and "also uses Monday's aubergine" on one
        card say the same thing twice, and counting both would inflate the
        reuse figure with the repeat count."""
        facts = shared_ingredient_facts(_week_with_a_leftover())

        assert 1 not in facts["entries"]


class TestTheLedgerRow:
    def test_the_week_says_how_many_meals_were_cooked_once(self):
        result = build_weekly_explainability(
            _week_with_a_leftover(), {"preferences": []}
        )
        row = next(
            r for r in result["constraints_applied"]
            if r["constraint"] == "cook once, eat twice"
        )

        assert row["status"] == "satisfied"
        assert "1 lunch(es)" in row["detail"]
        assert str(MAX_LEFTOVER_MEALS) in row["detail"]

    def test_it_says_outright_that_portions_are_not_tracked(self):
        """The one thing a member could reasonably assume and be wrong about.
        Saying it in the ledger is cheaper than a support ticket, and is the
        only honest version of a feature built without quantity data."""
        result = build_weekly_explainability(
            _week_with_a_leftover(), {"preferences": []}
        )
        row = next(
            r for r in result["constraints_applied"]
            if r["constraint"] == "cook once, eat twice"
        )

        assert "doesn't track portions" in row["detail"]

    def test_a_week_without_leftovers_says_nothing_about_them(self):
        result = build_weekly_explainability(
            [_entry(1, "dinner", "d0", "Aubergine Bake")], {"preferences": []}
        )

        assert not [
            r for r in result["constraints_applied"]
            if "eat twice" in r["constraint"]
        ]
        assert "eat twice" not in result["reasoning"]

    def test_the_justification_the_member_reads_says_it_too(self):
        result = build_weekly_explainability(
            _week_with_a_leftover(), {"preferences": []}
        )

        assert "eaten again at lunch" in result["reasoning"]


class TestAStaleMarker:
    """An edit can leave a leftover pointing at a slot that has moved on."""

    def test_a_marker_whose_slot_no_longer_holds_the_dish_is_dropped(self):
        entries = _week_with_a_leftover()
        entries[0]["recipe"] = {
            "recipe_id": "d9", "recipe_title": "Something Else",
            "recipe_ingredients": "rice",
        }

        facts = repeat_facts(entries)

        assert facts["count"] == 0
        assert facts["leftovers"] == 0

    def test_the_slot_is_checked_and_not_only_the_day(self):
        """Monday still serves the dish — at lunch. "Monday's dinner again" is
        false, and a day-only check would have let it through."""
        entries = _week_with_a_leftover()
        entries[0]["meal_type"] = "lunch"
        entries[0]["meal_idx"] = 1

        assert repeat_facts(entries)["count"] == 0

    def test_it_becomes_an_unexplained_duplicate_rather_than_a_silent_one(self):
        entries = _week_with_a_leftover()
        entries[0]["meal_type"] = "lunch"
        entries[0]["meal_idx"] = 1

        assert repeat_facts(entries)["unexplained"] == 1


# --------------------------------------------------------------------- #
# The whole week                                                          #
# --------------------------------------------------------------------- #


class TestAcrossAWholeWeek:
    @staticmethod
    def _week(**profile):
        built = _profile(**profile)
        space = RecipeActionSpace(built, additional_diet=[], pantry=())
        env = WeeklyMealPlanEnv(built, space, RewardCalculator())
        entries = WeeklyPlanner(env).generate_full_plan(
            user_query="a week", scorer=build_preference_scorer(built),
        )
        assert len(entries) == 21
        return entries, env

    def test_the_default_week_has_no_leftovers_at_all(self, offline):
        entries, _ = self._week()

        assert not [e for e in entries if e["recipe"].get("leftover_of")]

    def test_every_leftover_is_yesterdays_dinner(self, offline):
        entries, _ = self._week(plan_parameters=LEFTOVERS_ON)
        by_slot = {(e["day"], e["meal_type"]): e for e in entries}

        leftovers = [e for e in entries if e["recipe"].get("leftover_of")]
        assert leftovers

        for entry in leftovers:
            marker = entry["recipe"]["leftover_of"]
            assert entry["meal_type"] == "lunch"
            assert marker["day"] == entry["day"] - LEFTOVER_GAP_DAYS
            source = by_slot[(marker["day"], marker["meal_type"])]
            assert source["recipe"]["recipe_id"] == entry["recipe"]["recipe_id"]

    def test_the_week_stays_within_the_cap(self, offline):
        entries, _ = self._week(plan_parameters=LEFTOVERS_ON)

        leftovers = [e for e in entries if e["recipe"].get("leftover_of")]
        assert len(leftovers) <= MAX_LEFTOVER_MEALS

    def test_no_dish_is_served_more_than_twice_however_it_came_back(self, offline):
        entries, _ = self._week(plan_parameters=LEFTOVERS_ON)
        served: dict = {}
        for entry in entries:
            served.setdefault(entry["recipe"]["recipe_id"], []).append(entry["day"])

        for recipe_id, days in served.items():
            assert len(days) <= MAX_APPEARANCES, (recipe_id, days)

    def test_every_leftover_was_recorded_when_it_happened(self, offline):
        entries, env = self._week(plan_parameters=LEFTOVERS_ON)
        logged = {
            (e["day"], e["recipe_id"])
            for e in env.selection_events
            if e["type"] == "repeat_allowed" and e.get("leftover_of")
        }
        served = {
            (e["day"], e["recipe"]["recipe_id"])
            for e in entries if e["recipe"].get("leftover_of")
        }

        assert logged == served

    def test_the_finished_week_reports_no_unexplained_duplicate(self, offline):
        """Every second serving on the plate passed through a rule that
        recorded a reason for it — which is what makes the repeat labelling
        worth anything at all."""
        entries, env = self._week(plan_parameters=LEFTOVERS_ON)
        result = build_weekly_explainability(
            entries, _profile(plan_parameters=LEFTOVERS_ON),
            selection_events=env.selection_events,
        )

        assert result["metrics"]["variety"]["unexplained_repeats"] == 0

    def test_all_meals_distinct_when_the_member_asks_for_that(self, offline):
        entries, _ = self._week(plan_parameters={"repeat_meals": "off"})

        ids = [e["recipe"]["recipe_id"] for e in entries]
        assert len(set(ids)) == 21

    def test_at_all_a_lunch_or_dinner_may_come_back_spaced(self, offline):
        entries, _ = self._week(plan_parameters={"repeat_meals": "all"})
        for meal in ("lunch", "dinner"):
            days: dict = {}
            for entry in entries:
                if entry["meal_type"] == meal:
                    days.setdefault(entry["recipe"]["recipe_id"], []).append(entry["day"])
            for recipe_id, served in days.items():
                assert len(served) <= MAX_APPEARANCES, recipe_id
                assert all(
                    b - a >= REPEAT_MIN_GAP_DAYS
                    for a, b in zip(sorted(served), sorted(served)[1:])
                ), (recipe_id, served)


# --------------------------------------------------------------------- #
# Editing the dinner a lunch is eating                                    #
# --------------------------------------------------------------------- #


class TestEditingTheSourceDinner:
    """`IDEAS.md` named this as the ripple that could not be left silent:
    "either the edit cascades, or the reference breaks and the member is
    told — but it cannot silently keep pointing at a dish that is no longer
    there."

    It cascades. Changing what you cook changes what you eat the next day —
    that is what "cook once, eat twice" means, and it is the only reading in
    which the two slots still describe one act of cooking. The alternative,
    leaving the lunch alone, is not "the reference breaks": the lunch would
    keep serving a dish that is nowhere else in the week, which reads to
    every measurement as an unexplained duplicate.
    """

    @staticmethod
    def _plan(session_service, sample_profile, leftover_day=3):
        import uuid

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile
        )
        entries = []
        for day in range(1, 8):
            for idx, meal in enumerate(["breakfast", "lunch", "dinner"]):
                recipe = {
                    "recipe_id": f"w-{day}-{idx}",
                    "recipe_title": f"Meal {day}-{idx}",
                    "recipe_ingredients": "x", "recipe_directions": "y",
                }
                if (day, meal) == (leftover_day, "lunch"):
                    recipe = {
                        "recipe_id": f"w-{leftover_day - 1}-2",
                        "recipe_title": f"Meal {leftover_day - 1}-2",
                        "recipe_ingredients": "x", "recipe_directions": "y",
                        "repeat_of_day": leftover_day - 1,
                        "repeat_source": REPEAT_LEFTOVER,
                        "leftover_of": {
                            "day": leftover_day - 1, "meal_type": "dinner",
                        },
                    }
                entries.append({
                    "day": day, "meal_idx": idx, "meal_type": meal,
                    "recipe": recipe, "reward": 1.0,
                })
        session_service.add_weekly_meal_plan(session.session_id, entries)
        return session

    @staticmethod
    def _service(session_service, day, meal_type="dinner"):
        from models.recipe import CandidateRecipe
        from services.edit_service import EditService
        from test_edit_and_transparency import (  # noqa: E402
            FakeEditClient, FakeEditExtractor, _rich,
        )

        replacement = CandidateRecipe("new-1", "Poke bowl", "rice, fish", "assemble")
        return EditService(
            session_service,
            client=FakeEditClient(
                {meal_type: [replacement]},
                {"w-2-2": _rich("w-2-2", "Meal 2-2", kcal=800),
                 "w-3-1": _rich("w-3-1", "Meal 3-1", kcal=800),
                 "w-5-2": _rich("w-5-2", "Meal 5-2", kcal=800),
                 "new-1": _rich("new-1", "Poke bowl", kcal=450)},
            ),
            extractor=FakeEditExtractor({
                "meal_type": meal_type, "day": day, "directive": "lighter",
                "needs_slot_clarification": False, "question": None,
            }),
        )

    def test_the_lunch_eating_it_follows_the_swap(self, session_service, sample_profile):
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=2).process(
            session.session_id, "tuesday dinner lighter please"
        )

        entries = outcome.weekly_meal_plan.entries
        dinner = next(e for e in entries if (e["day"], e["meal_idx"]) == (2, 2))
        lunch = next(e for e in entries if (e["day"], e["meal_idx"]) == (3, 1))

        assert dinner["recipe"]["recipe_title"] == "Poke bowl"
        assert lunch["recipe"]["recipe_title"] == "Poke bowl"
        assert lunch["recipe"]["leftover_of"] == {"day": 2, "meal_type": "dinner"}

    def test_the_member_is_told_rather_than_surprised(self, session_service, sample_profile):
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=2).process(
            session.session_id, "tuesday dinner lighter please"
        )

        assert "Wednesday lunch was the leftovers" in outcome.text
        assert outcome.facts["leftovers_followed"] == [
            {"day": 3, "meal_type": "lunch"}
        ]

    def test_the_lunch_is_not_anchored_by_someone_elses_edit(
        self, session_service, sample_profile
    ):
        """The member asked to change a dinner. Marking the lunch `pinned`
        would tell every later turn they had requested that dish there."""
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=2).process(
            session.session_id, "tuesday dinner lighter please"
        )

        lunch = next(
            e for e in outcome.weekly_meal_plan.entries
            if (e["day"], e["meal_idx"]) == (3, 1)
        )
        assert not lunch["recipe"].get("pinned")

    def test_the_patched_week_carries_no_unexplained_duplicate(
        self, session_service, sample_profile
    ):
        """The whole reason to cascade rather than leave it: a lunch still
        serving a dish the week no longer cooks is a duplicate with nothing
        behind it, and the ledger says so in a `violated` row."""
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=2).process(
            session.session_id, "tuesday dinner lighter please"
        )

        metrics = outcome.weekly_meal_plan.metrics
        assert metrics["variety"]["unexplained_repeats"] == 0
        assert metrics["repeats"]["leftovers"] == 1

    def test_an_unrelated_dinner_changes_nothing_else(
        self, session_service, sample_profile
    ):
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=5).process(
            session.session_id, "friday dinner lighter please"
        )

        entries = outcome.weekly_meal_plan.entries
        lunch = next(e for e in entries if (e["day"], e["meal_idx"]) == (3, 1))

        assert lunch["recipe"]["recipe_title"] == "Meal 2-2"
        assert "leftovers" not in outcome.text
        assert "leftovers_followed" not in outcome.facts

    def test_editing_the_leftover_lunch_simply_makes_it_an_ordinary_lunch(
        self, session_service, sample_profile
    ):
        """Nothing eats a lunch, so there is nothing to cascade into — and the
        dinner it came from is untouched."""
        session = self._plan(session_service, sample_profile)
        outcome = self._service(session_service, day=3, meal_type="lunch").process(
            session.session_id, "wednesday lunch lighter please"
        )

        entries = outcome.weekly_meal_plan.entries
        lunch = next(e for e in entries if (e["day"], e["meal_idx"]) == (3, 1))
        dinner = next(e for e in entries if (e["day"], e["meal_idx"]) == (2, 2))

        assert lunch["recipe"]["recipe_title"] == "Poke bowl"
        assert "leftover_of" not in lunch["recipe"]
        assert dinner["recipe"]["recipe_title"] == "Meal 2-2"
        assert "leftovers_followed" not in outcome.facts


class TestItSurvivesStorageAndTheWire:
    """A leftover that only exists in memory is not a plan feature.

    Everything the member reads is rebuilt from the stored entries — chips
    included — so the marker has to survive a JSON round-trip and reach the
    gateway. It rides inside `recipe`, which is a free-form dict on the
    response model, so this is additive for every existing client.
    """

    @staticmethod
    def _stored(session_service):
        import uuid

        from services.weekly_planner.day_summary import build_day_summaries

        session = session_service.create_session(
            f"member-{uuid.uuid4()}",
            {"diet": [], "preferences": [], "plan_parameters": dict(LEFTOVERS_ON)},
        )
        entries = _week_with_a_leftover()
        summaries = build_day_summaries(entries)
        explainability = build_weekly_explainability(
            entries, session.user_profile, day_summaries=summaries
        )
        session_service.add_weekly_meal_plan(
            session.session_id, entries,
            day_summaries=summaries, explainability=explainability,
        )
        # Read back through a service with an empty cache, so the plan comes
        # off the database rather than out of memory.
        from services.session_service import SessionService

        fresh = SessionService().get_session(session.session_id)
        return fresh.get_current_weekly_plan()

    def test_the_marker_comes_back_off_the_database_intact(self, session_service):
        plan = self._stored(session_service)
        lunch = next(e for e in plan.entries if e["meal_type"] == "lunch")

        assert lunch["recipe"]["leftover_of"] == {"day": 1, "meal_type": "dinner"}
        assert lunch["recipe"]["repeat_source"] == REPEAT_LEFTOVER
        assert plan.metrics["repeats"]["leftovers"] == 1

    def test_the_router_hands_it_to_the_gateway(self, session_service):
        from routers.foodchat_router import WeeklyMealPlanResponse

        plan = self._stored(session_service)
        payload = WeeklyMealPlanResponse.from_weekly_meal_plan(plan).model_dump()
        lunch = next(e for e in payload["entries"] if e["meal_type"] == "lunch")

        assert lunch["recipe"]["leftover_of"] == {"day": 1, "meal_type": "dinner"}
        assert "Monday's dinner again" in str(lunch["recipe"]["match_reasons"])
        assert payload["metrics"]["variety"]["leftover_meals"] == 1


# --------------------------------------------------------------------- #
# Offering a repeat rather than waiting for one                           #
# --------------------------------------------------------------------- #


@pytest.fixture
def stingy(monkeypatch):
    """A source with a healthy pool that never volunteers a served dish back.

    This is the real-world behaviour, and the one the cooldown could not beat:
    a day's pool is a fresh fetch, and RecipeWrangler ranks recipes the member
    has not seen. Observed on a live plan — the policy allowed a breakfast
    repeat at all four remaining slots and the source offered one at none of
    them, so a member who had asked for repeated breakfasts got exactly one,
    for a reason nothing in the stored plan could name.
    """
    state = {"served": set(), "fetches": 0}
    catalogue = {
        "breakfast": [(f"b{i}", f"Breakfast{i}") for i in range(40)],
        "lunch": [(f"l{i}", f"Lunch{i}") for i in range(40)],
        "dinner": [(f"d{i}", f"Dinner{i}") for i in range(40)],
    }

    def fake_pool(*, profile, allergens, diet, cuisines, exclude_recipe_ids,
                  limit_per_slot):
        excluded = set(exclude_recipe_ids or [])
        state["fetches"] += 1
        return {
            slot: [
                _cand(p) for p in items
                if p[0] not in excluded and p[0] not in state["served"]
            ][:limit_per_slot]
            for slot, items in catalogue.items()
        }

    monkeypatch.setattr(action_adapter, "_fetch_candidate_pool", fake_pool)
    monkeypatch.setattr(action_adapter.CANDIDATES, "fetch_details", lambda ids: {})
    monkeypatch.setattr(
        action_adapter.CANDIDATES, "split_cuisines", lambda likes: ([], list(likes or []))
    )
    from services import pantry_service

    monkeypatch.setattr(
        pantry_service, "fetch_pantry_candidates", lambda profile, pantry, **kw: {}
    )
    return state


def _stingy_week(stingy, params, seed):
    """A full week against the stingy source, returning (entries, env)."""
    import random

    random.seed(seed)
    profile = _profile(plan_parameters=params)
    space = RecipeActionSpace(profile, additional_diet=[], pantry=())
    original = space.mark_committed

    def spy(recipe_id, day, meal_type, action=None):
        stingy["served"].add(recipe_id)
        return original(recipe_id, day, meal_type, action=action)

    space.mark_committed = spy
    env = WeeklyMealPlanEnv(profile, space, RewardCalculator())
    entries = WeeklyPlanner(env).generate_full_plan(
        user_query="a week", scorer=build_preference_scorer(profile),
    )
    return entries, env


def _repeats_in(entries, meal_type=None):
    rows = [e for e in entries if meal_type is None or e["meal_type"] == meal_type]
    ids = [e["recipe"]["recipe_id"] for e in rows]
    return len(ids) - len(set(ids))


class TestOnlyAnAskedForSettingIsActedOn:
    """The gate the whole feature hangs on.

    `repeat_mode` returns "breakfast" both for a member who chose it and for
    one who has never seen the card. Injecting on the second would change
    every existing member's week to satisfy a preference none of them
    expressed — the difference between honouring a request and inventing one.
    """

    def test_the_predicate_separates_a_choice_from_a_default(self):
        assert not plan_parameters.repeat_mode_is_explicit({})
        assert not plan_parameters.repeat_mode_is_explicit({"food_waste": "reuse"})
        assert plan_parameters.repeat_mode_is_explicit({"repeat_meals": "breakfast"})
        assert plan_parameters.repeat_mode_is_explicit({"repeat_meals": "off"})

    def test_a_value_we_could_not_read_is_not_a_request(self):
        """`repeat_mode` degrades it to the default; acting on it would be
        inventing a request out of a typo."""
        assert not plan_parameters.repeat_mode_is_explicit({"repeat_meals": "sometimes"})
        assert plan_parameters.repeat_mode({"repeat_meals": "sometimes"}) == "breakfast"

    def test_the_default_never_injects(self, offline):
        space = _space()
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(3, "breakfast", set()) == []

    def test_an_explicit_setting_does(self, offline):
        space = _space(plan_parameters={"repeat_meals": "breakfast"})
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        injected = space._injected_repeats(3, "breakfast", set())

        assert [a["recipe_id"] for a in injected] == ["b0"]
        assert injected[0]["repeat_of_day"] == 1
        assert injected[0]["repeat_source"] == action_adapter.REPEAT_PLAN
        assert "leftover_of" not in injected[0]


class TestWhatMayBeInjected:
    @staticmethod
    def _space():
        return _space(plan_parameters={"repeat_meals": "all"})

    def test_a_dish_the_source_already_returned_is_not_added_twice(self, offline):
        space = self._space()
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(3, "breakfast", {"b0"}) == []

    def test_the_cooldown_still_applies(self, offline):
        space = self._space()
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(2, "breakfast", set()) == []
        assert space._injected_repeats(3, "breakfast", set()) != []

    def test_the_cap_still_applies(self, offline):
        space = self._space()
        _serve(space, 1, "breakfast", BREAKFASTS[0])
        _serve(space, 3, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(5, "breakfast", set()) == []

    def test_a_pinned_or_downvoted_dish_still_has_no_way_back(self, offline):
        space = self._space()
        space.mark_selected("b0")
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(3, "breakfast", set()) == []

    def test_it_never_moves_a_dish_to_another_slot(self, offline):
        space = self._space()
        _serve(space, 1, "dinner", DINNERS[0])

        assert space._injected_repeats(3, "breakfast", set()) == []
        assert space._injected_repeats(3, "lunch", set()) == []

    def test_the_setting_still_decides_which_slots(self, offline):
        """At "Repeat breakfasts", a lunch is not offered back — widening the
        scope is the member's call, not the injector's."""
        space = _space(plan_parameters={"repeat_meals": "breakfast"})
        _serve(space, 1, "lunch", LUNCHES[0])
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(3, "lunch", set()) == []
        assert space._injected_repeats(3, "breakfast", set()) != []

    def test_off_injects_nothing_at_all(self, offline):
        space = _space(plan_parameters={"repeat_meals": "off"})
        _serve(space, 1, "breakfast", BREAKFASTS[0])

        assert space._injected_repeats(3, "breakfast", set()) == []

    def test_a_slot_is_not_flooded_with_old_dishes(self, offline):
        """Everything eligible would be legal. A slot whose pool is a third
        old dishes is a repetitive week arriving by the back door rather than
        by the member's setting."""
        space = self._space()
        for day, pair in enumerate(BREAKFASTS[:5], start=1):
            _serve(space, day, "breakfast", pair)

        injected = space._injected_repeats(7, "breakfast", set())

        assert len(injected) == action_adapter.INJECTED_REPEATS_PER_SLOT

    def test_the_most_recent_dishes_come_back_first(self, offline):
        """A routine is built out of what the week is already in the habit of."""
        space = self._space()
        for day, pair in enumerate(BREAKFASTS[:4], start=1):
            _serve(space, day, "breakfast", pair)

        injected = space._injected_repeats(7, "breakfast", set())

        assert [a["recipe_id"] for a in injected] == ["b3", "b2"]

    def test_it_is_the_dish_that_was_actually_served(self, offline):
        space = self._space()
        _serve(space, 1, "breakfast", BREAKFASTS[0],
               nutrition={"kcal": 310.0}, tags=["vegan"])

        injected = space._injected_repeats(3, "breakfast", set())[0]

        assert injected["recipe_title"] == "Breakfast0"
        assert injected["nutrition"] == {"kcal": 310.0}
        assert injected["tags"] == ["vegan"]


class TestTheDiagnosticSurvives:
    """`repeat_offered` exists to separate "the week repeated nothing" from
    "the source never offered anything to repeat" — opposite fixes, one here
    and one at RecipeWrangler. Injecting would erase exactly that signal if
    the plan's own additions were folded into the count.
    """

    def test_an_injected_offer_is_counted_apart(self, stingy):
        space = RecipeActionSpace(
            _profile(plan_parameters={"repeat_meals": "breakfast"}),
            additional_diet=[], pantry=(),
        )
        _serve(space, 1, "breakfast", ("b0", "Breakfast0"))
        stingy["served"].add("b0")
        space.get_candidate_actions("breakfast", {"day": 3})

        event = space.selection_events[-1]
        assert event["type"] == "repeat_offered"
        assert event["count"] == 1
        assert event["injected"] == 1

    def test_a_source_offer_carries_no_injected_key(self, offline):
        """The `offline` source returns everything not excluded, so day 3's
        pool already holds the earlier breakfast and nothing is added."""
        space = _space(plan_parameters={"repeat_meals": "breakfast"})
        _serve(space, 1, "breakfast", BREAKFASTS[0])
        _actions(space, 3, "breakfast")

        event = space.selection_events[-1]
        assert event["count"] == 1
        assert "injected" not in event

    def test_injecting_costs_no_request(self, offline):
        space = _space(plan_parameters={"repeat_meals": "all"})
        _serve(space, 1, "breakfast", BREAKFASTS[0])
        before = offline["fetches"]
        _actions(space, 3, "breakfast")
        _actions(space, 3, "lunch")

        assert offline["fetches"] - before == 1


class TestWhatARequestedRepeatIsWorth:
    FRESH = {"recipe_id": "x", "recipe_title": "Bean Stew",
             "recipe_ingredients": "beans, kale"}
    REPEAT = {"recipe_id": "b0", "recipe_title": "Oats",
              "recipe_ingredients": "oats, water", "repeat_of_day": 1,
              "repeat_source": action_adapter.REPEAT_PLAN}

    def test_a_default_pays_nothing_and_the_week_is_unchanged(self):
        scorer = build_preference_scorer(_profile())

        assert scorer(self.REPEAT, [], IngredientBasket(), 3) == 0.0

    def test_an_asked_for_repeat_beats_a_bare_fresh_dish(self):
        """Zero is the right price for a repeat the planner merely allowed. It
        is the wrong price for one the member requested: a bare profile leaves
        nearly every candidate at exactly 0.0, so "equal terms" in a pool of
        ten means a one-in-eleven share of the slot."""
        scorer = build_preference_scorer(
            _profile(plan_parameters={"repeat_meals": "breakfast"})
        )

        assert scorer(self.REPEAT, [], IngredientBasket(), 3) > scorer(
            self.FRESH, [], IngredientBasket(), 3
        )

    def test_it_still_loses_to_a_favourite(self):
        profile = _profile(
            plan_parameters={"repeat_meals": "all"}, favorite_recipe_ids=["x"]
        )
        scorer = build_preference_scorer(profile)

        assert scorer(self.FRESH, [], IngredientBasket(), 3) > scorer(
            self.REPEAT, [], IngredientBasket(), 3
        )

    def test_it_still_loses_to_a_stated_pantry(self):
        scorer = build_preference_scorer(
            _profile(plan_parameters={"repeat_meals": "all"}), pantry=("kale",)
        )

        assert scorer(self.FRESH, [], IngredientBasket(), 3) > scorer(
            self.REPEAT, [], IngredientBasket(), 3
        )

    def test_a_liked_ingredient_still_ties_rather_than_losing(self):
        """A fresh dish the member has a reason to like is not pushed out by a
        repeat; it joins the tie pool, which is where it belongs."""
        profile = _profile(
            plan_parameters={"repeat_meals": "all"}, food_likes=["kale"]
        )
        scorer = build_preference_scorer(profile)

        assert scorer(self.FRESH, [], IngredientBasket(), 3) == scorer(
            self.REPEAT, [], IngredientBasket(), 3
        )

    def test_a_leftover_is_paid_once_not_twice(self):
        """Both weights exist because the member moved the SAME single
        control. Stacking them prices one setting twice — enough, as it
        happens, to put a leftover level with a stated pantry item, which the
        documented ladder says must still win its slot."""
        scorer = build_preference_scorer(
            _profile(plan_parameters=LEFTOVERS_ON), pantry=("kale",)
        )
        leftover = {
            **self.REPEAT, "repeat_source": REPEAT_LEFTOVER,
            "leftover_of": {"day": 2, "meal_type": "dinner"},
        }

        assert scorer(leftover, [], IngredientBasket(), 3) == 2.0
        assert scorer(self.FRESH, [], IngredientBasket(), 3) == 3.0


class TestAgainstASourceThatNeverReOffers:
    """The end-to-end claim, on the behaviour that made this necessary."""

    def test_the_default_week_is_exactly_what_it_was(self, stingy):
        """Nothing is injected and nothing is paid, so a member who never
        touched the card keeps waiting for the source — and this source never
        offers. That is the week the report was written about, and it is
        deliberately still that week."""
        entries, _ = _stingy_week(stingy, {}, seed=0)

        assert _repeats_in(entries) == 0

    def test_asking_for_repeated_breakfasts_now_produces_them(self, stingy):
        entries, _ = _stingy_week(stingy, {"repeat_meals": "breakfast"}, seed=0)

        assert _repeats_in(entries, "breakfast") >= 2

    def test_and_only_breakfasts(self, stingy):
        entries, _ = _stingy_week(stingy, {"repeat_meals": "breakfast"}, seed=0)

        assert _repeats_in(entries, "lunch") == 0
        assert _repeats_in(entries, "dinner") == 0

    def test_off_still_means_twenty_one_different_recipes(self, stingy):
        entries, _ = _stingy_week(stingy, {"repeat_meals": "off"}, seed=0)

        assert _repeats_in(entries) == 0

    def test_the_caps_still_bound_it(self, stingy):
        entries, _ = _stingy_week(stingy, {"repeat_meals": "all"}, seed=0)
        served: dict = {}
        for entry in entries:
            served.setdefault(entry["recipe"]["recipe_id"], []).append(entry["day"])

        for recipe_id, days in served.items():
            assert len(days) <= MAX_APPEARANCES, (recipe_id, days)
            assert all(
                b - a >= REPEAT_MIN_GAP_DAYS
                for a, b in zip(sorted(days), sorted(days)[1:])
            ), (recipe_id, days)

    def test_every_injected_repeat_is_still_fully_accounted_for(self, stingy):
        entries, env = _stingy_week(stingy, {"repeat_meals": "all"}, seed=0)
        result = build_weekly_explainability(
            entries, _profile(plan_parameters={"repeat_meals": "all"}),
            selection_events=env.selection_events,
        )

        assert result["metrics"]["variety"]["unexplained_repeats"] == 0
        assert result["metrics"]["repeats"]["count"] == _repeats_in(entries)
