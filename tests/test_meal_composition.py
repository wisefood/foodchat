"""
A meal is more than one dish, and something has to decide which dishes.

Multi-plate planning had a renderer, a request format, a data model and a
`role` field, and no producer. Two different reasons:

* **Weekly** fetched one pool per meal SLOT and returned single recipes. There
  was no notion of a plate anywhere in the walk, so a dinner could never be a
  main and a salad — and a multi-plate week, once created, was flattened back
  to single dishes by its first refinement.
* **The structured path** did ask for one entry per plate, and then took
  RecipeWrangler's first recipe for each. Assembly was "delegated wholly to
  plan_meals", in that module's own words, so nothing was in a position to
  notice that a lasagne main and a macaroni salad side are two plates of pasta.

The split these tests pin down: three of the four things that make a set of
plates a meal can be MEASURED — same dish, repeated ingredient, portion share —
and those run always. Whether a dish SUITS another one cannot be measured from
what FoodChat holds, so it goes to a model, once for the whole plan, and only
when the turn can afford it.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec                       # noqa: E402
from models.recipe import CandidateRecipe                   # noqa: E402
from services import meal_composer                          # noqa: E402


def _c(rid, title, ingredients, kcal=None):
    return CandidateRecipe(
        recipe_id=rid, title=title, ingredients=ingredients, directions="cook",
        nutrition={"kcal": kcal} if kcal is not None else None,
    )


# ── reading the response back without losing the roles ──────────────────────

class TestPairingPlatesToRoles:
    """`to_candidates` buckets by slot name alone, which throws the plate
    distinction away — a main and a salad both land in `by_slot["lunch"]`,
    mixed, with nothing left to say which was which. That is why multi-plate
    had no producer even though the request was already right."""

    @staticmethod
    def _envelope(spec, per_plate=2, skip=None):
        from services.plan_client import PLANNER  # noqa: F401  (contract parity)

        slots = []
        for index, entry in enumerate(spec.to_request_slots()):
            recipes = [] if index == skip else [
                {"recipe_id": f"r{index}-{n}", "title": f"R{index}-{n}",
                 "ingredients": "x", "instructions": "y"}
                for n in range(per_plate)
            ]
            slots.append({"slot": entry["slot"], "recipes": recipes})
        return {"days": [{"day": 1, "slots": slots}]}

    def test_each_plate_gets_its_own_pool(self):
        from services.plan_client import PlanClient

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        pools = PlanClient.to_role_pools(self._envelope(spec), spec)

        assert set(pools[1]) == {("dinner", "main"), ("dinner", "side")}
        assert len(pools[1][("dinner", "main")]) == 2
        assert len(pools[1][("dinner", "side")]) == 2

    def test_the_pools_are_disjoint(self):
        """Both plates share the slot NAME. Pairing by slot name would put every
        recipe in both pools and the composer would offer a main as a side."""
        from services.plan_client import PlanClient

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        pools = PlanClient.to_role_pools(self._envelope(spec), spec)

        mains = {c.recipe_id for c in pools[1][("dinner", "main")]}
        sides = {c.recipe_id for c in pools[1][("dinner", "side")]}
        assert not mains & sides

    def test_an_empty_plate_does_not_shift_every_role_after_it(self):
        """The bug this reader replaces. `plan_structured` zipped the FLATTENED
        recipe list against the role sequence, which is correct only while every
        plate returns exactly one recipe — so a two-plate lunch whose main came
        back empty rendered the salad as the main, and the next slot's main as a
        side."""
        from services.plan_client import PlanClient

        spec = PlanSpec(
            meals=("lunch", "dinner"),
            plates={"lunch": ("main", "side")},
        )
        # Skip index 0 — lunch's MAIN comes back empty.
        pools = PlanClient.to_role_pools(self._envelope(spec, per_plate=1, skip=0), spec)

        assert pools[1][("lunch", "main")] == []
        # The salad stayed a salad, and dinner's main stayed dinner's main.
        assert [c.recipe_id for c in pools[1][("lunch", "side")]] == ["r1-0"]
        assert [c.recipe_id for c in pools[1][("dinner", "main")]] == ["r2-0"]

    def test_the_allergen_backstop_still_runs(self):
        from services.plan_client import PlanClient

        spec = PlanSpec(meals=("dinner",))
        envelope = {"days": [{"day": 1, "slots": [{
            "slot": "dinner", "recipes": [
                {"recipe_id": "bad", "title": "Almond tart", "ingredients": "almond"},
                {"recipe_id": "ok", "title": "Bean stew", "ingredients": "beans"},
            ],
        }]}]}
        pools = PlanClient.to_role_pools(envelope, spec, allergens=["tree nuts"])
        assert [c.recipe_id for c in pools[1][("dinner", "main")]] == ["ok"]


# ── the arithmetic ──────────────────────────────────────────────────────────

class TestWhatCanBeMeasured:
    def test_the_same_dish_twice_is_not_a_two_plate_meal(self):
        one = _c("r1", "Bean stew", "beans")
        pools = {("dinner", "main"): [one], ("dinner", "side"): [one]}
        assert meal_composer.compose("dinner", ("main", "side"), pools) == []

    def test_two_plates_leaning_on_the_same_thing_lose_to_two_that_do_not(self):
        """A main and a side both built on potatoes is the commonest way a
        two-plate meal reads as a mistake, and it is exactly measurable."""
        pools = {
            ("dinner", "main"): [_c("m", "Potato gratin", "potatoes, cream")],
            ("dinner", "side"): [
                _c("s1", "Potato salad", "potatoes, mayonnaise"),
                _c("s2", "Green salad", "lettuce, cucumber"),
            ],
        }
        best = meal_composer.compose("dinner", ("main", "side"), pools)[0]
        assert [p.title for p in best.plates] == ["Potato gratin", "Green salad"]

    def test_it_says_which_ingredient_it_objected_to(self):
        pools = {
            ("dinner", "main"): [_c("m", "Potato gratin", "potatoes, cream")],
            ("dinner", "side"): [_c("s", "Potato salad", "potatoes, mayonnaise")],
        }
        composed = meal_composer.compose("dinner", ("main", "side"), pools)[0]
        assert any("potatoes" in f for f in composed.findings)

    def test_staples_are_not_an_overlap(self):
        """Two plates sharing olive oil have nothing in common worth
        penalising, and counting staples would swamp the real signal."""
        pools = {
            ("dinner", "main"): [_c("m", "Bean stew", "beans, olive oil, salt")],
            ("dinner", "side"): [_c("s", "Green salad", "lettuce, olive oil, salt")],
        }
        composed = meal_composer.compose("dinner", ("main", "side"), pools)[0]
        assert composed.findings == []

    def test_a_side_heavier_than_its_share_loses(self):
        """`PlanSpec.kcal_split` was written for exactly this and has never had
        a consumer. A side that outweighs the main is not a side."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        pools = {
            ("dinner", "main"): [_c("m", "Grilled fish", "fish", kcal=400)],
            ("dinner", "side"): [
                _c("s1", "Huge gratin", "cheese", kcal=800),
                _c("s2", "Slaw", "cabbage", kcal=120),
            ],
        }
        best = meal_composer.compose(
            "dinner", ("main", "side"), pools,
            kcal_split=spec.kcal_split("dinner"), meal_kcal_target=600,
        )[0]
        assert best.plates[1].title == "Slaw"

    def test_no_calorie_target_means_no_calorie_judgement(self):
        """A plate marked down against a budget nobody set is being marked down
        for someone else's number."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        pools = {
            ("dinner", "main"): [_c("m", "Grilled fish", "fish", kcal=400)],
            ("dinner", "side"): [_c("s", "Huge gratin", "cheese", kcal=900)],
        }
        composed = meal_composer.compose(
            "dinner", ("main", "side"), pools,
            kcal_split=spec.kcal_split("dinner"), meal_kcal_target=None,
        )[0]
        assert composed.findings == [], composed.findings

    def test_a_plate_not_annotated_as_its_course_is_penalised_not_rejected(self):
        """The corpus's `dish_types` are incomplete, so absence of evidence is
        common and rejecting on it would empty plates that are fine."""
        from models.recipe import RecipeEnrichment

        pools = {
            ("dinner", "main"): [_c("m", "Bean stew", "beans")],
            ("dinner", "side"): [
                _c("s1", "Chocolate cake", "cocoa"),
                _c("s2", "Green salad", "lettuce"),
            ],
        }
        enrichment = {
            "s1": RecipeEnrichment("s1", "Chocolate cake", dish_types=["dessert"]),
            "s2": RecipeEnrichment("s2", "Green salad", dish_types=["salad"]),
        }
        composed = meal_composer.compose(
            "dinner", ("main", "side"), pools, enrichment=enrichment,
        )
        assert composed[0].plates[1].title == "Green salad"
        # And the cake is still offered — penalised, not ruled out.
        assert any(c.plates[1].title == "Chocolate cake" for c in composed)

    def test_missing_annotation_is_not_reported_as_a_mismatch(self):
        """Silence in the corpus is not evidence, and reporting it would fill
        the ledger with rows about missing data."""
        pools = {
            ("dinner", "main"): [_c("m", "Bean stew", "beans")],
            ("dinner", "side"): [_c("s", "Green salad", "lettuce")],
        }
        composed = meal_composer.compose(
            "dinner", ("main", "side"), pools, enrichment={},
        )[0]
        assert not any("annotated" in f for f in composed.findings)

    def test_a_plate_with_no_candidates_yields_no_meal(self):
        """A member who asked for a main AND a salad is not served a smaller
        meal without being told."""
        pools = {("dinner", "main"): [_c("m", "Bean stew", "beans")],
                 ("dinner", "side"): []}
        assert meal_composer.compose("dinner", ("main", "side"), pools) == []

    def test_a_single_plate_meal_still_composes(self):
        pools = {("lunch", "main"): [_c("m", "Soup", "lentils")]}
        composed = meal_composer.compose("lunch", ("main",), pools)
        assert len(composed) == 1 and composed[0].plates[0].title == "Soup"

    def test_already_used_dishes_are_not_offered(self):
        pools = {("lunch", "main"): [_c("a", "A", "x"), _c("b", "B", "y")]}
        composed = meal_composer.compose(
            "lunch", ("main",), pools, exclude_ids={"a"},
        )
        assert [c.plates[0].recipe_id for c in composed] == ["b"]

    def test_the_scan_is_bounded(self):
        """3 plates x 40 candidates is 64,000 combinations. A planner that
        stops responding is worse than one that considers the first few hundred
        in RecipeWrangler's own order."""
        pools = {
            ("dinner", role): [_c(f"{role}{n}", f"{role} {n}", "x") for n in range(40)]
            for role in ("main", "side", "dessert")
        }
        composed = meal_composer.compose(
            "dinner", ("main", "side", "dessert"), pools, limit=1000,
        )
        assert 0 < len(composed) <= meal_composer._COMBO_SCAN_LIMIT


# ── the judgement ───────────────────────────────────────────────────────────

class _Judge:
    def __init__(self, verdicts=None, boom=False):
        self.verdicts = verdicts or {}
        self.boom = boom
        self.offers = None

    def choose(self, message, offers):
        if self.boom:
            raise RuntimeError("groq is down")
        self.offers = offers
        return dict(self.verdicts)


def _two_options(slot="dinner"):
    pools = {
        (slot, "main"): [_c("m", "Pork belly", "pork")],
        (slot, "side"): [_c("s1", "Pickled slaw", "cabbage"),
                         _c("s2", "Buttered peas", "peas")],
    }
    return meal_composer.compose(slot, ("main", "side"), pools)


class TestWhatMustBeJudged:
    def test_the_judge_can_move_a_meal_off_the_measured_winner(self):
        options = _two_options()
        measured = options[0]
        other = next(i for i, c in enumerate(options) if c is not measured)
        judge = _Judge({"dinner": (other, "the slaw cuts the rich pork")})

        chosen = meal_composer.judge("something rich", {"dinner": options}, agent=judge)
        assert chosen["dinner"] is options[other]

    def test_the_reason_is_kept_apart_from_the_measurements(self):
        """It is the one thing here worth SAYING — "the slaw cuts the rich
        pork" is about the food. It used to be appended to `findings`, which
        put the marker this code invented for its own bookkeeping straight onto
        the member's plan: "day 1 lunch: chosen for the table: …"."""
        options = _two_options()
        judge = _Judge({"dinner": (1, "the slaw cuts the rich pork")})
        chosen = meal_composer.judge("rich", {"dinner": options}, agent=judge)
        assert chosen["dinner"].pairing == "the slaw cuts the rich pork"
        assert not any("chosen for the table" in f for f in chosen["dinner"].findings)

    def test_it_is_one_call_for_the_whole_plan(self):
        """A week with a side at dinner is seven judgements. Seven round trips
        inside one turn budget is how a plan stops arriving."""
        judge = _Judge()
        options = {f"day {d} dinner": _two_options() for d in range(1, 8)}
        meal_composer.judge("a week", options, agent=judge)
        assert len(judge.offers) == 7

    def test_a_meal_with_one_option_is_not_offered(self):
        judge = _Judge()
        single = meal_composer.compose(
            "lunch", ("main",), {("lunch", "main"): [_c("m", "Soup", "lentils")]},
        )
        chosen = meal_composer.judge("q", {"lunch": single}, agent=judge)
        assert judge.offers is None
        assert chosen["lunch"] is single[0]

    def test_a_judge_that_fails_keeps_the_measured_order(self):
        options = _two_options()
        chosen = meal_composer.judge("q", {"dinner": options}, agent=_Judge(boom=True))
        assert chosen["dinner"] is options[0]

    def test_every_meal_comes_back_whether_it_was_judged_or_not(self):
        """So no caller has to ask whether the judge ran."""
        options = {"dinner": _two_options(), "lunch": meal_composer.compose(
            "lunch", ("main",), {("lunch", "main"): [_c("m", "Soup", "lentils")]},
        )}
        chosen = meal_composer.judge("q", options, agent=_Judge())
        assert set(chosen) == {"dinner", "lunch"}

    def test_a_late_turn_does_not_pay_for_it(self):
        from services import turn_budget

        judge = _Judge({"dinner": (1, "no")})
        options = _two_options()
        with turn_budget.start(seconds=0.001):
            chosen = meal_composer.judge("q", {"dinner": options}, agent=judge)
        assert judge.offers is None, "it should not have asked"
        assert chosen["dinner"] is options[0]


class TestTheJudgeCannotBreakTheMeal:
    def test_an_option_that_was_not_offered_is_ignored(self):
        options = _two_options()
        chosen = meal_composer.judge(
            "q", {"dinner": options}, agent=_Judge({"dinner": (9, "nope")}),
        )
        assert chosen["dinner"] is options[0]

    def test_a_meal_it_was_not_asked_about_is_ignored(self):
        options = _two_options()
        chosen = meal_composer.judge(
            "q", {"dinner": options},
            agent=_Judge({"day 4 supper": (1, "?")}),
        )
        assert chosen["dinner"] is options[0]
        assert "day 4 supper" not in chosen

    def test_the_real_agent_drops_an_index_it_did_not_offer(self):
        from agents import MealJudge

        agent = MealJudge.__new__(MealJudge)

        class _Client:
            def invoke(self, messages, config=None):
                class _R:
                    content = (
                        '{"choices": [{"meal": "dinner", "pick": 7, "reason": "x"},'
                        ' {"meal": "ghost", "pick": 0, "reason": "y"}]}'
                    )
                return _R()

        agent.llm = _Client()
        out = agent.choose("q", [{"meal": "dinner", "options": ["a", "b"]}])
        assert out == {}

    def test_the_real_agent_matches_by_label_not_position(self):
        """A model answering in a different order is common and is not a reason
        to lose the answer."""
        from agents import MealJudge

        agent = MealJudge.__new__(MealJudge)

        class _Client:
            def invoke(self, messages, config=None):
                class _R:
                    content = (
                        '{"choices": [{"meal": "day 2 dinner", "pick": 1, "reason": "b"},'
                        ' {"meal": "day 1 dinner", "pick": 0, "reason": "a"}]}'
                    )
                return _R()

        agent.llm = _Client()
        out = agent.choose("q", [
            {"meal": "day 1 dinner", "options": ["a", "b"]},
            {"meal": "day 2 dinner", "options": ["c", "d"]},
        ])
        assert out["day 1 dinner"][0] == 0
        assert out["day 2 dinner"][0] == 1


class TestWhatTheJudgeIsShown:
    def test_it_sees_the_plates_by_role_with_their_ingredients(self):
        judge = _Judge()
        meal_composer.judge("q", {"dinner": _two_options()}, agent=judge)
        text = judge.offers[0]["options"][0]
        assert "main: Pork belly" in text and "side: " in text
        assert "pork" in text

    def test_staples_are_left_out_of_what_it_reads(self):
        pools = {
            ("dinner", "main"): [_c("m", "Stew", "beans, salt, olive oil")],
            ("dinner", "side"): [_c("s1", "Slaw", "cabbage"), _c("s2", "Peas", "peas")],
        }
        judge = _Judge()
        meal_composer.judge(
            "q", {"dinner": meal_composer.compose("dinner", ("main", "side"), pools)},
            agent=judge,
        )
        text = judge.offers[0]["options"][0]
        assert "beans" in text and "salt" not in text


class TestThePromptsAreNew:
    def test_they_are_registered_under_names_of_their_own(self):
        """`sync_prompts` creates only missing prompts, so a new capability
        under an existing name ships dead."""
        import prompts

        names = {p.name for p in prompts.ALL_PROMPTS}
        assert any(n.endswith("meal_composer_system") for n in names)
        assert any(n.endswith("meal_composer_user") for n in names)

    def test_the_meals_reach_the_prompt(self):
        from prompts import MEAL_COMPOSER_USER

        assert "{meals}" in MEAL_COMPOSER_USER.fallback


# ── the request actually asks for a choice ──────────────────────────────────

class TestTheDepthReachesTheRequest:
    """The near-miss worth a test of its own.

    `plan_meals` documented that "a spec supersedes slots/count_per_slot/days",
    and `to_request_slots` hardcoded `count: 1`. So a composer asking for four
    candidates per plate was silently handed one — one composition, no choice to
    make, the scoring rubric inert and the judge never invoked. Every unit test
    in this file would still have passed, because they build pools directly.
    """

    def test_a_spec_can_ask_for_more_than_one_per_plate(self):
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        assert [e["count"] for e in spec.to_request_slots(count=4)] == [4, 4]

    def test_one_is_still_the_default(self):
        """A plan needs one per plate. Only a composer needs more, and it has
        to say so."""
        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        assert [e["count"] for e in spec.to_request_slots()] == [1, 1]

    def test_plan_meals_no_longer_drops_it_when_a_spec_is_passed(self, monkeypatch):
        from services.plan_client import PlanClient

        sent = {}

        class _Response:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"days": []}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                sent.update(json or {})
                return _Response()

        import httpx
        monkeypatch.setattr(httpx, "Client", _Client)

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        PlanClient().plan_meals(spec=spec, count_per_slot=4)

        assert [s["count"] for s in sent["slots"]] == [4, 4]

    def test_the_composer_asks_for_a_pool_deep_enough_to_choose_from(self, monkeypatch):
        """End to end: what `role_pools` actually puts on the wire."""
        import services.plan_client as plan_module

        sent = {}

        class _Planner:
            @staticmethod
            def plan_meals(**kwargs):
                sent.update(kwargs)
                return {"days": []}

            describe_relaxations = staticmethod(lambda e: [])
            to_role_pools = staticmethod(lambda e, s, allergens=None: {})

        monkeypatch.setattr(plan_module, "PLANNER", _Planner())

        spec = PlanSpec(meals=("dinner",), plates={"dinner": ("main", "side")})
        meal_composer.role_pools({"allergies": []}, spec, per_plate=4)

        assert sent["count_per_slot"] == 4


class TestOnlyMealsWithMoreThanOnePlateAreJudged:
    """A single-plate meal inside a multi-plate spec has a pool too, and
    picking among four mains is RANKING, not composition. The prompt asks
    whether these dishes belong on a table together; for one dish that is not a
    question, and offering it made the judge answer "the slaw cuts it" about a
    lunch with no slaw in it."""

    def test_a_one_plate_meal_with_four_options_is_not_offered(self):
        pools = {("lunch", "main"): [_c(f"m{n}", f"M{n}", f"x{n}") for n in range(4)]}
        single = meal_composer.compose("lunch", ("main",), pools, limit=4)
        assert len(single) == 4

        judge = _Judge({"lunch": (2, "?")})
        chosen = meal_composer.judge("q", {"lunch": single}, agent=judge)
        assert judge.offers is None
        assert chosen["lunch"] is single[0]

    def test_a_mixed_plan_offers_only_its_multi_plate_meals(self):
        pools = {("lunch", "main"): [_c(f"m{n}", f"M{n}", f"x{n}") for n in range(3)]}
        judge = _Judge()
        meal_composer.judge("q", {
            "day 1 lunch": meal_composer.compose("lunch", ("main",), pools, limit=3),
            "day 1 dinner": _two_options(),
        }, agent=judge)
        assert [o["meal"] for o in judge.offers] == ["day 1 dinner"]


class TestDepthIsSpentWhereThereIsAChoice:
    """A single-plate meal in a shaped spec has nothing to compose — no second
    dish for it to sit beside — so four candidates for it are three recipes
    fetched, enriched and ranked by RecipeWrangler's own order to arrive at the
    one that was first anyway."""

    SPEC = PlanSpec(
        meals=("breakfast", "lunch", "dinner"), plates={"dinner": ("main", "side")},
    )

    def test_only_the_multi_plate_meal_gets_a_deep_pool(self):
        slots = self.SPEC.to_request_slots(count=4, only_multiplate=True)
        assert [(e["slot"], e["count"]) for e in slots] == [
            ("breakfast", 1), ("lunch", 1), ("dinner", 4), ("dinner", 4),
        ]

    def test_it_is_a_third_off_a_shaped_week(self):
        uniform = sum(e["count"] for e in self.SPEC.to_request_slots(count=4))
        trimmed = sum(
            e["count"] for e in self.SPEC.to_request_slots(count=4, only_multiplate=True)
        )
        assert (uniform, trimmed) == (16, 10)

    def test_an_all_single_plate_spec_is_untouched_by_the_flag(self):
        spec = PlanSpec(meals=("breakfast", "lunch"))
        assert [e["count"] for e in spec.to_request_slots(count=4, only_multiplate=True)] \
            == [1, 1]

    def test_the_flag_is_off_by_default(self):
        """A caller that wants a uniform pool still gets one."""
        assert [e["count"] for e in self.SPEC.to_request_slots(count=4)] == [4, 4, 4, 4]

    def test_the_composer_asks_for_it(self, monkeypatch):
        import services.plan_client as plan_module

        sent = {}

        class _Planner:
            @staticmethod
            def plan_meals(**kwargs):
                sent.update(kwargs)
                return {"days": []}

            describe_relaxations = staticmethod(lambda e: [])
            to_role_pools = staticmethod(lambda e, s, allergens=None: {})

        monkeypatch.setattr(plan_module, "PLANNER", _Planner())
        meal_composer.role_pools({"allergies": []}, self.SPEC, per_plate=4)
        assert sent["deepen_multiplate_only"] is True

    def test_it_reaches_the_wire(self, monkeypatch):
        from services.plan_client import PlanClient

        sent = {}

        class _Response:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"days": []}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                sent.update(json or {})
                return _Response()

        import httpx
        monkeypatch.setattr(httpx, "Client", _Client)

        PlanClient().plan_meals(
            spec=self.SPEC, count_per_slot=4, deepen_multiplate_only=True,
        )
        assert [s["count"] for s in sent["slots"]] == [1, 1, 4, 4]


class TestTheVerifierReadsWhatThePlanCarries:
    """`_check_kcal` and `_check_nutri_score` read the enrichment only, so a
    plan carrying its own macros was reported as `calories: unknown` — a
    measurement missing, nothing said, and the member reading silence as
    agreement. The planning envelope carries macros with every card and the
    composer now keeps them on the plate instead of throwing them away and
    re-fetching the same numbers.
    """

    @staticmethod
    def _plan(kcal=700.0, label="A"):
        from models.session import DayPlan, Meal, MealCourse, MealPlan

        plate = MealCourse(
            recipe_id="r1", title="Stew", ingredients="beans", directions="cook",
            nutrition={"kcal": kcal, "nutri_score_label": label},
        )
        return MealPlan.from_days(
            [DayPlan(day=1, meals=[Meal(meal_type="dinner", plates=[plate])])],
            reasoning="",
        )

    def _check(self, plan, requested, name):
        from services import plan_verifier

        report = plan_verifier.verify(plan, requested, {})
        return next((c for c in report.checks if c.name == name), None)

    def test_calories_are_measured_without_a_details_call(self):
        check = self._check(self._plan(kcal=700.0), {"kcal_target": 700}, "calories")
        assert check is not None and check.status == "passed"

    def test_a_miss_is_still_a_miss(self):
        check = self._check(self._plan(kcal=2400.0), {"kcal_target": 700}, "calories")
        assert check.status == "failed"

    def test_a_plate_with_no_macros_is_still_unknown(self):
        """The honest state, and it must survive: reporting 0 kcal for missing
        data would be a measurement of nothing presented as a measurement."""
        from models.session import DayPlan, Meal, MealCourse, MealPlan

        bare = MealCourse("r1", "Stew", "beans", "cook")
        plan = MealPlan.from_days(
            [DayPlan(day=1, meals=[Meal(meal_type="dinner", plates=[bare])])],
            reasoning="",
        )
        assert self._check(plan, {"kcal_target": 700}, "calories").status == "unknown"

    def test_the_nutri_score_floor_reads_the_plate_too(self):
        check = self._check(
            self._plan(label="D"), {"min_nutri_score": "B"}, "nutri-score",
        )
        assert check.status == "failed" and check.offenders == ("r1",)

    def test_a_fresh_details_call_still_wins(self):
        """Enrichment is a live fetch, so it is the more current of the two."""
        from models.recipe import RecipeEnrichment
        from services import plan_verifier

        plan = self._plan(kcal=700.0)
        report = plan_verifier.verify(
            plan, {"kcal_target": 700},
            {"r1": RecipeEnrichment("r1", "Stew", kcal=2400.0)},
        )
        calories = next(c for c in report.checks if c.name == "calories")
        assert calories.status == "failed", "the live figure should have been used"


class TestTheComposerJudgesAMainTheSameWay:
    """The fetch and the judgement have to agree about what a main is.

    The composer checked every main against `main-dish`, so the moment
    breakfast started returning breakfasts, a correct one would have been
    marked down for "not annotated as a main" — the same slot-blind assumption,
    one step later, penalising the fix.
    """

    def test_a_breakfast_in_a_main_role_is_not_a_mismatch(self):
        from models.recipe import RecipeEnrichment

        pools = {("breakfast", "main"): [_c("b", "Porridge", "oats")]}
        enrichment = {
            "b": RecipeEnrichment("b", "Porridge", dish_types=["breakfast"]),
        }
        composed = meal_composer.compose(
            "breakfast", ("main",), pools, enrichment=enrichment,
        )[0]
        assert not any("annotated" in f for f in composed.findings), (
            composed.findings
        )

    def test_a_side_is_still_checked(self):
        """Only `main` is slot-dependent; the rest keep their meaning."""
        from models.recipe import RecipeEnrichment

        pools = {
            ("dinner", "main"): [_c("m", "Bean stew", "beans")],
            ("dinner", "side"): [_c("s", "Chocolate cake", "cocoa")],
        }
        enrichment = {
            "s": RecipeEnrichment("s", "Chocolate cake", dish_types=["dessert"]),
        }
        composed = meal_composer.compose(
            "dinner", ("main", "side"), pools, enrichment=enrichment,
        )[0]
        assert any("not annotated as a side" in f for f in composed.findings)
