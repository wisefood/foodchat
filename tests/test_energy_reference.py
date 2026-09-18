"""
Whose calorie number is on this plan.

    > what if someone asks for 3 snacks?
    > can we suggest something healthy if something overshoot recommended
      daily intake?

Three snacks is a shape the planner will happily build, and nothing measured
what it added up to. Not because the measurement was missing — it was written,
tested and wired — but because the number it needed never arrived:

    weekly_planner.state_tracking   parses "2000 calories target"   ✓
    plan_scorer.scoring             parses the same string          ✓
    models.plan_brief._kcal_target  reads `profile["calorie_target"]` ✗

`_map_profile` turns the gateway's `nutritional_preferences.calories` into
that prose string and puts it in `preferences`. Nothing ever set
`calorie_target`. So on the daily path the target was None, and everything
downstream returned early: the plate critic ranked on no calorie fit, the
per-meal budget was empty, and `plan_verifier._check_kcal` stopped before it
measured a plate.

The other half is the number to use when the member set none. It must exist —
otherwise nobody who has not filled in a target can ever be told a day is
heavy — and it must never shape the plan or be attributed to them. That is the
weekly meat limit's lesson, which reported FoodChat's own default as the
member's "dietary preference" and apologised for failing to honour it.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_brief import PlanBrief                             # noqa: E402
from models.plan_spec import PlanSpec                               # noqa: E402
from models.session import DayPlan, Meal, MealCourse, MealPlan      # noqa: E402
from services import plan_verifier, reference_intake                # noqa: E402


def _course(recipe_id: str, title: str, kcal: float) -> MealCourse:
    return MealCourse(recipe_id=recipe_id, title=title, ingredients="x",
                      directions="y", nutrition={"kcal": kcal})


def _day(*plates: tuple[str, str, float]) -> MealPlan:
    meals = [
        Meal(slot, [_course(f"r-{index}", title, kcal)])
        for index, (slot, title, kcal) in enumerate(plates)
    ]
    return MealPlan.from_days([DayPlan(day=1, meals=meals)], reasoning="")


HEAVY = _day(
    ("breakfast", "Spiced lentil fritters", 418.0),
    ("snack", "Pea and mint toast", 530.0),
    ("lunch", "Barley salad", 491.0),
    ("snack_2", "Banana bread slice", 470.0),
    ("dinner", "Feta-crusted salmon", 235.0),
    ("snack_3", "Loaded nachos", 780.0),
)          # 2,924 kcal


def _calorie_rows(profile: dict, plan=HEAVY) -> list[dict]:
    """The energy rows of the ledger — under either name it can carry."""
    report = plan_verifier.verify(plan, PlanBrief.build(profile).to_requested(), None)
    return [
        r for r in report.as_ledger_rows()
        if r["constraint"] == "calories" or "kcal daily reference" in r["constraint"]
    ]


# ── the target that never arrived ────────────────────────────────────────

class TestTheMembersOwnTarget:
    def test_the_prose_form_is_read(self):
        """`_build_preferences` writes it; nothing wrote `calorie_target`."""
        assert reference_intake.stated_target(
            {"preferences": ["2000 calories target"]}
        ) == 2000.0

    @pytest.mark.parametrize("preference,expected", [
        ("1,800 calories target", 1800.0),
        ("2200 kcal target", 2200.0),
        ("target 1500 calories", 1500.0),
    ])
    def test_the_forms_it_arrives_in(self, preference, expected):
        assert reference_intake.stated_target({"preferences": [preference]}) == expected

    @pytest.mark.parametrize("preference", [
        "likes spicy food",
        "high protein (150g)",
        "40 calories target",        # implausible as a day
        "99000 calories target",
    ])
    def test_these_are_not_a_target(self, preference):
        assert reference_intake.stated_target({"preferences": [preference]}) is None

    def test_it_reaches_the_daily_planner(self):
        """The bug itself: this was None for every member who set a target."""
        assert PlanBrief.build({"preferences": ["2000 calories target"]}).kcal_target \
            == 2000.0

    def test_and_therefore_the_per_meal_budget(self):
        brief = PlanBrief.build(
            {"preferences": ["2000 calories target"]}, spec=PlanSpec(),
        )
        assert brief.kcal_by_slot
        assert sum(
            sum(split.values()) for split in brief.kcal_by_slot.values()
        ) == pytest.approx(2000.0, rel=0.01)

    def test_missing_it_is_a_failure_because_they_asked(self):
        rows = _calorie_rows({"preferences": ["2000 calories target"]})
        assert rows and rows[0]["status"] == "violated"
        assert "your 2000 kcal target" in rows[0]["detail"]


# ── the number nobody set ────────────────────────────────────────────────

class TestAReferenceIsNotATarget:
    def test_it_never_ranks_or_apportions(self):
        """`kcal_target` stays None, so the plate critic and the per-meal
        budget still see nothing — a plate marked down against a number the
        member never chose is being judged for somebody else."""
        brief = PlanBrief.build({"age_group": "adult", "sex": "female"}, spec=PlanSpec())
        assert brief.kcal_target is None
        assert brief.kcal_by_slot == {}
        assert brief.kcal_reference == 2000.0

    def test_going_over_it_is_not_a_broken_rule(self):
        """`relaxed`, not `violated`. Nobody set this number, and a red mark
        for it is the meat-limit apology in a new place."""
        rows = _calorie_rows({"age_group": "adult", "sex": "female"})
        assert rows and rows[0]["status"] == "relaxed"
        # Reads as a sentence in the reply's "could not be honoured" list,
        # which is handed the constraint NAME on its own.
        assert rows[0]["constraint"] == "a 2000 kcal daily reference"

    def test_every_sentence_says_whose_number_it_is(self):
        detail = _calorie_rows({"age_group": "adult", "sex": "female"})[0]["detail"]
        assert "a 2000 kcal reference" in detail
        assert "not a calculation for you" in detail
        assert "your" not in detail.lower().split("ask me")[0]

    def test_an_overshoot_names_the_plate_to_change(self):
        """"924 kcal over" is a fact nobody can act on. The suggestion is the
        point of the question that prompted this."""
        detail = _calorie_rows({"age_group": "adult", "sex": "female"})[0]["detail"]
        assert "Loaded nachos" in detail
        assert "780 kcal" in detail
        assert "lighter" in detail

    def test_being_under_does_not_suggest_eating_more(self):
        light = _day(("breakfast", "Toast", 120.0), ("dinner", "Soup", 180.0))
        detail = _calorie_rows({"age_group": "adult"}, plan=light)[0]["detail"]
        assert "under" in detail
        assert "lighter" not in detail

    def test_no_nutrition_measures_nothing(self):
        bare = MealPlan.from_days([DayPlan(day=1, meals=[
            Meal("lunch", [MealCourse("r-x", "Mystery", "", "")]),
        ])], reasoning="")
        rows = _calorie_rows({"age_group": "adult"}, plan=bare)
        assert rows and rows[0]["status"] == "unsupported"
        assert "no dish has nutrition data" in rows[0]["detail"]


class TestWhichReference:
    @pytest.mark.parametrize("stated,expected", [
        ("male", 2500.0), ("man", 2500.0), ("M", 2500.0),
        ("female", 2000.0), ("woman", 2000.0),
    ])
    def test_a_stated_sex_picks_its_row(self, stated, expected):
        profile = {"age_group": "adult", "nutritional_preferences": {"gender": stated}}
        assert reference_intake.reference_for(profile).kcal == expected

    @pytest.mark.parametrize("stated", ["non-binary", "other", "", "prefer not to say"])
    def test_anything_else_uses_the_ungendered_figure(self, stated):
        """The EU reference intake needs no such split, and guessing between
        two rows for somebody who did not answer is worse than not splitting."""
        profile = {"age_group": "adult", "nutritional_preferences": {"gender": stated}}
        reference = reference_intake.reference_for(profile)
        assert reference.kcal == reference_intake.EU_REFERENCE_INTAKE
        assert "EU reference intake" in reference.basis

    @pytest.mark.parametrize("age_group", ["child", "teen", "infant", "toddler"])
    def test_a_child_gets_no_number_at_all(self, age_group):
        """The range across childhood is several hundred kcal a year. One
        figure spanning it would be worse than saying nothing."""
        assert reference_intake.reference_for({"age_group": age_group}) is None
        assert _calorie_rows({"age_group": age_group}) == []

    def test_a_stated_target_always_wins(self):
        reference = reference_intake.reference_for({
            "age_group": "adult",
            "nutritional_preferences": {"gender": "male", "calories": 1800},
            "preferences": ["1800 calories target"],
        })
        assert reference.kcal == 1800.0
        assert reference.chosen is True
        assert reference.source == "your calorie target"


# ── the shape, before any recipe is chosen ───────────────────────────────

class TestThreeSnacks:
    def test_three_is_buildable(self):
        spec = PlanSpec().with_meal_count("snack", 3)
        assert len(spec.instances_of("snack")) == 3

    def test_and_it_says_something_first(self):
        """`concerns` is the existing seam for a shape worth a word before it
        is built. Six eating occasions is one."""
        notes = PlanSpec().with_meal_count("snack", 3).concerns()
        assert notes and "3 snacks" in notes[0]
        assert "keep them light" in notes[0]

    def test_two_is_an_ordinary_day(self):
        assert PlanSpec().with_meal_count("snack", 2).concerns() == []


# ── the weekly row beside the meat limit ─────────────────────────────────

class TestTheWeeklyCalorieRow:
    """The meat limit was fixed to say whose number it was. The calorie row
    directly beneath it still said `source: "calorie target"`, status
    `violated`, and "over your target" — about a flat 2,000 kcal a day that
    nobody had set."""

    @staticmethod
    def _rows(profile: dict, planned_kcal: float) -> list[dict]:
        from services.weekly_planner.explainability import weekly_constraints_ledger
        from services.weekly_planner.state_tracking import WeeklyNutritionalTracker

        targets = WeeklyNutritionalTracker(profile).targets
        nutrition = {
            "weekly_totals": {"kcal": planned_kcal},
            "coverage": {"meals_with_data": 21, "total_meals": 21},
        }
        ledger = weekly_constraints_ledger(
            profile, meat_count=0, targets=targets, selection_events=[],
            downvoted_count=0, nutrition=nutrition,
        )
        return [r for r in ledger if "calorie" in r["constraint"]]

    def test_a_default_is_not_reported_as_the_members_own(self):
        row = self._rows({}, 20000.0)[0]
        assert row["source"] == "a population reference"
        assert row["constraint"] == "weekly calorie reference"
        assert "your target" not in row["detail"]
        assert "reference" in row["detail"]

    def test_missing_a_number_nobody_set_is_not_a_violation(self):
        assert self._rows({}, 20000.0)[0]["status"] == "relaxed"

    def test_the_members_own_target_still_is(self):
        row = self._rows({"preferences": ["1800 calories target"]}, 20000.0)[0]
        assert row["status"] == "violated"
        assert row["source"] == "calorie target"
        assert "over your target" in row["detail"]

    def test_a_stated_sex_changes_the_reference(self):
        """"contextualize gender intakes" — on the weekly path too, where the
        figure was a flat 2,000 for everybody."""
        man = self._rows({"age_group": "adult", "sex": "male"}, 17000.0)[0]
        woman = self._rows({"age_group": "adult", "sex": "female"}, 17000.0)[0]
        assert "17,500 kcal" in man["detail"]
        assert "14,000 kcal" in woman["detail"]

    def test_the_verifier_adds_no_second_row_when_it_has_no_reference(self):
        """The weekly path drops `kcal_reference` before verifying, because
        the tracker already built a calorie row from the same figure — and a
        member reading two rows would have to work out whether they disagree.

        Asserted on the verifier's own contract rather than on the weekly
        service's source: with no reference and no target there is no row, so
        dropping the field is sufficient as well as necessary."""
        from models.plan_brief import PlanBrief

        requested = PlanBrief.build({"age_group": "adult"}).to_requested()
        assert requested["kcal_reference"]        # it would have produced one
        requested.pop("kcal_reference")
        report = plan_verifier.verify(HEAVY, requested, None)
        assert [c for c in report.checks if "calorie" in c.name] == []

    def test_and_the_weekly_path_drops_it(self):
        from pathlib import Path

        source = Path("src/services/weekly_plan_service.py").read_text()
        assert 'requested.pop("kcal_reference", None)' in source


# ── the wiring from the gateway profile ──────────────────────────────────

class TestWhatTheGatewayProfileCarries:
    """Both fields above are read off `profile`, and nothing put them there.

    Written after reverting each fix and watching the tests pass anyway: every
    case above builds its profile by hand, so the mapping from the gateway's
    `nutritional_preferences` blob — the only place either value comes from in
    production — was covered by nothing at all. That is the same hole as the
    one `_attach_context` had.
    """

    @staticmethod
    def _mapped(nutritional_preferences: dict) -> dict:
        from services.profile_service import ProfileService

        service = ProfileService.__new__(ProfileService)
        return service._map_profile({
            "dietary_groups": [],
            "allergies": [],
            "nutritional_preferences": nutritional_preferences,
            "properties": {},
        })

    def test_the_calorie_target_arrives_as_a_number(self):
        assert self._mapped({"calories": 1800})["calorie_target"] == 1800

    def test_and_still_as_the_prose_the_old_readers_parse(self):
        """Both forms, because the weekly tracker and the plan scorer read the
        string and removing it would break them to fix the daily path."""
        mapped = self._mapped({"calories": 1800})
        assert "1800 calories target" in mapped["preferences"]

    def test_either_form_reaches_the_planner(self):
        for profile in ({"calorie_target": 1800}, {"preferences": ["1800 calories target"]}):
            assert PlanBrief.build(profile).kcal_target == 1800.0

    @pytest.mark.parametrize("blob,expected", [
        ({"gender": "female"}, "female"),
        ({"sex": "male"}, "male"),
        # `sex` wins: it is the field that means what the reference table is
        # indexed by, where `gender` is being read as a proxy for it.
        ({"sex": "female", "gender": "male"}, "female"),
        ({}, None),
    ])
    def test_the_sex_proxy_arrives(self, blob, expected):
        assert self._mapped(blob)["sex"] == expected

    def test_and_it_picks_the_reference(self):
        """End to end: the gateway blob decides which figure the plan is
        reported against. Delete the mapping and this fails."""
        profile = dict(self._mapped({"gender": "male"}), age_group="adult")
        assert reference_intake.reference_for(profile).kcal == 2500.0

    def test_nothing_else_of_the_blob_comes_along(self):
        """`nutritional_preferences` is free-form. Carrying it whole would put
        arbitrary gateway keys into every prompt and log that renders a
        profile."""
        mapped = self._mapped({"gender": "male", "secret_note": "do not log me"})
        assert "secret_note" not in mapped
        assert "do not log me" not in str(mapped)
