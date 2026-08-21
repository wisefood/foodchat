"""
Real dietary guidelines, for the member's own region and life stage.

FoodChat grades plans against dietary guidelines and has never read one. The
checklist it reports — eat fish 1–2 times a week, limit red meat, make most
meals plant-based — is three rules hardcoded in the weekly explainability
module. They are real guidance. They are also the same three rules for a member
in Ireland, Slovenia, Hungary or Greece, and the same three for a pregnant
member, a teenager and a 70-year-old.

The catalog holds ~2,700 rules faceted by exactly the things that would make
them different for those people. Most of them cannot be checked — measured over
a 1,334-rule sample, 6.4% are food-group frequencies a plan can be counted
against and 73.4% are prose — and the tests that matter most here are the ones
pinning that a prose rule never acquires an invented target.

No network: the catalog client is stubbed.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from services import guidelines_service as gs        # noqa: E402


def _rule(text, **kw):
    base = {
        "rule_text": text,
        "guideline_type": kw.pop("guideline_type", "food_based"),
        "food_groups": kw.pop("food_groups", []),
        "guide_title": kw.pop("guide_title", "Healthy Eating Guidelines"),
        "status": "active",
    }
    base.update(kw)
    return base


# ── who the guidelines are for ────────────────────────────────────────────

class TestFacets:
    def test_the_region_shapes_the_query(self):
        assert gs.facets_for({"region": "IE"})["region"] == ["ie"]

    def test_a_life_stage_the_catalog_knows_is_passed_through(self):
        assert gs.facets_for({"life_stage": "pregnancy"})["life_stage"] == ["pregnancy"]

    @pytest.mark.parametrize("raw,expected", [
        ("65+", "older_adult"), ("13-18", "teen"), ("4-12", "child"),
        ("0-2", "toddler"), ("19-64", "adult"),
    ])
    def test_the_pickers_own_vocabulary_is_mapped(self, raw, expected):
        """Sending "65+" to a facet that stores "older_adult" matches nothing,
        and matching nothing looks identical to "no special guidance"."""
        assert gs.facets_for({"age_group": raw})["life_stage"] == [expected]

    def test_an_unknown_age_group_is_left_out_not_guessed(self):
        """Asking for `life_stage:adult` because most members are adults would
        silently exclude the rules that exist for the members who are not."""
        assert "life_stage" not in gs.facets_for({"age_group": "somewhere in the middle"})

    def test_an_empty_profile_asks_for_everything(self):
        assert gs.facets_for({}) == {}

    def test_health_conditions_narrow_it(self):
        facets = gs.facets_for({"health_conditions": ["Diabetes"]})
        assert facets["health_conditions"] == ["diabetes"]

    def test_only_the_food_groups_the_plan_is_shaped_around(self):
        """Asking for every group returns the corpus and tells nobody anything."""
        from models.plan_brief import PlanBrief

        facets = gs.facets_for({}, PlanBrief(food_groups=("legumes",)))
        assert facets["food_groups"] == ["legumes"]


# ── checkable vs prose ────────────────────────────────────────────────────

class TestSplit:
    def test_a_food_based_frequency_is_checkable(self):
        checkable, prose = gs.split([
            _rule("Eat fish twice a week", food_groups=["fish"]),
        ])
        assert len(checkable) == 1 and not prose
        assert checkable[0]["category"] == "fish"

    def test_prose_advice_stays_prose(self):
        """"Choose wholegrain varieties where possible" is advice, not an
        assertion with a truth value."""
        checkable, prose = gs.split([
            _rule("Choose wholegrain varieties where possible",
                  guideline_type="food_based", food_groups=["wholegrains"]),
        ])
        assert not checkable and len(prose) == 1

    def test_a_behavioural_rule_is_never_checkable(self):
        checkable, _ = gs.split([
            _rule("Eat slowly and mindfully, twice a day",
                  guideline_type="behavioral", food_groups=["fish"]),
        ])
        assert not checkable

    def test_a_rule_about_an_uncountable_group_is_prose(self):
        """A plan can count fish. It cannot count "added sugars"."""
        checkable, prose = gs.split([
            _rule("Limit added sugars to 6 a day", food_groups=["added sugars"]),
        ])
        assert not checkable and len(prose) == 1

    def test_a_frequency_only_in_the_text_is_still_found(self):
        checkable, _ = gs.split([_rule("Have fish at least 2 times per week")])
        assert checkable and checkable[0]["category"] == "fish"

    @pytest.mark.parametrize("text,direction,value", [
        ("Eat fish at least 2 times a week", "at least", 2),
        ("Limit red meat to at most 3 times a week", "at most", 3),
        ("No more than 1 portion of processed meat a week", "at most", 1),
        ("Eat fish twice a week", "about", 2),
        ("Have fish once a week", "about", 1),
    ])
    def test_the_direction_is_read_from_the_words(self, text, direction, value):
        checkable, _ = gs.split([_rule(text)])
        assert checkable, text
        parsed = checkable[0]
        assert parsed["direction"] == direction
        assert (parsed.get("low") or parsed.get("high")) == value

    def test_a_range_keeps_both_ends(self):
        checkable, _ = gs.split([_rule("Eat fish 1-2 times a week")])
        assert (checkable[0]["low"], checkable[0]["high"]) == (1, 2)


class TestTheStructuredQuantity:
    """`quantity` is a documented {operator, value, unit, period} triple with a
    full mapping, and nothing populates it: the write path exists and the
    producer does not. Read first anyway, so the day an import fills it this is
    the accurate source and the regex stops being the only one."""

    def test_it_is_preferred_over_the_sentence(self):
        checkable, _ = gs.split([_rule(
            "Eat fish twice a week",
            food_groups=["fish"],
            quantity={"operator": "at_least", "value": 3, "period": "week"},
        )])
        assert checkable[0]["direction"] == "at least"
        assert checkable[0]["low"] == 3

    def test_an_unusable_quantity_falls_back_to_the_sentence(self):
        checkable, _ = gs.split([_rule(
            "Eat fish twice a week", food_groups=["fish"],
            quantity={"operator": "at_least", "value": None},
        )])
        assert checkable[0]["direction"] == "about" and checkable[0]["low"] == 2

    def test_a_quantity_in_an_unsupported_period_is_ignored(self):
        checkable, prose = gs.split([_rule(
            "Fish is good", food_groups=["fish"],
            quantity={"operator": "at_least", "value": 2, "period": "month"},
        )])
        assert not checkable and len(prose) == 1


# ── the checklist ─────────────────────────────────────────────────────────

class TestChecklist:
    def _rows(self, rules, counts, total=21):
        checkable, _ = gs.split(rules)
        return gs.checklist(checkable, counts, total)

    def test_a_met_rule_says_so_with_the_real_count(self):
        rows = self._rows([_rule("Eat fish at least 2 times a week")], {"fish": 3})
        assert rows[0]["met"] is True and rows[0]["actual"] == 3
        assert "at least 2 a week" in rows[0]["target"]

    def test_an_unmet_rule_says_so(self):
        rows = self._rows([_rule("Eat fish at least 2 times a week")], {"fish": 0})
        assert rows[0]["met"] is False and rows[0]["actual"] == 0

    def test_an_upper_bound_is_met_by_staying_under(self):
        rows = self._rows([_rule("Limit red meat to at most 3 times a week")],
                          {"red meat": 2})
        assert rows[0]["met"] is True

    def test_a_range_is_met_inside_it_and_missed_outside(self):
        rule = [_rule("Eat fish 1-2 times a week")]
        assert self._rows(rule, {"fish": 2})[0]["met"] is True
        assert self._rows(rule, {"fish": 5})[0]["met"] is False

    def test_the_rows_keep_the_shape_the_ui_already_renders(self):
        rows = self._rows([_rule("Eat fish twice a week")], {"fish": 2})
        assert {"rule", "target", "actual", "met"} <= set(rows[0])

    def test_the_row_names_where_the_rule_came_from(self):
        """Three constants needed no attribution. Someone's national guidance
        does."""
        rows = self._rows(
            [_rule("Eat fish twice a week", guide_title="Irish Food Pyramid")],
            {"fish": 2},
        )
        assert rows[0]["source"] == "Irish Food Pyramid"


# ── prose and chips ───────────────────────────────────────────────────────

class TestProseAndChips:
    def test_prose_is_capped(self):
        """A grader handed 400 lines reads the first few and pads the rest."""
        rules = [_rule(f"Advice number {i}") for i in range(50)]
        assert gs.prose_context(rules).count("\n") < 10

    def test_prose_carries_its_source(self):
        text = gs.prose_context([_rule("Drink water", guide_title="HSE")])
        assert "Drink water" in text and "HSE" in text

    def test_an_empty_set_produces_no_text(self):
        assert gs.prose_context([]) == ""

    def test_a_matching_plate_gets_a_guideline_chip(self):
        """`guideline` has been a declared reason kind the UI renders with its
        own icon, that nothing ever emitted."""
        checkable, _ = gs.split([_rule("Eat fish at least 2 times a week")])
        chips = gs.reason_chips("salmon, fish stock, dill", checkable)
        assert chips and chips[0]["kind"] == "guideline"

    def test_a_limit_rule_never_becomes_a_chip(self):
        """A chip saying a dish helps you limit red meat, on a dish containing
        red meat, is nonsense."""
        checkable, _ = gs.split([_rule("Limit red meat to at most 3 a week")])
        assert gs.reason_chips("beef, red meat, onion", checkable) == []

    def test_a_plate_that_does_not_match_gets_nothing(self):
        checkable, _ = gs.split([_rule("Eat fish at least 2 times a week")])
        assert gs.reason_chips("lentils, carrot", checkable) == []


# ── it never breaks the plan ──────────────────────────────────────────────

class TestDegradation:
    def test_an_unconfigured_catalog_returns_nothing_quietly(self, monkeypatch):
        from backend import catalog

        monkeypatch.setattr(catalog, "DATA_API_URL", None)
        assert gs.fetch({"region": "IE"}) == []

    def test_the_weekly_checklist_falls_back_to_the_hardcoded_three(self, monkeypatch):
        from backend import catalog
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(catalog, "DATA_API_URL", None)
        rows = guideline_checklist({"fish": 2, "red meat": 1}, 21, {"region": "IE"})
        assert len(rows) == 3
        assert any("fish" in row["rule"] for row in rows)

    def test_a_catalog_failure_falls_back_rather_than_raising(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        def boom(*_a, **_k):
            raise RuntimeError("catalog is down")

        monkeypatch.setattr(gs, "fetch", boom)
        rows = guideline_checklist({"fish": 2}, 21, {"region": "IE"})
        assert len(rows) == 3

    def test_real_rules_replace_the_defaults_when_they_exist(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(gs, "fetch", lambda *a, **k: [
            _rule("Eat oily fish at least 1 time a week", guide_title="SI"),
        ])
        rows = guideline_checklist({"fish": 1}, 21, {"region": "SI"})
        assert len(rows) == 1 and rows[0]["source"] == "SI"

    def test_rules_that_are_all_prose_leave_the_defaults_in_place(self, monkeypatch):
        """A checklist of zero rows would look like "no guidance applies",
        which is the opposite of what a corpus of prose advice means."""
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(gs, "fetch", lambda *a, **k: [
            _rule("Choose wholegrain varieties where possible"),
        ])
        rows = guideline_checklist({"fish": 1}, 21, {"region": "SI"})
        assert len(rows) == 3

    def test_no_profile_means_the_defaults_without_a_lookup(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        def boom(*_a, **_k):
            raise AssertionError("must not query the catalog with no profile")

        monkeypatch.setattr(gs, "fetch", boom)
        assert len(guideline_checklist({"fish": 1}, 21)) == 3


class TestTheClient:
    def test_facets_are_anded_and_values_ored(self, monkeypatch):
        """An Irish adult wants Irish rules for adults, not Irish rules OR
        adult rules."""
        from backend import catalog

        captured = {}

        class _Client:
            def post(self, path, json=None):
                captured["path"] = path
                captured["json"] = json
                return {"results": []}

        monkeypatch.setattr(catalog, "DATA_API_URL", "http://catalog.test")
        monkeypatch.setattr(catalog.CatalogClient, "_get_client",
                            classmethod(lambda cls: _Client()))
        catalog.CatalogClient.clear_cache()
        catalog.CatalogClient.search_guidelines({
            "region": ["ie"], "life_stage": ["adult", "older_adult"],
        })
        fq = captured["json"]["fq"]
        assert "region:(ie)" in fq
        assert "life_stage:(adult OR older_adult)" in fq

    def test_only_published_rules_are_asked_for(self, monkeypatch):
        """A draft rule is someone's work in progress."""
        from backend import catalog

        captured = {}

        class _Client:
            def post(self, path, json=None):
                captured["json"] = json
                return {"results": []}

        monkeypatch.setattr(catalog, "DATA_API_URL", "http://catalog.test")
        monkeypatch.setattr(catalog.CatalogClient, "_get_client",
                            classmethod(lambda cls: _Client()))
        catalog.CatalogClient.clear_cache()
        catalog.CatalogClient.search_guidelines({"region": ["ie"]})
        assert "status:(active OR verified)" in captured["json"]["fq"]

    def test_a_failure_returns_nothing_rather_than_raising(self, monkeypatch):
        from backend import catalog

        class _Client:
            def post(self, *_a, **_k):
                raise RuntimeError("catalog is down")

        monkeypatch.setattr(catalog, "DATA_API_URL", "http://catalog.test")
        monkeypatch.setattr(catalog.CatalogClient, "_get_client",
                            classmethod(lambda cls: _Client()))
        catalog.CatalogClient.clear_cache()
        assert catalog.CatalogClient.search_guidelines({"region": ["ie"]}) == []

    def test_a_second_identical_query_is_served_from_cache(self, monkeypatch):
        from backend import catalog

        calls = []

        class _Client:
            def post(self, *_a, **_k):
                calls.append(1)
                return {"results": [{"rule_text": "Eat fish"}]}

        monkeypatch.setattr(catalog, "DATA_API_URL", "http://catalog.test")
        monkeypatch.setattr(catalog.CatalogClient, "_get_client",
                            classmethod(lambda cls: _Client()))
        catalog.CatalogClient.clear_cache()
        catalog.CatalogClient.search_guidelines({"region": ["ie"]})
        catalog.CatalogClient.search_guidelines({"region": ["ie"]})
        assert len(calls) == 1

    @pytest.mark.parametrize("payload,expected", [
        ({"results": [{"a": 1}]}, 1),
        ({"result": {"results": [{"a": 1}, {"b": 2}]}}, 2),
        ([{"a": 1}], 1),
        ({"unexpected": "shape"}, 0),
        (None, 0),
    ])
    def test_the_envelope_is_not_assumed(self, payload, expected):
        """An unrecognised shape returns nothing rather than a confident
        misreading."""
        from backend.catalog import _results_of

        assert len(_results_of(payload)) == expected
