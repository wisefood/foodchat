"""
Dietary guidelines from the WiseFood Data API: which rules, the text the judges
read, and the few rules a weekly checklist can honestly count.

The earlier version of these tests fed hand-written dicts to a client whose
filters matched nothing in the real catalog (`region:(ireland)`,
`life_stage:adult`) and passed. The parsing tests here read a recorded, trimmed
search response (`fixtures/guidelines_search.json`, demo catalog, 2026-09-17),
and the filter strings are pinned exactly as verified against the live API.

No network: the catalog client is stubbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from backend import catalog                              # noqa: E402
from models.guidelines import Guideline, GuidelineScope  # noqa: E402
from services import guidelines_service as gs            # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "guidelines_search.json").read_text(encoding="utf-8")
)
ROWS = FIXTURE["result"]["results"]


def _row(prefix: str) -> dict:
    (row,) = [r for r in ROWS if r["rule_text"].startswith(prefix)]
    return Guideline.model_validate(row).model_dump()


def _rule(text, **kw):
    base = {
        "id": kw.pop("id", text[:12]),
        "rule_text": text,
        "guideline_type": kw.pop("guideline_type", "food_based"),
        "topic": kw.pop("topic", []),
        "guide_urn": kw.pop("guide_urn", "urn:guide:healthy-eating-guidelines-20260330145212111"),
        "status": "active",
    }
    base.update(kw)
    return base


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_catalog(monkeypatch):
    """A configured catalog whose client records every request."""
    calls: list[dict] = []
    replies: list = []

    class _Client:
        def post(self, path, json=None):
            calls.append({"path": path, "json": json})
            reply = replies.pop(0) if replies else FIXTURE
            if isinstance(reply, Exception):
                raise reply
            return _Response(reply)

    monkeypatch.setattr(catalog, "DATA_API_URL", "http://catalog.test")
    monkeypatch.setattr(catalog.CatalogClient, "_get_client", classmethod(lambda cls: _Client()))
    catalog.CatalogClient.clear_cache()
    yield calls, replies
    catalog.CatalogClient.clear_cache()


# ── which rules ───────────────────────────────────────────────────────────

class TestTheScope:
    def test_everyone_is_an_irish_adult_until_the_profile_says_otherwise(self):
        scope = gs.resolve_scope({}, "daily")
        assert scope.regions == ("IE",)
        assert scope.life_stage == "adulthood"

    def test_the_daily_filter_is_exactly_what_the_live_api_accepted(self):
        assert gs.resolve_scope(None, "daily").fq() == [
            "status:active",
            "guide_region:(IE)",
            "life_stage:(adulthood) OR (*:* -life_stage:*)",
            "target_populations:(general_population OR adults) OR (*:* -target_populations:*)",
            "-guideline_type:(activity OR other)",
            "-frequency:(weekly OR monthly)",
        ]

    def test_a_week_keeps_its_weekly_rules(self):
        assert not any("frequency" in f for f in gs.resolve_scope(None, "weekly").fq())

    def test_untagged_rules_use_the_form_that_matches(self):
        """`(life_stage:(x) OR -life_stage:*)` is accepted by the catalog and
        silently matches nothing — which reads exactly like "no guidance"."""
        fq = gs.resolve_scope(None, "weekly").fq()
        assert not any("OR -" in f for f in fq)
        assert any("(*:* -life_stage:*)" in f for f in fq)

    def test_only_active_rules_are_asked_for(self):
        assert "status:active" in gs.resolve_scope(None, "weekly").fq()
        assert "verified" not in " ".join(gs.resolve_scope(None, "weekly").fq())

    @pytest.mark.parametrize("raw,code", [
        ("IE", "IE"), ("hu", "HU"), ("Hungary", "HU"), ("Slovenia", "SI"), ("Éire", "IE"),
    ])
    def test_a_region_in_the_profile_is_read_later(self, raw, code):
        assert gs.resolve_scope({"region": raw}).regions == (code,)

    def test_a_region_the_catalog_lacks_falls_back_to_the_default(self):
        assert gs.resolve_scope({"region": "France"}).regions == ("IE",)

    def test_an_age_group_becomes_a_life_stage_and_its_population(self):
        scope = gs.resolve_scope({"age_group": "senior"})
        assert scope.life_stage == "older_adulthood"
        assert "target_populations:(general_population OR elderly)" in " ".join(scope.fq())

    def test_an_override_wins_and_inherits_the_plan_type(self):
        scope = gs.resolve_scope({"region": "IE"}, "daily", GuidelineScope(regions=("HU", "SI")))
        assert scope.regions == ("HU", "SI") and scope.plan_type == "daily"
        assert "guide_region:(HU OR SI)" in scope.fq()

    def test_a_subset_of_rules_ignores_every_other_facet(self):
        scope = GuidelineScope(regions=("IE",), life_stage="adulthood", rule_ids=("a", "b"))
        assert scope.fq() == ['id:("a" OR "b")', "status:active"]

    def test_guide_urns_are_quoted(self):
        """URNs contain colons, which the query parser reads as field names."""
        scope = GuidelineScope(guide_urns=("urn:guide:okostanyer-diet-plate-20260407095500812",))
        assert 'guide_urn:("urn:guide:okostanyer-diet-plate-20260407095500812")' in scope.fq()

    def test_the_cache_tells_a_day_from_a_week(self):
        assert gs.resolve_scope(None, "daily").cache_key() != gs.resolve_scope(None, "weekly").cache_key()


# ── the client ────────────────────────────────────────────────────────────

class TestTheClient:
    def test_the_recorded_response_parses(self, fake_catalog):
        rules = catalog.CatalogClient.search(gs.resolve_scope(None, "weekly"))
        assert len(rules) == len(ROWS)
        assert all(isinstance(r, Guideline) for r in rules)

    def test_the_sdk_response_is_read_as_json(self, fake_catalog):
        """The SDK returns a `requests.Response`. Reading it as the payload
        itself returned [] on every call, and nothing complained."""
        assert catalog.CatalogClient.search(GuidelineScope())

    def test_the_request_never_sends_a_field_list(self, fake_catalog):
        """`fl` answers 500 on the live API."""
        calls, _ = fake_catalog
        scope = gs.resolve_scope(None, "daily")
        catalog.CatalogClient.search(scope)
        body = calls[0]["json"]
        assert calls[0]["path"] == "guidelines/search"
        assert body["fq"] == scope.fq()
        assert "fl" not in body
        assert body["fields"] == ["guide_region"]

    def test_pages_until_the_total(self, fake_catalog, monkeypatch):
        calls, replies = fake_catalog
        monkeypatch.setattr(catalog, "PAGE_SIZE", 5)
        first = {"success": True, "result": {"results": ROWS[:5], "total": len(ROWS)}}
        rest = {"success": True, "result": {"results": ROWS[5:], "total": len(ROWS)}}
        replies.extend([first, rest])
        rules = catalog.CatalogClient.search(GuidelineScope())
        assert len(rules) == len(ROWS)
        assert [c["json"]["offset"] for c in calls] == [0, 5]

    def test_the_same_sentence_twice_is_one_rule(self, fake_catalog):
        _, replies = fake_catalog
        twin = dict(ROWS[0], id="another")
        replies.append({"success": True, "result": {"results": [ROWS[0], twin], "total": 2}})
        assert len(catalog.CatalogClient.search(GuidelineScope())) == 1

    def test_null_lists_are_empty_lists(self):
        rule = Guideline.model_validate({"rule_text": "x", "topic": None, "life_stage": None})
        assert rule.topic == [] and rule.life_stage == []

    def test_an_api_error_backs_off_briefly_and_is_not_cached(self, fake_catalog):
        """A catalog that is down must not cost every plan turn a timeout, and
        must not be remembered as "no rules" once it is back."""
        calls, replies = fake_catalog
        replies.append({"success": False, "error": {"code": "server/internal"}})
        assert catalog.CatalogClient.search(GuidelineScope()) == []
        assert catalog.CatalogClient.search(GuidelineScope(regions=("HU",))) == []
        assert len(calls) == 1

        catalog.CatalogClient._down_until = 0.0
        assert catalog.CatalogClient.search(GuidelineScope())
        assert len(calls) == 2

    def test_a_failure_returns_nothing_rather_than_raising(self, fake_catalog):
        _, replies = fake_catalog
        replies.append(RuntimeError("catalog is down"))
        assert catalog.CatalogClient.search(GuidelineScope()) == []

    def test_a_second_identical_query_is_served_from_cache(self, fake_catalog):
        calls, _ = fake_catalog
        catalog.CatalogClient.search(GuidelineScope(regions=("IE",)))
        catalog.CatalogClient.search(GuidelineScope(regions=("IE",)))
        assert len(calls) == 1

    def test_an_unconfigured_catalog_asks_nobody(self, monkeypatch):
        def boom(cls):
            raise AssertionError("must not build a client")

        monkeypatch.setattr(catalog, "DATA_API_URL", None)
        monkeypatch.setattr(catalog.CatalogClient, "_get_client", classmethod(boom))
        assert catalog.CatalogClient.search(GuidelineScope()) == []
        assert gs.fetch({}) == []

    @pytest.mark.parametrize("payload,expected", [
        ({"results": [{"a": 1}]}, 1),
        ({"result": {"results": [{"a": 1}, {"b": 2}]}}, 2),
        ([{"a": 1}], 1),
        ({"unexpected": "shape"}, 0),
        (None, 0),
    ])
    def test_the_envelope_is_not_assumed(self, payload, expected):
        assert len(catalog._results_of(payload)) == expected


# ── the judges' text ──────────────────────────────────────────────────────

class TestTheJudgesText:
    def test_rules_are_numbered_under_a_header_naming_the_scope(self):
        text = gs.render([_row("Offer oily fish"), _row("Have 2 servings")],
                         gs.resolve_scope(None, "weekly"))
        lines = text.splitlines()
        assert lines[0].startswith("Dietary guidelines: Ireland, adults — 2 of 2 rules")
        assert "[G1] (weekly) Offer oily fish" in text
        assert "[G2] (daily) Have 2 servings a day" in text

    def test_the_source_is_a_readable_guide_name(self):
        text = gs.render([_rule("Eat fish", guide_urn="urn:guide:meat-shelf-irish-food-pyramid-20260320103505424")])
        assert "Sources: Meat shelf irish food pyramid." in text

    def test_the_most_checkable_rules_come_first(self):
        text = gs.render([
            _rule("Make mealtimes social", guideline_type="behavioral"),
            _rule("Choose wholegrains", guideline_type=None),
            _rule("Eat fish twice a week", frequency="weekly"),
        ])
        assert text.index("Eat fish") < text.index("Choose wholegrains") < text.index("Make mealtimes")

    def test_it_is_capped(self):
        """A judge handed 400 rules reads the first few and pads the rest."""
        rules = [_rule(f"Advice number {i}", id=str(i)) for i in range(400)]
        text = gs.render(rules)
        assert f"{gs.MAX_RULES} of 400 rules" in text.splitlines()[0]
        assert len(text) < gs.MAX_CHARS + 2000

    def test_a_subset_says_it_is_one(self):
        text = gs.render([_rule("Eat fish")], GuidelineScope(rule_ids=("x",)))
        assert text.startswith("Dietary guidelines: a selected subset")

    def test_no_rules_is_no_text(self):
        assert gs.render([]) == ""

    def test_the_text_never_raises(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("catalog is down")

        monkeypatch.setattr(gs, "fetch", boom)
        assert gs.guidelines_text("daily", {}) == ""

    def test_it_is_built_from_the_catalog(self, fake_catalog):
        text = gs.guidelines_text("weekly", {})
        assert "Offer oily fish" in text and "Eat fish at least once a week" in text

    def test_the_scorer_reads_the_same_function(self):
        from services import plan_scoring

        assert plan_scoring.guidelines_text is gs.guidelines_text


class TestTheDailyMetricsAreGivenTheRules:
    def test_the_daily_judge_receives_the_members_daily_rules(self, monkeypatch):
        from services import plan_quality
        from services.chat_service import ChatService

        seen = {}
        monkeypatch.setattr(gs, "guidelines_text",
                            lambda plan_type, profile=None: f"RULES:{plan_type}")
        monkeypatch.setattr(plan_quality, "metrics",
                            lambda plan, guidelines="": seen.update(g=guidelines) or {"fvs_count": 0})
        ChatService._compute_metrics(object(), "s1", object(), {"diet": []})
        assert seen["g"] == "RULES:daily"

    def test_an_n_day_plan_is_judged_by_weekly_rules(self, monkeypatch):
        from services import plan_quality
        from services.chat_service import ChatService

        seen = {}
        monkeypatch.setattr(gs, "guidelines_text",
                            lambda plan_type, profile=None: f"RULES:{plan_type}")
        monkeypatch.setattr(plan_quality, "metrics",
                            lambda plan, guidelines="": seen.update(g=guidelines) or {"fvs_count": 0})
        ChatService._compute_metrics(object(), "s1", object(), {}, plan_type="weekly")
        assert seen["g"] == "RULES:weekly"


# ── checkable vs prose ────────────────────────────────────────────────────

class TestSplitOnTheRealRules:
    def test_the_countable_weekly_rules(self):
        checkable, _ = gs.split([_row("Eat fish at least once a week"),
                                 _row("Eat sea fish regularly")])
        assert [(c["category"], c["direction"], c["low"]) for c in checkable] == [
            ("fish", "at least", 1), ("fish", "at least", 1),
        ]

    @pytest.mark.parametrize("prefix", ["Offer oily fish", "Offer red meat"])
    def test_a_bare_count_is_not_a_bound(self, prefix):
        """Checked as exact targets, "oily fish once a week" fails a week with
        two fish dinners and "red meat 3 times a week" fails a week with one."""
        checkable, prose = gs.split([_row(prefix)])
        assert not checkable and len(prose) == 1

    @pytest.mark.parametrize("prefix", [
        "Limit processed meat",               # every red-meat meal is not processed meat
        "Include at least 1 meat-free day",   # no counted category
        "Have sweets or desserts",            # behavioural
        "Limit foods and drinks high in fat", # behavioural, no category
        "Have 2 servings a day of meat",      # daily, and four categories at once
    ])
    def test_the_rest_is_prose(self, prefix):
        checkable, prose = gs.split([_row(prefix)])
        assert not checkable and len(prose) == 1

    def test_a_daily_fish_rule_is_not_counted_against_a_week(self):
        checkable, _ = gs.split([_rule("Eat fish twice a day", topic=["fish"])])
        assert not checkable

    def test_a_food_group_enum_names_no_category(self):
        """`protein_foods` is fish AND meat."""
        checkable, _ = gs.split([_rule("Eat these 2 times a week", food_groups=["protein_foods"])])
        assert not checkable

    @pytest.mark.parametrize("text,direction,value", [
        ("Eat fish at least 2 times a week", "at least", 2),
        ("Limit red meat to at most 3 times a week", "at most", 3),
        ("Limit red meat to 3 times a week", "at most", 3),
        ("Eat fish at least once a week", "at least", 1),
        ("Have fish no more than twice a week", "at most", 2),
    ])
    def test_the_direction_is_read_from_the_words(self, text, direction, value):
        checkable, _ = gs.split([_rule(text)])
        assert checkable, text
        parsed = checkable[0]
        assert parsed["direction"] == direction
        assert (parsed.get("low") or parsed.get("high")) == value

    @pytest.mark.parametrize("text,value", [
        ("Eat fish twice a week", 2), ("Have fish once a week", 1),
    ])
    def test_a_bare_count_is_read_but_not_checked(self, text, value):
        assert gs._parse_frequency(text) == {
            "direction": "about", "low": value, "high": value, "period": "week",
        }
        assert gs.split([_rule(text)])[0] == []

    def test_a_range_keeps_both_ends(self):
        checkable, _ = gs.split([_rule("Eat fish 1-2 times a week")])
        assert (checkable[0]["low"], checkable[0]["high"]) == (1, 2)

    def test_prose_advice_stays_prose(self):
        checkable, prose = gs.split([_rule("Choose wholegrain varieties where possible")])
        assert not checkable and len(prose) == 1


class TestTheStructuredQuantity:
    """`quantity` is a documented {operator, value, unit, period} object and
    empty on every rule today. Read first anyway, so the day an import fills it
    this is the accurate source."""

    def test_it_is_preferred_over_the_sentence(self):
        checkable, _ = gs.split([_rule(
            "Eat fish twice a week", topic=["fish"],
            quantity={"operator": "gte", "value": 3, "unit": "servings", "period": "week"},
        )])
        assert checkable[0]["direction"] == "at least" and checkable[0]["low"] == 3

    def test_an_unusable_quantity_falls_back_to_the_sentence(self):
        checkable, _ = gs.split([_rule(
            "Eat fish at least 2 times a week", topic=["fish"],
            quantity={"operator": "gte", "value": None},
        )])
        assert checkable[0]["direction"] == "at least" and checkable[0]["low"] == 2

    def test_a_quantity_in_an_unsupported_period_is_ignored(self):
        checkable, prose = gs.split([_rule(
            "Fish is good", topic=["fish"],
            quantity={"operator": "gte", "value": 2, "period": "month"},
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
        rows = self._rows([_rule("Limit red meat to at most 3 times a week")], {"red meat": 2})
        assert rows[0]["met"] is True

    def test_a_range_is_met_inside_it_and_missed_outside(self):
        rule = [_rule("Eat fish 1-2 times a week")]
        assert self._rows(rule, {"fish": 2})[0]["met"] is True
        assert self._rows(rule, {"fish": 5})[0]["met"] is False

    def test_the_rows_keep_the_shape_the_ui_already_renders(self):
        rows = self._rows([_rule("Eat fish at least 2 times a week")], {"fish": 2})
        assert {"rule", "target", "actual", "met"} <= set(rows[0])

    def test_the_row_names_where_the_rule_came_from(self):
        rows = self._rows([_row("Eat fish at least once a week")], {"fish": 1})
        assert rows[0]["source"] == "Okostanyer diet plate"


class TestChips:
    def test_a_matching_plate_gets_a_guideline_chip(self):
        checkable, _ = gs.split([_rule("Eat fish at least 2 times a week")])
        chips = gs.reason_chips("salmon, fish stock, dill", checkable)
        assert chips and chips[0]["kind"] == "guideline"

    def test_a_limit_rule_never_becomes_a_chip(self):
        checkable, _ = gs.split([_rule("Limit red meat to at most 3 times a week")])
        assert gs.reason_chips("beef, red meat, onion", checkable) == []


# ── it never breaks the plan ──────────────────────────────────────────────

class TestDegradation:
    def test_the_weekly_checklist_falls_back_to_the_hardcoded_three(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(catalog, "DATA_API_URL", None)
        rows = guideline_checklist({"fish": 2, "red meat": 1}, 21, {"diet": []})
        assert len(rows) == 3
        assert any("fish" in row["rule"] for row in rows)

    def test_a_catalog_failure_falls_back_rather_than_raising(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        def boom(*_a, **_k):
            raise RuntimeError("catalog is down")

        monkeypatch.setattr(gs, "fetch", boom)
        assert len(guideline_checklist({"fish": 2}, 21, {"diet": []})) == 3

    def test_real_rules_replace_the_defaults_when_they_exist(self, fake_catalog):
        from services.weekly_planner.explainability import guideline_checklist

        rows = guideline_checklist({"fish": 1, "red meat": 3}, 21, {"diet": []})
        assert [row["rule"][:14] for row in rows] == ["Eat fish at le", "Eat sea fish r"]
        assert all(row["met"] for row in rows)

    def test_the_irish_adult_set_keeps_the_built_in_three(self, monkeypatch):
        """Its only countable rules are bare counts, so nothing replaces the
        rows members already see."""
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(gs, "fetch", lambda *a, **k: [
            _row("Offer oily fish"), _row("Offer red meat"), _row("Limit processed meat"),
        ])
        rows = guideline_checklist({"fish": 2, "red meat": 1}, 21, {"diet": []})
        assert [row["rule"] for row in rows] == [
            "eat fish 1–2 times a week", "limit red meat", "make most meals plant-based",
        ]

    def test_the_checklist_asks_for_weekly_rules(self, fake_catalog):
        from services.weekly_planner.explainability import guideline_checklist

        calls, _ = fake_catalog
        guideline_checklist({"fish": 1}, 21, {"diet": []})
        assert not any("frequency" in f for f in calls[0]["json"]["fq"])

    def test_rules_that_are_all_prose_leave_the_defaults_in_place(self, monkeypatch):
        """A checklist of zero rows would look like "no guidance applies"."""
        from services.weekly_planner.explainability import guideline_checklist

        monkeypatch.setattr(gs, "fetch", lambda *a, **k: [
            _rule("Choose wholegrain varieties where possible"),
        ])
        assert len(guideline_checklist({"fish": 1}, 21, {"diet": []})) == 3

    def test_no_profile_means_the_defaults_without_a_lookup(self, monkeypatch):
        from services.weekly_planner.explainability import guideline_checklist

        def boom(*_a, **_k):
            raise AssertionError("must not query the catalog with no profile")

        monkeypatch.setattr(gs, "fetch", boom)
        assert len(guideline_checklist({"fish": 1}, 21)) == 3
