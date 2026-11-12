"""The household behind a session is kept one member at a time.

`merge_profiles` used to be the only record of who a session cooks for, and a
merged profile cannot answer "how good is this for Anna?" — the question every
group strategy other than satisfy-all starts from. `HouseholdContext` keeps
each diner's own profile; the merged dict is a view derived from it.

These tests pin three things: the per-member record is complete and survives
the session snapshot's JSON round-trip, the merged view is unchanged for every
existing consumer, and the two cannot drift when a memory is accepted
mid-session.
"""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from models.household import (  # noqa: E402
    HOUSEHOLD_KEY, SATISFY_ALL, DinerProfile, HouseholdContext,
)
from services.profile_service import ProfileService  # noqa: E402


def _me():
    return {
        "member_id": "m-me", "diet": ["omnivore"], "allergies": ["peanuts"],
        "food_likes": ["salmon"], "food_dislikes": ["olives"],
        "preferences": ["2000 calories target"], "dietary_goals": ["reduce_fat"],
        "calorie_target": 2000, "sex": "female", "age_group": "adult",
        "favorite_recipe_ids": ["r-1"], "memory_log": [{"kind": "like"}],
    }


def _anna():
    return {
        "member_id": "m-anna", "diet": ["vegetarian"], "allergies": [],
        "food_likes": ["halloumi", "salmon"], "food_dislikes": ["mushrooms"],
        "preferences": [], "dietary_goals": ["increase_protein"],
        "calorie_target": "1600", "age_group": "adult",
    }


def _tom():
    return {
        "member_id": "m-tom", "diet": [], "allergies": ["shellfish"],
        "food_likes": [], "food_dislikes": [], "preferences": [],
        "age_group": "child",
    }


def _merged():
    return ProfileService.merge_profiles(
        _me(), [_anna(), _tom()], ["Me", "Anna", "Tom"],
    )


class TestEveryDinerKeepsTheirOwnProfile:
    def test_the_merged_profile_carries_the_members_it_was_merged_from(self):
        household = HouseholdContext.from_profile(_merged())

        assert household is not None and household.is_group
        assert household.names == ["Me", "Anna", "Tom"]
        assert household.member_ids == ["m-me", "m-anna", "m-tom"]
        assert household.strategy == SATISFY_ALL

    def test_what_the_merge_flattens_is_still_answerable_per_member(self):
        """The union says the table avoids shellfish and eats vegetarian. Only
        the per-member record says WHO — and who is a child."""
        household = HouseholdContext.from_profile(_merged())
        anna, tom = household.diner("Anna"), household.diner("m-tom")

        assert anna.diet == ["vegetarian"] and anna.allergies == []
        assert anna.food_likes == ["halloumi", "salmon"]
        assert anna.dietary_goals == ["increase_protein"]
        assert anna.calorie_target == 1600.0, "the gateway sends strings too"
        assert tom.allergies == ["shellfish"] and tom.age_group == "child"
        assert household.primary.primary and not anna.primary

    def test_only_planning_fields_are_copied_per_diner(self):
        """Memory bookkeeping and favorites are the session member's own."""
        fields = set(DinerProfile.model_fields)

        assert "memory_log" not in fields and "favorite_recipe_ids" not in fields

    def test_it_survives_the_session_snapshot(self):
        """The profile is persisted as JSON on the session row — the context
        has to come back identical on another replica."""
        restored = json.loads(json.dumps(_merged()))

        assert HouseholdContext.from_profile(restored) == HouseholdContext.from_profile(_merged())

    @pytest.mark.parametrize("profile", [
        None, {}, {HOUSEHOLD_KEY: "nonsense"},
        {HOUSEHOLD_KEY: {"strategy": "telepathy", "diners": []}},
    ])
    def test_a_session_without_one_is_not_an_error(self, profile):
        """Sessions created before this existed, or a snapshot gone bad."""
        assert HouseholdContext.from_profile(profile) is None

    def test_an_unknown_strategy_is_refused_not_guessed(self):
        with pytest.raises(ValueError):
            HouseholdContext(strategy="telepathy")

    def test_messy_gateway_values_do_not_break_a_diner(self):
        diner = DinerProfile.from_profile(
            {"diet": "vegan", "allergies": None, "calorie_target": "about 2000"},
        )

        assert diner.diet == ["vegan"] and diner.allergies == []
        assert diner.calorie_target is None


class TestTheMergedViewIsDerivedNotStored:
    def test_the_view_is_what_the_merge_always_returned(self):
        merged = _merged()

        assert set(merged["allergies"]) == {"peanuts", "shellfish"}
        assert set(merged["diet"]) == {"omnivore", "vegetarian"}
        # Salmon is liked by two diners, one of whom is vegetarian: see
        # TestALikeSomebodyCannotEat.
        assert merged["food_likes"] == ["halloumi"]
        assert merged["favorite_recipe_ids"] == ["r-1"]
        assert merged["calorie_target"] == 2000, "the primary's, not Anna's"
        assert merged["constraint_origins"]["allergies"]["shellfish"] == ["Tom"]
        assert merged["goal_reconciliation"] == [
            {"slug": "reduce_fat", "member": "Me", "applied": "target"},
            {"slug": "increase_protein", "member": "Anna", "applied": "soft"},
        ]

    def test_rebuilding_the_view_from_the_stored_context_gives_it_back(self):
        merged = _merged()
        household = HouseholdContext.from_profile(merged)

        assert household.merged_view(_me()) == merged

    def test_the_primary_profile_is_not_mutated(self):
        me = _me()
        ProfileService.merge_profiles(me, [_anna()], ["Me", "Anna"])

        assert me == _me()


def _table(*diners):
    """`(name, profile)` pairs -> the merged view, first diner primary."""
    names = [name for name, _ in diners]
    profiles = [profile for _, profile in diners]
    return ProfileService.merge_profiles(profiles[0], profiles[1:], names)


class TestALikeSomebodyCannotEat:
    """A like is the table's only if everyone at it can eat the thing.

    Found live: Liza likes milk, cheese and yogurt and cooks for Miranda, who
    is allergic to milk. The likes stayed in the merged profile, so FoodChat
    asked Liza why "her" milk allergy disagreed with her liking for milk, and
    put "you like milk" on a coconut-milk korma.
    """

    LIZA = {"allergies": ["peanut"], "food_likes": ["lamb", "milk", "cheese", "yogurt", "greek"]}
    MIRANDA = {"diet": ["flexitarian"], "allergies": ["milk"]}
    VEGETARIAN = {"diet": ["vegetarian"], "food_likes": ["lentils"]}

    def test_an_allergy_at_the_table_sets_the_like_aside(self):
        merged = _table(("Liza", self.LIZA), ("Miranda", self.MIRANDA))

        assert merged["food_likes"] == ["lamb", "greek"]
        assert merged["like_conflicts"] == [
            {"like": like, "liked_by": ["Liza"], "blocked_by": "allergy",
             "value": "milk", "members": ["Miranda"]}
            for like in ("milk", "cheese", "yogurt")
        ]

    def test_a_diet_at_the_table_sets_the_like_aside(self):
        merged = _table(("Liza", self.LIZA), ("Sam", self.VEGETARIAN))

        assert merged["food_likes"] == ["milk", "cheese", "yogurt", "greek", "lentils"]
        assert merged["like_conflicts"] == [{
            "like": "lamb", "liked_by": ["Liza"], "blocked_by": "diet",
            "value": "vegetarian", "members": ["Sam"],
        }]

    def test_a_dislike_at_the_table_sets_the_like_aside(self):
        merged = _table(
            ("Liza", {"food_likes": ["smoked salmon", "rice"]}),
            ("Tom", {"food_dislikes": ["salmon"]}),
        )

        assert merged["food_likes"] == ["rice"]
        assert merged["like_conflicts"][0]["blocked_by"] == "dislike"
        assert merged["like_conflicts"][0]["members"] == ["Tom"]

    def test_the_strongest_reason_is_the_one_reported(self):
        merged = _table(
            ("Liza", {"food_likes": ["cheese"]}),
            ("Tom", {"food_dislikes": ["cheese"]}),
            ("Miranda", self.MIRANDA),
        )

        conflict = merged["like_conflicts"][0]
        assert (conflict["blocked_by"], conflict["members"]) == ("allergy", ["Miranda"])

    def test_plant_milk_is_not_what_a_milk_allergy_blocks(self):
        merged = _table(("Liza", {"food_likes": ["coconut milk", "peanut butter"]}),
                        ("Miranda", self.MIRANDA))

        assert merged["food_likes"] == ["coconut milk", "peanut butter"]
        assert merged["like_conflicts"] == []

    def test_a_label_that_filters_nothing_blocks_nothing(self):
        """Flexitarian is a description, not a filter."""
        merged = _table(("Liza", {"food_likes": ["lamb"]}), ("Miranda", self.MIRANDA))

        assert merged["food_likes"] == ["lamb"]

    def test_the_diner_keeps_the_like_on_their_own_record(self):
        """Only the table's view drops it — a later strategy can still serve
        Liza cheese on a night Miranda is out."""
        merged = _table(("Liza", self.LIZA), ("Miranda", self.MIRANDA))

        assert "cheese" in HouseholdContext.from_profile(merged).primary.food_likes

    def test_a_like_shared_by_two_diners_names_both(self):
        merged = _merged()

        assert merged["like_conflicts"] == [{
            "like": "salmon", "liked_by": ["Me", "Anna"], "blocked_by": "diet",
            "value": "vegetarian", "members": ["Anna"],
        }]

    def test_the_ledger_says_what_was_set_aside_and_for_whom(self):
        from services.transparency import constraints_ledger, split_ledger

        ledger = constraints_ledger(_table(("Liza", self.LIZA), ("Miranda", self.MIRANDA)))
        row = next(r for r in ledger if "cheese" in r["constraint"])

        assert row["status"] == "relaxed" and row["members"] == ["Liza"]
        assert "Miranda's allergy" in row["constraint"]
        assert row["constraint"] in split_ledger(ledger, limit=20)[1]


class TestWhyADishIsOnTheTable:
    def _reasons(self, profile, ingredients):
        from services.transparency import match_reasons

        return match_reasons("r-9", ingredients, profile, set())

    def test_another_diners_like_is_not_yours(self):
        merged = _table(("Liza", {"food_likes": ["lamb"]}), ("Sam", {"food_likes": ["lentils"]}))

        assert self._reasons(merged, "red lentils, onion") == [
            {"kind": "diner", "label": "Sam likes lentils"},
        ]
        assert self._reasons(merged, "lamb shoulder")[0]["label"] == "you like lamb"

    def test_a_like_you_share_is_still_yours(self):
        merged = _table(("Liza", {"food_likes": ["rice"]}), ("Sam", {"food_likes": ["rice"]}))

        assert self._reasons(merged, "brown rice")[0]["label"] == "you like rice"

    def test_coconut_milk_is_not_the_milk_you_like(self):
        solo = {"food_likes": ["milk"]}

        assert self._reasons(solo, "1.5 cups coconut milk, mushrooms") == []
        assert self._reasons(solo, "200 ml milk, oats")[0]["label"] == "you like milk"


class _Profiles(ProfileService):
    """The gateway, replaced: three members and a call log."""

    def __init__(self):
        self.calls = []
        self._members = {"m-me": ("Me", _me), "m-anna": ("Anna", _anna), "m-tom": ("Tom", _tom)}

    def get_member_profile(self, member_id):
        self.calls.append(member_id)
        return self._members[member_id][1]()

    def get_member_name(self, member_id):
        return self._members[member_id][0]

    def get_member_favorites(self, member_id):
        return ["r-1"]

    def get_member_adapted_recipes(self, member_id):
        return {}


class TestTheSessionProfile:
    """`POST /sessions` and `PUT /diners` build the snapshot in one place."""

    def test_a_solo_session_is_a_household_of_one(self):
        profile = _Profiles().build_session_profile("m-me")
        household = HouseholdContext.from_profile(profile)

        assert not household.is_group and household.member_ids == ["m-me"]
        assert "cooking_for_names" not in profile, "creation never named a solo diner"
        assert "constraint_origins" not in profile, "and never invented a household"

    def test_extra_diners_are_merged_and_named(self):
        profile = _Profiles().build_session_profile("m-me", ["m-anna", "m-tom"])

        assert profile["cooking_for"] == ["m-me", "m-anna", "m-tom"]
        assert profile["cooking_for_names"] == ["Me", "Anna", "Tom"]
        assert HouseholdContext.from_profile(profile).member_ids == profile["cooking_for"]

    def test_the_owner_and_duplicates_in_the_selection_are_not_extra_diners(self):
        """The UI sends the current member among the selected ones."""
        service = _Profiles()
        profile = service.build_session_profile("m-me", ["m-me", "m-anna", "m-anna", ""])

        assert profile["cooking_for"] == ["m-me", "m-anna"]
        assert service.calls == ["m-me", "m-anna"], "one fetch per diner"

    def test_setting_diners_back_to_solo_still_echoes_the_selection(self):
        profile = _Profiles().build_session_profile("m-me", [], always_name_diners=True)

        assert profile["cooking_for"] == ["m-me"]
        assert profile["cooking_for_names"] == ["Me"]
        assert HouseholdContext.from_profile(profile).names == ["Me"]


class TestTheReconcilerIsToldWhoseRestrictionItIs:
    """Live: with Miranda at the table, Liza was asked "Your profile lists
    allergies to milk, but you also indicated you like milk…". The allergy is
    Miranda's, and the reconciler had been handed a flat list."""

    def _prompt(self, profile):
        from agents import QueryReconciler

        seen = {}

        class _Client:
            def invoke(self, messages, config=None):
                seen["text"] = "\n".join(str(m.content) for m in messages)
                return type("R", (), {"content": "{}"})()

        reconciler = QueryReconciler.__new__(QueryReconciler)
        reconciler.query_reconciler = _Client()
        reconciler.reconcile("Plan my meals for tomorrow.", profile)
        return seen["text"]

    def test_a_household_prompt_names_the_allergic_diner_and_drops_the_clash(self):
        text = self._prompt(_table(
            ("Liza", TestALikeSomebodyCannotEat.LIZA),
            ("Miranda", TestALikeSomebodyCannotEat.MIRANDA),
        ))

        assert "milk (Miranda)" in text and "peanut (Liza)" in text
        assert "the user is Liza" in text
        assert "likes milk" not in text and "likes cheese" not in text
        assert "likes lamb" in text

    def test_a_solo_prompt_is_unchanged(self):
        text = self._prompt({"allergies": ["peanut"], "diet": [], "food_likes": ["lamb"]})

        assert "Allergies: ['peanut']" in text and "the user is" not in text


class TestADinerWhoCannotBeLoadedOrIsNotOurs:
    def _service(self, broken=(), households=None):
        households = households or {}

        class _Flaky(_Profiles):
            def get_member_profile(self, member_id):
                if member_id in broken:
                    raise RuntimeError("gateway timeout")
                profile = super().get_member_profile(member_id)
                if member_id in households:
                    profile["household_id"] = households[member_id]
                return profile

        return _Flaky()

    def test_one_unreachable_diner_does_not_sink_the_session(self):
        profile = self._service(broken={"m-tom"}).build_session_profile("m-me", ["m-anna", "m-tom"])

        assert profile["cooking_for"] == ["m-me", "m-anna"], "Tom is not planned for"
        assert profile["diners_unavailable"] == [{"member_id": "m-tom", "name": "Tom"}]
        assert "shellfish" not in profile["allergies"], "and nothing is claimed on his behalf"

    def test_the_plan_says_who_it_does_not_account_for(self):
        from services.transparency import constraints_ledger, split_ledger

        profile = self._service(broken={"m-tom"}).build_session_profile("m-me", ["m-anna", "m-tom"])
        ledger = constraints_ledger(profile)
        row = next(r for r in ledger if r["source"] == "household")

        assert row["members"] == ["Tom"] and row["status"] != "satisfied"
        assert row["constraint"] in split_ledger(ledger, limit=20)[1]

    def test_the_only_extra_diner_failing_leaves_a_solo_session_that_says_so(self):
        profile = self._service(broken={"m-anna"}).build_session_profile(
            "m-me", ["m-anna"], always_name_diners=True,
        )

        assert profile["cooking_for"] == ["m-me"]
        assert [d["name"] for d in profile["diners_unavailable"]] == ["Anna"]

    def test_the_session_member_failing_is_still_an_error(self):
        with pytest.raises(RuntimeError):
            self._service(broken={"m-me"}).build_session_profile("m-me", ["m-anna"])

    def test_a_member_of_another_household_is_refused(self):
        from services.profile_service import DinerNotInHousehold

        service = self._service(households={"m-me": "h-1", "m-anna": "h-1", "m-tom": "h-2"})

        with pytest.raises(DinerNotInHousehold):
            service.build_session_profile("m-me", ["m-anna", "m-tom"])
        assert issubclass(DinerNotInHousehold, ValueError), "the router's 404, never 403"

    def test_an_unknown_household_is_not_a_mismatch(self):
        """Older gateway rows carry no household id; refusing them would lock
        every such family out of household planning."""
        service = self._service(households={"m-me": "h-1"})

        assert service.build_session_profile("m-me", ["m-anna"])["cooking_for"] == ["m-me", "m-anna"]


class TestWhatTheOtherDinersRejected:
    class _Feedback:
        def __init__(self, by_member):
            self.by_member = by_member

        def get_signals(self, member_id):
            from services.feedback_service import FeedbackSignals

            down, text = self.by_member.get(member_id, ([], ""))
            return FeedbackSignals(downvoted_recipe_ids=list(down), history_text=text)

    def _signals(self, profile, by_member):
        from services.feedback_service import household_signals

        return household_signals(self._Feedback(by_member), "m-me", profile)

    def test_a_dish_anyone_at_the_table_voted_down_stays_out(self):
        signals = self._signals(_Profiles().build_session_profile("m-me", ["m-anna"]), {
            "m-me": (["r-1"], "Disliked: Stew"),
            "m-anna": (["r-2", "r-1"], "Disliked: Curry"),
        })

        assert signals.downvoted_recipe_ids == ["r-1", "r-2"]
        assert signals.history_text == "Disliked: Stew\nAnna - Disliked: Curry"

    def test_someone_not_at_the_table_has_no_say(self):
        signals = self._signals(_Profiles().build_session_profile("m-me"), {
            "m-me": (["r-1"], ""), "m-anna": (["r-2"], ""),
        })

        assert signals.downvoted_recipe_ids == ["r-1"]


class TestTheOtherDinersFavourites:
    """A ranking boost for the table — never the session member's own list,
    which also drives the favourites offer and its decline."""

    def _profile(self):
        class _WithFavourites(_Profiles):
            def get_member_favorites(self, member_id):
                return {"m-me": ["r-1"], "m-anna": ["r-2", "r-1"], "m-tom": ["r-2"]}[member_id]

        return _WithFavourites().build_session_profile("m-me", ["m-anna", "m-tom"])

    def test_they_are_kept_apart_from_the_session_members(self):
        profile = self._profile()

        assert profile["favorite_recipe_ids"] == ["r-1"]
        assert profile["household_favorites"] == {"r-2": ["Anna", "Tom"], "r-1": ["Anna"]}

    def test_they_join_the_boost_and_survive_a_declined_offer(self):
        from models.household import table_favorites

        profile = self._profile()

        assert table_favorites(profile) == ["r-1", "r-2"]
        assert table_favorites(profile, own=False) == ["r-2", "r-1"]
        assert table_favorites({"favorite_recipe_ids": ["r-1"]}) == ["r-1"], "solo: unchanged"

    def test_the_chip_says_whose_favourite_it_is(self):
        from services.transparency import match_reasons

        profile = self._profile()

        assert match_reasons("r-2", "", profile, set()) == [
            {"kind": "diner", "label": "one of Anna's and Tom's favorites"},
        ]
        assert match_reasons("r-1", "", profile, set())[0]["label"] == "one of your favorites"


class TestWhoseGuidelinesThePlanIsJudgedBy:
    def _rows(self, profile):
        from services.transparency import constraints_ledger

        return [r for r in constraints_ledger(profile) if r["source"] == "guidelines"]

    def test_a_child_at_an_adults_table_is_named_as_not_judged(self):
        """Tom is a child; the scope is resolved for the session member."""
        rows = self._rows(_merged())

        assert [r["members"] for r in rows] == [["Tom"]]
        assert rows[0]["status"] == "unsupported"
        assert "school-age children" in rows[0]["constraint"]
        assert "adults" in rows[0]["detail"]

    def test_a_table_of_one_life_stage_says_nothing(self):
        merged = ProfileService.merge_profiles(_me(), [_anna()], ["Me", "Anna"])

        assert self._rows(merged) == []
        assert self._rows(_Profiles().build_session_profile("m-me")) == []


class TestAnEmptyPlanSaysWhoseConstraintsStoodInTheWay:
    def test_the_table_s_constraints_carry_their_diner(self):
        from services.chat_service import no_plan_message

        text = no_plan_message(_merged())

        assert "this table's current constraints" in text
        assert "peanuts (Me)" in text and "shellfish (Tom)" in text
        assert "mushrooms (Anna)" in text

    def test_a_solo_message_is_as_it_was(self):
        from services.chat_service import no_plan_message

        text = no_plan_message({"allergies": ["peanuts"], "food_dislikes": ["olives"]})

        assert "your current constraints" in text
        assert "allergens excluded: peanuts;" in text and "(" not in text.split("constraints", 1)[0]


class TestAStatementAboutAnotherDiner:
    """"Miranda hates mushrooms" was offered to the session member as "It
    seems you don't like mushrooms — remember this?"."""

    def _service(self, candidates):
        from services.memory_service import MemoryService

        class _Extractor:
            def extract(self, message):
                return [dict(c) for c in candidates]

        class _Sessions:
            persisted = 0

            def persist_profile(self, session_id):
                _Sessions.persisted += 1

        service = MemoryService.__new__(MemoryService)
        service.extractor = _Extractor()
        service.session_service = _Sessions()
        return service

    def _session(self, profile):
        return type("S", (), {"session_id": "s-1", "member_id": "m-me", "user_profile": profile})()

    def _candidate(self, **over):
        return {"kind": "dislike", "value": "olives", "statement": "…", "evidence": "",
                "confidence": "high", **over}

    def test_it_is_not_offered_as_the_session_members_memory(self):
        session = self._session(_merged())
        service = self._service([self._candidate(value="beetroot", subject="Anna")])

        assert service.suggest(session, "Anna hates beetroot") == []

    def test_it_applies_to_that_diner_for_this_session(self):
        session = self._session(_merged())
        self._service([self._candidate(value="beetroot", subject="anna")]).suggest(session, "…")
        profile = session.user_profile
        household = HouseholdContext.from_profile(profile)

        assert "beetroot" in profile["food_dislikes"]
        assert profile["constraint_origins"]["food_dislikes"]["beetroot"] == ["Anna"]
        assert "beetroot" in household.diner("Anna").food_dislikes
        assert "beetroot" not in household.primary.food_dislikes

    def test_an_allergy_applies_at_any_confidence_and_a_dislike_does_not(self):
        session = self._session(_merged())
        self._service([
            self._candidate(kind="allergy_hint", value="sesame", subject="Tom", confidence="low"),
            self._candidate(value="kale", subject="Tom", confidence="medium"),
        ]).suggest(session, "…")

        assert "sesame" in session.user_profile["allergies"]
        assert "kale" not in session.user_profile["food_dislikes"]

    def test_someone_who_is_not_at_the_table_changes_nothing(self):
        session = self._session(_merged())
        before = json.dumps(session.user_profile, sort_keys=True)
        suggestions = self._service([self._candidate(subject="my boss")]).suggest(session, "…")

        assert suggestions == [] and json.dumps(session.user_profile, sort_keys=True) == before

    def test_a_statement_about_the_user_is_still_offered(self):
        session = self._session(_merged())
        service = self._service([
            self._candidate(value="kale", subject="user"),
            self._candidate(value="leeks"),  # no subject: an older extractor
        ])

        assert [s["value"] for s in service.suggest(session, "…")] == ["kale", "leeks"]


class TestAMemoryAcceptedMidSession:
    """`MemoryService` updates the merged profile directly, bypassing the
    merge — the per-member record must not be left behind."""

    def _apply(self, profile, kind, value):
        from services.memory_service import MemoryService

        class _Session:
            session_id = "s-1"
            user_profile = profile

        class _Sessions:
            def persist_profile(self, session_id):
                pass

        service = MemoryService.__new__(MemoryService)
        service.session_service = _Sessions()
        service._apply_to_session_profile(_Session(), kind, value)
        return HouseholdContext.from_profile(profile)

    def test_it_lands_on_the_primary_diner(self):
        household = self._apply(_merged(), "allergy_hint", "sesame")

        assert "sesame" in household.primary.allergies
        assert "sesame" not in household.diner("Anna").allergies

    def test_a_stated_diet_replaces_omnivore_for_that_member_only(self):
        household = self._apply(_merged(), "diet", "pescatarian")

        assert household.primary.diet == ["pescatarian"]
        assert household.diner("Anna").diet == ["vegetarian"]

    def test_a_dislike_leaves_the_members_likes(self):
        household = self._apply(_merged(), "dislike", "salmon")

        assert "salmon" in household.primary.food_dislikes
        assert "salmon" not in household.primary.food_likes
        assert "salmon" in household.diner("Anna").food_likes, "Anna still likes it"

    def test_a_session_with_no_context_is_left_alone(self):
        profile = {"allergies": []}

        assert self._apply(profile, "allergy_hint", "sesame") is None
        assert HOUSEHOLD_KEY not in profile
