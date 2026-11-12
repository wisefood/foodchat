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
        assert merged["food_likes"] == ["salmon", "halloumi"]
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
