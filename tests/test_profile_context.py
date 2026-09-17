"""Who the member is, not only what they eat.

`guidelines_service.resolve_scope` has always read `profile["region"]` and
`profile["age_group"]`, and the profile carried neither. `_map_profile` maps
the member's PROFILE object, where the region is not — `age_group` is a
first-class field on the member and `region` belongs to their household — so
every member fell through to the deployment default. Regional guidelines were
applied and an Irish household and a Hungarian one were judged by the same
rules, which is worse than not filtering at all: it looks like personalisation.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from services.guidelines_service import resolve_scope       # noqa: E402
from services.profile_service import ProfileService         # noqa: E402


class _Household:
    def __init__(self, region):
        self.region = region


class _Client:
    """Just the two lookups `_attach_context` makes."""

    def __init__(self, region="IE"):
        self.region = region
        self.household_calls = 0

    @property
    def households(self):
        client = self

        class _Proxy:
            @staticmethod
            def get(household_id):
                client.household_calls += 1
                if client.region is None:
                    raise RuntimeError("catalog down")
                return _Household(client.region)
        return _Proxy()


class _Member:
    def __init__(self, age_group="adult", household_id="h-1"):
        self.age_group = age_group
        self.household_id = household_id


def _svc():
    service = ProfileService.__new__(ProfileService)
    service._region_cache = {}
    return service


class TestTheProfileCarriesTheContext:
    def test_the_age_group_comes_from_the_member(self):
        profile = {}
        _svc()._attach_context(_Client(), _Member(age_group="Child"), profile)

        assert profile["age_group"] == "child"

    def test_the_region_comes_from_the_household(self):
        profile = {}
        _svc()._attach_context(_Client(region="HU"), _Member(), profile)

        assert profile["region"] == "HU"
        assert profile["household_id"] == "h-1"

    def test_the_household_is_read_once_per_hour_not_once_per_plan(self):
        """This sits on the profile fetch, which is on the path of every plan,
        and a household's country does not change between turns."""
        service, client = _svc(), _Client()
        for _ in range(4):
            service._attach_context(client, _Member(), {})

        assert client.household_calls == 1

    def test_a_household_lookup_that_fails_costs_only_the_region(self):
        """Best-effort by construction, like everything the catalog feeds."""
        profile = {}
        _svc()._attach_context(_Client(region=None), _Member(age_group="senior"), profile)

        assert "region" not in profile
        assert profile["age_group"] == "senior", "the rest of the context survives"

    def test_a_member_with_no_household_is_not_an_error(self):
        profile = {}
        _svc()._attach_context(_Client(), _Member(household_id=""), profile)

        assert "region" not in profile and "household_id" not in profile


class TestAndTheScopeFollows:
    """The point of the two fields: different members, different rules."""

    @pytest.mark.parametrize("region,age_group,expected_region,expected_stage", [
        ("IE", "adult", "IE", "adulthood"),
        ("HU", "child", "HU", "school_age"),
        ("Hungary", "senior", "HU", "older_adulthood"),
        ("SI", "teen", "SI", "adolescence"),
    ])
    def test_the_scope_is_the_members_own(self, region, age_group,
                                          expected_region, expected_stage):
        profile = {}
        _svc()._attach_context(_Client(region=region), _Member(age_group=age_group), profile)
        scope = resolve_scope(profile, "daily")

        assert scope.regions == (expected_region,)
        assert scope.life_stage == expected_stage

    def test_two_households_are_not_judged_alike(self):
        irish, hungarian = {}, {}
        _svc()._attach_context(_Client(region="IE"), _Member(), irish)
        _svc()._attach_context(_Client(region="HU"), _Member(), hungarian)

        assert resolve_scope(irish).regions != resolve_scope(hungarian).regions

    def test_without_the_context_everyone_gets_the_default(self):
        """What the behaviour WAS, kept as the fallback it should have been."""
        scope = resolve_scope({}, "daily")

        assert scope.regions and scope.life_stage, "a default, not nothing"


class TestTheWiringItself:
    """The tests above call `_attach_context` directly, so every one of them
    passes with the CALL SITE deleted. This one runs `get_member_profile`."""

    def test_get_member_profile_attaches_it(self):
        from contextlib import contextmanager

        class _Members:
            @staticmethod
            def get(member_id):
                member = _Member(age_group="child", household_id="h-9")
                member.profile = {
                    "dietary_groups": [], "allergies": [],
                    "nutritional_preferences": {}, "properties": {},
                }
                return member

        client = _Client(region="HU")
        client.members = _Members()

        class _Pool:
            @staticmethod
            @contextmanager
            def client():
                yield client

        service = ProfileService.__new__(ProfileService)
        service._region_cache = {}
        service.client_pool = _Pool()

        profile = service.get_member_profile("m-1")

        assert profile["region"] == "HU"
        assert profile["age_group"] == "child"
        assert resolve_scope(profile, "daily").regions == ("HU",)
