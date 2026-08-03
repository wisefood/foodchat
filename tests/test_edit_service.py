"""Edit-service behaviours pinned at the unit level."""

import sys

import pytest

sys.path.insert(0, "src")


class TestNamedDishDirectives:
    """"i want apple pie for breakfast" must chase the pie, not the slot.

    The directive classified as an unverified predicate that every breakfast
    candidate trivially passes, so the member got the slot's top-ranked
    muffins while the reply claimed a "best match for apple pie". A name is
    the member overriding the slot taxonomy, and it resolves by name first.
    """

    @pytest.mark.parametrize("directive", [
        "apple pie", "moussaka", "chicken fried rice", "beef stew",
    ])
    def test_dish_names_are_recognised(self, directive):
        from services.edit_service import _names_a_dish
        assert _names_a_dish(directive)

    @pytest.mark.parametrize("directive", [
        "different", "something else", "surprise me", "another one",
        "change it", "", "  ",
    ])
    def test_change_requests_are_not_names(self, directive):
        """These mean "pick for me" — the slot candidates are the right pool."""
        from services.edit_service import _names_a_dish
        assert not _names_a_dish(directive)
