"""ProfileService: dietary_goals from properties -> planner signals."""

from services.profile_service import (
    ProfileService,
    GOAL_TO_DIET_TAG,
    goal_preference_strings,
)


def _map(raw: dict) -> dict:
    # _map_profile handles dict-like profiles directly (no WiseFood client).
    return ProfileService.__new__(ProfileService)._map_profile(raw)


def test_tag_mappable_goal_becomes_hard_diet_filter():
    mapped = _map({
        "dietary_groups": ["vegetarian"],
        "properties": {"dietary_goals": [{"slug": "reduce_fat", "label": "reduce fat"}]},
    })
    # reduce_fat -> low-fat, folded into the hard `diet` filter alongside groups.
    assert "low-fat" in mapped["diet"]
    assert "vegetarian" in mapped["diet"]
    # And surfaced as a first-class field.
    assert mapped["dietary_goals"] == ["reduce_fat"]


def test_soft_goal_only_adds_preference_string():
    mapped = _map({
        "properties": {"dietary_goals": [{"slug": "reduce_sugar", "label": "less sugar"}]},
    })
    # reduce_sugar has no RW diet tag -> not a hard filter...
    assert "low-fat" not in mapped["diet"]
    assert mapped["diet"] == []
    # ...but rides the soft preferences channel for the LLM grader.
    assert "prefers lower-sugar meals" in mapped["preferences"]
    assert mapped["dietary_goals"] == ["reduce_sugar"]


def test_multiple_goals_dedup_and_mixed():
    mapped = _map({
        "dietary_groups": ["low-fat"],  # already present -> no dup
        "properties": {"dietary_goals": [
            {"slug": "reduce_fat"},        # -> low-fat (dup-safe)
            {"slug": "increase_protein"},  # -> high-protein
            {"slug": "lose_weight"},       # soft only
        ]},
    })
    assert mapped["diet"].count("low-fat") == 1
    assert "high-protein" in mapped["diet"]
    assert "lose_weight" not in mapped["diet"]
    assert "goal: weight loss (favor lighter, filling meals)" in mapped["preferences"]
    assert set(mapped["dietary_goals"]) == {"reduce_fat", "increase_protein", "lose_weight"}


def test_no_goals_is_noop():
    mapped = _map({"dietary_groups": ["vegan"], "properties": {}})
    assert mapped["diet"] == ["vegan"]
    assert mapped["dietary_goals"] == []


def test_malformed_goal_entries_ignored():
    mapped = _map({
        "properties": {"dietary_goals": ["not-a-dict", {}, {"slug": ""}, {"slug": "reduce_fat"}]},
    })
    assert mapped["dietary_goals"] == ["reduce_fat"]


def test_helpers_stay_in_sync():
    # Every tag-mapped slug must also have a preference string.
    for slug in GOAL_TO_DIET_TAG:
        assert goal_preference_strings([slug]), slug
