"""ProfileService: dietary_goals from properties -> planner signals.

Goals map to NUMERIC nutrition_profile targets (the corpus has no
low-fat/low-carb/high-protein tags — hard tag filters returned zero
candidates for every goal-carrying member). Hard diet tags are reserved
for real diets from dietary_groups.
"""

from services.profile_service import (
    ProfileService,
    GOAL_TO_NUTRITION_PROFILE,
    goal_preference_strings,
    goals_nutrition_profile,
)


def _map(raw: dict) -> dict:
    # _map_profile handles dict-like profiles directly (no WiseFood client).
    return ProfileService.__new__(ProfileService)._map_profile(raw)


def test_numeric_goal_becomes_nutrition_profile_not_diet_tag():
    mapped = _map({
        "dietary_groups": ["vegetarian"],
        "properties": {"dietary_goals": [{"slug": "reduce_fat", "label": "reduce fat"}]},
    })
    # reduce_fat -> numeric fat cap; NEVER a hard tag (no such tag exists).
    assert mapped["nutrition_profile"] == {"max_fat_g": 20}
    assert mapped["diet"] == ["vegetarian"]
    # And surfaced as a first-class field.
    assert mapped["dietary_goals"] == ["reduce_fat"]


def test_soft_goal_only_adds_preference_string():
    mapped = _map({
        "properties": {"dietary_goals": [{"slug": "reduce_sugar", "label": "less sugar"}]},
    })
    # reduce_sugar has no numeric mapping -> no nutrition profile...
    assert mapped["nutrition_profile"] is None
    assert mapped["diet"] == []
    # ...but rides the soft preferences channel for the LLM grader.
    assert "prefers lower-sugar meals" in mapped["preferences"]
    assert mapped["dietary_goals"] == ["reduce_sugar"]


def test_multiple_goals_merge_strictest_bounds():
    mapped = _map({
        "dietary_groups": ["vegan"],
        "properties": {"dietary_goals": [
            {"slug": "reduce_fat"},        # max_fat_g 20
            {"slug": "increase_protein"},  # min_protein_g 20
            {"slug": "lose_weight"},       # max_calories 650
            {"slug": "reduce_calories"},   # max_calories 650 (same bound, merged)
        ]},
    })
    assert mapped["nutrition_profile"] == {
        "max_fat_g": 20,
        "min_protein_g": 20,
        "max_calories": 650,
    }
    # Real diets stay hard tags; goals never leak into diet.
    assert mapped["diet"] == ["vegan"]
    assert "goal: weight loss (favor lighter, filling meals)" in mapped["preferences"]


def test_merge_takes_strictest_bound_per_key():
    # max_* takes the minimum, min_* takes the maximum.
    merged = goals_nutrition_profile(["reduce_calories", "lose_weight"])
    assert merged == {"max_calories": 650}
    assert goals_nutrition_profile([]) == {}
    assert goals_nutrition_profile(["unknown_goal"]) == {}


def test_no_goals_is_noop():
    mapped = _map({"dietary_groups": ["vegan"], "properties": {}})
    assert mapped["diet"] == ["vegan"]
    assert mapped["nutrition_profile"] is None
    assert mapped["dietary_goals"] == []


def test_malformed_goal_entries_ignored():
    mapped = _map({
        "properties": {"dietary_goals": ["not-a-dict", {}, {"slug": ""}, {"slug": "reduce_fat"}]},
    })
    assert mapped["dietary_goals"] == ["reduce_fat"]


def test_helpers_stay_in_sync():
    # Every numerically-mapped slug must also have a preference string.
    for slug in GOAL_TO_NUTRITION_PROFILE:
        assert goal_preference_strings([slug]), slug
