"""
Follow-up fixes to the pantry-planning review (PR #1).

Each class here corresponds to a finding, and each test states the failure in
member-visible terms: what the plan did versus what the reply said about it.
"""

from services.edit_service import DirectivePredicate
from services.pantry_service import _badge_reasons, matched_items


class TestIngredientDirectiveIsNotGreedy:
    """A directive naming an ingredient must not swallow the rest of the
    sentence. A junk term hard-fails the swap: the hard include finds nothing,
    the text match finds nothing, and the member is told no candidate
    satisfies a request that previously produced an ordinary swap."""

    def test_trailing_politeness_is_not_part_of_the_ingredient(self):
        p = DirectivePredicate("something with zucchini please")
        assert (p.kind, p.tag) == ("uses_ingredient", "zucchini")

    def test_trailing_instead_is_not_part_of_the_ingredient(self):
        p = DirectivePredicate("something with chicken instead")
        assert (p.kind, p.tag) == ("uses_ingredient", "chicken")

    def test_a_two_item_request_verifies_the_first_item(self):
        # The natural phrasing for this very feature. Verifying one item beats
        # searching for an ingredient literally named "zucchini and spinach".
        p = DirectivePredicate("something with zucchini and spinach")
        assert (p.kind, p.tag) == ("uses_ingredient", "zucchini")

    def test_a_two_word_ingredient_still_survives(self):
        p = DirectivePredicate("something with ground beef")
        assert (p.kind, p.tag) == ("uses_ingredient", "ground beef")

    def test_a_quantity_directive_is_not_an_ingredient(self):
        # "less salt" is about an amount; there is no ingredient by that name,
        # so this must fall through to unverified and still yield a swap.
        p = DirectivePredicate("replace lunch with something with less salt")
        assert p.kind == "unverified"

    def test_determiners_and_leftover_are_stripped(self):
        p = DirectivePredicate("one using the leftover chicken")
        assert (p.kind, p.tag) == ("uses_ingredient", "chicken")

    def test_earlier_classifiers_still_win(self):
        # These are matched before the ingredient rule and must stay that way.
        assert DirectivePredicate("something with more protein").kind == "more_protein"
        assert DirectivePredicate("make it lighter with fewer calories").kind == "lighter"
        assert DirectivePredicate("swap it with something quicker").kind == "quicker"

    def test_a_placeholder_names_no_ingredient(self):
        assert DirectivePredicate("something with something").kind == "unverified"


class TestMatcherIsNumberSymmetric:
    """Plural is how people name fridge contents. A one-directional suffix
    made the plan use the item while the reply said "I couldn't work in your
    tomatoes" and the ledger recorded the coverage as relaxed — under-reporting
    that produces an affirmative false claim."""

    def test_plural_pantry_matches_singular_recipe(self):
        assert matched_items("Tomato Basil Soup — tomato, basil", ["tomatoes"]) == ["tomatoes"]

    def test_singular_pantry_matches_plural_recipe(self):
        assert matched_items("Tomato Salad — tomatoes, oil", ["tomato"]) == ["tomato"]

    def test_oes_plural(self):
        assert matched_items("Roast Potato — potato wedges", ["potatoes"]) == ["potatoes"]

    def test_ies_plural(self):
        assert matched_items("Berry Tart — berries, sugar", ["berries"]) == ["berries"]

    def test_multi_word_items_still_match_as_a_phrase(self):
        assert matched_items("Beef Ragu — ground beef, onion", ["ground beef"]) == ["ground beef"]

    def test_word_boundaries_are_still_respected(self):
        # The whole reason for the boundary: "rice" must not match "price".
        assert matched_items("Priceless Pie — price, sugar", ["rice"]) == []

    def test_an_absent_item_does_not_match(self):
        assert matched_items("Chicken Pie — chicken, cream", ["spinach"]) == []


class TestFanOutCarriesTheCallersConstraints:
    """The pantry pool is merged into the ordinary one and sorted
    coverage-first, so a constraint the fan-out drops does not merely appear —
    it appears at the TOP."""

    def test_fetch_accepts_cuisines_and_max_minutes(self):
        import inspect

        from services.pantry_service import fetch_pantry_candidates

        params = inspect.signature(fetch_pantry_candidates).parameters
        assert "cuisines" in params, "cuisine filter cannot be threaded"
        assert "max_minutes" in params, "cooking-time slider cannot be threaded"

    def test_the_daily_pipeline_passes_both(self):
        import inspect

        from services import planning_pipeline

        src = inspect.getsource(planning_pipeline.PlanningPipeline.generate)
        assert "cuisines=cuisines" in src
        assert "max_minutes=max_minutes" in src

    def test_the_weekly_adapter_passes_both(self):
        import inspect

        from services.weekly_planner import action_adapter

        src = inspect.getsource(action_adapter.RecipeActionSpace.get_candidate_actions)
        pantry_call = src.split("fetch_pantry_candidates", 1)[1]
        assert "cuisines=cuisines" in pantry_call
        assert "max_minutes=" in pantry_call

    def test_the_edit_path_matches_its_fallback_pool(self):
        import inspect

        from services.edit_service import EditService

        src = inspect.getsource(EditService._find_replacement)
        pantry_call = src.split("fetch_pantry_candidates(", 1)[1]
        assert "cuisines=" in pantry_call


class TestBadgesClaimOnlyWhatWasMeasured:
    def test_one_chip_per_match_and_no_history_claim(self):
        reasons = _badge_reasons(["zucchini"])
        assert len(reasons) == 1
        label = reasons[0]["label"]
        assert "zucchini" in label
        # Nothing records whether an item is a leftover or bought this morning.
        assert "leftover" not in label.lower()

    def test_the_chip_names_every_matched_item(self):
        label = _badge_reasons(["zucchini", "spinach"])[0]["label"]
        assert "zucchini" in label and "spinach" in label
