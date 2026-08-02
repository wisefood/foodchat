"""Standing plan constraints, and the conversation that motivated them.

This is the transcript that prompted the change, turn by turn:

    "no" (to the favourites offer)  -> a favourite appeared in the plan
    "apple pie for breakfast"       -> scrambled eggs
    "add salads as side dishes"     -> a different breakfast, no salads
    "that has eggs, I'm vegan"      -> poached eggs

Every turn rewrote the whole request into a fresh query and regenerated from
scratch, so what the member had already said stopped existing between turns.
The tests below are that conversation, asserted.

The distinction the whole design rests on is **silence versus retraction**. A
turn that does not mention favourites must leave the favourites decision alone.
Collapsing the two is what made "no" last exactly one request.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from models.plan_spec import PlanSpec  # noqa: E402
from models.planning_state import PlanningState, PlanningStateDelta  # noqa: E402
from services.planning_delta import extract_state_delta  # noqa: E402


class TestTheConversationThatBrokeIt:
    """Each turn of the reported transcript, in order."""

    def test_a_decline_outlives_the_turn_that_made_it(self):
        state = PlanningState().merge(PlanningStateDelta(use_favorites=False))

        # Two further turns that say nothing about favourites.
        state = state.merge(PlanningStateDelta(anchors={"breakfast": "r-pie"}))
        state = state.merge(PlanningStateDelta())

        assert state.use_favorites is False, (
            "the member said no; a favourite reappearing one turn later reads "
            "as not listening"
        )

    def test_an_anchor_survives_a_later_shape_change(self):
        """"apple pie for breakfast", then "add salads as side dishes"."""
        state = PlanningState().merge(PlanningStateDelta(anchors={"breakfast": "r-pie"}))
        state = state.merge(
            PlanningStateDelta(
                spec=PlanSpec.from_spec({"plates": {"dinner": ["main", "salad"]}})
            )
        )

        assert state.anchors == {"breakfast": "r-pie"}
        assert state.spec.roles_for("dinner") == ("main", "salad")

    def test_a_shape_survives_a_later_anchor(self):
        state = PlanningState().merge(
            PlanningStateDelta(
                spec=PlanSpec.from_spec({"plates": {"lunch": ["main", "salad"]}})
            )
        )
        state = state.merge(PlanningStateDelta(anchors={"breakfast": "r-pie"}))

        assert state.spec.roles_for("lunch") == ("main", "salad")

    def test_all_four_turns_accumulate(self):
        state = PlanningState()
        state = state.merge(PlanningStateDelta(use_favorites=False))
        state = state.merge(PlanningStateDelta(anchors={"breakfast": "r-pie"}))
        state = state.merge(
            PlanningStateDelta(
                spec=PlanSpec.from_spec({
                    "meals": ["breakfast", "lunch", "dinner"],
                    "plates": {"lunch": ["main", "salad"], "dinner": ["main", "salad"]},
                })
            )
        )
        state = state.merge(PlanningStateDelta(excluded_recipe_ids=("r-eggs",)))

        assert state.use_favorites is False
        assert state.anchors == {"breakfast": "r-pie"}
        assert state.spec.roles_for("lunch") == ("main", "salad")
        assert "r-eggs" in state.excluded_recipe_ids


class TestSilenceIsNotRetraction:
    """An unmentioned field must not be reset to its default."""

    @pytest.fixture
    def loaded(self):
        return PlanningState().merge(
            PlanningStateDelta(
                use_favorites=False,
                anchors={"breakfast": "r-pie"},
                excluded_recipe_ids=("r-1",),
                spec=PlanSpec.from_spec({"num_days": 3}),
                notes=("nothing heavy in the evening",),
            )
        )

    def test_an_empty_delta_changes_nothing(self, loaded):
        assert loaded.merge(PlanningStateDelta()) == loaded

    def test_a_shape_change_does_not_clear_the_decline(self, loaded):
        after = loaded.merge(PlanningStateDelta(spec=PlanSpec.default()))

        assert after.use_favorites is False
        assert after.anchors == {"breakfast": "r-pie"}

    def test_an_empty_delta_is_recognisable_as_empty(self):
        assert PlanningStateDelta().is_empty
        assert not PlanningStateDelta(use_favorites=False).is_empty


class TestExplicitChanges:
    def test_an_empty_anchor_value_clears_that_slot(self):
        """"actually never mind the apple pie" — distinct from silence."""
        state = PlanningState().merge(PlanningStateDelta(anchors={"breakfast": "r-pie"}))
        after = state.merge(PlanningStateDelta(anchors={"breakfast": ""}))

        assert "breakfast" not in after.anchors

    def test_reset_clears_everything(self, ):
        state = PlanningState().merge(
            PlanningStateDelta(use_favorites=False, anchors={"breakfast": "r-pie"})
        )

        assert state.merge(PlanningStateDelta(reset=True)) == PlanningState()

    def test_exclusions_accumulate_without_duplicating(self):
        state = PlanningState().merge(PlanningStateDelta(excluded_recipe_ids=("a", "b")))
        after = state.merge(PlanningStateDelta(excluded_recipe_ids=("b", "c")))

        assert after.excluded_recipe_ids == ("a", "b", "c")

    def test_accepting_favourites_is_distinct_from_never_asking(self):
        """A plain bool cannot tell "they said no" from "we never offered", and
        only the first should suppress the offer."""
        assert PlanningState().use_favorites is None
        assert PlanningState().merge(
            PlanningStateDelta(use_favorites=True)
        ).use_favorites is True


class TestPersistence:
    def test_a_loaded_state_round_trips(self):
        state = PlanningState().merge(
            PlanningStateDelta(
                use_favorites=False,
                anchors={"breakfast": "r-pie"},
                excluded_recipe_ids=("r-1",),
                spec=PlanSpec.from_spec({"num_days": 2, "plates": {"dinner": ["main", "salad"]}}),
                notes=("no heavy dinners",),
            )
        )

        assert PlanningState.from_dict(state.to_dict()) == state

    @pytest.mark.parametrize("junk", [None, "", 42, [], {"spec": "nonsense"}])
    def test_unreadable_stored_state_yields_a_fresh_one(self, junk):
        """Losing accumulated constraints is a bad turn; raising is a broken
        product."""
        assert PlanningState.from_dict(junk).spec.is_default

    def test_describe_names_what_is_in_force(self):
        """The member should be able to ask what the system is working with."""
        state = PlanningState().merge(
            PlanningStateDelta(use_favorites=False, anchors={"breakfast": "r-pie"})
        )
        text = state.describe()

        assert "favourites declined" in text
        assert "breakfast=r-pie" in text


class TestDeltaExtraction:
    """What a turn's message changes — and what it must not."""

    class _Abstains:
        def extract(self, text):
            return PlanSpec.default()

    class _Shapes:
        def extract(self, text):
            return PlanSpec.from_spec({"plates": {"dinner": ["main", "salad"]}})

    @pytest.mark.parametrize(
        "message",
        ["start over", "forget about this", "scrap that", "reset", "never mind all that"],
    )
    def test_reset_phrases_are_recognised_without_a_model(self, message):
        """"Start over" has to work when the model is down."""
        assert extract_state_delta(message).reset

    def test_reset_is_anchored_to_the_start(self):
        """"forget the salt" must not wipe a member's whole session."""
        assert not extract_state_delta("add pasta but forget the salt").reset

    def test_an_abstaining_extractor_produces_no_shape_change(self):
        """Otherwise every unrelated message resets the standing shape."""
        delta = extract_state_delta("something spicy", extractor=self._Abstains())

        assert delta.spec is None
        assert delta.is_empty

    def test_a_stated_shape_is_carried(self):
        delta = extract_state_delta("salads on the side", extractor=self._Shapes())

        assert delta.spec is not None
        assert delta.spec.roles_for("dinner") == ("main", "salad")

    def test_a_failing_extractor_changes_nothing(self):
        class Boom:
            def extract(self, text):
                raise RuntimeError("groq down")

        assert extract_state_delta("salads please", extractor=Boom()).is_empty

    def test_an_empty_message_changes_nothing(self):
        assert extract_state_delta("   ").is_empty
