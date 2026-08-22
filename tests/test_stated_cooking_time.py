"""
"Keep it under 20 minutes" is the same constraint as the cooking-time slider.

The persona has promised for a long time that a member can steer by cooking
time. Only the slider ever could: all seven fetch sites read the limit from
`profile["plan_parameters"]["cooking_time"]` through `max_duration_minutes`,
and nothing but the slider ever wrote it. A sentence reached the grader as
prose over a pool that had not been narrowed at all.

Two decisions these tests pin down, because both are judgement calls:

* **A number becomes a filter; an adjective becomes a tag.** "Under 30 minutes"
  is a ceiling. "Something quick" is not a number, and turning it into one
  would be the invented-constraint bug — so it becomes the corpus's own
  `30_minutes_or_less` annotation instead.
* **Looser than the card can express means no ceiling.** "Under two hours"
  sets nothing rather than 90, because 90 is tighter than what was said.
"""

from __future__ import annotations

import sys
import uuid

import pytest

sys.path.insert(0, "src")

from models.planning_state import PlanningState, PlanningStateDelta  # noqa: E402
from services import plan_parameters, turn_intake                    # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_turn():
    turn_intake.forget()
    yield
    turn_intake.forget()


class TestReadingATimeOutOfASentence:
    @pytest.mark.parametrize("text,minutes", [
        ("keep it under 30 minutes", 30),
        ("in 20 min", 20),
        ("30 minutes or less", 30),
        ("I want 20-minute meals", 20),
        ("no longer than 25 minutes", 25),
        ("nothing over 45 mins please", 45),
        ("max 15 minutes", 15),
        ("within half an hour", 30),
        ("under an hour", 60),
    ])
    def test_a_stated_ceiling_is_read(self, text, minutes):
        assert plan_parameters.extract_time_delta(text).max_minutes == minutes

    @pytest.mark.parametrize("text", [
        "plan my week",
        "swap the dinner",
        "simmer for 30 minutes",          # recipe prose, not a request
        "we eat at 7",
        "",
    ])
    def test_silence_stays_silence(self, text):
        delta = plan_parameters.extract_time_delta(text)
        assert delta.max_minutes is None and not delta.claim_tags

    def test_the_tightest_limit_wins(self):
        """"Under an hour, ideally 20 minutes" states a limit and a preference.
        Honouring the looser one answers the wrong half."""
        delta = plan_parameters.extract_time_delta("under an hour, ideally in 20 minutes")
        assert delta.max_minutes == 20

    def test_tighter_than_the_card_snaps_up_to_the_floor(self):
        """Nothing in the corpus is a five-minute meal. The choice is the
        tightest limit the product supports, or no plan."""
        assert plan_parameters.extract_time_delta("in 5 minutes").max_minutes == 10

    def test_looser_than_the_card_sets_no_ceiling(self):
        """90 would be TIGHTER than "under two hours" — a constraint the member
        did not state. Above the card's range the limit filters out almost
        nothing anyway."""
        assert plan_parameters.extract_time_delta("under 2 hours").max_minutes is None

    def test_it_lands_on_a_value_the_slider_can_show(self):
        """One constraint, one number: if the card cannot render it, pressing
        Apply would silently overwrite what the member said."""
        minutes = plan_parameters.extract_time_delta("in 22 minutes").max_minutes
        assert minutes == plan_parameters.sanitize({"cooking_time": minutes})["cooking_time"]


class TestVagueSpeedIsNotANumber:
    @pytest.mark.parametrize("text", [
        "something quick", "I'm in a hurry", "weeknight dinners please",
    ])
    def test_it_becomes_the_corpus_tag_not_an_invented_ceiling(self, text):
        delta = plan_parameters.extract_time_delta(text)
        assert delta.claim_tags == ("30_minutes_or_less",)
        assert delta.max_minutes is None, "an adjective is not a measurement"


class TestTakingItBack:
    def test_no_rush_clears_a_standing_ceiling(self):
        state = PlanningState().merge(PlanningStateDelta(max_minutes=20))
        assert state.max_minutes == 20
        cleared = state.merge(plan_parameters.extract_time_delta("take as long as you like"))
        assert cleared.max_minutes is None

    def test_silence_does_not_clear_it(self):
        state = PlanningState().merge(PlanningStateDelta(max_minutes=20))
        assert state.merge(plan_parameters.extract_time_delta("plan my day")).max_minutes == 20


class TestItReachesWhatFetches:
    def test_the_profile_the_fetch_sites_read_carries_it(self):
        """`max_duration_minutes` is the single accessor all seven use."""
        state = PlanningState().merge(PlanningStateDelta(max_minutes=25))
        profile = plan_parameters.apply_state({"diet": ["vegan"]}, state)
        assert plan_parameters.max_duration_minutes(profile["plan_parameters"]) == 25
        assert profile["diet"] == ["vegan"], "it should touch nothing else"

    def test_no_stated_ceiling_leaves_the_slider_alone(self):
        profile = {"plan_parameters": {"cooking_time": 45}}
        plan_parameters.apply_state(profile, PlanningState())
        assert profile["plan_parameters"]["cooking_time"] == 45

    def test_a_spoken_limit_overrides_a_stale_slider(self):
        """The member just said it. The slider is what they said last time."""
        profile = {"plan_parameters": {"cooking_time": 90, "goal": "energy"}}
        state = PlanningState().merge(PlanningStateDelta(max_minutes=20))
        plan_parameters.apply_state(profile, state)
        assert profile["plan_parameters"] == {"cooking_time": 20, "goal": "energy"}

    def test_the_brief_the_verifier_checks_against_carries_it(self):
        from models.plan_brief import PlanBrief

        state = PlanningState().merge(PlanningStateDelta(max_minutes=25))
        profile = plan_parameters.apply_state({}, state)
        assert PlanBrief.build(profile).max_minutes == 25


class TestItSurvivesTheTurn:
    def test_intake_puts_it_on_the_standing_state(self, session_service, sample_profile,
                                                  monkeypatch):
        import services.diet_intent as diet_intent
        import services.intent_facets as intent_facets
        import services.pantry_service as pantry_service
        import services.planning_delta as planning_delta

        for module, name in ((planning_delta, "extract_state_delta"),
                             (pantry_service, "extract_pantry_delta"),
                             (diet_intent, "extract_diet_delta"),
                             (intent_facets, "extract_facet_delta")):
            monkeypatch.setattr(module, name, lambda *a, **k: PlanningStateDelta())

        session = session_service.create_session(f"member-{uuid.uuid4()}", sample_profile)
        state = turn_intake.intake(
            session.session_id, "plan my day, nothing over 20 minutes",
            session_service=session_service,
        )
        assert state.max_minutes == 20

        turn_intake.forget()
        assert session_service.get_planning_state(session.session_id).max_minutes == 20

    def test_it_round_trips_through_the_session_row(self):
        state = PlanningState().merge(PlanningStateDelta(max_minutes=25))
        assert PlanningState.from_dict(state.to_dict()).max_minutes == 25

    def test_a_junk_stored_value_is_dropped_not_raised(self):
        assert PlanningState.from_dict({"max_minutes": "soon"}).max_minutes is None

    def test_it_is_in_what_the_member_can_be_told(self):
        state = PlanningState().merge(PlanningStateDelta(max_minutes=25))
        assert "25 min" in state.describe()


class TestThePersonaDescribesWhatExists:
    """The persona is what the member is told the assistant can do, and it has
    now been wrong in both directions.

    v1 promised mood, flavour, food group, calorie and protein steering when
    none of it was wired — a promise the next turn broke. v2 removed the four
    it could not keep, and then the facet extractor made three of them real,
    leaving the prompt actively instructing the model to refuse capabilities
    the planner had gained.

    These assertions are deliberately tied to the code that implements each
    claim, so the prompt fails the build rather than drifting again.
    """

    @staticmethod
    def _persona() -> str:
        from prompts import CHATBOT_SYSTEM_INSTRUCTIONS

        return CHATBOT_SYSTEM_INSTRUCTIONS.lower()

    def test_it_does_not_refuse_facets_that_are_real_filters(self):
        from models.planning_state import PlanningState
        from services import intent_facets

        # The families that reach a fetch, straight from the code that sends
        # them — not a list written out here to go stale.
        wired = set(intent_facets.facet_kwargs({}, []))
        assert {"moods", "flavor_profiles", "food_groups"} <= wired
        assert set(PlanningState.FACET_FIELDS) <= wired

        persona = self._persona()
        assert "do not promise to steer by mood" not in persona
        for word in ("mood", "flavour", "food group"):
            assert word in persona, f"{word} is a real filter and goes unmentioned"

    def test_it_mentions_cooking_time_now_that_a_sentence_sets_one(self):
        assert plan_parameters.extract_time_delta("under 20 minutes").max_minutes
        assert "cooking time" in self._persona()

    def test_it_still_refuses_the_target_it_cannot_take_from_chat(self):
        """A calorie target is CHECKED when the member's profile carries one,
        and still cannot be SET by asking: the standing state has nowhere to
        put a number from a sentence, so no extractor can produce one."""
        import dataclasses

        from models.planning_state import PlanningState

        names = {f.name for f in dataclasses.fields(PlanningState)}
        assert not [n for n in names if "kcal" in n or "calorie" in n or "protein" in n]
        assert "do not promise calorie or protein targets" in self._persona()


class TestTheSliderAndTheSentenceCannotFight:
    """One constraint means the newer statement wins, whichever channel it
    arrived on.

    The spoken limit is standing state, and the planning paths write it over
    `profile["plan_parameters"]["cooking_time"]` on every turn. So a member who
    said "under 20 minutes" on Monday and dragged the knob to 60 on Tuesday
    would watch their drag undo itself: the state would re-apply 20 over the
    slider value that had just been saved.
    """

    def _orch(self, session_service):
        from services.orchestrator_service import OrchestratorService

        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service
        orch._owned_session = lambda sid, mid: session_service.get_session(sid)
        orch._limit_turn = staticmethod(lambda _s: None)
        orch._handle_plan = lambda *a, **k: _StubTurn()
        orch._handle_weekly = lambda *a, **k: _StubTurn()
        return orch

    def test_moving_the_knob_replaces_a_spoken_limit(self, session_service,
                                                    sample_profile):
        from models.planning_state import PlanningStateDelta

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(max_minutes=20),
            ),
        )

        self._orch(session_service).apply_plan_parameters(
            session.session_id, session.member_id, {"cooking_time": 60},
        )

        state = session_service.get_planning_state(session.session_id)
        assert state.max_minutes == 60
        profile = plan_parameters.apply_state(
            dict(session_service.get_session(session.session_id).user_profile), state,
        )
        assert profile["plan_parameters"]["cooking_time"] == 60

    def test_a_slider_apply_that_says_nothing_about_time_leaves_it_alone(
        self, session_service, sample_profile,
    ):
        from models.planning_state import PlanningStateDelta

        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.set_planning_state(
            session.session_id,
            session_service.get_planning_state(session.session_id).merge(
                PlanningStateDelta(max_minutes=20),
            ),
        )
        self._orch(session_service).apply_plan_parameters(
            session.session_id, session.member_id, {"goal": "energy"},
        )
        assert session_service.get_planning_state(session.session_id).max_minutes == 20


class _StubTurn:
    """Enough of a ChatTurn for `apply_plan_parameters` to finish."""
    plan_parameters = None
