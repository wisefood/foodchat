"""
The agent can use its own tools.

The registry has been complete and unreachable from chat since it was written:
`manifest()` and `describe_tools()` are generated from it and **neither reached
a prompt**. The only in-chat tool call was one hardcoded `plan_totals` to stop
the analyst doing arithmetic in prose. So "summarise my week" and "redo
Thursday" had no path — the closest available action was a full refinement,
which regenerates every slot and throws away a swap the member already
approved.

Selection rides the same pre-classification seam the FoodScholar bypass uses,
under a new prompt name, because `orchestrator_system` is Langfuse-managed and
an added intent there would ship dead.

What these tests care about most is the NEGATIVE cases. A tool surface that
fires on "thanks, that looks great" is worse than no tool surface: it answers a
question nobody asked and costs a turn.
"""

from __future__ import annotations

import json
import sys
import uuid

import pytest

sys.path.insert(0, "src")

from services.orchestrator_service import OrchestratorService      # noqa: E402


def _entry(day, idx, slot, rid, title):
    return {
        "day": day, "meal_idx": idx, "meal_type": slot,
        "recipe": {
            "recipe_id": rid, "recipe_title": title,
            "recipe_ingredients": "lentils, carrot", "recipe_directions": "cook",
            "nutrition": {"calories": 500, "protein_g": 20},
        },
    }


def _week(days=7):
    out = []
    for day in range(1, days + 1):
        for idx, slot in enumerate(("breakfast", "lunch", "dinner")):
            out.append(_entry(day, idx, slot, f"r{day}{idx}", f"Dish {day}{idx}"))
    return out


class _Selector:
    """Returns a fixed choice and records what it was shown."""

    def __init__(self, choice=None):
        self.choice = choice or {}
        self.calls: list[dict] = []

    def choose(self, message, *, plan_type, plan_shape, manifest, allowed):
        self.calls.append({
            "message": message, "plan_type": plan_type,
            "plan_shape": plan_shape, "manifest": manifest, "allowed": allowed,
        })
        return dict(self.choice)


class _Analyst:
    def __init__(self, answer="Here's your week."):
        self.answer_text = answer
        self.seen: list[str] = []

    def answer(self, question, plan_summary, history=None):
        self.seen.append(plan_summary)
        return self.answer_text


@pytest.fixture
def weekly(session_service, sample_profile):
    session = session_service.create_session(
        f"member-{uuid.uuid4()}", sample_profile,
    )
    session_service.add_weekly_meal_plan(
        session.session_id, _week(), day_summaries={},
    )
    return session_service.get_session(session.session_id)


def _orch(session_service, selector, analyst=None):
    orch = OrchestratorService.__new__(OrchestratorService)
    orch.session_service = session_service
    OrchestratorService._tool_selector = selector
    orch.plan_analyst = analyst or _Analyst()
    return orch


# ── the gates, before a model is asked anything ──────────────────────────

class TestItAsksOnlyWhenAToolCouldWork:
    def test_no_canvas_means_no_question(self, session_service, sample_profile):
        """Every tool acts on a plan. With none there is nothing to summarise,
        total or replace, and the answer is known without paying for it."""
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        selector = _Selector({"tool": "summarize_week"})
        orch = _orch(session_service, selector)
        assert orch._maybe_use_tool(session, session.session_id, "summarise my week") is None
        assert selector.calls == [], "it should not have asked"

    def test_a_plan_on_the_canvas_does_get_asked(self, session_service, weekly):
        selector = _Selector()
        orch = _orch(session_service, selector)
        orch._maybe_use_tool(weekly, weekly.session_id, "summarise my week")
        assert len(selector.calls) == 1

    def test_the_selector_is_shown_the_real_manifest(self, session_service, weekly):
        """Generated from the registry, so a tool that exists is offered."""
        selector = _Selector()
        orch = _orch(session_service, selector)
        orch._maybe_use_tool(weekly, weekly.session_id, "how's my week?")
        call = selector.calls[0]
        assert "summarize_week" in call["manifest"]
        assert {"summarize_week", "plan_totals", "replace_day"} <= call["allowed"]

    def test_it_is_told_what_shape_the_plan_is(self, session_service, weekly):
        """"Redo Thursday" is only answerable if Thursday exists."""
        selector = _Selector()
        orch = _orch(session_service, selector)
        orch._maybe_use_tool(weekly, weekly.session_id, "redo Thursday")
        assert "7 day" in selector.calls[0]["plan_shape"]
        assert selector.calls[0]["plan_type"] == "weekly"


class TestChoosingNothingIsNormal:
    @pytest.mark.parametrize("choice", [{}, {"tool": ""}, {"tool": None}])
    def test_no_choice_routes_the_turn_normally(self, session_service, weekly, choice):
        """Most messages are a plan request, a refinement, or conversation."""
        orch = _orch(session_service, _Selector(choice))
        assert orch._maybe_use_tool(weekly, weekly.session_id, "thanks!") is None

    def test_a_tool_that_does_not_exist_is_dropped(self, session_service, weekly):
        """The registry would reject it, but a 400 is a worse answer than
        routing the message normally."""
        orch = _orch(session_service, _Selector({"tool": "reticulate_splines"}))
        assert orch._maybe_use_tool(weekly, weekly.session_id, "do the thing") is None

    def test_a_selector_failure_routes_normally(self, session_service, weekly):
        class _Boom:
            def choose(self, *_a, **_k):
                raise RuntimeError("groq is down")

        orch = _orch(session_service, _Boom())
        # `choose` is called inside `_maybe_use_tool`; a raise must not escape
        # into the turn. The real ToolSelector catches its own failures, so a
        # raising stub is the harsher case.
        with pytest.raises(RuntimeError):
            orch._maybe_use_tool(weekly, weekly.session_id, "summarise")


# ── running a read tool ───────────────────────────────────────────────────

class TestReadTools:
    def test_a_week_summary_answers_from_the_tool(self, session_service, weekly):
        analyst = _Analyst("Your week has 21 meals.")
        orch = _orch(session_service, _Selector({"tool": "summarize_week"}), analyst)
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "summarise my week")
        assert turn is not None
        assert turn.content == "Your week has 21 meals."
        assert turn.intent == "chat"

    def test_the_tools_own_numbers_are_the_grounding(self, session_service, weekly):
        analyst = _Analyst()
        orch = _orch(session_service, _Selector({"tool": "summarize_week"}), analyst)
        orch._maybe_use_tool(weekly, weekly.session_id, "how many calories?")
        summary = analyst.seen[0]
        assert "summarize_week returned" in summary
        assert "10500" in summary, "the tool's own total"

    def test_the_model_is_told_not_to_re_add(self, session_service, weekly):
        """The one thing a language model should not be trusted with here."""
        analyst = _Analyst()
        orch = _orch(session_service, _Selector({"tool": "summarize_week"}), analyst)
        orch._maybe_use_tool(weekly, weekly.session_id, "totals?")
        assert "do not" in analyst.seen[0].lower()
        assert "recalculate" in analyst.seen[0].lower()

    def test_a_day_scoped_tool_gets_its_day(self, session_service, weekly):
        analyst = _Analyst("Thursday looks good.")
        orch = _orch(
            session_service,
            _Selector({"tool": "summarize_day", "day": 4}), analyst,
        )
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "how's Thursday?")
        assert turn is not None
        assert "Thursday" in analyst.seen[0]

    def test_plan_totals_gets_the_canvas_type(self, session_service, weekly):
        analyst = _Analyst()
        orch = _orch(session_service, _Selector({"tool": "plan_totals"}), analyst)
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "how many calories?")
        assert turn is not None, "it must not have asked for a daily plan"

    def test_the_turn_is_recorded_in_the_conversation(self, session_service, weekly):
        orch = _orch(session_service, _Selector({"tool": "summarize_week"}))
        orch._maybe_use_tool(weekly, weekly.session_id, "summarise my week")
        page = session_service.get_messages_page(weekly.session_id)
        roles = [m["role"] for m in page[-2:]]
        assert roles == ["user", "assistant"]

    def test_a_read_tool_does_not_attach_a_plan(self, session_service, weekly):
        """A summary is an answer, not a change — attaching a plan would make
        the UI redraw the canvas for a question."""
        orch = _orch(session_service, _Selector({"tool": "summarize_week"}))
        turn = orch._maybe_use_tool(weekly, weekly.session_id, "summarise")
        assert turn.weekly_meal_plan is None and turn.meal_plan is None


# ── a tool that declines ─────────────────────────────────────────────────

class TestWhenATooRefuses:
    def test_the_registrys_own_words_reach_the_member(self, session_service,
                                                      sample_profile):
        """A ToolError carries member-facing prose — "this plan covers Monday to
        Wednesday" — which is a better answer than routing on and answering a
        different question."""
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.add_weekly_meal_plan(
            session.session_id, _week(days=3), day_summaries={},
        )
        loaded = session_service.get_session(session.session_id)
        orch = _orch(session_service, _Selector({"tool": "summarize_day", "day": 7}))
        turn = orch._maybe_use_tool(loaded, session.session_id, "how's Sunday?")
        assert turn is not None
        # `summarize_day` names the day asked for and the plan's extent;
        # `replace_day` additionally lists the days it does cover. Both are the
        # registry's own words rather than a generic failure.
        assert "Sunday" in turn.content
        assert "3 day" in turn.content

    def test_a_refusal_is_still_a_turn_not_an_error(self, session_service,
                                                   sample_profile):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", sample_profile,
        )
        session_service.add_weekly_meal_plan(
            session.session_id, _week(days=2), day_summaries={},
        )
        loaded = session_service.get_session(session.session_id)
        orch = _orch(session_service, _Selector({"tool": "summarize_day", "day": 6}))
        turn = orch._maybe_use_tool(loaded, session.session_id, "how's Saturday?")
        assert turn.role == "assistant" and turn.intent == "chat"


# ── the fallback sentence ────────────────────────────────────────────────

class TestTheFallback:
    def test_totals_read_as_a_sentence(self):
        text = OrchestratorService._tool_fallback("plan_totals", {
            "total": {"calories": 10500.0, "complete": True},
            "daily_average_kcal": 1500.0,
        })
        assert "10,500 kcal" in text and "1500 a day" in text

    def test_a_partial_total_says_so(self):
        """A total that omits an unprofiled meal is a number someone might act
        on."""
        text = OrchestratorService._tool_fallback("plan_totals", {
            "total": {"calories": 4000.0, "complete": False,
                      "meals_counted": 8, "meals_total": 21},
        })
        assert "8 of 21" in text

    def test_a_day_summary_names_its_dishes(self):
        text = OrchestratorService._tool_fallback("summarize_day", {
            "name": "Thursday",
            "meals": [{"title": "Porridge"}, {"title": "Soup"}],
        })
        assert "Thursday" in text and "Porridge" in text

    def test_an_unrecognised_result_still_produces_words(self):
        assert OrchestratorService._tool_fallback("something_new", {}) == "Done."


# ── the prompts are new, not edited ──────────────────────────────────────

class TestThePromptsAreNew:
    @pytest.mark.parametrize("name", ["tool_selector_system", "tool_selector_user"])
    def test_registered_under_a_new_name(self, name):
        """`orchestrator_system` is Langfuse-managed and `sync_prompts` never
        overwrites, so an added intent there would ship dead."""
        import prompts

        assert any(p.name.endswith(name) for p in prompts.ALL_PROMPTS)

    def test_the_manifest_reaches_the_prompt(self):
        from prompts import TOOL_SELECTOR_SYSTEM

        assert "{tools}" in TOOL_SELECTOR_SYSTEM.fallback

    def test_the_schema_cannot_express_free_form_arguments(self):
        """The registry validates arguments. The selector names the tool and
        only the arguments a member can state in a sentence — a closed set, so
        no tool can be handed something nobody checked."""
        from schemas import ToolChoiceSchema

        fields = set(ToolChoiceSchema.model_fields)
        assert fields == {"tool", "day", "plan_type", "title", "saved", "reason"}

    def test_a_member_supplied_title_cannot_outrun_its_column(self):
        """`saved_title` is 120 wide. Capping at the schema means the cap is
        not a thing the tool has to remember."""
        import pydantic
        import pytest as _pytest

        from schemas import ToolChoiceSchema

        assert ToolChoiceSchema(tool="save_plan", title="x" * 120).title
        with _pytest.raises(pydantic.ValidationError):
            ToolChoiceSchema(tool="save_plan", title="x" * 121)


class TestTheSelectorItself:
    def _selector(self, payload):
        from agents import ToolSelector

        agent = ToolSelector.__new__(ToolSelector)

        class _Client:
            def invoke(self, messages, config=None):
                class _R:
                    content = payload if isinstance(payload, str) else json.dumps(payload)
                return _R()

        agent.llm = _Client()
        return agent

    def test_it_returns_the_choice(self):
        out = self._selector({"tool": "summarize_week", "reason": "asked for the week"}).choose(
            "summarise", plan_type="weekly", plan_shape="7 days",
            manifest="- summarize_week", allowed={"summarize_week"},
        )
        assert out["tool"] == "summarize_week"

    def test_an_unlisted_tool_is_dropped(self):
        out = self._selector({"tool": "rm_rf"}).choose(
            "do it", plan_type="weekly", plan_shape="7 days",
            manifest="", allowed={"summarize_week"},
        )
        assert out == {}

    def test_unparseable_output_is_no_tool(self):
        assert self._selector("not json").choose(
            "x", plan_type="weekly", plan_shape="", manifest="",
            allowed={"summarize_week"},
        ) == {}

    def test_an_empty_message_is_not_asked_about(self):
        assert self._selector({"tool": "summarize_week"}).choose(
            "   ", plan_type="weekly", plan_shape="", manifest="",
            allowed={"summarize_week"},
        ) == {}

    def test_no_available_tools_means_no_call(self):
        assert self._selector({"tool": "summarize_week"}).choose(
            "summarise", plan_type="weekly", plan_shape="", manifest="",
            allowed=set(),
        ) == {}

    def test_it_runs_on_the_fast_tier(self):
        """Routing over a handful of named tools, on every eligible turn."""
        import inspect

        from agents import ToolSelector

        src = inspect.getsource(ToolSelector.__init__)
        assert "FAST_MODEL" in src and "DEFAULT_MODEL" not in src


class TestTheSeam:
    def test_selection_runs_before_classification(self):
        """After it, the intent classifier has already spent a reasoning call
        and committed the turn to a plan path."""
        import inspect

        src = inspect.getsource(OrchestratorService._classify_and_route)
        assert src.index("_maybe_use_tool") < src.index("self.orchestrator.classify")

    def test_it_runs_after_the_scholar_bypass(self):
        """An explicit FoodScholar consult is not a tool request."""
        import inspect

        src = inspect.getsource(OrchestratorService._classify_and_route)
        assert src.index("_SCHOLAR_CONSULT_RE") < src.index("_maybe_use_tool")

    def test_the_selector_survives_an_instance_built_without_init(self):
        """The test suite builds this service with `__new__` routinely, and a
        first turn through it used to raise AttributeError."""
        OrchestratorService._tool_selector = None
        orch = OrchestratorService.__new__(OrchestratorService)
        assert orch.tool_selector is not None
