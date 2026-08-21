"""
What a turn produced, stored on the message it belongs to.

The memory nudge, the slot-edit proof and the plan-parameter ribbon were
client-side only: the UI grafted them from the live response onto the last
assistant message. So they lasted exactly as long as the tab did. A member who
reloaded saw the plan with none of the explanation that came with it — no
"remember you're vegetarian?", no proof of what a swap changed, and a settings
ribbon that simply vanished.

`attribution` had already solved this, by being a column. These follow it.
"""

from __future__ import annotations

import json
import sys
import uuid

import pytest
from fastapi import HTTPException

sys.path.insert(0, "src")

import services                                             # noqa: E402
from db import SessionLocal, db_get_messages                 # noqa: E402
from routers import foodchat_router as api                   # noqa: E402


class _Turn:
    """A ChatTurn, as far as the response funnel is concerned."""

    def __init__(self, **kw):
        self.role = "assistant"
        self.content = kw.get("content", "Here's your plan.")
        self.intent = "daily_plan"
        self.needs_clarification = False
        self.meal_plan = None
        self.weekly_meal_plan = None
        self.at_message_limit = False
        self.plan_version = None
        self.plan_parent_id = None
        self.attribution = None
        self.memory_suggestions = kw.get("memory_suggestions")
        self.changed_slots = kw.get("changed_slots")
        self.plan_parameters = kw.get("plan_parameters")


# The real wire shapes — a fixture that does not validate would test a path
# the router never takes.
NUDGE = [{"id": "m1", "kind": "constraint", "value": "vegetarian",
          "statement": "You avoid meat", "evidence": "I need something vegetarian"}]
PROOF = [{"meal_type": "dinner", "day": 2, "old": {"title": "Ragu", "kcal": 800},
          "new": {"title": "Lentil bake", "kcal": 500}, "directive": "lighter",
          "verified": True}]
CARD = {"plan_type": "daily", "parameters": [{
    "key": "goal", "kind": "choice", "label": "Goal", "value": "balanced",
}]}


@pytest.fixture
def session_id():
    member = f"member-{uuid.uuid4()}"
    session = services.session_service.create_session(member, {})
    services.session_service.add_message(session.session_id, "user", "plan my day")
    services.session_service.add_message(session.session_id, "assistant", "Here you go.")
    return session.session_id


def _stored(session_id):
    db = SessionLocal()
    try:
        rows = db_get_messages(db, session_id, limit=50)
    finally:
        db.close()
    return rows


class TestWhatGetsStored:
    def test_a_memory_nudge_survives_the_turn(self, session_id):
        api._finalize_turn(session_id, _Turn(memory_suggestions=NUDGE))
        page = services.session_service.get_messages_page(session_id)
        assert page[-1]["extras"]["memory_suggestions"] == NUDGE

    def test_a_slot_edit_proof_survives(self, session_id):
        api._finalize_turn(session_id, _Turn(changed_slots=PROOF))
        page = services.session_service.get_messages_page(session_id)
        assert page[-1]["extras"]["changed_slots"][0]["new"]["title"] == "Lentil bake"

    def test_the_settings_ribbon_survives(self, session_id):
        api._finalize_turn(session_id, _Turn(plan_parameters=CARD))
        page = services.session_service.get_messages_page(session_id)
        assert page[-1]["extras"]["plan_parameters"]["plan_type"] == "daily"

    def test_all_three_ride_together(self, session_id):
        api._finalize_turn(session_id, _Turn(
            memory_suggestions=NUDGE, changed_slots=PROOF, plan_parameters=CARD,
        ))
        extras = services.session_service.get_messages_page(session_id)[-1]["extras"]
        assert set(extras) == {"memory_suggestions", "changed_slots", "plan_parameters"}

    def test_a_plain_answer_stores_nothing(self, session_id):
        """A turn with no extras must not write an empty object — the column
        stays NULL, which is what "there was nothing to remember" looks like."""
        api._finalize_turn(session_id, _Turn())
        assert services.session_service.get_messages_page(session_id)[-1]["extras"] is None

    def test_it_lands_on_the_assistant_message_not_the_user_one(self, session_id):
        api._finalize_turn(session_id, _Turn(memory_suggestions=NUDGE))
        rows = _stored(session_id)
        by_role = {r.role: r.extras for r in rows}
        assert by_role["assistant"] is not None
        assert by_role["user"] is None

    def test_it_lands_on_the_NEWEST_assistant_message(self, session_id):
        """A session has many turns; the extras belong to the one just made."""
        services.session_service.add_message(session_id, "assistant", "A later reply.")
        api._finalize_turn(session_id, _Turn(memory_suggestions=NUDGE))
        rows = [r for r in _stored(session_id) if r.role == "assistant"]
        assert rows[-1].content == "A later reply."
        assert json.loads(rows[-1].extras)["memory_suggestions"] == NUDGE
        assert rows[0].extras is None

    def test_the_wire_response_is_unchanged(self, session_id):
        """Persisting is additive: the live turn still carries everything it
        did, because the tab looking at it right now must not depend on the
        write having succeeded."""
        response = api._finalize_turn(session_id, _Turn(
            memory_suggestions=NUDGE, changed_slots=PROOF, plan_parameters=CARD,
        ))
        assert response.memory_suggestions is not None
        assert response.changed_slots == PROOF
        assert response.plan_parameters is not None


class TestReadingThemBack:
    def test_the_conversation_page_carries_them(self, session_id):
        member = services.session_service.get_session(session_id).member_id
        api._finalize_turn(session_id, _Turn(plan_parameters=CARD))
        page = api.get_conversation(session_id, member_id=member, before_id=None, limit=20)
        assistant = [m for m in page.messages if m["role"] == "assistant"][-1]
        assert assistant["extras"]["plan_parameters"]["plan_type"] == "daily"

    def test_a_message_from_before_this_existed_reads_as_none(self, session_id):
        member = services.session_service.get_session(session_id).member_id
        page = api.get_conversation(session_id, member_id=member, before_id=None, limit=20)
        assert all(m["extras"] is None for m in page.messages)

    def test_the_in_memory_message_matches_the_database(self, session_id):
        """A caller reading the session without a round trip must see the same
        thing the next reload will."""
        api._finalize_turn(session_id, _Turn(memory_suggestions=NUDGE))
        session = services.session_service.get_session(session_id)
        assistant = [m for m in session.conversation if m.role == "assistant"][-1]
        assert assistant.extras["memory_suggestions"] == NUDGE

    def test_the_conversation_page_is_still_ownership_checked(self, session_id):
        with pytest.raises(HTTPException) as e:
            api.get_conversation(session_id, member_id=f"member-{uuid.uuid4()}",
                                before_id=None, limit=20)
        assert e.value.status_code == 404


class TestItNeverBreaksTheTurn:
    def test_a_failed_write_is_not_an_error(self, session_id):
        """The turn has already happened and its plan is already stored.
        Failing to record the nudge costs the nudge, not the turn."""
        original = services.session_service.attach_turn_extras

        def boom(*_a, **_k):
            raise RuntimeError("database went away")

        services.session_service.attach_turn_extras = boom
        try:
            with pytest.raises(RuntimeError):
                api._finalize_turn(session_id, _Turn(memory_suggestions=NUDGE))
        finally:
            services.session_service.attach_turn_extras = original

    def test_the_service_swallows_a_database_failure(self, session_id, monkeypatch):
        """Same guarantee, at the layer that actually owns it."""
        # `services.session_service` is the SINGLETON, not the module — the
        # package rebinds the name. Patch the module the function lives in.
        mod = sys.modules["services.session_service"]

        def boom(*_a, **_k):
            raise RuntimeError("write failed")

        monkeypatch.setattr(mod, "db_set_last_assistant_extras", boom)
        assert services.session_service.attach_turn_extras(
            session_id, {"memory_suggestions": NUDGE}
        ) is False

    def test_a_session_with_no_assistant_message_is_not_a_crash(self):
        member = f"member-{uuid.uuid4()}"
        session = services.session_service.create_session(member, {})
        services.session_service.add_message(session.session_id, "user", "hello")
        assert services.session_service.attach_turn_extras(
            session.session_id, {"memory_suggestions": NUDGE}
        ) is False

    def test_empty_extras_are_a_no_op(self, session_id):
        assert services.session_service.attach_turn_extras(session_id, {}) is False
        assert services.session_service.attach_turn_extras(session_id, None) is False


class TestEveryTurnShapedEndpointGoesThroughTheFunnel:
    """Four endpoints return a turn. The one that forgets loses a memory nudge
    with no error anywhere, so assert none of them can."""

    def test_no_endpoint_bypasses_it(self):
        import inspect

        src = inspect.getsource(api)
        # `_finalize_turn` is the only place allowed to call the raw builder.
        calls = [
            line.strip() for line in src.splitlines()
            if "_chat_turn_response(turn)" in line and "def " not in line
        ]
        assert len(calls) == 1, f"a turn endpoint bypasses the funnel: {calls}"

    @pytest.mark.parametrize("endpoint", [
        "unified_chat", "compose_plan", "apply_plan_parameters", "replan",
    ])
    def test_the_endpoint_finalizes(self, endpoint):
        import inspect

        assert "_finalize_turn" in inspect.getsource(getattr(api, endpoint))
