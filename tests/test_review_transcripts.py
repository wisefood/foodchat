"""A conversation read back whole, for a reviewer: what was said, what came
back, which plan each turn made, and what each plan was asked for.

The research team could see in the console that a FoodChat turn happened and
how long it took, but not what the person typed or how FoodChat read it. Three
pieces make that possible, each tested here: turns stamp their intent and plan
on their messages (the columns were always empty), every plan keeps a copy of
its request (the planning state is overwritten by the next turn), and the
review service returns one conversation with both.
"""

import json
from types import SimpleNamespace

import pytest

from db import (
    Base,
    MealPlanRow,
    MessageRow,
    SessionLocal,
    SessionRow,
    db_save_meal_plan,
    db_stamp_last_turn,
    engine,
)
from services.chat_review_service import CHAT_REVIEW_SERVICE, MAX_CONTENT_CHARS

SESSION = "s-transcript"
OTHER = "s-transcript-other"


def _clear(db):
    for sid in (SESSION, OTHER):
        db.query(MealPlanRow).filter(MealPlanRow.session_id == sid).delete()
        db.query(MessageRow).filter(MessageRow.session_id == sid).delete()
        db.query(SessionRow).filter(SessionRow.session_id == sid).delete()
    db.commit()


@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    _clear(session)
    for sid in (SESSION, OTHER):
        session.add(SessionRow(session_id=sid, member_id="m-1", user_profile="{}"))
    session.commit()
    try:
        yield session
    finally:
        _clear(session)
        session.close()


def _say(db, role, content, session_id=SESSION, **kw):
    row = MessageRow(session_id=session_id, role=role, content=content, **kw)
    db.add(row)
    db.commit()
    return row


class TestATurnStampsItsMessages:
    def test_the_reply_gets_the_intent_and_the_plan(self, db):
        _say(db, "user", "Plan my week")
        reply = _say(db, "assistant", "Here is your week")
        assert db_stamp_last_turn(db, SESSION, "generate_weekly_plan", "plan-1") is True
        db.refresh(reply)
        assert (reply.intent, reply.plan_id) == ("generate_weekly_plan", "plan-1")

    def test_the_message_it_answered_gets_the_intent(self, db):
        asked = _say(db, "user", "Plan my week")
        _say(db, "assistant", "Here is your week")
        db_stamp_last_turn(db, SESSION, "generate_weekly_plan", "plan-1")
        db.refresh(asked)
        assert asked.intent == "generate_weekly_plan"
        assert asked.plan_id is None  # the plan is the reply's

    def test_only_the_latest_turn_is_stamped(self, db):
        first_q = _say(db, "user", "hello")
        first_a = _say(db, "assistant", "hi")
        _say(db, "user", "Plan my day")
        _say(db, "assistant", "Your day")
        db_stamp_last_turn(db, SESSION, "generate_meal_plan", "plan-2")
        db.refresh(first_q)
        db.refresh(first_a)
        assert first_q.intent is None and first_a.intent is None and first_a.plan_id is None

    def test_a_value_already_there_is_kept(self, db):
        reply = _say(db, "assistant", "x", intent="answer_question", plan_id="plan-0")
        db_stamp_last_turn(db, SESSION, "generate_meal_plan", "plan-9")
        db.refresh(reply)
        assert (reply.intent, reply.plan_id) == ("answer_question", "plan-0")

    def test_another_conversation_is_untouched(self, db):
        other = _say(db, "assistant", "elsewhere", session_id=OTHER)
        _say(db, "assistant", "here")
        db_stamp_last_turn(db, SESSION, "generate_meal_plan", "plan-3")
        db.refresh(other)
        assert other.intent is None and other.plan_id is None

    def test_a_conversation_with_no_reply_is_not_an_error(self, db):
        assert db_stamp_last_turn(db, SESSION, "generate_meal_plan", None) is False


class TestTheSessionServiceStamp:
    def test_nothing_to_stamp_costs_nothing(self):
        import services

        assert services.session_service.stamp_turn(SESSION, None, None) is False

    def test_a_database_failure_is_swallowed(self, monkeypatch):
        """A reviewer losing the link is a pity; a member losing a turn is a bug."""
        import importlib

        import services

        # `services.session_service` is the service instance, not the module.
        module = importlib.import_module("services.session_service")

        def boom(*_a, **_k):
            raise RuntimeError("db down")

        monkeypatch.setattr(module, "db_stamp_last_turn", boom)
        assert services.session_service.stamp_turn(SESSION, "generate_meal_plan", "p") is False


class TestAPlanKeepsItsRequest:
    def test_the_request_is_stored_beside_the_plan_not_inside_it(self, db):
        request = {"message": "Plan my week", "reading": {"spec": {"num_days": 7}}}
        db_save_meal_plan(db, "plan-r", SESSION, "weekly", {"entries": []}, request=json.dumps(request))
        row = db.query(MealPlanRow).filter(MealPlanRow.id == "plan-r").one()
        assert json.loads(row.request) == request
        # A shared plan is built from the payload: what somebody typed must
        # never be in it.
        assert "Plan my week" not in row.payload

    def test_the_snapshot_is_the_message_the_reading_and_the_profile(self):
        import services
        from models.session import Message

        service = services.session_service
        profile = {
            "allergies": ["peanut"], "food_dislikes": ["fish"], "calorie_target": 2000,
            "member_id": "m-1", "memory_log": ["private"],
            service.PLANNING_STATE_KEY: {"spec": {"num_days": 7, "meals": ["breakfast", "lunch", "dinner"]}},
        }
        fake = SimpleNamespace(
            conversation=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
                Message(role="user", content="Plan my week"),
            ],
            user_profile=profile,
        )
        service._sessions["s-snapshot"] = fake
        try:
            snapshot = json.loads(service._plan_request("s-snapshot"))
        finally:
            service._sessions.pop("s-snapshot", None)
        assert snapshot["message"] == "Plan my week"
        assert snapshot["reading"]["spec"]["num_days"] == 7
        assert snapshot["profile"] == {"allergies": ["peanut"], "food_dislikes": ["fish"], "calorie_target": 2000}

    def test_no_session_still_saves_the_plan(self):
        import services

        snapshot = json.loads(services.session_service._plan_request("s-not-in-memory"))
        assert snapshot["message"] is None and snapshot["reading"] is None


class TestTheTranscript:
    def test_every_turn_in_order_with_intent_and_plan(self, db):
        _say(db, "user", "Plan my week", intent="generate_weekly_plan")
        _say(db, "assistant", "Your week", intent="generate_weekly_plan", plan_id="plan-w")
        _say(db, "user", "Go away")
        _say(db, "assistant", "I'm sorry you're upset.")
        out = CHAT_REVIEW_SERVICE.transcript(SESSION)
        assert [(m["role"], m["content"]) for m in out["messages"]] == [
            ("user", "Plan my week"), ("assistant", "Your week"),
            ("user", "Go away"), ("assistant", "I'm sorry you're upset."),
        ]
        assert out["messages"][1]["plan_id"] == "plan-w"
        assert out["messages"][0]["intent"] == "generate_weekly_plan"
        assert out["truncated"] is False

    def test_plans_come_with_what_they_were_asked_for(self, db):
        request = {"message": "Plan my week", "reading": {"spec": {"num_days": 7}}, "profile": {}}
        db_save_meal_plan(db, "plan-a", SESSION, "weekly", {"entries": []}, request=json.dumps(request))
        db_save_meal_plan(db, "plan-b", SESSION, "weekly", {"entries": []}, version=2, parent_id="plan-a")
        plans = CHAT_REVIEW_SERVICE.transcript(SESSION)["plans"]
        assert [p["plan_id"] for p in plans] == ["plan-a", "plan-b"]
        assert plans[0]["request"] == request
        assert plans[1]["request"] is None  # made before requests were kept
        assert plans[1]["parent_id"] == "plan-a"

    def test_it_stays_inside_its_conversation(self, db):
        _say(db, "user", "mine")
        _say(db, "user", "not mine", session_id=OTHER)
        contents = [m["content"] for m in CHAT_REVIEW_SERVICE.transcript(SESSION)["messages"]]
        assert contents == ["mine"]

    def test_a_missing_conversation_is_none(self, db):
        assert CHAT_REVIEW_SERVICE.transcript("s-never-was") is None
        assert CHAT_REVIEW_SERVICE.transcript("") is None

    def test_long_messages_are_truncated_and_say_so(self, db):
        _say(db, "user", "x" * (MAX_CONTENT_CHARS + 10))
        message = CHAT_REVIEW_SERVICE.transcript(SESSION)["messages"][0]
        assert len(message["content"]) == MAX_CONTENT_CHARS and message["truncated"] is True

    def test_the_plan_summary_carries_the_request_too(self, db):
        db_save_meal_plan(db, "plan-s", SESSION, "weekly", {"entries": []},
                          request=json.dumps({"message": "Plan my week"}))
        assert CHAT_REVIEW_SERVICE.plan_summary("plan-s")["request"] == {"message": "Plan my week"}


class TestTheRoute:
    def test_a_missing_conversation_is_a_404(self):
        from fastapi import HTTPException

        from routers import review_router

        with pytest.raises(HTTPException) as caught:
            review_router.get_session_transcript("s-never-was")
        assert caught.value.status_code == 404
