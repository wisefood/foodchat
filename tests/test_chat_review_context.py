"""Reading a chat exchange back for a reviewer judging feedback about it.

A thumbs-down used to arrive in the shared inbox as a rating, a target type and
an opaque message id. These tests cover the route that turns that id back into
the conversation — and, as much, the ways it is allowed to fail: a pruned
session and an unreachable service are ordinary outcomes here, and both have to
degrade to a sentence rather than an error page.
"""

import pytest

import json

from db import Base, MealPlanRow, MessageRow, SessionLocal, SessionRow, engine
from services.chat_review_service import (
    CHAT_REVIEW_SERVICE,
    DEFAULT_WINDOW,
    MAX_CONTENT_CHARS,
    MAX_WINDOW,
    ChatReviewService,
)


@pytest.fixture()
def conversation():
    """Twelve turns in one session, and three in another."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(MessageRow).delete()
        db.query(SessionRow).filter(SessionRow.session_id.in_(["s-main", "s-other"])).delete(
            synchronize_session=False
        )
        db.commit()
        # Messages are FK-bound to a session, so the parents come first.
        for session_id in ("s-main", "s-other"):
            db.add(
                SessionRow(session_id=session_id, member_id="m-1", user_profile="{}")
            )
        db.commit()
        for index in range(12):
            db.add(
                MessageRow(
                    session_id="s-main",
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"turn {index}",
                    intent="plan" if index % 2 else None,
                )
            )
        for index in range(3):
            db.add(
                MessageRow(
                    session_id="s-other",
                    role="user",
                    content=f"other {index}",
                )
            )
        db.commit()
        rows = (
            db.query(MessageRow)
            .filter(MessageRow.session_id == "s-main")
            .order_by(MessageRow.id.asc())
            .all()
        )
        yield [row.id for row in rows]
    finally:
        db.query(MessageRow).delete()
        db.query(SessionRow).filter(SessionRow.session_id.in_(["s-main", "s-other"])).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


class TestTheExchangeComesBack:
    def test_the_rated_message_is_marked(self, conversation):
        """So the console does not have to compare ids to highlight it."""
        target = conversation[6]
        result = CHAT_REVIEW_SERVICE.message_context(str(target))
        rated = [m for m in result["messages"] if m["is_rated"]]
        assert len(rated) == 1
        assert rated[0]["id"] == target

    def test_turns_either_side_are_included_in_order(self, conversation):
        target = conversation[6]
        result = CHAT_REVIEW_SERVICE.message_context(str(target), window=2)
        ids = [m["id"] for m in result["messages"]]
        assert ids == conversation[4:9]
        # Ordered by id, not timestamp: two turns inside one second are common
        # and their order is what makes the exchange readable.
        assert ids == sorted(ids)

    def test_it_does_not_reach_into_another_session(self, conversation):
        result = CHAT_REVIEW_SERVICE.message_context(str(conversation[0]))
        assert all(m["id"] in conversation for m in result["messages"])

    def test_the_start_of_a_conversation_has_no_preceding_turns(self, conversation):
        result = CHAT_REVIEW_SERVICE.message_context(str(conversation[0]), window=4)
        assert result["messages"][0]["is_rated"] is True

    def test_the_end_of_a_conversation_has_no_following_turns(self, conversation):
        result = CHAT_REVIEW_SERVICE.message_context(str(conversation[-1]), window=4)
        assert result["messages"][-1]["is_rated"] is True

    def test_it_says_how_much_it_is_not_showing(self, conversation):
        """So the console can say '5 of 42 turns' rather than implying the
        excerpt is the whole conversation."""
        result = CHAT_REVIEW_SERVICE.message_context(str(conversation[6]), window=2)
        assert result["session_messages"] == 12
        assert len(result["messages"]) == 5


class TestItIsNarrowOnPurpose:
    def test_the_window_is_capped(self, conversation):
        """Enough to judge a complaint, not a window onto someone's history."""
        result = CHAT_REVIEW_SERVICE.message_context(
            str(conversation[6]), window=MAX_WINDOW * 10
        )
        assert result["window"] == MAX_WINDOW

    def test_a_missing_message_is_not_an_error(self, conversation):
        """Sessions are pruned and a complaint outlives the conversation it was
        about, so an inbox item pointing at nothing is expected."""
        assert CHAT_REVIEW_SERVICE.message_context("999999") is None

    def test_a_non_numeric_id_costs_no_query(self):
        """Feedback carries whatever the client sent."""
        assert CHAT_REVIEW_SERVICE.message_context("not-a-number") is None
        assert CHAT_REVIEW_SERVICE.message_context("") is None
        assert CHAT_REVIEW_SERVICE.message_context(None) is None

    def test_long_content_is_truncated_and_says_so(self, conversation):
        db = SessionLocal()
        try:
            row = db.query(MessageRow).filter(MessageRow.id == conversation[6]).first()
            row.content = "x" * (MAX_CONTENT_CHARS + 500)
            db.commit()
        finally:
            db.close()
        result = CHAT_REVIEW_SERVICE.message_context(str(conversation[6]), window=1)
        rated = [m for m in result["messages"] if m["is_rated"]][0]
        assert len(rated["content"]) == MAX_CONTENT_CHARS
        assert rated["truncated"] is True

    def test_defaults_are_conservative(self):
        assert DEFAULT_WINDOW <= 6
        assert MAX_WINDOW <= 25


class TestItLivesApartFromTheMemberScopedRouter:
    """`foodchat_router` guarantees every route checks session ownership, and
    `test_route_authorization.py` proves it by enumeration. A reviewer reads
    conversations they do not own, so this route cannot satisfy that check —
    and rather than weaken a router whose whole value is that guarantee, it
    lives under its own prefix."""

    def test_the_review_route_is_not_in_the_member_router(self):
        from routers import foodchat_router

        paths = {route.path for route in foodchat_router.router.routes}
        assert not any("review" in path for path in paths)

    def test_the_review_router_exposes_only_the_two_reads(self):
        from routers import review_router

        methods = {
            (route.path, tuple(sorted(route.methods)))
            for route in review_router.router.routes
        }
        assert methods == {
            ("/foodchat/review/messages/{message_id}/context", ("GET",)),
            ("/foodchat/review/plans/{plan_id}", ("GET",)),
        }

    def test_it_is_documented_as_gateway_gated(self):
        """The authorization is real but lives one layer up. If that stops
        being written down, the next person removes the gateway check."""
        import services.chat_review_service as module

        assert "gateway" in (module.__doc__ or "").lower()


DAILY = {
    "id": "p-daily",
    "reasoning": "Light and quick, with fish twice.",
    "llm_score": 7,
    "llm_reasoning": "Balanced, a little repetitive.",
    "fvs_count": 4,
    "fvs_reasoning": "Four portions of fruit and veg.",
    "diversity_llm_score": 0,
    "diversity_llm_reasoning": "",
    "guideline_adherence_score": 6,
    "guideline_adherence_reasoning": "Salt is high at dinner.",
    "constraints_applied": [
        {"constraint": "no peanuts", "type": "hard", "status": "satisfied"},
        {"constraint": "3 servings of dairy", "type": "soft", "status": "unmet", "kind": "guideline"},
    ],
    "personalization_summary": {"name": "Maria", "allergies": ["peanuts"]},
    "breakfast": {"recipe_id": "r1", "title": "Oat porridge", "ingredients": "oats, milk",
                  "directions": "Stir.", "nutrition": {"kcal": 320},
                  "match_reasons": [{"kind": "diet", "label": "Vegetarian"}]},
    "lunch": {"recipe_id": "r2", "title": "Tuna salad", "ingredients": "tuna",
              "directions": "Mix.", "nutrition": None, "match_reasons": []},
    "dinner": {"recipe_id": "r3", "title": "Baked cod", "ingredients": "cod",
               "directions": "Bake.", "nutrition": {"kcal": 510}, "match_reasons": []},
}

WEEKLY = {
    "id": "p-weekly",
    "entries": [
        {"day": 1, "meal_idx": 0, "meal_type": "breakfast", "recipe": {"recipe_id": "r1", "title": "Oat porridge"}, "reward": 1},
        {"day": 1, "meal_idx": 2, "meal_type": "dinner", "recipe": {"recipe_id": "r3", "title": "Baked cod"}, "reward": 1},
        {"day": 1, "meal_idx": 2, "meal_type": "dinner", "recipe": {"recipe_id": "r4", "title": "Green salad"}, "reward": 1, "role": "side"},
        {"day": 2, "meal_idx": 0, "meal_type": "breakfast", "recipe": {"recipe_id": "r5", "title": "Yoghurt"}, "reward": 1},
    ],
    "metrics": {"variety": 0.8},
    "constraints_applied": [],
    "personalization_summary": {"name": "Maria"},
    "reasoning": "A week of quick dinners.",
}


@pytest.fixture()
def plans():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(MealPlanRow).filter(MealPlanRow.session_id == "s-plans").delete()
        db.query(SessionRow).filter(SessionRow.session_id == "s-plans").delete()
        db.commit()
        db.add(SessionRow(session_id="s-plans", member_id="m-1", user_profile="{}"))
        db.commit()
        db.add(MealPlanRow(id="p-daily", session_id="s-plans", plan_type="daily", version=2,
                           parent_id="p-v1", saved=1, saved_title="Maria's Tuesday",
                           payload=json.dumps(DAILY)))
        db.add(MealPlanRow(id="p-weekly", session_id="s-plans", plan_type="weekly",
                           payload=json.dumps(WEEKLY)))
        db.commit()
        yield
    finally:
        db.query(MealPlanRow).filter(MealPlanRow.session_id == "s-plans").delete()
        db.query(SessionRow).filter(SessionRow.session_id == "s-plans").delete()
        db.commit()
        db.close()


class TestWhatAPlanWas:
    def test_a_daily_plan_is_its_dishes_per_meal(self, plans):
        plan = CHAT_REVIEW_SERVICE.plan_summary("p-daily")
        assert [m["meal_type"] for m in plan["meals"]] == ["breakfast", "lunch", "dinner"]
        assert plan["meals"][0]["plates"] == [{
            "recipe_id": "r1", "title": "Oat porridge", "role": "main", "kcal": 320,
            "match_reasons": ["Vegetarian"],
        }]
        assert plan["version"] == 2 and plan["parent_id"] == "p-v1" and plan["saved"] is True

    def test_the_judges_say_why(self, plans):
        scores = {s["name"]: s for s in CHAT_REVIEW_SERVICE.plan_summary("p-daily")["scores"]}
        assert scores["guidelines"] == {"name": "guidelines", "score": 6,
                                        "reasoning": "Salt is high at dinner."}
        # A judge that did not run is not a verdict of zero.
        assert "variety" not in scores

    def test_the_rows_the_member_saw_come_back(self, plans):
        rows = CHAT_REVIEW_SERVICE.plan_summary("p-daily")["constraints_applied"]
        assert [r["constraint"] for r in rows] == ["no peanuts", "3 servings of dairy"]

    def test_a_weekly_plan_groups_plates_into_meals(self, plans):
        plan = CHAT_REVIEW_SERVICE.plan_summary("p-weekly")
        dinner = [m for m in plan["meals"] if m["day"] == 1 and m["meal_type"] == "dinner"][0]
        assert [p["title"] for p in dinner["plates"]] == ["Baked cod", "Green salad"]
        assert dinner["plates"][1]["role"] == "side"
        assert len(plan["meals"]) == 3
        assert plan["scores"] == [] and plan["metrics"] == {"variety": 0.8}

    def test_nothing_the_member_wrote_or_their_profile_leaves(self, plans):
        flat = repr(CHAT_REVIEW_SERVICE.plan_summary("p-daily"))
        for private in ("Maria", "allergies", "personalization", "Stir.", "oats, milk"):
            assert private not in flat, private

    def test_a_pruned_plan_is_not_an_error(self, plans):
        assert CHAT_REVIEW_SERVICE.plan_summary("gone") is None
        assert CHAT_REVIEW_SERVICE.plan_summary("") is None

    def test_long_reasoning_is_clipped(self, plans):
        from services.chat_review_service import MAX_REASONING_CHARS, _clip

        assert len(_clip("x" * (MAX_REASONING_CHARS + 50))) == MAX_REASONING_CHARS + 1


class TestOnlyTheGatewaysReviewerGetsIn:
    """Where assertions are configured, a review read must be signed as the
    reviewer. A member's assertion opened these routes before, and the
    gateway's own reads — which name no member — were refused."""

    SECRET = "review-test-secret"

    def _call(self, monkeypatch, header):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import auth
        from routers import review_router

        # The review router behind the real middleware, without the rest of
        # the service's startup.
        app = FastAPI()
        app.middleware("http")(auth.assertion_middleware)
        app.include_router(review_router.router)
        monkeypatch.setenv("FOODCHAT_ASSERTION_SECRET", self.SECRET)
        client = TestClient(app)
        headers = {auth.HEADER: header} if header else {}
        return client.get("/foodchat/review/plans/nope", headers=headers)

    def test_the_reviewer_reaches_the_route(self, monkeypatch):
        import auth

        response = self._call(monkeypatch, auth.sign(auth.REVIEWER, key=self.SECRET))
        assert response.status_code == 404  # through the gate, to a missing plan

    def test_a_member_is_refused(self, monkeypatch):
        import auth

        assert self._call(monkeypatch, auth.sign("m-1", key=self.SECRET)).status_code == 401

    def test_nobody_is_refused(self, monkeypatch):
        assert self._call(monkeypatch, None).status_code == 401

    def test_without_a_secret_nothing_is_checked(self, monkeypatch):
        import auth

        monkeypatch.delenv("FOODCHAT_ASSERTION_SECRET", raising=False)
        auth.set_asserted_member(None)
        auth.check_reviewer()  # no raise
