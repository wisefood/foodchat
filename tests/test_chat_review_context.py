"""Reading a chat exchange back for a reviewer judging feedback about it.

A thumbs-down used to arrive in the shared inbox as a rating, a target type and
an opaque message id. These tests cover the route that turns that id back into
the conversation — and, as much, the ways it is allowed to fail: a pruned
session and an unreachable service are ordinary outcomes here, and both have to
degrade to a sentence rather than an error page.
"""

import pytest

from db import Base, MessageRow, SessionLocal, SessionRow, engine
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

    def test_the_review_router_exposes_only_the_context_read(self):
        from routers import review_router

        methods = {
            (route.path, tuple(sorted(route.methods)))
            for route in review_router.router.routes
        }
        assert methods == {
            ("/foodchat/review/messages/{message_id}/context", ("GET",))
        }

    def test_it_is_documented_as_gateway_gated(self):
        """The authorization is real but lives one layer up. If that stops
        being written down, the next person removes the gateway check."""
        import services.chat_review_service as module

        assert "gateway" in (module.__doc__ or "").lower()
