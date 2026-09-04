"""
No member may act as another member. Enforced, and enforced for every route.

FoodChat was internally unauthenticated: every session-scoped endpoint took
`member_id` as DATA and trusted it. The gateway in front does authenticate, but
anything that could reach FoodChat's port could name any member alive and be
believed — which inside a cluster is one misconfigured NetworkPolicy, one
port-forward, or one other pod away.

Two checks now stand between a caller and someone else's data:

    identity   the gateway signs "this caller IS this member"; FoodChat
               verifies the signature and requires the request to match  -> 401
    ownership  this member owns this session                             -> 404

The second was always there and was never sufficient on its own: a caller free
to name any member can always name the owner.

This file is deliberately built the hard way — it CALLS every route with a
mismatched identity rather than reading the source for a guard. A test that
greps for `_require_member` passes for a route that calls it and ignores the
result. The last test then walks the router's own route table and fails for any
member-scoped route that is not exercised above, so a new route cannot quietly
skip the audit.
"""

from __future__ import annotations

import inspect
import sys
import uuid

import pytest
from fastapi import HTTPException

sys.path.insert(0, "src")

import auth                                                  # noqa: E402
import services                                              # noqa: E402
from routers import foodchat_router as api                    # noqa: E402

SECRET = "test-assertion-secret"


@pytest.fixture(autouse=True)
def enforcing(monkeypatch):
    """Run this whole file with assertions configured.

    The rest of the suite runs without a secret, which is the unconfigured
    deployment — and proving THAT still works is its own test below.
    """
    monkeypatch.setenv("FOODCHAT_ASSERTION_SECRET", SECRET)
    auth.set_asserted_member(None)
    yield
    auth.set_asserted_member(None)


@pytest.fixture
def victim():
    """A member with a session, a plan and a message — something worth taking."""
    member = f"victim-{uuid.uuid4()}"
    session = services.session_service.create_session(member, {"diet": []})
    services.session_service.add_message(session.session_id, "user", "plan my day")
    services.session_service.add_message(session.session_id, "assistant", "Here you go.")
    return member, session.session_id


@pytest.fixture
def attacker():
    return f"attacker-{uuid.uuid4()}"


# ── the assertion itself ──────────────────────────────────────────────────

class TestTheAssertion:
    def test_a_signed_assertion_round_trips(self):
        assert auth.verify(auth.sign("m1", key=SECRET), key=SECRET) == "m1"

    def test_a_forged_signature_is_refused(self):
        good = auth.sign("m1", key=SECRET)
        forged = good.rsplit(".", 1)[0] + "." + "0" * 64
        with pytest.raises(auth.AssertionError_):
            auth.verify(forged, key=SECRET)

    def test_editing_the_member_invalidates_it(self):
        """The whole point: a caller cannot rewrite whose assertion this is."""
        good = auth.sign("victim", key=SECRET)
        tampered = "attacker" + good[len("victim"):]
        with pytest.raises(auth.AssertionError_):
            auth.verify(tampered, key=SECRET)

    def test_a_different_secret_is_refused(self):
        with pytest.raises(auth.AssertionError_):
            auth.verify(auth.sign("m1", key="someone-elses-key"), key=SECRET)

    def test_an_expired_assertion_is_refused(self):
        stale = auth.sign("m1", key=SECRET, ttl=-3600)
        with pytest.raises(auth.AssertionError_) as e:
            auth.verify(stale, key=SECRET)
        assert "xpired" in str(e.value)

    def test_a_little_clock_skew_is_tolerated(self):
        """A pod a second ahead must not reject every assertion it is minted."""
        fresh = auth.sign("m1", key=SECRET, ttl=0)
        assert auth.verify(fresh, key=SECRET, now=None) == "m1"

    def test_a_member_id_containing_dots_survives(self):
        """The format splits from the right, so an id with dots is not a bug
        waiting for the first member whose id has one."""
        weird = "a.b.c-123"
        assert auth.verify(auth.sign(weird, key=SECRET), key=SECRET) == weird

    def test_a_member_id_cannot_be_shifted_into_the_expiry(self):
        """The separator is inside the signed payload, so `a|1` and `a.1`
        cannot be re-cut into a different meaning with the same digest."""
        one = auth._digest("a.1", 2, SECRET)
        two = auth._digest("a", 12, SECRET)
        assert one != two

    @pytest.mark.parametrize("bad", [None, "", "garbage", "a.b", "m1.notanumber.ff"])
    def test_malformed_assertions_are_refused(self, bad):
        with pytest.raises(auth.AssertionError_):
            auth.verify(bad, key=SECRET)

    def test_no_secret_means_nothing_verifies(self, monkeypatch):
        monkeypatch.delenv("FOODCHAT_ASSERTION_SECRET", raising=False)
        assert auth.enforcing() is False
        with pytest.raises(auth.AssertionError_):
            auth.verify("anything")


# ── the routes ────────────────────────────────────────────────────────────
#
# Each entry builds the arguments for one handler. `member` is substituted with
# whichever identity the test is exercising, so one table drives both the
# "wrong member is refused" and "right member is allowed" passes.

def _routes(session_id: str):
    return {
        "get_session": lambda m: (api.get_session, {"session_id": session_id, "member_id": m}),
        "delete_session": lambda m: (api.delete_session, {"session_id": session_id, "member_id": m}),
        "rename_session": lambda m: (api.rename_session, {
            "session_id": session_id,
            "request": api.RenameSessionRequest(member_id=m, title="Renamed"),
        }),
        "create_session": lambda m: (api.create_session, {
            "request": api.CreateSessionRequest(member_id=m),
        }),
        "save_meal_plan": lambda m: (api.save_meal_plan, {
            "session_id": session_id, "plan_id": "p1",
            "request": api.SavePlanRequest(member_id=m, saved=True),
        }),
        "get_member_saved_plans": lambda m: (api.get_member_saved_plans, {"member_id": m}),
        # Someone's ratings and the comments they wrote, from their id alone —
        # exactly the shape of route this audit exists for.
        "get_member_feedback": lambda m: (api.get_member_feedback, {"member_id": m}),
        "get_member_sessions": lambda m: (api.get_member_sessions, {"member_id": m}),
        "get_member_current_plans": lambda m: (api.get_member_current_plans, {"member_id": m}),
        "unified_chat": lambda m: (api.unified_chat, {
            "session_id": session_id,
            "request": api.ChatRequest(member_id=m, content="hello"),
        }),
        # A real pick: compose validates the payload before it reaches the
        # orchestrator, and an empty one would 400 before ownership is asked.
        "compose_plan": lambda m: (api.compose_plan, {
            "session_id": session_id,
            "request": api.ComposeRequest(member_id=m, picks=[
                api.ComposePick(meal_type="dinner", recipe_id="r1", title="Ragu"),
            ]),
        }),
        "apply_plan_parameters": lambda m: (api.apply_plan_parameters, {
            "session_id": session_id,
            "request": api.PlanParametersRequest(member_id=m, values={"goal": "balanced"}),
        }),
        "replan": lambda m: (api.replan, {
            "session_id": session_id,
            "request": api.RegenerateRequest(member_id=m),
        }),
        "get_conversation": lambda m: (api.get_conversation, {
            "session_id": session_id, "member_id": m, "before_id": None, "limit": 20,
        }),
        "get_meal_plans": lambda m: (api.get_meal_plans, {"session_id": session_id, "member_id": m}),
        "get_current_meal_plan": lambda m: (api.get_current_meal_plan, {
            "session_id": session_id, "member_id": m}),
        "get_daily_plan_history": lambda m: (api.get_daily_plan_history, {
            "session_id": session_id, "member_id": m}),
        "get_weekly_meal_plans": lambda m: (api.get_weekly_meal_plans, {
            "session_id": session_id, "member_id": m}),
        "get_current_weekly_meal_plan": lambda m: (api.get_current_weekly_meal_plan, {
            "session_id": session_id, "member_id": m}),
        "get_weekly_plan_history": lambda m: (api.get_weekly_plan_history, {
            "session_id": session_id, "member_id": m}),
        "submit_feedback": lambda m: (api.submit_feedback, {
            "session_id": session_id, "message_id": 1,
            "request": api.FeedbackRequest(member_id=m, rating="up"),
        }),
        "decide_memory": lambda m: (api.decide_memory, {
            "session_id": session_id,
            "request": api.MemoryDecisionRequest(
                member_id=m, decision="decline",
                suggestion=api.MemorySuggestionModel(
                    id="s1", kind="like", value="tofu", statement="likes tofu"),
            ),
        }),
        "set_diners": lambda m: (api.set_diners, {
            "session_id": session_id,
            "request": api.SetDinersRequest(member_id=m, cooking_for=[]),
        }),
        "get_planning_state": lambda m: (api.get_planning_state, {
            "session_id": session_id, "member_id": m}),
        "set_pantry": lambda m: (api.set_pantry, {
            "session_id": session_id,
            "request": api.PantryRequest(member_id=m, items=["zucchini"]),
        }),
        "add_pantry_items": lambda m: (api.add_pantry_items, {
            "session_id": session_id,
            "request": api.PantryRequest(member_id=m, items=["zucchini"]),
        }),
        "remove_pantry_item": lambda m: (api.remove_pantry_item, {
            "session_id": session_id, "item": "zucchini", "member_id": m}),
        "remove_facet": lambda m: (api.remove_facet, {
            "session_id": session_id, "value": "light", "member_id": m}),
        "add_facets": lambda m: (api.add_facets, {
            "session_id": session_id,
            "request": api.FacetRequest(member_id=m, values=["light"]),
        }),
        "invoke_tool": lambda m: (api.invoke_tool, {
            "tool_name": "summarize_week",
            "request": api.ToolInvokeRequest(
                member_id=m, arguments={"session_id": session_id}),
        }),
    }


# Routes that carry no member identity, with the reason each one is open.
OPEN_ROUTES = {
    "health_check": "kubelet has no assertion to send",
    "readiness_check": "same — and a probe that 401s takes the deployment down",
    "list_tools": "the manifest is identical for every member",
    "get_vocabularies": "the corpus vocabulary is identical for every member",
}


# Routes that name a session. For these, an attacker asking honestly — as
# themselves, about a session they do not own — must still be refused, and the
# refusal is a 404 so the session id is not confirmed.
SESSION_SCOPED = sorted(
    name for name in _routes("x")
    if "session_id" in _routes("x")[name]("m")[1]
    or "session_id" in getattr(_routes("x")[name]("m")[1].get("request"), "arguments", {})
)


# The four routes whose ownership check lives one layer down, in the
# orchestrator. The router only maps the refusal; the refusal itself is tested
# against the real `_owned_session` below.
ORCHESTRATOR_ROUTED = {
    "unified_chat", "compose_plan", "apply_plan_parameters", "replan",
}


class TestAnotherMemberIsRefused:
    """The headline guarantee, executed against every member-scoped route."""

    @pytest.mark.parametrize(
        "name", [n for n in SESSION_SCOPED if n not in ORCHESTRATOR_ROUTED]
    )
    def test_an_honest_attacker_still_cannot_reach_someone_elses_session(
        self, name, victim, attacker
    ):
        """The gateway vouched for the attacker and the attacker asked as
        themselves — so identity passes, and ownership is what has to hold."""
        _, session_id = victim
        handler, kwargs = _routes(session_id)[name](attacker)
        auth.set_asserted_member(attacker)
        with pytest.raises(HTTPException) as e:
            handler(**kwargs)
        assert e.value.status_code == 404, f"{name} returned {e.value.status_code}"

    def test_the_orchestrator_refuses_a_session_it_does_not_own(self, victim, attacker):
        """The real check behind the four orchestrator-routed endpoints."""
        from services.orchestrator_service import OrchestratorService, SessionAccessError

        _, session_id = victim
        owned = OrchestratorService.__new__(OrchestratorService)
        owned.session_service = services.session_service
        with pytest.raises(SessionAccessError):
            owned._owned_session(session_id, attacker)

    @pytest.mark.parametrize("name", sorted(ORCHESTRATOR_ROUTED))
    def test_the_router_turns_that_refusal_into_a_404(self, name, victim, attacker):
        """A 403 would confirm the session exists; a 500 would leak a stack."""
        from services.orchestrator_service import SessionAccessError

        _, session_id = victim

        class _Refusing:
            def __getattr__(self, _name):
                def call(*_a, **_k):
                    raise SessionAccessError("Session not found or access denied")
                return call

        handler, kwargs = _routes(session_id)[name](attacker)
        auth.set_asserted_member(attacker)
        saved = services.orchestrator_service
        services.orchestrator_service = _Refusing()
        try:
            with pytest.raises(HTTPException) as e:
                handler(**kwargs)
        finally:
            services.orchestrator_service = saved
        assert e.value.status_code == 404, f"{name} returned {e.value.status_code}"

    @pytest.mark.parametrize("name", sorted(_routes("x")))
    def test_claiming_to_be_someone_else_is_a_401(self, name, victim, attacker):
        """The attack the assertion exists to stop: reach the port, put the
        victim's id in the body, and be believed."""
        victim_id, session_id = victim
        handler, kwargs = _routes(session_id)[name](victim_id)
        auth.set_asserted_member(attacker)   # the gateway vouched for someone else
        with pytest.raises(HTTPException) as e:
            handler(**kwargs)
        assert e.value.status_code == 401, f"{name} returned {e.value.status_code}"

    @pytest.mark.parametrize("name", sorted(_routes("x")))
    def test_no_assertion_at_all_is_a_401(self, name, victim):
        """Belt and braces: the middleware rejects these before a handler sees
        them, but a handler must not depend on the middleware having run."""
        victim_id, session_id = victim
        handler, kwargs = _routes(session_id)[name](victim_id)
        auth.set_asserted_member(None)
        with pytest.raises(HTTPException) as e:
            handler(**kwargs)
        assert e.value.status_code == 401, f"{name} returned {e.value.status_code}"


class TestTheOwnerStillWorks:
    """A guarantee is worthless if it also blocks the person it protects."""

    @pytest.mark.parametrize("name", [
        "get_session", "get_conversation", "get_meal_plans", "get_planning_state",
        "get_member_sessions", "get_member_saved_plans", "get_member_current_plans",
        "set_pantry", "remove_facet",
    ])
    def test_the_owner_is_let_through(self, name, victim):
        victim_id, session_id = victim
        handler, kwargs = _routes(session_id)[name](victim_id)
        auth.set_asserted_member(victim_id)
        handler(**kwargs)   # must not raise

    def test_a_404_still_does_not_confirm_a_session_exists(self, victim, attacker):
        """Ownership failures stay indistinguishable from "no such session" —
        otherwise a 403 would confirm the id is real."""
        _, session_id = victim
        auth.set_asserted_member(attacker)
        with pytest.raises(HTTPException) as real:
            api.get_session(session_id=session_id, member_id=attacker)
        with pytest.raises(HTTPException) as fake:
            api.get_session(session_id="no-such-session", member_id=attacker)
        assert real.value.status_code == fake.value.status_code == 404
        assert real.value.detail == fake.value.detail


class TestUnconfiguredDeploymentStillWorks:
    """Without a secret, FoodChat behaves exactly as it did before.

    Deliberate: the gateway and FoodChat deploy independently, and a control
    that breaks the service when only one side has shipped is a control someone
    reverts under pressure. Setting the secret on both sides IS the enable step.
    """

    def test_no_secret_means_no_identity_check(self, monkeypatch, victim):
        monkeypatch.delenv("FOODCHAT_ASSERTION_SECRET", raising=False)
        victim_id, session_id = victim
        auth.set_asserted_member(None)
        api.get_session(session_id=session_id, member_id=victim_id)  # must not raise

    def test_ownership_is_still_enforced_without_a_secret(self, monkeypatch, victim, attacker):
        """The layer that always existed does not go away when the new one is
        off — otherwise turning the secret off would open everything."""
        monkeypatch.delenv("FOODCHAT_ASSERTION_SECRET", raising=False)
        _, session_id = victim
        auth.set_asserted_member(None)
        with pytest.raises(HTTPException) as e:
            api.get_session(session_id=session_id, member_id=attacker)
        assert e.value.status_code == 404


class TestNoRouteEscapesTheAudit:
    """The test that keeps this file honest as routes are added."""

    def test_every_route_is_either_audited_or_declared_open(self):
        audited = set(_routes("x")) | set(OPEN_ROUTES)
        registered = {
            route.endpoint.__name__
            for route in api.router.routes
            if getattr(route, "endpoint", None) is not None
        }
        missing = sorted(registered - audited)
        assert not missing, (
            "these routes are neither audited above nor declared open: "
            + ", ".join(missing)
        )

    def test_the_open_list_holds_no_route_that_takes_a_member(self):
        """A route cannot be declared open if it names a member — that is
        exactly the thing that needs checking."""
        for name in OPEN_ROUTES:
            handler = getattr(api, name)
            params = inspect.signature(handler).parameters
            assert "member_id" not in params, name
            for param in params.values():
                fields = getattr(param.annotation, "model_fields", {})
                assert "member_id" not in fields, f"{name} takes a member in its body"

    def test_the_audit_table_has_not_gone_stale(self):
        """Every name in the table is a real handler on the router."""
        registered = {
            route.endpoint.__name__
            for route in api.router.routes
            if getattr(route, "endpoint", None) is not None
        }
        stale = sorted(set(_routes("x")) - registered)
        assert not stale, f"audited routes that no longer exist: {stale}"


class TestTheProbesStayReachable:
    """A probe that 401s takes the deployment down.

    The kubelet has no member assertion and never will. If enforcement covered
    the probe paths, setting the shared secret — the act of SECURING the
    service — would fail every liveness and readiness check and roll the
    deployment into a crash loop.
    """

    @pytest.mark.parametrize("path", ["/foodchat/health", "/foodchat/ready"])
    def test_a_probe_path_is_never_gated(self, path):
        assert auth._is_open_path(path)

    @pytest.mark.parametrize("path", [
        "/foodchat/vocabularies", "/foodchat/tools",
    ])
    def test_the_memberless_reads_are_open_too(self, path):
        assert auth._is_open_path(path)

    @pytest.mark.parametrize("path", [
        "/foodchat/sessions", "/foodchat/sessions/abc/chat",
        "/foodchat/members/m1/sessions", "/foodchat/sessions/abc/pantry",
    ])
    def test_everything_member_scoped_is_gated(self, path):
        assert not auth._is_open_path(path)

    def test_a_trailing_slash_does_not_open_a_gated_path(self):
        assert not auth._is_open_path("/foodchat/sessions/abc/chat/")
