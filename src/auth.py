"""
Who the caller actually is.

FoodChat has been internally unauthenticated since it was written: every
session-scoped endpoint takes `member_id` as DATA and trusts it. The gateway in
front of it does authenticate — Keycloak token, then a household-ownership check
— but nothing stopped a caller who could reach FoodChat's port from acting as
any member alive. Inside a cluster that is one misconfigured NetworkPolicy, one
port-forward, or one other pod away.

The fix is defence in depth, and the shape of it follows from one fact: **only
the gateway can answer "does this Keycloak user own this member".** The
household tables live there. FoodChat validating the bearer token itself would
prove "some authenticated user" and nothing more — which is not the question.

So the gateway signs its own answer and FoodChat verifies the signature:

    X-WiseFood-Member: <member_id>.<expires_at>.<hmac-sha256>

The assertion is the gateway saying "I checked, and this caller is this member",
in a form the caller cannot forge or edit. FoodChat then requires that the
member_id in the request MATCHES the asserted one — which is the actual
isolation guarantee: reaching the port is no longer enough to act as someone
else.

**Enabling it is setting the secret.** There is deliberately no separate on/off
flag: a security control with its own feature flag is a security control that
ships disabled and stays that way. With `FOODCHAT_ASSERTION_SECRET` set on both
sides, assertions are required; without it, FoodChat logs a loud warning at
boot and behaves exactly as it did before, so the two services can be deployed
in either order without an outage. The rollout is: deploy the gateway (sending
the header is harmless to a FoodChat that ignores it), then set the secret here.

Replay is bounded by the expiry rather than by a nonce store: the assertion
carries no authority beyond "act as this member for the next few minutes", the
gateway has already authenticated the caller, and a nonce store would be shared
mutable state on the request path for a threat the expiry already covers.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

HEADER = "X-WiseFood-Member"

# How long an assertion stays valid. Long enough to survive a slow plan
# generation and any clock skew between pods; short enough that a captured
# header is worth little.
DEFAULT_TTL_SECONDS = 300

# Tolerated clock skew between the gateway and FoodChat. Without it, a pod a
# second ahead rejects every assertion the moment it is minted.
CLOCK_SKEW_SECONDS = 30


class AssertionError_(Exception):
    """The assertion was absent, malformed, expired or not ours."""


# The member the gateway vouched for, for the duration of one request.
#
# A ContextVar rather than a parameter threaded through 31 handlers: the
# identity is a property of the request, not of any one endpoint's signature,
# and a handler that forgets to accept the parameter is a handler that silently
# skips the check. The middleware sets it; `_require_session` reads it.
_asserted_member: ContextVar[Optional[str]] = ContextVar(
    "asserted_member", default=None
)


def secret() -> Optional[str]:
    """The shared signing key, or None when assertions are not configured."""
    value = (os.getenv("FOODCHAT_ASSERTION_SECRET") or "").strip()
    return value or None


def enforcing() -> bool:
    """Whether a valid assertion is required on member-scoped requests."""
    return secret() is not None


def sign(member_id: str, *, ttl: int = DEFAULT_TTL_SECONDS,
         key: Optional[str] = None, now: Optional[float] = None) -> str:
    """Mint an assertion. Lives here so both sides sign the same bytes.

    FoodChat never calls this in production — the gateway does. It is here so
    the format has exactly one definition, and so the tests that prove a forged
    assertion is rejected can build a real one to compare against.
    """
    signing_key = key or secret()
    if not signing_key:
        raise ValueError("No assertion secret configured")
    expires = int((now if now is not None else time.time()) + ttl)
    return f"{member_id}.{expires}.{_digest(member_id, expires, signing_key)}"


def _digest(member_id: str, expires: int, key: str) -> str:
    # The separator is inside the signed payload, so a member id containing a
    # dot cannot be shifted into the expiry field and re-signed.
    payload = f"{member_id}|{expires}".encode()
    return hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()


def verify(header_value: Optional[str], *, key: Optional[str] = None,
           now: Optional[float] = None) -> str:
    """The member id this assertion vouches for. Raises otherwise."""
    signing_key = key or secret()
    if not signing_key:
        raise AssertionError_("Assertions are not configured")
    if not header_value:
        raise AssertionError_("Missing member assertion")

    # rsplit: a member id may contain dots; the last two fields never do.
    parts = header_value.rsplit(".", 2)
    if len(parts) != 3:
        raise AssertionError_("Malformed member assertion")
    member_id, expires_raw, provided = parts
    if not member_id:
        raise AssertionError_("Malformed member assertion")
    try:
        expires = int(expires_raw)
    except ValueError:
        raise AssertionError_("Malformed member assertion") from None

    # Signature BEFORE expiry: an attacker must not learn whether a forged id
    # would have been in date, and comparing the digest first means the answer
    # is the same either way.
    expected = _digest(member_id, expires, signing_key)
    if not hmac.compare_digest(expected, provided):
        raise AssertionError_("Invalid member assertion")

    current = now if now is not None else time.time()
    if current > expires + CLOCK_SKEW_SECONDS:
        raise AssertionError_("Expired member assertion")
    return member_id


def set_asserted_member(member_id: Optional[str]) -> None:
    """Record who the gateway vouched for, for this request."""
    _asserted_member.set(member_id)


def asserted_member() -> Optional[str]:
    """Who the gateway vouched for, or None when not enforcing."""
    return _asserted_member.get()


def check_member(member_id: str) -> None:
    """Refuse a request that claims to be a member the gateway did not vouch for.

    Called wherever a member id is first known — `_require_session` and the
    `/members/{id}/…` handlers — because that is where the identity actually
    enters the decision, and a check placed anywhere else is a check some future
    route will be written around.

    A no-op when assertions are not configured, which is what keeps the
    deployment order free.
    """
    if not enforcing():
        return
    vouched = asserted_member()
    if vouched is None or vouched != member_id:
        # Deliberately the same message either way: whether the assertion was
        # missing or simply named someone else is not the caller's business.
        raise AssertionError_("Member assertion does not match this request")


async def assertion_middleware(request, call_next):
    """Verify the assertion once per request and stash who it names.

    Verification happens here — early, uniformly, and without any handler
    having to remember. Identity MATCHING happens later, where the member id is
    known. Splitting them that way means a new route inherits the verification
    for free and can only get the matching wrong by not checking ownership at
    all, which the route audit test catches.
    """
    from fastapi.responses import JSONResponse

    set_asserted_member(None)
    if enforcing() and not _is_open_path(request.url.path):
        try:
            set_asserted_member(verify(request.headers.get(HEADER)))
        except AssertionError_ as exc:
            logger.warning(
                "Rejected %s %s: %s", request.method, request.url.path, exc,
            )
            return JSONResponse(status_code=401, content={"detail": str(exc)})
    return await call_next(request)


# Paths that carry no member identity and must stay reachable: the health probe
# (kubelet has no assertion to send), the corpus vocabulary and tool manifest
# (identical for everyone), and the docs.
_OPEN_PATHS = frozenset({
    "/", "/docs", "/redoc", "/openapi.json",
    "/foodchat/health", "/foodchat/vocabularies", "/foodchat/tools",
})


def _is_open_path(path: str) -> bool:
    return path.rstrip("/") in _OPEN_PATHS or path in _OPEN_PATHS
