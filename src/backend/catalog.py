"""
The WiseFood Data API, for dietary guidelines.

FoodChat judges plans for guideline adherence, and the guidelines come from
here: ~2,800 rules across 31 national guides (Ireland, Hungary, Slovenia),
faceted by region, life stage, target population, frequency and food group.
Which rules a plan is judged against is a `models.guidelines.GuidelineScope`;
this module only runs the query.

    POST {DATA_API_URL}/api/v1/guidelines/search
         {"limit", "offset", "fq": scope.fq(), "sort", "fields"}

Not proxied by the gateway: the data API is its own service with its own
``/system/login`` and ``/system/mtm``, so FoodChat talks to it directly with the
SDK and the SAME credentials it uses for member profiles
(`backend.platform.credentials_from_env` — client credentials if set, else
username/password).

**In the cluster this is the INTERNAL address**, which is what
`platform-deployment` sets: ``http://data-catalog:8000``. The public one
(``https://demo.wisefood-project.eu/dc``) is the same service through the
ingress, and reaching it from inside would leave the cluster, cross TLS and
come back for a lookup between two pods on the same network — slower, and
dependent on the ingress being up for a plan that is not being served through
it. Point `DATA_API_URL` at the public host only from outside the cluster.

What the API does that a caller would not guess (probed 2026-09-17):

- ``fl`` (a field list) answers 500, so it is never sent.
- Facets cannot be switched off — omitting ``fields`` or sending ``[]`` returns
  every facet, ~14 KB — so one cheap facet is always asked for.
- The SDK returns a ``requests.Response``; the rows are inside
  ``{success, result: {results, total, facets}}``.

Everything here is best-effort by construction. A catalog that is down, slow,
or not configured costs the plan its guideline text and nothing else:
`DATA_API_URL` unset means `search` answers ``[]`` without a request.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

from pydantic import ValidationError

from models.guidelines import Guideline, GuidelineScope

logger = logging.getLogger(__name__)

DATA_API_URL = os.getenv("DATA_API_URL")
DATA_API_TIMEOUT = float(os.getenv("DATA_API_TIMEOUT", "8"))

# Guidelines change when a national body publishes, which is a multi-year
# cadence. An hour is short enough that a correction is not stuck for a day and
# long enough that a busy pod asks once.
CACHE_TTL_SECONDS = int(os.getenv("DATA_API_CACHE_TTL", "3600"))

# After a failed request, how long every scope answers `[]` without asking.
# Metrics run on every plan turn, and a catalog that is down would otherwise
# cost each one a full timeout — twice for a week (checklist and judge).
FAILURE_BACKOFF_SECONDS = 30

# One page is the catalog's maximum; a scope's limit is below it in practice
# (Ireland, adults, active: 81 rules).
PAGE_SIZE = 1000


class CatalogClient:
    """Read-only guideline lookups against the data API.

    One lazily-created client behind a lock, mirroring `backend/platform.py`:
    the SDK client holds a token and a connection pool, and building one per
    request would re-authenticate on every plan.
    """

    _client = None
    _lock = threading.Lock()
    _cache: dict[str, tuple[float, Any]] = {}
    _down_until: float = 0.0

    @classmethod
    def available(cls) -> bool:
        """Whether the catalog is configured at all.

        Checked before every call so an unconfigured deployment costs nothing —
        not a failed request, not a log line per plan.
        """
        return bool(DATA_API_URL)

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client
        with cls._lock:
            if cls._client is not None:
                return cls._client
            from wisefood import Client

            from backend.platform import credentials_from_env

            # The platform talking to itself, not a user — as in
            # `backend.platform`. With username/password credentials the SDK
            # would otherwise report every guideline lookup as member usage.
            cls._client = Client(
                DATA_API_URL, credentials_from_env(),
                default_timeout=DATA_API_TIMEOUT,
                # The platform talking to itself, not a user — the same call
                # `backend.platform` makes for member profiles. A guideline
                # lookup on behalf of a plan is not platform usage worth
                # reporting, and reporting it puts a second request behind
                # every one of these.
                telemetry=False,
            )
            logger.info("Data catalog client ready (%s)", DATA_API_URL)
            return cls._client

    @classmethod
    def search(cls, scope: GuidelineScope) -> list[Guideline]:
        """The active rules in `scope`, deduplicated. `[]` on any failure."""
        if not cls.available():
            return []

        key = scope.cache_key()
        cached = cls._cached(key)
        if cached is not None:
            return cached
        if time.time() < cls._down_until:
            return []

        fq = scope.fq()
        rows: list[dict] = []
        try:
            client = cls._get_client()
            while len(rows) < scope.limit:
                payload = _json_of(client.post(
                    "guidelines/search",
                    json={
                        "limit": min(PAGE_SIZE, scope.limit - len(rows)),
                        "offset": len(rows),
                        "fq": fq,
                        "sort": "sequence_no asc",
                        "fields": ["guide_region"],
                    },
                ))
                page = _results_of(payload)
                rows.extend(page)
                if not page or len(rows) >= _total_of(payload, default=len(rows)):
                    break
        except Exception as exc:  # noqa: BLE001
            cls._down_until = time.time() + FAILURE_BACKOFF_SECONDS
            logger.warning(
                "Guideline search failed (%s): %s — not retrying for %ds",
                fq, exc, FAILURE_BACKOFF_SECONDS,
            )
            return []

        rules = _deduplicated(_parsed(rows))
        cls._store(key, rules)
        logger.info("Catalog returned %d guideline(s) for %s", len(rules), fq)
        return rules

    # -- cache ---------------------------------------------------------- #

    @classmethod
    def _cached(cls, key: str):
        entry = cls._cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > CACHE_TTL_SECONDS:
            cls._cache.pop(key, None)
            return None
        return value

    @classmethod
    def _store(cls, key: str, value) -> None:
        # An empty result is cached too: a scope the catalog has no rules for
        # is a stable fact, and re-asking on every plan would be a request per
        # turn for an answer that will not change. A FAILED request is not
        # cached — it only starts the short backoff.
        cls._cache[key] = (time.time(), value)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._down_until = 0.0


def _json_of(response) -> Any:
    """The body, whether the SDK handed back a Response or already-parsed JSON."""
    if hasattr(response, "json") and callable(response.json):
        return response.json()
    return response


def _results_of(payload) -> list[dict]:
    """The rows, whatever envelope the API wrapped them in.

    An unrecognised envelope returns nothing rather than a confident misreading.
    """
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if payload.get("success") is False:
        raise RuntimeError(f"catalog error: {payload.get('error')}")
    for key in ("results", "items", "guidelines", "docs"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    inner = payload.get("result")
    if isinstance(inner, (dict, list)):
        return _results_of(inner)
    return []


def _total_of(payload, default: int) -> int:
    if isinstance(payload, dict):
        inner = payload.get("result", payload)
        if isinstance(inner, dict) and isinstance(inner.get("total"), int):
            return inner["total"]
    return default


def _parsed(rows: list[dict]) -> list[Guideline]:
    rules = []
    for row in rows:
        try:
            rule = Guideline.model_validate(row)
        except ValidationError as exc:
            logger.debug("Skipping an unreadable guideline row: %s", exc)
            continue
        if rule.rule_text.strip():
            rules.append(rule)
    return rules


def _deduplicated(rules: list[Guideline]) -> list[Guideline]:
    """One rule per wording.

    Guides republish each other's messages ("Eat 3 servings a day of milk,
    yogurt, and cheese." / "Have 3 servings a day of milk, yogurt and cheese."
    differ; two identical sentences do not), and a judge handed the same
    sentence twice weighs it twice.
    """
    seen: set[str] = set()
    out: list[Guideline] = []
    for rule in rules:
        key = re.sub(r"[^a-z0-9]+", " ", rule.rule_text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(rule)
    return out


CATALOG = CatalogClient
