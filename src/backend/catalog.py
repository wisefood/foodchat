"""
The WiseFood data catalog, for dietary guidelines.

FoodChat grades plans against dietary guidelines and has never read one. The
`guideline_checklist` it reports is three rules hardcoded in
`weekly_planner/explainability.py` — eat fish 1–2 times a week, limit red meat,
make most meals plant-based — which are real guidance, and are the same three
for a member in Ireland, Slovenia, Hungary or Greece. The catalog holds ~2,700
rules across 31 guides, faceted by region, life stage, audience, food group and
nutrient, and nothing here has ever asked it for any of them.

Not proxied by the gateway: the catalog's guideline routes have no
`/api/v1/...` passthrough, so FoodChat talks to the data API directly with its
own Keycloak client — the same `WISEFOOD_CLIENT_ID` / `WISEFOOD_CLIENT_SECRET`
pair it already uses for member profiles, pointed at a different base URL.

Everything here is best-effort by construction. A catalog that is down, slow,
or not configured must cost the plan its guideline detail and nothing else:
`DATA_API_URL` unset simply means the hardcoded rules keep being used, which is
today's behaviour and a working product.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DATA_API_URL = os.getenv("DATA_API_URL") or os.getenv("WISEFOOD_DATA_API_URL")
DATA_API_TIMEOUT = float(os.getenv("DATA_API_TIMEOUT", "8"))

# Guidelines change when a national body publishes, which is a multi-year
# cadence. An hour is short enough that a correction is not stuck for a day and
# long enough that a busy pod asks once.
CACHE_TTL_SECONDS = int(os.getenv("DATA_API_CACHE_TTL", "3600"))


class CatalogClient:
    """Read-only guideline lookups against the data API.

    One lazily-created client behind a lock, mirroring `backend/platform.py`:
    the SDK client holds a Keycloak token and a connection pool, and building
    one per request would re-authenticate on every plan.
    """

    _client = None
    _lock = threading.Lock()
    _cache: dict[str, tuple[float, Any]] = {}

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
            from wisefood import Client, Credentials

            client_id = os.getenv("WISEFOOD_CLIENT_ID")
            client_secret = os.getenv("WISEFOOD_CLIENT_SECRET")
            if not (client_id and client_secret):
                raise RuntimeError(
                    "WISEFOOD_CLIENT_ID / WISEFOOD_CLIENT_SECRET are required "
                    "to read the data catalog"
                )
            cls._client = Client(
                DATA_API_URL,
                Credentials(client_id=client_id, client_secret=client_secret),
                default_timeout=DATA_API_TIMEOUT,
            )
            logger.info("Data catalog client ready (%s)", DATA_API_URL)
            return cls._client

    @classmethod
    def search_guidelines(
        cls, filters: dict[str, list[str]], limit: int = 40,
    ) -> list[dict]:
        """Guidelines matching every facet given. `[]` on any failure.

        `filters` is `{facet: [values]}` — `region`, `life_stage`, `audience`,
        `food_groups`, `nutrients`, `guideline_type`. Values within a facet are
        OR'd, facets are AND'd, which is how the catalog's `fq` behaves and how
        a member's context actually composes: an Irish adult wants Irish rules
        for adults, not Irish rules or adult rules.
        """
        if not cls.available():
            return []

        fq = [
            f"{field}:({' OR '.join(str(v) for v in values if v)})"
            for field, values in (filters or {}).items()
            if values
        ]
        # Only what a member should see. `active`/`verified` is the catalog's
        # own published state; a draft rule is someone's work in progress.
        fq.append("status:(active OR verified)")

        key = f"guidelines::{limit}::" + "|".join(sorted(fq))
        cached = cls._cached(key)
        if cached is not None:
            return cached

        try:
            client = cls._get_client()
            payload = client.post(
                "/guidelines/search",
                json={"limit": limit, "fq": fq, "sort": "created_at desc"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Guideline search failed (%s): %s", fq, exc)
            return []

        results = _results_of(payload)
        cls._store(key, results)
        logger.info("Catalog returned %d guideline(s) for %s", len(results), fq)
        return results

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
        # An empty result is cached too: a region the catalog has no rules for
        # is a stable fact, and re-asking on every plan would be a request per
        # turn for an answer that will not change.
        cls._cache[key] = (time.time(), value)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()


def _results_of(payload) -> list[dict]:
    """The rows, whatever envelope the API wrapped them in.

    The data API answers `{"result": {...}}` through its renderer and the SDK
    may unwrap one level, so the shape at this boundary is not something to
    assume — an unrecognised envelope returns nothing rather than a confident
    misreading.
    """
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "items", "guidelines", "docs"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    inner = payload.get("result")
    if isinstance(inner, (dict, list)):
        return _results_of(inner)
    return []


CATALOG = CatalogClient
