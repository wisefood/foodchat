#!/usr/bin/env python3
"""
Seed the RecSys '26 demo household through the WiseFood gateway.

Creates (idempotently, by name) a household with three contrasting members —
the multi-diner, memory, and favorites features need real variety to shine:

    Dimitris (adult)  — omnivore, high-protein target, likes salmon/chickpeas,
                        dislikes olives; the primary demo driver
    Anna     (adult)  — vegetarian (hard-constraint demo: one diner turns the
                        whole plan vegetarian)
    Tom      (child)  — peanut allergy (safety demo: no peanuts anywhere,
                        attributed to Tom in the constraint ledger)

Also favorites the Greek anchor dishes (pastitsio, fakes, moussaka) for
Dimitris — resolved live against RecipeWrangler through the gateway so the
script doubles as the "are the demo dishes resolvable?" preflight check.

Usage:
    WISEFOOD_API_URL=https://demo.wisefood-project.eu/rest \\
    DEMO_USERNAME=... DEMO_PASSWORD=... python3 scripts/seed_demo.py

Prints the member ids and a resolution report; exits non-zero when a demo
prerequisite is missing (unresolvable dish, failed profile write).
"""

import os
import sys

import httpx

BASE = os.environ.get("WISEFOOD_API_URL", "").rstrip("/")
USERNAME = os.environ.get("DEMO_USERNAME")
PASSWORD = os.environ.get("DEMO_PASSWORD")

DEMO_DISHES = ["pastitsio", "fakes", "moussaka"]

MEMBERS = [
    {
        "name": "Dimitris", "age_group": "adult",
        "profile": {
            "dietary_groups": ["omnivore", "high_protein"],
            "allergies": [],
            "nutritional_preferences": {
                "calories": 2200, "protein": 120,
                "food_likes": ["salmon", "chickpeas"],
                "food_dislikes": ["olives"],
            },
        },
    },
    {
        "name": "Anna", "age_group": "adult",
        "profile": {
            "dietary_groups": ["vegetarian"],
            "allergies": [],
            "nutritional_preferences": {"food_likes": ["halloumi"], "food_dislikes": ["mushrooms"]},
        },
    },
    {
        "name": "Tom", "age_group": "child",
        "profile": {
            "dietary_groups": [],
            "allergies": ["peanuts"],
            "nutritional_preferences": {"food_likes": ["pasta"], "food_dislikes": []},
        },
    },
]


def die(message: str) -> None:
    print(f"✗ {message}", file=sys.stderr)
    sys.exit(1)


def result(response: httpx.Response) -> dict | list:
    response.raise_for_status()
    body = response.json()
    return body.get("result", body)


def main() -> None:
    if not (BASE and USERNAME and PASSWORD):
        die("Set WISEFOOD_API_URL, DEMO_USERNAME, DEMO_PASSWORD")

    client = httpx.Client(base_url=f"{BASE}/api/v1", timeout=60)

    # 1. Login
    token = result(client.post("/system/login", json={"username": USERNAME, "password": PASSWORD}))
    access = token.get("token") or token.get("access_token")
    if not access:
        die(f"Login response carried no token: {token}")
    client.headers["Authorization"] = f"Bearer {access}"
    print("✓ Logged in")

    # 2. Household (reuse when it exists)
    try:
        household = result(client.get("/households/me"))
    except httpx.HTTPStatusError:
        household = None
    if not household:
        household = result(client.post("/households", json={
            "name": "RecSys Demo Household", "region": "GR",
        }))
        print(f"✓ Household created: {household['id']}")
    else:
        print(f"✓ Household exists: {household['id']}")

    # 3. Members + profiles (idempotent by name)
    existing = {m["name"]: m for m in result(client.get(
        "/members", params={"household_id": household["id"]},
    ))}
    member_ids: dict[str, str] = {}
    for spec in MEMBERS:
        member = existing.get(spec["name"])
        if not member:
            member = result(client.post("/members", json={
                "name": spec["name"], "age_group": spec["age_group"],
                "household_id": household["id"],
            }))
            print(f"✓ Member created: {spec['name']}")
        member_ids[spec["name"]] = member["id"]

        # Profile: POST first (create), PATCH on conflict (update)
        try:
            result(client.post(f"/members/{member['id']}/profile", json=spec["profile"]))
        except httpx.HTTPStatusError:
            result(client.patch(f"/members/{member['id']}/profile", json=spec["profile"]))
        print(f"✓ Profile set: {spec['name']}")

    # 4. Resolve + favorite the Greek anchor dishes for Dimitris
    dimitris = member_ids["Dimitris"]
    unresolved = []
    for dish in DEMO_DISHES:
        suggestions = result(client.get(
            "/recipewrangler/recipes/autocomplete", params={"q": dish, "limit": 3},
        )).get("suggestions", {})
        if not suggestions:
            unresolved.append(dish)
            print(f"✗ '{dish}' NOT resolvable in RecipeWrangler")
            continue
        recipe_id, title = next(iter(suggestions.items()))
        result(client.put(f"/members/{dimitris}/favorites/{recipe_id}"))
        print(f"✓ Favorited '{title}' ({recipe_id}) for Dimitris")

    print("\nMember ids:")
    for name, mid in member_ids.items():
        print(f"  {name}: {mid}")

    if unresolved:
        die(f"Demo prerequisite missing — unresolvable dishes: {unresolved} "
            "(import them into RecipeWrangler before the demo)")
    print("\n✓ Demo household ready.")


if __name__ == "__main__":
    main()
