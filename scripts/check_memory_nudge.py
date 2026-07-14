#!/usr/bin/env python3
"""
Preflight: does the FoodScholar heart-health beat produce its memory nudge?

Asks the demo question ("We're worried about our family's heart health…")
through the gateway as a real member, answers a clarification round if one
comes back, and reports whether `memory_suggestions` rode on the answer —
plus the profile state that could legitimately suppress it (an already-
accepted goal, or a recorded opt-out).

Read the verdict it prints:
  NUDGE OK        — the chip will appear; the demo beat works.
  SUPPRESSED      — profile already carries the goal / an opt-out; forget it
                    in the memory panel (or use a fresh guest) and re-run.
  NO SUGGESTIONS  — the pipeline ran but extracted nothing. Almost always the
                    Langfuse prompt: open foodscholar/qa-memory-extractor and
                    confirm the production version contains "Concern vs
                    topic". If it does, check foodscholar logs for
                    "Memory suggestion failed" (profile fetch / LLM errors
                    are swallowed by design).
  NO FIELD        — the response has no memory_suggestions key at all: the
                    deployed foodscholar image predates the nudge feature.

Usage:
    WISEFOOD_API_URL=https://demo.wisefood-project.eu/rest \\
    DEMO_USERNAME=... DEMO_PASSWORD=... python3 scripts/check_memory_nudge.py
"""

import os
import sys

import httpx

BASE = os.environ.get("WISEFOOD_API_URL", "").rstrip("/")
USERNAME = os.environ.get("DEMO_USERNAME")
PASSWORD = os.environ.get("DEMO_PASSWORD")

QUESTION = "We're worried about our family's heart health — is eating red meat often harmful?"
CLARIFICATION_FREE_TEXT = "several times a week"


def die(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def result(response: httpx.Response) -> dict | list:
    response.raise_for_status()
    payload = response.json()
    return payload.get("result", payload) if isinstance(payload, dict) else payload


def main() -> None:
    if not BASE or not USERNAME or not PASSWORD:
        die("Set WISEFOOD_API_URL, DEMO_USERNAME, DEMO_PASSWORD")

    client = httpx.Client(base_url=BASE, timeout=180)
    token = result(client.post("/system/login", json={"username": USERNAME, "password": PASSWORD}))
    access = token.get("token") or token.get("access_token")
    if not access:
        die(f"Login response carried no token: {token}")
    client.headers["Authorization"] = f"Bearer {access}"

    household = result(client.get("/households/me"))
    members = result(client.get("/members", params={"household_id": household["id"]}))
    member = next((m for m in members if m.get("name") == "Dimitris"), members[0] if members else None)
    if not member:
        die("No household members found — run seed_demo.py first")
    member_id = member["id"]
    print(f"Member: {member.get('name')} ({member_id})")

    # Anything here legitimately suppresses the nudge (same-value dedupe).
    profile = result(client.get(f"/members/{member_id}/profile"))
    props = (profile.get("properties") or {}) if isinstance(profile, dict) else {}
    goals = props.get("dietary_goals") or []
    optouts = props.get("memory_optouts") or []
    print(f"Existing dietary_goals: {goals or '—'}")
    print(f"Memory opt-outs:        {optouts or '—'}")

    payload: dict = {"question": QUESTION, "member_id": member_id}
    answer = result(client.post("/foodscholar/qa/ask", json=payload))

    if answer.get("needs_clarification") and answer.get("clarification"):
        clarif = answer["clarification"]
        options = clarif.get("options") or []
        print(f"Clarification round: {clarif.get('question') or clarif.get('id')}")
        response = {
            "question_id": clarif.get("id"),
            "selected_values": [options[0]["value"]] if options else [],
            "free_text": None if options else CLARIFICATION_FREE_TEXT,
        }
        payload["qa_thread_id"] = answer.get("qa_thread_id")
        payload["clarification_response"] = response
        answer = result(client.post("/foodscholar/qa/ask", json=payload))

    if answer.get("needs_clarification"):
        die("Still clarifying after one round — answer manually and inspect the UI")

    print(f"Answer confidence: {answer.get('confidence') or answer.get('primary_answer', {}).get('confidence')}")

    if "memory_suggestions" not in answer:
        print("VERDICT: NO FIELD — deployed foodscholar image predates the nudge feature; rebuild + redeploy foodscholar.")
        sys.exit(1)

    suggestions = answer.get("memory_suggestions") or []
    if not suggestions:
        blocked = [g for g in ([g.get("slug") if isinstance(g, dict) else g for g in goals]) if g]
        if "reduce_fat" in blocked or "reduce_fat" in [str(v).lower() for v in optouts]:
            print("VERDICT: SUPPRESSED — reduce_fat already on the profile (goal or opt-out). Forget it in the memory panel and re-run.")
        else:
            print("VERDICT: NO SUGGESTIONS — extractor found nothing. Check the Langfuse prompt "
                  "foodscholar/qa-memory-extractor (production version must contain 'Concern vs topic'), "
                  "then foodscholar logs for 'Memory suggestion failed'.")
        sys.exit(1)

    print("VERDICT: NUDGE OK")
    for s in suggestions:
        print(f"  → [{s.get('kind')}] {s.get('value')}: {s.get('statement')}")


if __name__ == "__main__":
    main()
