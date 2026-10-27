#!/usr/bin/env python3
"""Export conversations from people who agreed, for testing WiseFood's answers.

Runs inside the FoodChat pod, which has the database and the platform's service
credentials:

    kubectl exec -n wf-prod deploy/foodchat -- \\
        python /app/scripts/export_research_conversations.py --out /tmp/conversations.jsonl
    kubectl cp wf-prod/<foodchat-pod>:/tmp/conversations.jsonl ./conversations.jsonl

Who may be included is asked of the gateway every time
(`GET /api/v1/research/eligible-members`): members of households whose account
holder has research consent and has not withdrawn it, guests never. Nobody else
is read. Each line of the output is one conversation, shaped and scrubbed by
`research_export.conversation_record`; a summary goes to stderr.

The ids in the file are codes keyed by a salt made for this run and never
written anywhere, so two exports cannot be joined and neither leads back to a
person. Pass `--salt-file` only when one test set really must be extended
later with the same codes, and keep that file away from the export.

Read the file before sharing it. The scrubber removes names it knows and
identifiers with a shape; a person can still describe themselves in words.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def eligible_from_gateway() -> Set[str]:
    """Member ids whose conversations may be used, from the consent ledger."""
    from wisefood import Client

    from backend.platform import WISEFOOD_API_URL, credentials_from_env

    if not WISEFOOD_API_URL:
        raise SystemExit("WISEFOOD_API_URL is not set")
    client = Client(WISEFOOD_API_URL, credentials_from_env(), telemetry=False)
    response = client.get("research/eligible-members")
    payload = response.json()
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    return {row["member_id"] for row in result.get("members", []) if row.get("member_id")}


def eligible_from_file(path: str) -> Set[str]:
    """Member ids from a file, one per line — for a run without network access."""
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip() and not line.startswith("#")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="JSONL file to write")
    parser.add_argument("--since", help="Only conversations started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--min-user-turns", type=int, default=1,
                        help="Skip conversations where the member said less than this (default 1)")
    parser.add_argument("--no-replies", action="store_true",
                        help="Only the member's own words, not FoodChat's answers")
    parser.add_argument("--members-file",
                        help="Eligible member ids, one per line, instead of asking the gateway")
    parser.add_argument("--salt-file",
                        help="Reuse (or create) a salt so codes match an earlier export")
    args = parser.parse_args()

    from db import FeedbackRow, MealPlanRow, MessageRow, SessionLocal, SessionRow
    from research_export import conversation_record, worth_keeping

    members = eligible_from_file(args.members_file) if args.members_file else eligible_from_gateway()
    if not members:
        print("Nobody has agreed to research use yet; nothing exported.", file=sys.stderr)
        return 0

    if args.salt_file:
        if os.path.exists(args.salt_file):
            with open(args.salt_file, "rb") as handle:
                salt = handle.read()
        else:
            salt = secrets.token_bytes(32)
            with open(args.salt_file, "wb") as handle:
                handle.write(salt)
    else:
        salt = secrets.token_bytes(32)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    written = skipped = 0
    redactions: Dict[str, int] = {}
    db = SessionLocal()
    try:
        query = db.query(SessionRow).filter(SessionRow.member_id.in_(list(members)))
        if since is not None:
            query = query.filter(SessionRow.created_at >= since)
        with open(args.out, "w", encoding="utf-8") as out:
            # Listed first and read one by one: streaming the sessions while
            # querying their messages on the same connection is not something
            # every driver allows.
            for session in query.order_by(SessionRow.created_at.asc()).all():
                messages: List[Dict] = [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "intent": m.intent,
                        "plan_id": m.plan_id,
                        "timestamp": m.timestamp,
                    }
                    for m in db.query(MessageRow)
                    .filter(MessageRow.session_id == session.session_id)
                    .order_by(MessageRow.timestamp.asc(), MessageRow.id.asc())
                ]
                plans = {
                    p.id: {"plan_type": p.plan_type, "version": p.version, "saved": p.saved}
                    for p in db.query(MealPlanRow).filter(MealPlanRow.session_id == session.session_id)
                }
                ratings = {
                    f.message_id: f.rating
                    for f in db.query(FeedbackRow).filter(FeedbackRow.session_id == session.session_id)
                }
                record = conversation_record(
                    conversation_id=session.session_id,
                    member_id=session.member_id,
                    profile=session.user_profile,
                    created_at=session.created_at,
                    messages=messages,
                    plans=plans,
                    ratings=ratings,
                    salt=salt,
                    with_replies=not args.no_replies,
                )
                if not worth_keeping(record, min_user_turns=args.min_user_turns):
                    skipped += 1
                    continue
                for kind, count in record["redactions"].items():
                    redactions[kind] = redactions.get(kind, 0) + count
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    finally:
        db.close()

    import pii

    print(
        json.dumps(
            {
                "members_eligible": len(members),
                "conversations_written": written,
                "conversations_skipped": skipped,
                "redactions": redactions,
                # Without a local NER model only names the profile knows are
                # removed; say so rather than let a zero look like a clean run.
                "names_beyond_profile_detected": pii.names_detected(),
                "out": args.out,
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
