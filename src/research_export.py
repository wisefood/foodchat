"""Conversations for testing WiseFood, from the people who agreed to it.

The platform's single consent covers using a person's conversations, with
identifying details removed, to test and compare WiseFood's answers — never to
train models. Who agreed is decided by the gateway, where the consent ledger
is; what leaves FoodChat, and in what form, is decided here.

What a record keeps: what was asked and answered, in order, with each turn's
intent, the kind of plan it produced and the rating it got; the member's diet,
allergies, goals and food likes as categories; the month it happened.

What it does not: names — the member's own and those of the people they cook
for are replaced exactly, then the shared redactor removes emails, phone, card
and bank numbers, addresses and postcodes (and further names, where a local
NER model is installed); account, profile and conversation ids, which become
codes made for one export and meaningless outside it; timestamps, which become
the month and seconds since the conversation began; plan contents and
provenance blobs.

Pure functions only: the script that runs an export owns the database and the
network, so everything that shapes a record can be tested without either.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pii

#: Profile fields kept, as categories. Everything else in the stored profile —
#: names, free-text notes, memory provenance, internal state — stays behind.
PROFILE_FIELDS = (
    "diet",
    "allergies",
    "dietary_goals",
    "age_group",
    "life_stage",
    "region",
    "food_likes",
    "food_dislikes",
)

#: Items kept per list field. A profile's likes can run long, and a long,
#: specific list is closer to a fingerprint than to a category.
MAX_LIST_ITEMS = 15

#: What a message becomes when it could not be checked for identifiers.
UNCHECKED = "[removed: could not be checked]"

#: A name shorter than this is not replaced on its own: "Al" would turn every
#: "almond" into "[NAME]mond" were it not for word boundaries, and still turns
#: up as a word in other languages.
MIN_NAME_LENGTH = 3


def pseudonym(kind: str, value: str, salt: bytes) -> str:
    """A code for an id, stable within one export and unlinkable across them.

    Keyed with the export's own salt, so two exports made with different salts
    cannot be joined on it, and the code cannot be turned back into the id
    without the salt — which is never written to the export.
    """
    digest = hmac.new(salt, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{kind[0]}_{digest[:12]}"


def _as_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except ValueError:
            return {}
    return {}


def known_names(profile: Mapping[str, Any]) -> List[str]:
    """The names a profile knows: the member's own and the people they cook for.

    Full names and each of their parts, longest first, so "Maria Papadopoulou"
    is replaced whole before "Maria" is tried on what is left.
    """
    found: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= MIN_NAME_LENGTH:
                found.append(text)
                found.extend(part for part in re.split(r"\s+", text) if len(part) >= MIN_NAME_LENGTH)
        elif isinstance(value, Mapping):
            add(value.get("name"))
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    add(profile.get("name"))
    add(profile.get("cooking_for_names"))
    add(profile.get("cooking_for"))
    return sorted(set(found), key=len, reverse=True)


def scrub(text: Optional[str], names: Sequence[str]) -> Tuple[str, Dict[str, int]]:
    """The text with names and identifiers replaced, and a count per kind."""
    counts: Dict[str, int] = {}
    # Shaped identifiers first. Names first turned "maria@example.com" into
    # "[NAME]@example.com", which no longer looks like an email address, and
    # the domain left with the text.
    redacted, found = pii.redact(text or "", force=True)
    for kind, count in found.items():
        counts[kind] = counts.get(kind, 0) + count
    # Text the redactor could not scan — too long, or it failed — comes back
    # unchanged. On the request path that is the right call; here it would be
    # the one way identifiers leave unchecked, so the text goes instead.
    if "unscanned" in found or "failed" in found:
        return UNCHECKED, counts
    out = redacted or ""
    for name in names:
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        out, replaced = pattern.subn("[NAME]", out)
        if replaced:
            counts["NAME"] = counts.get("NAME", 0) + replaced
    return out, counts


def _category(value: Any) -> Any:
    """A profile value reduced to plain, lower-case categories."""
    if isinstance(value, str):
        return value.strip().lower() or None
    if isinstance(value, Mapping):
        # A labelled chip ({"id": "gluten", "label": "Gluten"}) is its id.
        return _category(value.get("id") or value.get("name") or value.get("label"))
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            category = _category(item)
            if isinstance(category, str) and category not in items:
                items.append(category)
        return sorted(items)[:MAX_LIST_ITEMS]
    return None


def profile_summary(raw_profile: Any, names: Sequence[str]) -> Dict[str, Any]:
    """Diet, allergies, goals and likes as categories; nothing else."""
    profile = _as_dict(raw_profile)
    summary: Dict[str, Any] = {}
    for field in PROFILE_FIELDS:
        value = _category(profile.get(field))
        if value in (None, [], ""):
            continue
        # Free text can hide in a "like" as easily as in a message.
        if isinstance(value, list):
            value = [scrub(item, names)[0] for item in value]
        elif isinstance(value, str):
            value = scrub(value, names)[0]
        summary[field] = value
    return summary


def _month(at: Optional[datetime]) -> Optional[str]:
    return at.strftime("%Y-%m") if at else None


def conversation_record(
    *,
    conversation_id: str,
    member_id: str,
    profile: Any,
    created_at: Optional[datetime],
    messages: Iterable[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    ratings: Mapping[Any, str],
    salt: bytes,
    with_replies: bool = True,
) -> Dict[str, Any]:
    """One conversation, ready to leave the platform.

    ``messages`` carry ``id``, ``role``, ``content``, ``intent``, ``plan_id``
    and ``timestamp``; ``plans`` maps a plan id to its ``plan_type``,
    ``version`` and ``saved``; ``ratings`` maps a message id to "up"/"down".
    """
    profile_dict = _as_dict(profile)
    names = known_names(profile_dict)
    rows = sorted(
        messages,
        key=lambda m: (
            m.get("timestamp") is None,
            m.get("timestamp").timestamp() if m.get("timestamp") else 0.0,
            m.get("id") or 0,
        ),
    )
    start = next((m.get("timestamp") for m in rows if m.get("timestamp")), None) or created_at

    turns: List[Dict[str, Any]] = []
    redactions: Dict[str, int] = {}
    for row in rows:
        role = row.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not with_replies:
            continue
        text, found = scrub(row.get("content"), names)
        for kind, count in found.items():
            redactions[kind] = redactions.get(kind, 0) + count
        at = row.get("timestamp")
        turn: Dict[str, Any] = {
            "turn": len(turns),
            "role": role,
            "text": text,
            "seconds": int((at - start).total_seconds()) if at and start else None,
        }
        if role == "assistant":
            if row.get("intent"):
                turn["intent"] = row["intent"]
            plan = plans.get(row.get("plan_id") or "")
            if plan:
                turn["plan"] = {
                    "type": plan.get("plan_type"),
                    "version": plan.get("version"),
                    "saved": bool(plan.get("saved")),
                }
            rating = ratings.get(row.get("id"))
            if rating:
                turn["rating"] = rating
        turns.append(turn)

    return {
        "conversation": pseudonym("conversation", conversation_id, salt),
        "member": pseudonym("member", member_id, salt),
        "month": _month(start),
        "profile": profile_summary(profile_dict, names),
        "turns": turns,
        "redactions": redactions,
    }


def worth_keeping(record: Mapping[str, Any], *, min_user_turns: int = 1) -> bool:
    """Whether a record has anything to test with: enough of the member's words."""
    said = sum(1 for turn in record.get("turns", []) if turn.get("role") == "user" and turn.get("text"))
    return said >= min_user_turns
