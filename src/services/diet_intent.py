"""
Diet stated in chat → standing planning state.

"I need something vegetarian" was the request that exposed why this module has
to exist. The conflict WAS detected, the member WAS asked, they said yes — and
the plan was still generated from a pool filtered on `profile["diet"]` alone,
because the only consumer of `DietaryIntentExtractor` was the weekly service.
The stated diet reached the grader as prose over candidates that had never been
filtered for it, and the failure message then blamed `diet: omnivore` — a value
dropped before the fetch for not being a restriction at all.

Contract, mirroring `pantry_service`:

    extract_diet_delta(message)  → PlanningStateDelta (never raises)

Two kinds come back from the extractor and they must not be conflated:

* **Filterable diets** (vegetarian, vegan, gluten_free, …) → `diet_tags`, which
  every fetch site unions with the profile via `candidates_client.effective_diet`.
* **Nutrition claims** (low-carb, low-fat, high-protein) → `notes`, i.e. grader
  prose. They are NOT diet tags: no recipe in the corpus carries them, so
  sending one as a filter empties every slot. They become real numeric targets
  once the planning endpoint grows nutrition targets.

A retraction is explicit. Silence never clears a stated diet — same rule as
every other field in `PlanningState`.
"""

import logging
import re
from typing import Optional

from models.planning_state import PlanningStateDelta
from services.candidates_client import split_diet_intent

logger = logging.getLogger(__name__)


def extract_diet_delta(message: str, *, extractor=None) -> PlanningStateDelta:
    """What this turn says about diet. Never raises.

    Unlike the pantry extractor there is no regex gate: a diet can be stated in
    too many ways to pattern-match ("no meat for me", "keep it plant based",
    "I've gone veggie"), and a missed diet is the failure this module exists to
    prevent. The call runs on the fast tier, and the weekly path has always
    paid it unconditionally.
    """
    text = (message or "").strip()
    if not text:
        return PlanningStateDelta()

    try:
        if extractor is None:
            from agents import DietaryIntentExtractor

            extractor = DietaryIntentExtractor()
        raw = extractor.extract(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diet extraction failed: %s", exc)
        return PlanningStateDelta()

    filterable, claims = split_diet_intent(raw)
    if filterable or claims:
        logger.info("Diet intent: filters=%s claims=%s", filterable, claims)
    return PlanningStateDelta(
        diet_tags=tuple(filterable),
        # A claim carries no filter, so say it out loud to the grader rather
        # than dropping it — the member asked for something.
        notes=tuple(f"prefers {c} meals" for c in claims),
    )


def describe_applied(diet_tags, profile_diet) -> str:
    """One honest sentence about which diet was actually applied.

    Used by the empty-pool apology, which previously listed the RAW profile —
    including `omnivore`, a value that never reached the request. Naming a
    constraint that was not applied sends the member to relax the wrong thing.
    """
    stated = [str(t) for t in (diet_tags or ())]
    stored = profile_diet or []
    stored = [stored] if isinstance(stored, str) else [str(d) for d in stored]

    from services.candidates_client import normalize_diet_tags

    applied = {a.lower() for a in normalize_diet_tags(stored)}
    kept = [d for d in stored if d.lower() in applied]
    dropped = [d for d in stored if d.lower() not in applied]

    parts = []
    if stated:
        parts.append("you asked for " + ", ".join(stated))
    if kept:
        parts.append("your profile's " + ", ".join(kept))
    # The "not a restriction" note earns its place only as a CONTRAST — it
    # explains why a stored value isn't in the list beside it. With nothing
    # else to contrast against there is no diet clause to qualify, and saying
    # it alone reads like an accusation against a setting that did nothing.
    if dropped and parts:
        parts.append(
            "your '" + ", ".join(dropped) + "' setting isn't a restriction, "
            "so it wasn't applied as one"
        )
    return "; ".join(parts)


# Answers to "your request conflicts with your profile — adjust the plan?" that
# mean "no, follow my profile". Deliberately narrow: an unrecognised answer
# KEEPS the stated diet, because the member did say it out loud, and discarding
# a stated diet on a guess is the failure this module exists to fix.
_REFUSAL_RE = re.compile(
    r"^\s*(no|nope|nah|don'?t|do not)\b"
    r"|\b(follow|use|keep|stick (?:to|with))\s+(my|the)\s+profile\b"
    r"|\bnever\s*mind\b|\bforget\s+it\b|\bas\s+(?:is|before)\b",
    re.IGNORECASE,
)


def is_conflict_refusal(answer: str) -> bool:
    """Whether a dietary-conflict answer retracts the stated diet.

    The conflict question asks whether to adjust the plan to what the member
    just asked for. "Yes please" needs no action — the tags are already in
    force. Only a refusal has to undo them, and only when it is unambiguous.
    """
    text = (answer or "").strip()
    if not text:
        return False
    return bool(_REFUSAL_RE.search(text))


def suggest_diet_memory(diet_tags, message: str, profile: dict) -> Optional[dict]:
    """A consent nudge for a diet the member stated but hasn't stored.

    Deterministic on purpose — no LLM call and, critically, no change to the
    `preference_extractor` managed prompt, which a deploy would never overwrite
    (an edit there ships dead to production). The evidence is the member's own
    sentence, so the memory panel can answer "why am I seeing this?".

    Returns None when there is nothing to offer: no stated diet, or the profile
    already says so.
    """
    tags = [str(t) for t in (diet_tags or ()) if t]
    if not tags:
        return None

    stored = profile.get("diet") or []
    stored = [stored] if isinstance(stored, str) else stored
    known = {str(d).strip().lower() for d in stored}
    fresh = [t for t in tags if t.lower() not in known]
    if not fresh:
        return None

    value = fresh[0]
    label = value.replace("_", " ")
    return {
        "kind": "diet",
        "value": value,
        "statement": (
            f"You asked for {label} meals — should I remember that you eat "
            f"{label}, so every plan starts there?"
        ),
        "evidence": (message or "").strip()[:280],
        "confidence": "high",
    }
