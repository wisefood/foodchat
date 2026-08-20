"""
Consented memory (M3) — "It seems you don't like blueberries. Remember this?"

Principle: **session-scoped adaptation is automatic; durable memory requires
an explicit yes.** The PreferenceExtractor detects candidates in each user
turn; this service applies the nudge policy and, on the user's decision
(POST /sessions/{id}/memory), writes to the durable member profile with
provenance — or records an opt-out so the same thing is never suggested again.

Nudge policy (deliberately conservative):
  - only high-confidence, explicitly-stated candidates are suggested —
    EXCEPT allergy hints, which are suggested at any confidence and are the
    ONLY path that ever touches the allergies field (safety data demands
    explicit consent);
  - nothing already in the profile (likes/dislikes/allergies/standing seeds)
    is re-suggested;
  - nothing the user previously declined (properties.memory_optouts) is
    re-suggested;
  - one-off constraints never nudge (the extractor filters them, this
    service double-checks kind validity).

Cross-app payoff: FoodScholar reads the same profile, so an accepted memory
personalizes its answers with no extra work.
"""

import logging
import uuid
from typing import Optional

from agents import PreferenceExtractor
from .profile_service import (
    GOAL_PREFERENCE_STRINGS,
    goals_nutrition_profile,
    ProfileService,
    goal_preference_strings,
)
from .session_service import SessionService

logger = logging.getLogger(__name__)

VALID_KINDS = {
    "like", "dislike", "cuisine", "constraint", "allergy_hint",
    "standing_seed", "dietary_goal",
}
# At most this many nudges per turn — more reads as surveillance, not help.
MAX_SUGGESTIONS_PER_TURN = 2


class MemoryService:
    """Suggestion policy + consented write-back for durable preferences."""

    def __init__(
        self,
        session_service: SessionService,
        profile_service: ProfileService,
        extractor: Optional[PreferenceExtractor] = None,
    ):
        self.session_service = session_service
        self.profile_service = profile_service
        self.extractor = extractor or PreferenceExtractor()

    # ------------------------------------------------------------------ #
    # Suggestion (attached to chat turns by the orchestrator)              #
    # ------------------------------------------------------------------ #

    def suggest(self, session, message: str) -> list[dict]:
        """Return nudge-worthy memory suggestions for this user turn."""
        candidates = self.extractor.extract(message)
        if not candidates:
            return []

        profile = session.user_profile
        # Same-KIND dedupe only: a dislike of something currently in the
        # LIKES list must still nudge — it's a contradiction the user should
        # resolve, and filtering it against a flat "known" set silently
        # swallowed exactly the most valuable suggestions (pre-fix bug).
        likes = {str(v).lower() for v in (profile.get("food_likes") or [])}
        dislikes = {str(v).lower() for v in (profile.get("food_dislikes") or [])}
        allergies = {str(v).lower() for v in (profile.get("allergies") or [])}
        seeds = {str(s.get("name", "")).lower() for s in (profile.get("standing_seeds") or [])}
        goals = {str(v).lower() for v in (profile.get("dietary_goals") or [])}
        known_by_kind = {
            "like": likes, "cuisine": likes,
            "dislike": dislikes,
            "allergy_hint": allergies,
            "standing_seed": seeds,
            "dietary_goal": goals,
            "constraint": set(),
        }
        optouts = {str(v).lower() for v in (profile.get("memory_optouts") or [])}

        suggestions = []
        for cand in candidates:
            kind = cand.get("kind")
            value = str(cand.get("value", "")).strip().lower()
            if kind not in VALID_KINDS or not value:
                continue
            # Goals must be canonical slugs the planner understands — an
            # off-list value would be written but never acted on.
            if kind == "dietary_goal" and value not in GOAL_PREFERENCE_STRINGS:
                continue
            if value in known_by_kind.get(kind, set()) or value in optouts:
                continue
            # Allergy hints always nudge; everything else needs an explicit statement.
            if kind != "allergy_hint" and cand.get("confidence") != "high":
                continue
            statement = cand.get("statement") or (
                f"It seems “{value}” matters to you — want me to remember this?"
            )
            # Contradiction with the existing profile deserves an explicit callout.
            if kind == "dislike" and value in likes:
                statement = (
                    f"“{value}” is currently in your likes, but it sounds like "
                    f"you've gone off it — update your profile?"
                )
            elif kind in ("like", "cuisine") and value in dislikes:
                statement = (
                    f"“{value}” is currently in your dislikes, but it sounds like "
                    f"you enjoy it now — update your profile?"
                )
            suggestions.append({
                "id": str(uuid.uuid4()),
                "kind": kind,
                "value": value,
                "statement": statement,
                # The extractor's own justification, carried to the client and
                # back so an accepted memory can say what it was inferred from.
                "evidence": str(cand.get("evidence") or "")[:240],
            })
            if len(suggestions) >= MAX_SUGGESTIONS_PER_TURN:
                break

        if suggestions:
            logger.info(
                "[%s] %d memory suggestion(s): %s",
                session.session_id, len(suggestions),
                [(s["kind"], s["value"]) for s in suggestions],
            )
        return suggestions

    # ------------------------------------------------------------------ #
    # Decision (POST /sessions/{id}/memory)                                #
    # ------------------------------------------------------------------ #

    def decide(self, session, suggestion: dict, decision: str) -> bool:
        """Apply an accepted suggestion or record a declined one.

        The client echoes the suggestion payload back (no server-side pending
        store) — kind/value are re-validated here before any write.
        Returns True if a durable change was persisted.
        """
        kind = suggestion.get("kind")
        value = str(suggestion.get("value", "")).strip().lower()
        if kind not in VALID_KINDS or not value:
            raise ValueError("Invalid memory suggestion payload")

        if decision == "accept":
            applied = self.profile_service.apply_memory(
                session.member_id, kind, value,
                session_id=session.session_id,
                evidence=str(suggestion.get("evidence") or "")[:240],
            )
            if applied:
                # Keep the live session consistent so the very next plan
                # already honors the new memory.
                self._apply_to_session_profile(session, kind, value)
            return applied

        # decline → never suggest this value again
        declined = self.profile_service.record_memory_optout(session.member_id, value)
        if declined:
            optouts = list(session.user_profile.get("memory_optouts") or [])
            if value not in optouts:
                optouts.append(value)
            session.user_profile["memory_optouts"] = optouts
            self.session_service.persist_profile(session.session_id)
        return False

    @staticmethod
    def _attribute_to_session_member(profile: dict, key: str, value: str) -> None:
        """Record the accepting member as the origin of a just-added constraint.

        ``merge_profiles`` builds ``constraint_origins`` when the diner set is
        chosen; a memory accepted mid-session never passes through it, so its
        row rendered with ``members: []`` beside properly attributed siblings —
        which reads as "nobody at this table asked for this".

        Solo sessions are skipped on purpose: ``transparency`` attributes only
        when several people are eating, so there is nothing to say and writing
        the record would be inventing a household.
        """
        diners = profile.get("cooking_for_names") or []
        if len(diners) < 2:
            return
        primary = str(diners[0] or "")
        if not primary:
            return
        origins = dict(profile.get("constraint_origins") or {})
        by_value = dict(origins.get(key) or {})
        who = list(by_value.get(value) or [])
        if primary not in who:
            who.append(primary)
        by_value[value] = who
        origins[key] = by_value
        profile["constraint_origins"] = origins

    def _apply_to_session_profile(self, session, kind: str, value: str) -> None:
        profile = session.user_profile
        if kind in ("like", "cuisine"):
            items = list(profile.get("food_likes") or [])
            if value not in [str(v).lower() for v in items]:
                items.append(value)
            profile["food_likes"] = items
            profile["food_dislikes"] = [
                v for v in (profile.get("food_dislikes") or [])
                if str(v).lower() != value
            ]
        elif kind == "dislike":
            items = list(profile.get("food_dislikes") or [])
            if value not in [str(v).lower() for v in items]:
                items.append(value)
            profile["food_dislikes"] = items
            profile["food_likes"] = [
                v for v in (profile.get("food_likes") or [])
                if str(v).lower() != value
            ]
            self._attribute_to_session_member(profile, "food_dislikes", value)
        elif kind == "allergy_hint":
            items = list(profile.get("allergies") or [])
            if value not in [str(v).lower() for v in items]:
                items.append(value)
            profile["allergies"] = items
            self._attribute_to_session_member(profile, "allergies", value)
        elif kind == "standing_seed":
            seeds = list(profile.get("standing_seeds") or [])
            if value not in [s.get("name", "").lower() for s in seeds]:
                seeds.append({"name": value})
            profile["standing_seeds"] = seeds
        elif kind == "constraint":
            history = profile.get("history", "") or ""
            profile["history"] = (history + "\n" if history else "") + value
        elif kind == "dietary_goal":
            # Mirror ProfileService._map_profile so the very next plan sees
            # the goal exactly as a fresh profile fetch would: the slug, its
            # soft preference string, and (when mapped) the hard diet tag.
            goals = list(profile.get("dietary_goals") or [])
            if value not in goals:
                goals.append(value)
            profile["dietary_goals"] = goals
            prefs = list(profile.get("preferences") or [])
            for pref in goal_preference_strings([value]):
                if pref not in prefs:
                    prefs.append(pref)
            profile["preferences"] = prefs
            goal_nutrition = goals_nutrition_profile(goals)
            if goal_nutrition:
                existing = dict(profile.get("nutrition_profile") or {})
                for key, bound in goal_nutrition.items():
                    if key.startswith("max_"):
                        existing[key] = min(existing[key], bound) if key in existing else bound
                    else:
                        existing[key] = max(existing[key], bound) if key in existing else bound
                profile["nutrition_profile"] = existing
            # The ledger reads goals from `goal_reconciliation`, which
            # `merge_profiles` wrote when the diners were chosen — so a goal
            # accepted afterwards had no row at all, not even a demoted one.
            # Only extended when a record already exists: with none,
            # `_goal_rows` derives rows from `dietary_goals` directly, and
            # creating one here would render a solo session as a household.
            reconciliation = list(profile.get("goal_reconciliation") or [])
            if reconciliation and not any(
                str(r.get("slug") or "").lower() == value for r in reconciliation
            ):
                diners = profile.get("cooking_for_names") or []
                reconciliation.append({
                    "slug": value,
                    "member": str(diners[0]) if diners else "",
                    # The session member is the primary, and merge_profiles
                    # treats every primary goal as a target.
                    "applied": "target",
                })
                profile["goal_reconciliation"] = reconciliation
        self.session_service.persist_profile(session.session_id)
