import datetime as _dt
import logging
from typing import Any

from backend.platform import WISEFOOD, WiseFoodPool

logger = logging.getLogger(__name__)


class ProfileService:
    """Service for fetching and mapping WiseFood member profiles."""

    def __init__(self, client_pool: WiseFoodPool = None):
        self.client_pool = WISEFOOD

    def get_member_profile(self, member_id: str) -> dict:
        logger.info("Fetching profile for member %s from WiseFood.", member_id)
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = self._map_profile(member.profile)
            logger.info(
                "Profile fetched for member %s — diet=%s allergies=%s food_likes=%d food_dislikes=%d.",
                member_id,
                profile.get("diet"),
                profile.get("allergies"),
                len(profile.get("food_likes", [])),
                len(profile.get("food_dislikes", [])),
            )
            return profile
        except Exception as e:
            logger.error("Failed to fetch profile for member %s: %s", member_id, e, exc_info=True)
            raise

    def get_member_favorites(self, member_id: str) -> list[str]:
        """Fetch the member's favorited recipe ids from the gateway (M2).

        Best-effort: favorites are a soft personalization signal, so any
        failure degrades to an empty list rather than blocking the session.
        """
        try:
            with self.client_pool.client() as client:
                response = client.get(f"members/{member_id}/favorites")
                payload = response.json()
            rows = payload.get("result", payload) or []
            favorites = [r["recipe_id"] for r in rows if isinstance(r, dict) and r.get("recipe_id")]
            logger.info("Fetched %d favorites for member %s.", len(favorites), member_id)
            return favorites
        except Exception as e:
            logger.warning("Could not fetch favorites for member %s: %s", member_id, e)
            return []

    # ------------------------------------------------------------------ #
    # Consented memory write-back (M3)                                     #
    # ------------------------------------------------------------------ #
    # The SDK profile object auto-PATCHes the gateway on attribute
    # assignment (sync=True), so each field write below persists
    # immediately. Every durable write carries provenance in
    # properties.memory_log — personalization must stay auditable.

    def apply_memory(self, member_id: str, kind: str, value: str, session_id: str) -> bool:
        """Persist a user-CONSENTED memory to the durable member profile.

        kind → destination:
          like/cuisine   → nutritional_preferences.food_likes
          dislike        → nutritional_preferences.food_dislikes
          allergy_hint   → allergies (safety field — only ever via consent)
          standing_seed  → properties.standing_seeds [{name}]
          constraint     → properties.feedback_history (appended line)
        """
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = member.profile
                prefs = dict(profile.nutritional_preferences or {})
                props = dict(profile.properties or {})

                value_norm = value.strip().lower()
                if kind in ("like", "cuisine"):
                    likes = list(prefs.get("food_likes") or [])
                    if value_norm not in [v.lower() for v in likes]:
                        likes.append(value_norm)
                    prefs["food_likes"] = likes
                    profile.nutritional_preferences = prefs
                elif kind == "dislike":
                    dislikes = list(prefs.get("food_dislikes") or [])
                    if value_norm not in [v.lower() for v in dislikes]:
                        dislikes.append(value_norm)
                    prefs["food_dislikes"] = dislikes
                    profile.nutritional_preferences = prefs
                elif kind == "allergy_hint":
                    allergies = list(profile.allergies or [])
                    if value_norm not in [a.lower() for a in allergies]:
                        allergies.append(value_norm)
                    profile.allergies = allergies
                elif kind == "standing_seed":
                    seeds = list(props.get("standing_seeds") or [])
                    if value_norm not in [s.get("name", "").lower() for s in seeds]:
                        seeds.append({"name": value_norm})
                    props["standing_seeds"] = seeds
                elif kind == "constraint":
                    history = props.get("feedback_history", "") or ""
                    props["feedback_history"] = (history + "\n" if history else "") + value
                else:
                    logger.warning("Unknown memory kind %r — not applied", kind)
                    return False

                # Provenance: who learned what, where, when.
                log = list(props.get("memory_log") or [])
                log.append({
                    "kind": kind, "value": value_norm,
                    "source": "foodchat", "session_id": session_id,
                    "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                })
                props["memory_log"] = log
                profile.properties = props

            logger.info("Memory applied for member %s: %s=%r", member_id, kind, value_norm)
            return True
        except Exception as e:
            logger.error("Failed to apply memory for member %s: %s", member_id, e, exc_info=True)
            return False

    def record_memory_optout(self, member_id: str, value: str) -> bool:
        """Remember a declined suggestion so we never nag about it again."""
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = member.profile
                props = dict(profile.properties or {})
                optouts = list(props.get("memory_optouts") or [])
                value_norm = value.strip().lower()
                if value_norm not in optouts:
                    optouts.append(value_norm)
                    props["memory_optouts"] = optouts
                    profile.properties = props
            return True
        except Exception as e:
            logger.error("Failed to record opt-out for member %s: %s", member_id, e, exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Household diners (M3 — "who are we cooking for?")                    #
    # ------------------------------------------------------------------ #

    def get_member_name(self, member_id: str) -> str:
        try:
            with self.client_pool.client() as client:
                return client.members.get(member_id).name or member_id
        except Exception:
            return member_id

    @staticmethod
    def merge_profiles(primary: dict, others: list[dict], diner_names: list[str]) -> dict:
        """Merge diner profiles into one plan-generation profile.

        Hard constraints (safety/identity) are the UNION across all diners:
        allergies, dietary groups, dislikes-as-exclusions. Soft preferences
        keep the primary member's weighting (their likes lead, other diners'
        likes follow). Macro targets stay the primary member's. The primary
        member's favorites keep their boost.
        """
        merged = dict(primary)

        def union(key: str) -> list:
            seen: dict[str, str] = {}
            for profile in [primary, *others]:
                for item in profile.get(key) or []:
                    seen.setdefault(str(item).lower(), item)
            return list(seen.values())

        merged["allergies"] = union("allergies")
        merged["diet"] = union("diet")
        merged["food_dislikes"] = union("food_dislikes")

        likes = list(primary.get("food_likes") or [])
        for other in others:
            for like in other.get("food_likes") or []:
                if str(like).lower() not in [str(l).lower() for l in likes]:
                    likes.append(like)
        merged["food_likes"] = likes

        merged["cooking_for_names"] = diner_names
        return merged

    def _map_profile(self, wisefood_profile: Any) -> dict:
        """Map WiseFood profile structure to FoodChat format.

        WiseFood structure:
        - dietary_groups: list (vegetarian, gluten_free, etc.)
        - allergies: list (shellfish, sesame, etc.)
        - nutritional_preferences: dict (calories, protein, fat, carbs,
          food_likes, food_dislikes)
        - properties: dict (age_group, feedback_history, liked_recipes)

        FoodChat structure:
        - diet: list[str]
        - allergies: list[str]
        - preferences: list[str]
        - history: str
        - food_likes: list[str]
        - food_dislikes: list[str]
        """
        # Handle both object attributes and dict-like access
        if hasattr(wisefood_profile, "dietary_groups"):
            dietary_groups = wisefood_profile.dietary_groups or []
            allergies = wisefood_profile.allergies or []
            nutritional_preferences = wisefood_profile.nutritional_preferences or {}
            properties = wisefood_profile.properties or {}
        else:
            dietary_groups = wisefood_profile.get("dietary_groups", []) or []
            allergies = wisefood_profile.get("allergies", []) or []
            nutritional_preferences = (
                wisefood_profile.get("nutritional_preferences", {}) or {}
            )
            properties = wisefood_profile.get("properties", {}) or {}

        return {
            "diet": list(dietary_groups),
            "allergies": list(allergies),
            "preferences": self._build_preferences(nutritional_preferences, properties),
            "history": properties.get("feedback_history", "") or "",
            "food_likes": list(nutritional_preferences.get("food_likes", []) or []),
            "food_dislikes": list(nutritional_preferences.get("food_dislikes", []) or []),
            # M3 memory fields (written only through the consent flow)
            "standing_seeds": list(properties.get("standing_seeds", []) or []),
            "memory_optouts": list(properties.get("memory_optouts", []) or []),
            "memory_log": list(properties.get("memory_log", []) or []),
        }

    def _build_preferences(
        self, nutritional_prefs: dict, properties: dict
    ) -> list[str]:
        """Convert nutritional preferences and properties to preference strings."""
        prefs = []

        # Nutritional preferences
        if nutritional_prefs.get("calories"):
            prefs.append(f"{nutritional_prefs['calories']} calories target")

        if nutritional_prefs.get("protein"):
            prefs.append(f"high protein ({nutritional_prefs['protein']}g)")

        if nutritional_prefs.get("fiber"):
            prefs.append(f"high fiber ({nutritional_prefs['fiber']}g)")

        if nutritional_prefs.get("carbs"):
            prefs.append(f"{nutritional_prefs['carbs']}g carbs")

        if nutritional_prefs.get("fat"):
            prefs.append(f"{nutritional_prefs['fat']}g fat")

        # Additional preferences from properties.notes if structured
        notes = properties.get("notes", "")
        if notes and "prefers" in notes.lower():
            # Extract cuisine preferences from notes if present
            prefs.append(notes)

        return prefs

    def format_profile_for_display(self, profile: dict) -> str:
        """Format a profile dict for display/logging."""
        lines = []
        if profile.get("diet"):
            lines.append(f"Diet: {', '.join(profile['diet'])}")
        if profile.get("allergies"):
            lines.append(f"Allergies: {', '.join(profile['allergies'])}")
        if profile.get("preferences"):
            lines.append(f"Preferences: {', '.join(profile['preferences'])}")
        if profile.get("history"):
            lines.append(f"Notes: {profile['history']}")
        return "\n".join(lines) if lines else "No profile information"
