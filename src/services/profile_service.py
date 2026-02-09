from typing import Any

from backend.platform import WISEFOOD, WiseFoodPool


class ProfileService:
    """Service for fetching and mapping WiseFood member profiles."""

    def __init__(self, client_pool: WiseFoodPool = None):
        """
        Initialize ProfileService.

        Args:
            client_pool: Optional WiseFoodPool instance. If not provided,
                        uses the global WISEFOOD pool.
        """
        self.client_pool = WISEFOOD

    def get_member_profile(self, member_id: str) -> dict:
        """Fetch profile from WiseFood and map to FoodChat format.

        Args:
            member_id: The WiseFood member ID

        Returns:
            FoodChat-formatted profile dict with keys:
            - diet: list of dietary restrictions
            - allergies: list of allergies
            - preferences: list of preference strings
            - history: feedback/notes history
        """
        with self.client_pool.client() as client:
            member = client.members.get(member_id)
            return self._map_profile(member.profile)

    def update_member_history(self, member_id: str, history_update: str) -> dict:
        """
        Update the history/notes within a member's nutritional preferences.

        Args:
            member_id: The WiseFood member ID
            history_update: The new history string to save

        Returns:
            The updated, mapped profile dict.
        """
        with self.client_pool.client() as client:
            member = client.members.get(member_id)
            profile = member.profile

            # 1. Get existing prefs (default to dict if None)
            # We use .copy() to ensure we have a mutable dict we can work with
            current_prefs = (profile.nutritional_preferences or {}).copy()

            # 2. Update the history key
            current_prefs["history"] = history_update

            # 3. Re-assign to trigger the API update (Auto-syncs per SDK example)
            profile.nutritional_preferences = current_prefs

            # 4. Return the updated mapped profile
            return self._map_profile(profile)

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
