from typing import Any

from services.wisefood_client import WiseFoodClientPool


class ProfileService:
    """Service for fetching and mapping WiseFood member profiles."""

    def __init__(self, client_pool: WiseFoodClientPool):
        self.client_pool = client_pool

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
        client = self.client_pool.get_client()
        member = client.members.get(member_id)
        return self._map_profile(member.profile)

    def _map_profile(self, wisefood_profile: Any) -> dict:
        """Map WiseFood profile structure to FoodChat format.

        WiseFood structure:
        - dietary_groups: list (vegetarian, gluten_free, etc.)
        - nutritional_preferences: dict (calories, protein, fiber)
        - properties: dict (allergies, notes)

        FoodChat structure:
        - diet: list[str]
        - allergies: list[str]
        - preferences: list[str]
        - history: str
        """
        # Handle both object attributes and dict-like access
        if hasattr(wisefood_profile, "dietary_groups"):
            dietary_groups = wisefood_profile.dietary_groups or []
            nutritional_preferences = wisefood_profile.nutritional_preferences or {}
            properties = wisefood_profile.properties or {}
        else:
            dietary_groups = wisefood_profile.get("dietary_groups", []) or []
            nutritional_preferences = (
                wisefood_profile.get("nutritional_preferences", {}) or {}
            )
            properties = wisefood_profile.get("properties", {}) or {}

        return {
            "diet": list(dietary_groups),
            "allergies": self._extract_allergies(properties),
            "preferences": self._build_preferences(nutritional_preferences, properties),
            "history": properties.get("notes", "") or "",
        }

    def _extract_allergies(self, properties: dict) -> list[str]:
        """Extract allergies from properties."""
        allergies = properties.get("allergies", [])
        if isinstance(allergies, str):
            return [a.strip() for a in allergies.split(",") if a.strip()]
        return list(allergies) if allergies else []

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
