"""
Seeded / anchored planning (M2) — user-named dishes become pinned plan slots.

Flow: SeedExtractor pulls named dishes from the request → this service
resolves each name against RecipeWrangler (autocomplete → detail), runs the
safety check (allergens are hard constraints — a seed that violates one is
NEVER pinned), and assigns each accepted seed to a plan slot:

    resolve_seeds(seeds, profile)                → list[SeedResolution]
    place_weekly(resolutions)                    → {(day, meal_idx): CandidateRecipe}
    place_daily(resolutions)                     → {slot_name: CandidateRecipe}

Placement rules: an explicit hint ("Sunday dinner") is honored; otherwise the
slot comes from the recipe's dish-type tags, defaulting to dinner, and weekly
seeds are spread across the week (Mon, Wed, Fri, …) so anchors don't cluster.

Pinned slots are excluded from candidate fetch, never touched by refinements
unless targeted, and reported back so responses can explain what was pinned
and why anything was skipped.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from agents import SeedExtractor
from models.recipe import CandidateRecipe, ResolvedRecipe
from services.candidates_client import CANDIDATES, RecipeCandidatesClient, allergen_conflict

logger = logging.getLogger(__name__)

MEAL_TYPE_TO_IDX = {"breakfast": 0, "lunch": 1, "dinner": 2}
# Spread unplaced weekly seeds across the week: Mon, Wed, Fri, Sun, Tue, Thu, Sat
WEEKLY_DAY_SPREAD = [1, 3, 5, 7, 2, 4, 6]


@dataclass(frozen=True)
class SeedResolution:
    """Outcome of resolving one requested dish."""

    requested_name: str
    status: str                              # "resolved" | "not_found" | "conflict"
    recipe: Optional[ResolvedRecipe] = None
    meal_type: Optional[str] = None          # user hint, if any
    day: Optional[int] = None                # user hint (1–7), if any
    note: Optional[str] = None               # human-readable skip reason


class SeedService:
    """Resolve, safety-check, and place user-requested anchor dishes."""

    def __init__(self, client: Optional[RecipeCandidatesClient] = None,
                 extractor: Optional[SeedExtractor] = None):
        self.client = client or CANDIDATES
        self.extractor = extractor or SeedExtractor()

    def extract_seeds(self, message: str) -> list[dict]:
        """Named dishes in the message (delegates to the SeedExtractor agent)."""
        return self.extractor.extract(message)

    # ------------------------------------------------------------------ #
    # Resolution                                                           #
    # ------------------------------------------------------------------ #

    def resolve_seeds(self, seeds: list[dict], profile: dict) -> list[SeedResolution]:
        """Resolve dish names to recipes; enforce allergy hard constraints."""
        resolutions: list[SeedResolution] = []
        for seed in seeds:
            name = seed["name"]
            suggestions = self._autocomplete_tolerant(name)
            if not suggestions:
                resolutions.append(SeedResolution(
                    requested_name=name, status="not_found",
                    note=f"I couldn't find a recipe matching “{name}”.",
                ))
                continue

            recipe_id, _title = suggestions[0]  # best prefix match
            resolved = self.client.fetch_recipe(recipe_id)
            if resolved is None:
                resolutions.append(SeedResolution(
                    requested_name=name, status="not_found",
                    note=f"I couldn't load the details for “{name}”.",
                ))
                continue

            conflict = self._allergy_conflict(resolved, profile)
            if conflict:
                resolutions.append(SeedResolution(
                    requested_name=name, status="conflict", recipe=resolved,
                    note=(
                        f"I skipped “{resolved.recipe.title}” — it contains "
                        f"{conflict}, which is on your allergy list."
                    ),
                ))
                continue

            resolutions.append(SeedResolution(
                requested_name=name, status="resolved", recipe=resolved,
                meal_type=seed.get("meal_type"), day=seed.get("day"),
            ))
        return resolutions

    def _autocomplete_tolerant(self, name: str) -> list[tuple[str, str]]:
        """Autocomplete with trailing-typo tolerance.

        RecipeWrangler autocomplete is an ES bool_prefix match — no fuzziness,
        so "bolognesse" finds nothing while "bolognes" prefix-matches
        "bolognese". Retrying with progressively truncated queries recovers
        the common trailing-typo case cheaply. (Proper fuzziness belongs in
        the RW autocomplete endpoint.)
        """
        query = (name or "").strip()
        for cut in range(0, 4):
            attempt = query[: len(query) - cut] if cut else query
            if len(attempt) < 4:
                break
            suggestions = self.client.autocomplete(attempt)
            if suggestions:
                if cut:
                    logger.info("Seed %r resolved via truncated query %r", name, attempt)
                return suggestions
        return []

    @staticmethod
    def _allergy_conflict(resolved: ResolvedRecipe, profile: dict) -> Optional[str]:
        """Return the first profile allergen present in the recipe, if any.

        Checks RecipeWrangler's allergen tags first, then the shared
        synonym-expanded ingredient scan (allergen_conflict) — graph tags have
        been proven wrong in production ("nut_free" almond dishes), and
        allergies are a safety constraint, so over-blocking a seed is the
        correct failure mode.
        """
        allergies = [a.lower().strip() for a in profile.get("allergies", []) if a]
        if not allergies:
            return None
        for allergen in allergies:
            if allergen in resolved.allergens:
                return allergen
        return allergen_conflict(
            f"{resolved.recipe.title} {resolved.recipe.ingredients}", allergies
        )

    # ------------------------------------------------------------------ #
    # Placement                                                            #
    # ------------------------------------------------------------------ #

    def place_weekly(self, resolutions: list[SeedResolution]) -> dict[tuple[int, int], CandidateRecipe]:
        """Assign resolved seeds to (day, meal_idx) slots for the 7-day plan."""
        pinned: dict[tuple[int, int], CandidateRecipe] = {}
        spread = iter(WEEKLY_DAY_SPREAD)

        for res in resolutions:
            if res.status != "resolved":
                continue
            meal_idx = self._meal_idx_for(res)
            day = res.day
            if day is None:
                # Prefer a day with no anchors at all so seeds spread across
                # the week; fall back to any day where this slot is free.
                pinned_days = {d for (d, _idx) in pinned}
                day = next(
                    (d for d in WEEKLY_DAY_SPREAD if d not in pinned_days), None,
                ) or next(
                    (d for d in WEEKLY_DAY_SPREAD if (d, meal_idx) not in pinned), None,
                )
            if day is None or (day, meal_idx) in pinned:
                logger.info("No free slot for seed %r — skipping pin", res.requested_name)
                continue
            pinned[(day, meal_idx)] = res.recipe.recipe
        return pinned

    def place_daily(self, resolutions: list[SeedResolution]) -> dict[str, CandidateRecipe]:
        """Assign resolved seeds to daily slots ("breakfast"/"lunch"/"dinner")."""
        pinned: dict[str, CandidateRecipe] = {}
        for res in resolutions:
            if res.status != "resolved":
                continue
            slot = res.meal_type or self._slot_from_dish_types(res.recipe)
            if slot in pinned:
                # One anchor per slot in a daily plan; extras are dropped with a note
                logger.info("Daily slot %s already pinned — skipping %r", slot, res.requested_name)
                continue
            pinned[slot] = res.recipe.recipe
        return pinned

    def _meal_idx_for(self, res: SeedResolution) -> int:
        slot = res.meal_type or self._slot_from_dish_types(res.recipe)
        return MEAL_TYPE_TO_IDX.get(slot, 2)

    @staticmethod
    def _slot_from_dish_types(resolved: Optional[ResolvedRecipe]) -> str:
        """Best slot for a recipe from its dish-type tags (default: dinner)."""
        if resolved:
            for slot in ("breakfast", "lunch", "dinner"):
                if slot in resolved.dish_types:
                    return slot
        return "dinner"

    # ------------------------------------------------------------------ #
    # Response helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def describe(resolutions: list[SeedResolution]) -> str:
        """One-line summary of pinned/skipped seeds for the assistant response."""
        pinned = [r.recipe.recipe.title for r in resolutions if r.status == "resolved"]
        notes = [r.note for r in resolutions if r.note]
        parts = []
        if pinned:
            parts.append("I've worked in " + ", ".join(f"“{t}”" for t in pinned) + " as you asked.")
        parts.extend(notes)
        return " ".join(parts)
