"""
Seeded / anchored planning (M2) — user-named dishes become pinned plan slots.

Flow: SeedExtractor pulls named dishes from the request → this service
resolves each name against RecipeWrangler (autocomplete → detail), runs the
safety check (allergens are hard constraints — a seed that violates one is
NEVER pinned), and assigns each accepted seed to a plan slot:

    resolve_seeds(seeds, profile)                → list[SeedResolution]
    place_weekly(resolutions)                    → ({(day, meal_idx): CandidateRecipe}, dropped)
    place_daily(resolutions)                     → ({slot_name: CandidateRecipe}, dropped)

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
from services.adapted_recipes import overlay_resolved
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
    kept: bool = False                       # carried over from an earlier turn


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
        """Resolve dish names to recipes; enforce allergy hard constraints.

        A seed carrying ``recipe_id`` (favorites: the offer already resolved
        it) is fetched DIRECTLY — no autocomplete round-trip that could land
        on a different recipe with a similar title. Name-only seeds (dishes
        typed in chat) still resolve through tolerant autocomplete.
        """
        resolutions: list[SeedResolution] = []
        for seed in seeds:
            name = seed["name"]

            known_id = str(seed.get("recipe_id") or "").strip()
            if known_id:
                resolved = self.client.fetch_recipe(known_id)
                if resolved is not None:
                    resolutions.append(self._finalize_resolution(seed, name, resolved, profile))
                    continue
                # Known id unexpectedly gone — fall through to name search.

            # Search first, autocomplete second.
            #
            # `/api/v2/tools/find_recipes` is an analysed full-text search with
            # relevance ranking, so "bolognesse" finds bolognese without the
            # truncation retries below — and, more usefully, it applies the
            # member's allergens and diet, so a dish they cannot eat is never
            # offered as an anchor in the first place. Autocomplete is a
            # prefix matcher for a type-ahead box; it was doing this job only
            # because it was the only search FoodChat knew about.
            candidates = self._search_by_name(name, profile)
            resolved = None
            for candidate in candidates[:3]:
                resolved = self.client.fetch_recipe(candidate.recipe_id)
                if resolved is not None:
                    break

            if resolved is None:
                # Autocomplete fallback. Kept because it walks a different
                # index: a recipe absent from search for any reason is still
                # reachable by exact prefix, and losing a user's named dish is
                # worse than a second round trip.
                suggestions = self._autocomplete_tolerant(name)
                for recipe_id, _title in suggestions[:3]:
                    resolved = self.client.fetch_recipe(recipe_id)
                    if resolved is not None:
                        break

            if resolved is None:
                resolutions.append(SeedResolution(
                    requested_name=name, status="not_found",
                    note=f"I couldn't load the details for “{name}”.",
                ))
                continue

            resolutions.append(self._finalize_resolution(seed, name, resolved, profile))
        return resolutions

    def _finalize_resolution(self, seed: dict, name: str, resolved, profile: dict) -> SeedResolution:
        """Adapted-version overlay + allergy gate, shared by id and name paths."""
        # If the member saved an adapted version of this dish, anchor on
        # that instead of the freshly fetched original (before the allergy
        # check, so the check sees the ingredients we'll actually plan).
        resolved = overlay_resolved(resolved, profile)

        conflict = self._allergy_conflict(resolved, profile)
        if conflict:
            return SeedResolution(
                requested_name=name, status="conflict", recipe=resolved,
                note=(
                    f"I skipped “{resolved.recipe.title}” — it contains "
                    f"{conflict}, which is on your allergy list."
                ),
            )
        return SeedResolution(
            requested_name=name, status="resolved", recipe=resolved,
            meal_type=seed.get("meal_type"), day=seed.get("day"),
            kept=bool(seed.get("kept")),
        )

    def _search_by_name(self, name: str, profile: dict) -> list:
        """Recipes matching a dish name, filtered to what the member can eat.

        The allergy gate in `_finalize_resolution` still runs — this narrows
        the field rather than replacing the check. Filtering here means the
        gate rejects a genuinely borderline dish instead of the first three
        results all being things the member is allergic to.
        """
        from services.plan_client import PLANNER

        return PLANNER.find_recipes(
            name,
            limit=5,
            allergens=profile.get("allergies") or [],
            diet=profile.get("diet") or [],
            exclude_ingredients=profile.get("food_dislikes") or [],
        )

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

    def place_weekly(
        self, resolutions: list[SeedResolution]
    ) -> tuple[dict[tuple[int, int], CandidateRecipe], list[SeedResolution]]:
        """Assign resolved seeds to (day, meal_idx) slots for the 7-day plan.

        Returns (pinned, dropped) — a seed with no free slot (its meal is
        already anchored every day) is dropped and returned so the reply can
        acknowledge it instead of silently omitting it.
        """
        pinned: dict[tuple[int, int], CandidateRecipe] = {}
        dropped: list[SeedResolution] = []

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
                logger.info("No free slot for seed %r — dropping pin", res.requested_name)
                dropped.append(res)
                continue
            pinned[(day, meal_idx)] = res.recipe.recipe
        return pinned, dropped

    def place_daily(
        self, resolutions: list[SeedResolution]
    ) -> tuple[dict[str, CandidateRecipe], list[SeedResolution]]:
        """Assign resolved seeds to daily slots ("breakfast"/"lunch"/"dinner").

        Returns (pinned, dropped). A daily plan holds one dish per slot today,
        so a second dish for an already-filled slot ("pasta AND a salad for
        lunch") is DROPPED — but returned so the reply can own it honestly
        rather than silently claiming it was worked in. (Multi-plate meals
        will let these land — see DYNAMIC_MEALS_PLAN.md.)
        """
        pinned: dict[str, CandidateRecipe] = {}
        dropped: list[SeedResolution] = []
        for res in resolutions:
            if res.status != "resolved":
                continue
            slot = res.meal_type or self._slot_from_dish_types(res.recipe)
            if slot in pinned:
                logger.info("Daily slot %s already pinned — dropping extra %r", slot, res.requested_name)
                dropped.append(res)
                continue
            pinned[slot] = res.recipe.recipe
        return pinned, dropped

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
    def describe(
        resolutions: list[SeedResolution],
        dropped: Optional[list[SeedResolution]] = None,
    ) -> str:
        """One-line summary of pinned/skipped seeds for the assistant response.

        Dishes carried over from an earlier turn (``kept``) are announced
        separately: the member didn't ask for them THIS turn, so claiming
        they did reads as a misunderstanding — and saying nothing would hide
        that a pin outranked the refinement they just typed.

        ``dropped`` are dishes that resolved but had no free slot (a second
        dish for one meal, today's one-dish-per-meal limit). They must NOT be
        claimed as "worked in" — the reply owns the limit honestly instead of
        silently omitting a dish the member asked for.
        """
        dropped = dropped or []
        dropped_ids = {
            getattr(r.recipe.recipe, "recipe_id", None) for r in dropped
            if r.recipe and r.recipe.recipe
        }
        dropped_ids.discard(None)

        def is_dropped(res: SeedResolution) -> bool:
            if not dropped_ids or not (res.recipe and res.recipe.recipe):
                return False
            return getattr(res.recipe.recipe, "recipe_id", None) in dropped_ids

        asked = [r.recipe.recipe.title for r in resolutions
                 if r.status == "resolved" and not r.kept and not is_dropped(r)]
        kept = [r.recipe.recipe.title for r in resolutions
                if r.status == "resolved" and r.kept and not is_dropped(r)]
        dropped_titles = [r.recipe.recipe.title for r in dropped
                          if r.recipe and r.recipe.recipe]
        notes = [r.note for r in resolutions if r.note]
        parts = []
        if asked:
            parts.append("I've worked in " + ", ".join(f"“{t}”" for t in asked) + " as you asked.")
        if kept:
            parts.append(
                "I kept " + ", ".join(f"“{t}”" for t in kept)
                + " that you picked — say the word if you'd like those changed too."
            )
        if dropped_titles:
            parts.append(
                "You also asked for " + ", ".join(f"“{t}”" for t in dropped_titles)
                + " — I plan one dish per meal for now, so I've noted it rather than"
                " dropping it. Multi-dish meals are on the way."
            )
        parts.extend(notes)
        return " ".join(parts)
