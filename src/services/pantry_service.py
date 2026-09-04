"""
Pantry-driven planning ("cook from what I have") — reduce food waste.

The member states ingredients they already have ("I've got zucchini, spinach
and some ground beef"); the pipeline boosts recipes that use them and reports
COVERAGE — which items the finished plan actually uses — honestly, including
the items it could not place. See PANTRY_PLANNING_PLAN.md (Tier A: no
RecipeWrangler changes are assumed).

The one semantic this module exists to bridge: `/api/v2/tools/plan_meals`
treats `include_ingredients` as a hard AND — the whole pantry sent at once
would demand every item in every recipe and empty each slot. So sourcing is a
capped per-item fan-out (a SINGLE-item hard include is exactly "must use this
one thing"), merged into the ordinary pool and ranked by client-side coverage.

Contracts:

    extract_pantry_delta(message)         → PlanningStateDelta (never raises;
                                            regex-gated so most turns cost no
                                            LLM call)
    fetch_pantry_candidates(profile, …)   → {slot: [CandidateRecipe]} best-effort
    merge_pantry_pool(base, pantry, …)    → coverage-ranked slot pools
    pantry_boost_ids(profile, pantry)     → recipe ids for the structured path's
                                            soft favourites-style boost
    annotate_daily_plan(plan, pantry)     → UI badges + ledger row, returns facts
    annotate_weekly_entries(entries, …)   → same for the 21-entry weekly plan

Every user-facing claim ("uses your zucchini") comes from the deterministic
word-boundary matcher below, never from a model. A missed match under-reports
coverage; it never fabricates it. Pantry items are inputs to search, not
bypasses — every fetch still carries the member's allergens, diet and
dislikes, so an item the member cannot eat simply finds nothing usable.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

from models.planning_state import PlanningStateDelta
from models.recipe import CandidateRecipe

logger = logging.getLogger(__name__)

# Items forwarded to per-item candidate search. More items than this still
# count toward coverage; they just don't each get their own fetch.
PANTRY_ITEM_LIMIT = int(os.getenv("FOODCHAT_PANTRY_ITEM_LIMIT", "6"))
# Candidates fetched per pantry item per fan-out call (kept small — each item
# is one extra `plan_meals` round-trip).
PANTRY_PER_ITEM_CANDIDATES = int(os.getenv("FOODCHAT_PANTRY_PER_ITEM_CANDIDATES", "3"))

# The UI badge text (shared contract with the UI, kind="pantry").
BADGE_FOODWASTE = "reducing food waste"

# Cheap gate for the LLM extraction: most turns say nothing about a pantry,
# and an extractor call per turn would price the feature into every message.
# Broad on purpose — a false positive costs one abstaining LLM call.
#
# The subject and the verb are allowed up to two words apart. The original
# pattern required them adjacent, so "I **already** have avocado, tomatoes and
# pasta" — about the most natural way anyone says this — never reached the
# extractor at all, and the plan silently ignored the member's fridge while
# the reply still thanked them for it. "still", "just", "only", "also" and
# "now" fail the same way. The stated trade is explicitly in this direction:
# a miss costs the whole feature, a false positive costs one LLM call that
# abstains.
_PANTRY_HINT_RE = re.compile(
    r"\b(?:i|we)(?:'ve)?\s+(?:\w+\s+){0,2}(?:have|got)\b"
    r"|\bthere(?:'s| is| are)\b"
    r"|\b(fridge|freezer|pantry|cupboard|leftover|left\s*over|left-over"
    r"|scraps?|use\s+up|using\s+up|lying\s+around|going\s+(?:bad|off)"
    r"|about\s+to\s+(?:spoil|expire)|food\s*waste|used\s+up|finished\s+the)\b",
    re.IGNORECASE,
)


def normalize_items(raw: Iterable) -> tuple[str, ...]:
    """Lowercased, trimmed, deduped ingredient names, insertion order kept."""
    seen: dict[str, None] = {}
    for item in raw or ():
        value = str(item or "").strip().lower()
        if value and len(value) >= 2 and value not in seen:
            seen[value] = None
    return tuple(seen)


def singular(word: str) -> str:
    """A crude singular stem, enough for ingredient names.

    "tomatoes" → "tomato", "berries" → "berri" (which still matches "berries"
    once the suffix is optional again), "peas" → "pea". Deliberately not a
    real stemmer: it only has to make the match symmetric.
    """
    for suffix, stem in (("ies", "i"), ("oes", "o"), ("es", ""), ("s", "")):
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            return word[: -len(suffix)] + stem
    return word


def _item_pattern(term: str) -> str:
    """A word-boundary pattern matching ``term`` in either number.

    The original pattern appended an optional "s", which only worked in one
    direction: a member who said "tomatoes" — the natural way to name what is
    in a fridge — never matched a recipe listing "tomato". The plan then used
    the item while the reply said "I couldn't work in your tomatoes" and the
    ledger recorded the coverage as relaxed. Under-reporting is supposed to be
    the safe direction, but here it produced an affirmative false claim, so
    each word is stemmed and re-inflected instead.
    """
    words = [w for w in term.split() if w]
    return r"\b" + r"\s+".join(
        re.escape(singular(w)) + r"(?:e?s)?" for w in words
    ) + r"\b"


def matched_items(text: str, pantry: Iterable[str]) -> list[str]:
    """Pantry items present in the text (word-boundary, number-insensitive).

    The same matching posture as the allergen backstop: word boundaries so
    "rice" does not match "price". Singular and plural match each other in
    both directions. Multi-word items match as a phrase.
    """
    haystack = (text or "").lower()
    hits: list[str] = []
    for item in pantry or ():
        term = str(item).strip().lower()
        if not term:
            continue
        if re.search(_item_pattern(term), haystack):
            hits.append(term)
    return hits


# --------------------------------------------------------------------------- #
# Extraction (this turn's pantry statements → a PlanningStateDelta)
# --------------------------------------------------------------------------- #

def extract_pantry_delta(message: str, *, extractor=None) -> PlanningStateDelta:
    """What this turn says about on-hand ingredients. Never raises.

    Regex-gated: the LLM extractor runs only when the message plausibly talks
    about having/using ingredients. A turn whose pantry cannot be read changes
    nothing — the member keeps the pantry they already stated, which is the
    safe direction (same posture as `planning_delta.extract_state_delta`).
    """
    text = (message or "").strip()
    if not text or not _PANTRY_HINT_RE.search(text):
        return PlanningStateDelta()

    try:
        if extractor is None:
            from agents import PantryExtractor

            extractor = PantryExtractor()
        payload = extractor.extract(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pantry extraction failed: %s", exc)
        return PlanningStateDelta()

    have = normalize_items(payload.get("have") or [])
    used_up = normalize_items(payload.get("used_up") or [])
    if have or used_up:
        logger.info("Pantry delta: +%s -%s", list(have), list(used_up))
    return PlanningStateDelta(pantry_add=have, pantry_remove=used_up)


# --------------------------------------------------------------------------- #
# Candidate sourcing (per-item fan-out; Tier A — no RecipeWrangler changes)
# --------------------------------------------------------------------------- #

def fetch_pantry_candidates(
    profile: dict,
    pantry: Iterable[str],
    slots: tuple[str, ...] = ("breakfast", "lunch", "dinner"),
    exclude_recipe_ids: Optional[list[str]] = None,
    per_item: Optional[int] = None,
    diet: Optional[list[str]] = None,
    cuisines: Optional[list[str]] = None,
    max_minutes: Optional[int] = None,
) -> dict[str, list[CandidateRecipe]]:
    """Recipes that use each pantry item, under the member's constraints.

    One `plan_meals` call per item (capped), each with a SINGLE-item hard
    include — "must use this one thing" is exactly the AND semantics the
    endpoint has. Calls run in parallel and each failure means only "that
    item found nothing"; the plan never blocks on the pantry.

    ``cuisines`` and ``max_minutes`` MUST be threaded from the caller, because
    this pool is merged into the ordinary one and sorted coverage-first — so
    anything omitted here does not merely appear, it appears at the TOP. Left
    out, a member with the cooking-time slider at 20 minutes who mentioned a
    courgette got a 90-minute bake ranked first. The pantry is an input to
    search, never a way around what the member asked for.
    """
    from services.candidates_client import normalize_diet_tags
    from services.plan_client import PLANNER

    items = list(normalize_items(pantry))[:PANTRY_ITEM_LIMIT]
    if not items:
        return {}
    count = per_item if per_item is not None else PANTRY_PER_ITEM_CANDIDATES
    # A caller with a tightened diet (weekly action space: profile diet +
    # query-level tags) passes it explicitly; otherwise the profile's own.
    diet_tags = normalize_diet_tags(
        diet if diet is not None else profile.get("diet")
    )

    def fetch_one(item: str) -> dict[str, list[CandidateRecipe]]:
        try:
            envelope = PLANNER.plan_meals(
                days=1,
                slots=slots,
                count_per_slot=count,
                allergens=profile.get("allergies") or [],
                diet=diet_tags,
                cuisines=list(cuisines or []),
                include_ingredients=[item],
                exclude_ingredients=profile.get("food_dislikes") or [],
                exclude_recipe_ids=list(exclude_recipe_ids or []),
                max_minutes=max_minutes,
                min_nutri_score=profile.get("min_nutri_score"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Pantry fetch for %r found nothing usable: %s", item, exc)
            return {}
        return PLANNER.to_candidates(
            envelope, allergens=profile.get("allergies") or []
        )

    merged: dict[str, list[CandidateRecipe]] = {slot: [] for slot in slots}
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
        for by_slot in pool.map(fetch_one, items):
            for slot in slots:
                for candidate in by_slot.get(slot, []):
                    if candidate.recipe_id in seen:
                        continue
                    seen.add(candidate.recipe_id)
                    merged[slot].append(candidate)
    return merged


def merge_pantry_pool(
    base: dict[str, list[CandidateRecipe]],
    pantry_pools: dict[str, list[CandidateRecipe]],
    pantry: Iterable[str],
    limit_per_slot: int,
) -> dict[str, list[CandidateRecipe]]:
    """Fold pantry candidates into the ordinary pool, coverage first.

    Ranking inside each slot: coverage count descending, then the upstream
    deterministic order (planning tier / Nutri-Score) as the tiebreak — base
    candidates that happen to use pantry items rank by the same rule. The pool
    stays capped at `limit_per_slot` so the grading space does not grow.
    """
    items = list(normalize_items(pantry))
    out: dict[str, list[CandidateRecipe]] = {}
    for slot, base_list in base.items():
        combined: list[CandidateRecipe] = []
        seen: set[str] = set()
        for candidate in list(pantry_pools.get(slot, [])) + list(base_list):
            if candidate.recipe_id in seen:
                continue
            seen.add(candidate.recipe_id)
            combined.append(candidate)
        scored = sorted(
            enumerate(combined),
            key=lambda pair: (
                -len(matched_items(
                    f"{pair[1].title} {pair[1].ingredients}", items
                )),
                pair[0],  # stable: keep upstream order among equals
            ),
        )
        out[slot] = [candidate for _idx, candidate in scored][:limit_per_slot]
    return out


def pantry_boost_ids(
    profile: dict, pantry: Iterable[str], limit_per_item: int = 5
) -> list[str]:
    """Recipe ids that use pantry items, for a favourites-style soft boost.

    The structured path (`plan_structured`) delegates assembly wholly to
    `plan_meals`, which offers exactly one soft rank signal with no RW change:
    `favorite_recipe_ids` float within their slot while hard filters still
    decide eligibility. These ids ride that signal.
    """
    from services.candidates_client import normalize_diet_tags
    from services.plan_client import PLANNER

    ids: list[str] = []
    for item in list(normalize_items(pantry))[:PANTRY_ITEM_LIMIT]:
        try:
            hits = PLANNER.find_recipes(
                item,
                limit=limit_per_item,
                allergens=profile.get("allergies") or [],
                diet=normalize_diet_tags(profile.get("diet")),
                exclude_ingredients=profile.get("food_dislikes") or [],
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Pantry boost lookup failed for %r: %s", item, exc)
            continue
        for hit in hits:
            if hit.recipe_id and hit.recipe_id not in ids:
                ids.append(hit.recipe_id)
    return ids


# --------------------------------------------------------------------------- #
# Coverage + UI badges
# --------------------------------------------------------------------------- #

def coverage_facts(
    recipes: Iterable[tuple[str, str]], pantry: Iterable[str]
) -> dict:
    """Which pantry items the plan uses, per the deterministic matcher.

    `recipes` is (title, ingredients_text) pairs. Returns
    {"items", "used": {item: [titles]}, "unused", "used_count", "total"} —
    the response writer phrases it; unused items get a sentence, not silence.
    """
    items = list(normalize_items(pantry))
    used: dict[str, list[str]] = {}
    for title, ingredients in recipes:
        for item in matched_items(f"{title} {ingredients}", items):
            titles = used.setdefault(item, [])
            if title and title not in titles:
                titles.append(title)
    unused = [item for item in items if item not in used]
    return {
        "items": items,
        "used": used,
        "unused": unused,
        "used_count": len(used),
        "total": len(items),
    }


def _badge_reasons(matches: list[str]) -> list[dict]:
    """The per-course UI chip for a recipe that uses pantry items.

    One chip, and it names what the matcher actually found. There was a second,
    constant chip reading "cooked from your leftovers" — emitted on every match
    regardless of what the member said. "I picked up courgettes today" is a
    pantry statement too, and nothing in the state records whether an item was
    a leftover, so the claim had no measurement behind it. This module's rule
    is that every user-facing claim comes from the matcher; the food-waste chip
    already carries the intent without asserting the item's history.
    """
    label = "uses your " + ", ".join(matches) + f" — {BADGE_FOODWASTE}"
    return [{"kind": "pantry", "label": label}]


def _ledger_row(facts: dict) -> dict:
    """Plan-level constraint row: honest about partial coverage."""
    status = "satisfied" if not facts["unused"] else "relaxed"
    return {
        "constraint": (
            f"using {facts['used_count']} of {facts['total']} on-hand "
            f"ingredient(s) — {BADGE_FOODWASTE}"
        ),
        "type": "soft",
        "status": status,
        "source": "your pantry",
    }


def annotate_daily_plan(meal_plan, pantry: Iterable[str]) -> Optional[dict]:
    """Attach pantry badges + ledger row to a daily plan in place.

    Handles both plan shapes through `day_plans` (legacy three-course and
    day-oriented structured plans read the same way). Runs AFTER
    `apply_transparency`, appending to the chips it built. Returns the
    coverage facts, or None when there is no pantry.
    """
    items = list(normalize_items(pantry))
    if not items:
        return None

    pairs: list[tuple[str, str]] = []
    for day in meal_plan.day_plans:
        for meal in day.meals:
            for plate in meal.plates:
                pairs.append((plate.title, plate.ingredients))
                matches = matched_items(
                    f"{plate.title} {plate.ingredients}", items
                )
                if matches:
                    reasons = list(plate.match_reasons or [])
                    reasons.extend(_badge_reasons(matches))
                    plate.match_reasons = reasons

    facts = coverage_facts(pairs, items)
    ledger = list(meal_plan.constraints_applied or [])
    ledger.append(_ledger_row(facts))
    meal_plan.constraints_applied = ledger
    return facts


def annotate_weekly_entries(
    plan_entries: list[dict],
    pantry: Iterable[str],
    explainability: Optional[dict] = None,
) -> Optional[dict]:
    """Same as `annotate_daily_plan`, for the weekly entry-dict shape.

    Runs AFTER `build_weekly_explainability` so the chips it attached are
    appended to, not overwritten. The ledger row lands on the explainability
    payload's `constraints_applied` when one is passed.
    """
    items = list(normalize_items(pantry))
    if not items:
        return None

    pairs: list[tuple[str, str]] = []
    for entry in plan_entries:
        recipe = entry.get("recipe") or {}
        title = str(recipe.get("recipe_title") or recipe.get("title") or "")
        ingredients = str(recipe.get("recipe_ingredients") or recipe.get("ingredients") or "")
        pairs.append((title, ingredients))
        matches = matched_items(f"{title} {ingredients}", items)
        if matches:
            reasons = list(recipe.get("match_reasons") or [])
            reasons.extend(_badge_reasons(matches))
            recipe["match_reasons"] = reasons

    facts = coverage_facts(pairs, items)
    if explainability is not None:
        ledger = list(explainability.get("constraints_applied") or [])
        ledger.append(_ledger_row(facts))
        explainability["constraints_applied"] = ledger
    return facts


def describe_coverage(facts: Optional[dict]) -> str:
    """One honest sentence about coverage, for fallback texts and facts.

    Empty string when there is nothing to say. Never claims an item was used
    unless the matcher found it in a recipe on the plan.
    """
    if not facts or not facts.get("items"):
        return ""
    used = facts.get("used") or {}
    parts: list[str] = []
    if used:
        parts.append(
            "To cut food waste, the plan uses your "
            + ", ".join(used) + "."
        )
    unused = facts.get("unused") or []
    if unused:
        parts.append(
            "I couldn't work in your " + ", ".join(unused)
            + " — nothing that fits your requirements uses "
            + ("them" if len(unused) > 1 else "it")
            + "; it stays on your list."
        )
    return " ".join(parts)
