"""
Meal classification and per-day summaries for the weekly plan (M6).

Two jobs, both LLM-free and network-free:

1. **Shared food taxonomy.** One set of red-meat / poultry / fish keyword
   lists (fish reuses ``candidates_client.ALLERGEN_SYNONYMS``) used both by
   the day-summary classifier here and by the weekly meat-limit tracker
   (``state_tracking.WeeklyNutritionalTracker``). Previously the tracker
   kept its own ``MEAT_KEYWORDS`` copy that could silently drift from the
   allergen synonym lists.

2. **Per-day headlines.** ``classify_meal`` labels one plan recipe
   (RecipeWrangler diet tags first, ingredient/title keywords as backstop),
   ``summarize_day`` composes a short 2-4 word headline per day
   ("dinner with red meat", "light vegetarian day"), and
   ``build_day_summaries`` maps a full entry list to ``{day: headline}``
   for the API response and the chat reply facts.

Matching is word-boundary (like ``candidates_client.allergen_conflict``),
not substring — "meatless" or "graham" must not count as meat.
"""

import re
from typing import Iterable, Optional

from services.candidates_client import ALLERGEN_SYNONYMS

RED_MEAT_KEYWORDS = {
    "beef", "pork", "lamb", "steak", "bacon", "ham", "hamburger", "burger",
    "sausage", "mutton", "veal", "venison", "rabbit", "meat", "meatball",
    "meatloaf", "pepperoni", "salami", "gelatin",
}
POULTRY_KEYWORDS = {"chicken", "turkey", "duck", "goose", "quail"}
FISH_KEYWORDS = (
    {"fish", "seafood"}
    | set(ALLERGEN_SYNONYMS["fish"])
    | set(ALLERGEN_SYNONYMS["shellfish"])
)
MEAT_KEYWORDS = RED_MEAT_KEYWORDS | POULTRY_KEYWORDS | FISH_KEYWORDS

VEG_TAGS = {"vegetarian", "vegan"}

# kcal thresholds for the light/hearty day qualifier (whole-day estimate).
LIGHT_DAY_KCAL = 1400.0
HEARTY_DAY_KCAL = 2400.0


def _matches(text: str, keywords: set[str]) -> bool:
    return any(
        re.search(r"\b" + re.escape(k) + r"s?\b", text) for k in keywords
    )


def is_meat_meal(
    title: str,
    ingredients: str,
    tags: Optional[Iterable[str]] = None,
    count_fish: bool = True,
) -> bool:
    """Whether a recipe counts toward the weekly meat limit.

    A vegetarian/vegan RecipeWrangler tag overrides keyword hits (tags are
    authoritative); ``count_fish=False`` exempts fish/seafood — used for
    pescatarian profiles, whose meals would otherwise all count as meat.
    """
    if tags and VEG_TAGS & {str(t).lower() for t in tags}:
        return False
    keywords = MEAT_KEYWORDS if count_fish else (RED_MEAT_KEYWORDS | POULTRY_KEYWORDS)
    return _matches(f"{title} {ingredients}".lower(), keywords)


def classify_meal(recipe: dict) -> str:
    """Category for one plan recipe dict.

    Returns one of ``"vegan" | "vegetarian" | "fish" | "red meat" |
    "poultry"``. Accepts both weekly-entry keys (``recipe_title`` /
    ``recipe_ingredients``) and overlay/display keys (``title`` /
    ``ingredients``). With no tags and no meat keyword hit the recipe is
    treated as vegetarian — the same coverage assumption the meat tracker
    has always made, just in the other direction.
    """
    tags = {str(t).lower() for t in (recipe.get("tags") or [])}
    if "vegan" in tags:
        return "vegan"
    if "vegetarian" in tags:
        return "vegetarian"

    text = " ".join(
        str(recipe.get(k) or "")
        for k in ("recipe_title", "title", "recipe_ingredients", "ingredients")
    ).lower()
    if _matches(text, FISH_KEYWORDS):
        return "fish"
    if _matches(text, RED_MEAT_KEYWORDS):
        return "red meat"
    if _matches(text, POULTRY_KEYWORDS):
        return "poultry"
    return "vegetarian"


def _recipe_kcal(recipe: dict) -> Optional[float]:
    nutrition = recipe.get("nutrition") or {}
    for key in ("kcal", "calories"):
        value = nutrition.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize_day(day_entries: list[dict]) -> str:
    """Short headline for one day's entries (any number of meals).

    Composition rules (deliberately simple — see IDEAS.md for the caveat
    that these templates should be iterated against real generated weeks):
    all-veg day -> "vegetarian day" / "vegan day" with a light/hearty kcal
    qualifier when nutrition supports it; exactly one non-veg meal names it
    ("lunch with fish"); one dominant non-veg category -> "fish day"; two
    categories name both; anything busier falls back to "varied meals".
    """
    if not day_entries:
        return ""
    ordered = sorted(day_entries, key=lambda e: e.get("meal_idx", 0))
    labelled = [
        (str(e.get("meal_type", "meal")), classify_meal(e.get("recipe") or {}),
         _recipe_kcal(e.get("recipe") or {}))
        for e in ordered
    ]

    non_veg = [(meal, cat) for meal, cat, _ in labelled if cat not in VEG_TAGS]
    if not non_veg:
        label = "vegan day" if all(cat == "vegan" for _, cat, _ in labelled) else "vegetarian day"
        known = [k for _, _, k in labelled if k is not None]
        if len(known) >= 2:
            day_estimate = sum(known) * len(labelled) / len(known)
            if day_estimate < LIGHT_DAY_KCAL:
                return f"light {label}"
            if day_estimate > HEARTY_DAY_KCAL:
                return f"hearty {label}"
        return label

    if len(non_veg) == 1:
        meal, cat = non_veg[0]
        return f"{meal} with {cat}"

    categories = {cat for _, cat in non_veg}
    if len(categories) == 1:
        return f"{next(iter(categories))} day"
    if len(categories) == 2:
        first, second = sorted(categories)
        return f"{first} and {second}"
    return "varied meals"


def build_day_summaries(plan_entries: list[dict]) -> dict[int, str]:
    """{day -> headline} over a full weekly entry list, days sorted."""
    by_day: dict[int, list[dict]] = {}
    for entry in plan_entries:
        by_day.setdefault(int(entry.get("day", 0)), []).append(entry)
    return {day: summarize_day(by_day[day]) for day in sorted(by_day)}
