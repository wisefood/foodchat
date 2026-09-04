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

# Tag spellings RecipeWrangler uses to assert a recipe contains no meat.
# `vegetarian_or_vegan` is the one it sets most often and was not listed here,
# so the authoritative signal was being ignored on exactly the recipes that
# need it: "Chickpea and egg burgers" carried it, hit the "burger" keyword,
# and was counted as a red-meat meal — which spent a meat allowance, forced a
# `meat_limit_relaxed` event, and had the reply apologise that Thursday's
# dinner "required red meat". It is chickpeas and egg.
VEG_TAGS = {"vegetarian", "vegan", "vegetarian_or_vegan"}

# kcal thresholds for the light/hearty day qualifier (whole-day estimate).
LIGHT_DAY_KCAL = 1400.0
HEARTY_DAY_KCAL = 2400.0


def _matches(text: str, keywords: set[str]) -> bool:
    return any(
        re.search(r"\b" + re.escape(k) + r"s?\b", text) for k in keywords
    )


# Keyword matching cannot tell "beef burger" from "chickpea burger", and the
# recipe tag that could is not always present. Two kinds of qualifier, with
# deliberately different reach.

# Says outright that the thing is not meat, so it may qualify ANY meat word:
# "vegan bacon", "mock duck", "tofu sausage".
_VEG_QUALIFIERS = (
    r"vegan|vegetarian|veggie|plant[-\s]?based|meat[-\s]?free|meatless|"
    r"mock|faux|imitation|soya?|tofu|tempeh|quorn|seitan"
)

# Names an ingredient, and may only qualify a word describing a SHAPE. A
# vegetable next to a form word is a vegetarian version of that form
# ("chickpea burger", "lentil meatballs"); a vegetable next to an animal is a
# dish containing that animal, and "mushroom chicken" must stay poultry.
_INGREDIENT_QUALIFIERS = (
    r"lentil|bean|chickpea|falafel|mushroom|nut|quinoa|aubergine|eggplant|"
    r"cauliflower|jackfruit"
)
_FORM_WORDS = sorted(
    {"burger", "hamburger", "sausage", "meatball", "meatloaf"} & MEAT_KEYWORDS
)

_ANY_MEAT = "|".join(re.escape(k) for k in sorted(MEAT_KEYWORDS))
_QUALIFIED_NOT_MEAT = re.compile(
    r"\b(?:(?:" + _VEG_QUALIFIERS + r")s?[-\s]+(?P<word>" + _ANY_MEAT + r")"
    r"|(?:" + _INGREDIENT_QUALIFIERS + r")s?[-\s]+(?P<form>"
    + "|".join(re.escape(k) for k in _FORM_WORDS) + r"))s?\b"
)
# "meat-free" on its own, where there is no following meat word to swallow.
_MEAT_FREE = re.compile(r"\bmeat[-\s]?free\b")

# Food-composition-database wording for a hen's egg: "eggs, chicken, whole,
# raw" and "chicken egg". Read literally it makes every shakshuka a poultry
# meal. The first pattern only fires when "chicken" is a bare comma-delimited
# fragment, so "2 eggs, chicken breast, flour" is still poultry.
_EGG_NOT_POULTRY = (
    re.compile(r"\beggs?\s*,\s*chicken\s*(?=,|$)"),
    re.compile(r"\bchicken\s+eggs?\b"),
)


def meat_text(title: str, ingredients: str) -> str:
    """Lowercased recipe text with the non-meat uses of meat words removed.

    Every claim this module makes about meat is a keyword match, and a keyword
    match cannot read a qualifier. Stripping the qualified uses before matching
    keeps one vocabulary rather than growing a second list of exceptions inside
    each caller.
    """
    text = f"{title} {ingredients}".lower()
    for pattern in _EGG_NOT_POULTRY:
        text = pattern.sub(" ", text)

    # A word qualified once is qualified throughout the recipe: "Veggie
    # burger" with "burger, bun" in the ingredients is one burger, described
    # twice. RecipeWrangler's ingredient strings are full of that shape — the
    # chickpea burger's read "Burger patties burger patty".
    #
    # So every occurrence of the qualified word goes, not just the qualified
    # one. Only that word: "Beef burger with a veggie sausage" loses "burger"
    # and "sausage" and still counts, on "beef".
    qualified = {
        match.group("word") or match.group("form")
        for match in _QUALIFIED_NOT_MEAT.finditer(text)
    }
    for word in qualified:
        text = re.sub(r"\b" + re.escape(word) + r"s?\b", " ", text)
    return _MEAT_FREE.sub(" ", text)


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

    Keyword hits are taken from ``meat_text``, not the raw string, so a
    qualified use ("chickpea burger", "meat-free chilli", the food-composition
    database's "eggs, chicken, whole, raw") does not spend a meat allowance.
    """
    if tags and VEG_TAGS & {str(t).lower() for t in tags}:
        return False
    keywords = MEAT_KEYWORDS if count_fish else (RED_MEAT_KEYWORDS | POULTRY_KEYWORDS)
    return _matches(meat_text(title, ingredients), keywords)


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
    if tags & VEG_TAGS:
        # `vegetarian_or_vegan` cannot say which, so it says the weaker of the
        # two. Calling a vegan dish vegetarian understates it; the reverse
        # would be a claim about a recipe nobody made.
        return "vegetarian"

    text = meat_text(
        " ".join(str(recipe.get(k) or "") for k in ("recipe_title", "title")),
        " ".join(str(recipe.get(k) or "") for k in ("recipe_ingredients", "ingredients")),
    )
    if _matches(text, FISH_KEYWORDS):
        return "fish"
    if _matches(text, RED_MEAT_KEYWORDS):
        return "red meat"
    if _matches(text, POULTRY_KEYWORDS):
        return "poultry"
    return "vegetarian"


def recipe_kcal(source: dict) -> Optional[float]:
    """Per-serving kcal from a recipe or candidate dict, None when unknown.

    **Zero is unknown, not a zero-calorie meal.** RecipeWrangler returns
    ``kcal: 0`` for recipes it has no composition data for, and every caller
    here read that as a real measurement: a week containing two of them
    reported ``meals_with_data: 21 of 21``, suppressed the "based on N meals"
    note, and put the missing calories into the weekly total as if the member
    would eat nothing. No recipe is 0 kcal.

    The canonical reader for the whole weekly planner —
    ``reward_logic.candidate_kcal`` delegates here — so the rule is stated
    once rather than in each place that happens to divide by it.
    """
    nutrition = source.get("nutrition") or {}
    for key in ("kcal", "calories"):
        value = nutrition.get(key)
        if isinstance(value, (int, float)) and value > 0:
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
         recipe_kcal(e.get("recipe") or {}))
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
