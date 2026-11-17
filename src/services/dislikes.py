"""What a member dislikes, in the words recipes actually use.

A dislike reaches RecipeWrangler as `exclude_ingredients`, which it applies as
a phrase match on ingredient names. "fish" matches "white fish fillets" and
never "tuna", so on 2026-11-17 a member who listed fish among eighteen
dislikes was served Tuna filo pies — while the plan's rule list, which only
restated the first five dislikes and marked each "satisfied" without looking,
said all was well.

Two things here. `expand_dislikes` turns a category into its members before
the search (fish -> tuna, salmon, cod ...), reusing the allergen screen's
groups where they exist. `dislike_hits` checks a finished dish against the
same words, so the plan verifier can measure dislikes instead of asserting
them.

Dislikes stay soft: a dish that slips through is reported, not withheld.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional

from services.candidates_client import ALLERGEN_SYNONYMS

_FISH = [
    "fish", "salmon", "tuna", "cod", "haddock", "hake", "pollock", "pollack",
    "trout", "sardine", "anchovy", "anchovies", "mackerel", "herring",
    "halibut", "sea bass", "seabass", "bream", "tilapia", "sole", "plaice",
    "swordfish", "monkfish", "snapper", "kipper", "pilchard", "whitebait",
    "catfish", "basa", "coley", "whiting", "sprat", "barramundi", "mahi",
    "bonito", "turbot", "perch",
]
_POULTRY = ["chicken", "turkey", "duck", "goose", "quail", "poussin"]
_RED_MEAT = ["beef", "pork", "lamb", "veal", "venison", "mutton", "goat",
             "mince", "steak"]
_CURED = ["ham", "bacon", "sausage", "salami", "chorizo", "prosciutto",
          "pancetta", "pepperoni"]

#: A dislike named as a category, and the words a recipe uses instead. The
#: allergen screen's groups first (nuts, shellfish, dairy, eggs, ...), so the
#: two lists cannot disagree about what shellfish is; then the categories
#: people dislike that are not allergens.
DISLIKE_GROUPS: dict[str, list[str]] = {
    **{name: list(words) for name, words in ALLERGEN_SYNONYMS.items()},
    "fish": _FISH,
    "seafood": _FISH + list(ALLERGEN_SYNONYMS.get("shellfish", [])),
    "poultry": _POULTRY,
    "red meat": _RED_MEAT,
    "meat": _RED_MEAT + _POULTRY + _CURED,
    "pork": ["pork", "ham", "bacon", "pancetta", "prosciutto", "chorizo"],
}


def _normalise(term: object) -> str:
    """'Almond-milk' and 'almond_milk' are 'almond milk'."""
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", str(term or ""))).strip().lower()


def _singular(word: str) -> Optional[str]:
    """'mushrooms' -> 'mushroom', 'chickpeas' -> 'chickpea'; None if no plural."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return None


def _group(term: str) -> list[str]:
    words = DISLIKE_GROUPS.get(term)
    if words is None:
        single = _singular(term)
        words = DISLIKE_GROUPS.get(single) if single else None
    return list(words or [])


def expand_dislikes(terms: Optional[Iterable[object]]) -> list[str]:
    """The dislikes plus everything they stand for, deduplicated, in order.

    Each dislike is kept as written (RecipeWrangler matched it before and
    must still), its spaced form if it had hyphens, and its singular; a
    category adds its members.
    """
    out: list[str] = []

    def add(word: str) -> None:
        if word and word not in out:
            out.append(word)

    for raw in terms or []:
        original = str(raw or "").strip().lower()
        term = _normalise(raw)
        if not term:
            continue
        add(original)
        add(term)
        single = _singular(term)
        if single:
            add(single)
        for word in _group(term):
            add(_normalise(word))
    return out


@lru_cache(maxsize=512)
def _pattern(word: str) -> re.Pattern:
    # Whole words with an optional plural. A prefix match would read "ham" in
    # "hamburger", "mince" in "mincemeat" and "cod" in "coddled egg"; missing a
    # rare "fishcake" is the cheaper mistake for a soft constraint.
    return re.compile(r"\b" + re.escape(word) + r"(?:e?s)?\b", re.IGNORECASE)


def dislike_hits(text: str, dislikes: Optional[Iterable[object]]) -> list[str]:
    """Which of these dislikes a dish's title or ingredients contain.

    Returns the dislikes as the member wrote them, each at most once.
    """
    if not text:
        return []
    hits: list[str] = []
    for raw in dislikes or []:
        name = str(raw or "").strip()
        term = _normalise(raw)
        if not term or name in hits:
            continue
        words = [term] + ([s] if (s := _singular(term)) else []) + [_normalise(w) for w in _group(term)]
        if any(_pattern(word).search(text) for word in words if word):
            hits.append(name)
    return hits
