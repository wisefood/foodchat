"""
Does this dish belong in this slot, at this size?

A member got this day:

    Breakfast   Easter breakfast baskets                     226 kcal
    Lunch       1 recipe = 4 dinners: Herby onion rice        899 kcal
    Dinner      Southwestern Stuffed Potatoes                859 kcal
                "Assembled directly from your constraints
                 (not ranked — grader unavailable)."

Every constraint was honoured and the day total was 1,984 kcal against a 2,000
target, so every check in the system passed. And it is not a day anyone would
plan: an 11%-of-the-day breakfast, a batch-cooking article served as lunch, and
45% of the calories in one meal.

The reason is in the last line. When the grader is unavailable the pipeline
takes the FIRST candidate per slot — RecipeWrangler's own deterministic order,
which is planning tier, then Nutri-Score, then curated source. That order is a
reasonable tiebreak and a poor decision, and it has two consequences the member
sees directly:

* **Nothing reasons about the pick.** The pool is constraint-correct, so the
  plan is legal, and no step asks whether it is any good.
* **The same plan comes back.** A fixed order plus "take the first" is a
  function with one output. Ask twice, refine once — same three dishes.

This is the reasoning step, and it costs nothing: every signal it uses is
already on the candidate. No extra fetch, no model call, so it can run on every
plan rather than only when there is budget.

    rank(slot, candidates, kcal_share=...)  → the same candidates, best first

Used to reorder the pool BEFORE anything selects from it — so the grader ranks
a pool whose head already fits the slot, and the fallback's "first" is a
reasoned first. It never removes a candidate: a pool it emptied would turn a
quality opinion into "no meals exist".
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# How a day's calories divide across meals, normalised over the slots actually
# being planned — the same rule `PlanSpec.kcal_split` applies across a meal's
# plates, one level up.
#
# Not a nutrition standard. It is the shape of an ordinary day, and its whole
# job is to notice a 226-calorie breakfast beside a 900-calorie lunch. A member
# who wants a big lunch is not wrong; a planner that cannot see the difference
# between that and an accident is.
SLOT_KCAL_WEIGHT: dict[str, float] = {
    "breakfast": 0.25,
    "brunch": 0.35,
    "lunch": 0.35,
    "dinner": 0.40,
    "snack": 0.10,
    "dessert": 0.10,
    "side": 0.15,
    "drink": 0.05,
}

# Points lost per unit of relative calorie miss, where 1.0 means double the
# slot's share. In the same units as the weekly preference scorer (favourite
# +5, variety −2 per shared token) so the two can be added without one
# silently deciding everything.
KCAL_MISS_WEIGHT = 6.0

# A miss smaller than this is not worth an opinion. Portions vary, stored
# figures are per-serving estimates, and a planner that reorders a pool over 8%
# is pretending to a precision it does not have.
KCAL_TOLERANCE = 0.20

# Nutri-Score, as a mild preference. Mild on purpose: it is one axis of one
# scoring system, and the member's own constraints have already decided what is
# eligible. This only sorts within what is already allowed.
NUTRI_BONUS = {"A": 1.5, "B": 0.75, "C": 0.0, "D": -0.75, "E": -1.5}

# Titles that are an article, not a dish.
#
# The corpus is scraped, and some of it is editorial: "1 recipe = 4 dinners:
# Herby onion rice", "5 ways to use up a cabbage", "Batch cook: chilli". Served
# as a meal these read as a mistake even when the underlying recipe is fine,
# because the title is telling the member about a magazine feature.
#
# A penalty, never a rejection — the recipe behind the headline is usually
# real, and this is a workaround for someone else's data. The proper fix is
# upstream, in what the catalogue stores as a title.
_ARTICLE_TITLE = re.compile(
    r"(^\s*\d+\s*(recipes?|ways?|dinners?|meals?|ideas?)\b"      # "5 ways to…"
    r"|\brecipe\s*=\s*\d+"                                       # "1 recipe = 4 dinners"
    r"|\b\d+\s*(recipes?|dinners?|meals?)\s*(from|in|out of)\b"   # "4 dinners from one…"
    r"|^\s*(batch[- ]cook|meal[- ]prep|how to)\b)",
    re.IGNORECASE,
)
ARTICLE_TITLE_PENALTY = 2.5

# Bigger than the whole Nutri-Score spread below, smaller than either penalty:
# curation beats a grade, and does not rescue a dish that is wrong for the slot.
CURATED_BONUS = 2.0

# How far a stable per-member tiebreak may move a candidate.
#
# Smaller than one Nutri-Score step (0.75), so it can never reorder across a
# grade, a curated source or a penalty — it only separates candidates this
# module considers EQUAL, which RecipeWrangler's ranking then settles by
# recipe id, identically for everybody.
#
# That identical settling is the report: "i see recipes fetched in order across
# plans, i havent seen diverse breakfast recipes". The corpus is read top-down
# by every member of every session, so the same handful of breakfasts leads the
# pool forever. A per-member offset would fix that by paging deeper, and paging
# deeper walks past the curated recipes this same session asked to see MORE of.
# Varying the order WITHIN a quality tier costs nothing and keeps the tier.
TIE_JITTER = 0.25


def _tiebreak(key: str, recipe_id: str) -> float:
    """A stable number in [-1, 1) from a member key and a recipe.

    `hashlib`, not `hash()`: Python randomises string hashing per process, so
    the same member would get a different plan from each pod — and a member who
    regenerates must get the same plan, or they cannot tell a regeneration from
    a bug.
    """
    if not key:
        return 0.0
    digest = hashlib.blake2b(
        f"{key}:{recipe_id}".encode(), digest_size=4,
    ).digest()
    return (int.from_bytes(digest, "big") / 0xFFFFFFFF) * 2 - 1


@dataclass
class Verdict:
    """One candidate, scored for one slot, with the reasons."""

    candidate: object
    score: float = 0.0
    #: Plain statements about what was measured — for the log and, when a plan
    #: is served unranked, for the reasoning the member reads.
    findings: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return str(getattr(self.candidate, "title", "") or "")


def slot_share(slot: str, slots: tuple[str, ...]) -> float:
    """This slot's share of the day, normalised over the slots being planned.

    Normalised rather than absolute so a two-meal day still adds to one day:
    a member who skips breakfast is not on a 75% diet.
    """
    weights = {s: SLOT_KCAL_WEIGHT.get(str(s).lower(), 0.25) for s in slots}
    total = sum(weights.values()) or 1.0
    return weights.get(slot, SLOT_KCAL_WEIGHT.get(str(slot).lower(), 0.25)) / total


def critique(
    slot: str,
    candidate,
    *,
    kcal_share: Optional[float] = None,
    variety_key: str = "",
) -> Verdict:
    """Score one candidate for one slot. Never raises.

    `kcal_share` is the calories this slot should carry. None when the member
    set no target — and then the calorie axis is silent rather than guessed,
    because a dish marked down against a budget nobody chose is being marked
    down for someone else's number.
    """
    verdict = Verdict(candidate=candidate)

    kcal = _kcal(candidate)
    if kcal_share and kcal is not None and kcal_share > 0:
        miss = abs(kcal - kcal_share) / kcal_share
        if miss > KCAL_TOLERANCE:
            verdict.score -= KCAL_MISS_WEIGHT * (miss - KCAL_TOLERANCE)
            direction = "over" if kcal > kcal_share else "under"
            verdict.findings.append(
                f"{int(kcal)} kcal is {int(miss * 100)}% {direction} what "
                f"{slot} should carry (~{int(kcal_share)} kcal)"
            )

    grade = str(getattr(candidate, "nutri_score", "") or "").strip().upper()
    if grade and grade[-1] in NUTRI_BONUS:
        verdict.score += NUTRI_BONUS[grade[-1]]

    title = verdict.title
    if title and _ARTICLE_TITLE.search(title):
        verdict.score -= ARTICLE_TITLE_PENALTY
        verdict.findings.append(f"{title!r} is written as an article, not a dish")

    # A recipe somebody wrote for this project outranks one that was scraped.
    #
    # RecipeWrangler already ranks curated corpora first, and this reordering
    # was undoing it: a Nutri-Score bonus of a point and a half is enough to
    # lift a scraped recipe over a living-lab one, and nothing here could tell
    # them apart — the candidate did not carry its source. The bonus is larger
    # than the Nutri-Score spread on purpose, so curation survives a grade
    # difference, and smaller than the article and portion penalties, so a
    # curated recipe that is wrong for the slot is still wrong for the slot.
    if _is_curated(getattr(candidate, "source", None)):
        verdict.score += CURATED_BONUS

    if variety_key:
        verdict.score += TIE_JITTER * _tiebreak(
            variety_key, str(getattr(candidate, "recipe_id", "") or ""),
        )

    return verdict


def _is_curated(source: Optional[str]) -> bool:
    """Whether this corpus is one of the curated ones, per the live manifest.

    Read from RecipeWrangler rather than listed here. Duplicating that registry
    is exactly how five sources sat unfilterable for a release, and a list here
    would fall behind the next living lab to join. An unreachable manifest
    returns False for everything, which is the behaviour this replaces.
    """
    slug = str(source or "").strip().lower()
    if not slug:
        return False
    try:
        from services.candidates_client import CANDIDATES

        rows = (CANDIDATES.vocabularies() or {}).get("sources") or []
    except Exception:  # noqa: BLE001 — vocabulary is best-effort everywhere
        return False
    return any(
        str(row.get("slug", "")).lower() == slug and row.get("curated")
        for row in rows
        if isinstance(row, dict)
    )


def rank(
    slot: str,
    candidates: list,
    *,
    kcal_share: Optional[float] = None,
) -> list:
    """The same candidates, best first. Nothing is dropped.

    A stable sort, so candidates the critic cannot separate keep
    RecipeWrangler's own order — planning tier, then Nutri-Score, then curated
    source, which is a good tiebreak and not a good decision.
    """
    if not candidates:
        return candidates
    verdicts = [critique(slot, c, kcal_share=kcal_share) for c in candidates]
    ordered = sorted(verdicts, key=lambda v: -v.score)
    moved = [v.title for v in ordered[:1] if v is not verdicts[0]]
    if moved:
        logger.info(
            "%s: preferred %r over %r — %s",
            slot, ordered[0].title, verdicts[0].title,
            "; ".join(verdicts[0].findings) or "better fit for the slot",
        )
    return [v.candidate for v in ordered]


def rank_pool(
    candidates: dict,
    *,
    kcal_target: Optional[float] = None,
    variety_key: str = "",
) -> tuple[dict, list[str]]:
    """Reorder every slot's pool, and say what was wrong with the old head.

    Returns `(pool, findings)`. The findings describe candidates that were
    demoted — which is exactly what a member is owed when the plan is served
    unranked, because then this ordering IS the reasoning.
    """
    slots = tuple(
        k[0] if isinstance(k, tuple) else k for k in candidates
    )
    ranked: dict = {}
    findings: list[str] = []
    for key, pool in candidates.items():
        # Role pools are keyed `(slot, role)`; a plain pool by slot.
        slot = key[0] if isinstance(key, tuple) else key
        share = (kcal_target * slot_share(slot, slots)) if kcal_target else None
        verdicts = [
            critique(slot, c, kcal_share=share, variety_key=variety_key)
            for c in (pool or [])
        ]
        ordered = sorted(verdicts, key=lambda v: -v.score)
        ranked[key] = [v.candidate for v in ordered]
        if ordered and verdicts and ordered[0] is not verdicts[0]:
            findings.extend(f"{slot}: {f}" for f in verdicts[0].findings[:1])
    return ranked, findings


def _kcal(candidate) -> Optional[float]:
    nutrition = getattr(candidate, "nutrition", None) or {}
    for key in ("kcal", "calories"):
        value = nutrition.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
