"""
Building a MEAL out of plates, and judging what got built.

Multi-plate planning had a renderer, a request format and a data model, and no
producer. Two different reasons, one per path:

* **Weekly** fetched one pool per meal slot and picked one dish from it. There
  was no notion of a plate, so a dinner could never be a main and a salad.
* **The structured path** did ask for one entry per plate — and then took
  RecipeWrangler's first recipe for each. Assembly was "delegated wholly to
  `plan_meals`", in that module's own words. So nothing ever asked whether the
  main and the side went together: lasagne with a side of macaroni salad is two
  plates of pasta, and no step in the system was in a position to notice.

This module is the missing step. It composes — chooses which recipe fills which
plate, considering the others — and it judges what it composed.

    pools    = role_pools(profile, spec, ...)      one pool per plate
    meal     = compose(slot, roles, pools, ...)    plates that go together
    ranked   = judge(query, options)               a model, when it earns its keep

**Deterministic first, model second, and the split is not arbitrary.** Three of
the four things that make a set of plates a meal can be MEASURED: whether two
plates are the same dish, whether they repeat an ingredient, whether they add up
to roughly the meal's share of the day. Those are arithmetic and they run
always. What cannot be measured from the data FoodChat holds is whether a dish
*suits* another one — a corpus cuisine annotation does not survive into this
envelope in any shape this service reads, and texture and richness are not
annotated at all. That is a judgement, so it goes to a judgement, and it goes
there ONCE for the whole plan rather than once per meal.

**The judge is optional by construction.** It ranks compositions the
deterministic pass already produced and already ordered, so a model that is
down, slow, or out of budget costs the plan its polish and never its existence.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Recipes fetched per plate. There is no composition to make without a choice —
# one candidate per plate is the behaviour this module exists to replace — and
# the count multiplies the payload by the number of plates, so it stays small.
POOL_PER_PLATE = 4

# Combinations scored per meal. 4 candidates over a 3-plate meal is 64, which is
# fine; the cap is here because `itertools.product` over a spec nobody sanity
# checked is how a planner stops responding. Same reasoning as the daily
# grader's own scan limit.
_COMBO_SCAN_LIMIT = 256

# Compositions handed to the judge per meal. Three is enough for a real choice
# and keeps one prompt readable for a whole week of meals.
JUDGE_OPTIONS = 3

# ── the rubric ──────────────────────────────────────────────────────────────
#
# Weights are in the same units as the weekly preference scorer (favourite
# +5.0, variety −2.0 per shared title token) so a composition score can be
# added to it without one silently dominating the other.

# Per ingredient shared between two plates of the same meal. A main and a side
# that both lean on potatoes is the single most common way a two-plate meal
# reads as a mistake, and it is exactly measurable.
OVERLAP_PENALTY = 1.5

# Per unit of relative calorie miss, where 1.0 means "double the plate's share
# of the meal". A side that is heavier than the main it accompanies is not a
# side; this is what `PlanSpec.kcal_split` was written for and it has never had
# a consumer.
KCAL_MISS_WEIGHT = 4.0

# A plate whose course annotations do not include any course type its role
# accepts. Not a rejection: the corpus's `dish_types` are incomplete, so
# absence of evidence is common and treating it as a failure would empty
# plates that are perfectly fine. A penalty says "prefer the one we can
# confirm" without ruling out the one we cannot.
ROLE_MISMATCH_PENALTY = 2.0

# Ingredients in every kitchen. Two plates sharing olive oil have nothing in
# common worth penalising, and counting staples would swamp the signal from the
# ingredients that actually make two dishes the same dish.
STAPLES: frozenset = frozenset({
    "salt", "pepper", "water", "sugar", "flour", "oil", "olive oil", "butter",
    "vinegar", "olive", "stock", "broth", "sauce", "soy sauce", "honey",
    "mustard", "cumin", "paprika", "oregano", "thyme", "cinnamon", "vanilla",
    "baking powder", "cornflour", "cornstarch", "lemon", "lemon juice",
    "juice", "cloves", "seeds", "chilli", "chili", "ginger", "parsley",
    "coriander", "basil", "extract", "yeast", "breadcrumbs", "garlic",
    "onion", "onions", "black pepper", "sea salt", "milk", "eggs", "egg",
})


@dataclass
class Plate:
    """One recipe in one role, with the reasons it was chosen."""

    role: str
    candidate: object
    # Why this plate scored the way it did, for the ledger and the reply.
    notes: list[str] = field(default_factory=list)

    @property
    def recipe_id(self) -> str:
        return str(getattr(self.candidate, "recipe_id", "") or "")

    @property
    def title(self) -> str:
        return str(getattr(self.candidate, "title", "") or "")


@dataclass
class Composition:
    """One way of filling a meal's plates, and how well it holds together."""

    slot: str
    plates: list[Plate]
    score: float = 0.0
    # Measured, not asserted: what the arithmetic actually found. Rendered into
    # the ledger, so it has to describe the plan that exists.
    findings: list[str] = field(default_factory=list)

    @property
    def recipe_ids(self) -> list[str]:
        return [p.recipe_id for p in self.plates]

    def describe(self) -> str:
        roles = ", ".join(f"{p.role}: {p.title}" for p in self.plates)
        return f"{self.slot} — {roles}"


# ── fetching one pool per plate ─────────────────────────────────────────────

def role_pools(
    profile: dict,
    spec,
    *,
    exclude_recipe_ids: Optional[list[str]] = None,
    boost_ids: Optional[list[str]] = None,
    per_plate: int = POOL_PER_PLATE,
    days: Optional[int] = None,
) -> tuple[dict[int, dict[tuple[str, str], list]], list[str]]:
    """`({day: {(slot, role): [CandidateRecipe]}}, relaxation notes)`.

    `({}, [])` on failure.

    One `plan_meals` call for the whole plan, because that endpoint already
    takes one request entry per plate with its own `course_types` — the
    role-scoped fetch this was said to be blocked on has been available the
    whole time. What was missing was asking for more than one candidate per
    plate, and reading the answer back without collapsing the roles.

    The relaxation lines come back with the pools rather than only reaching a
    log: they are RecipeWrangler's own account of a preference it had to drop
    to fill a plate, and the member is entitled to hear it. Losing them here
    would be the third time in this codebase that a service explained itself
    and nobody passed it on.

    An empty pool map is what every caller already handles as "no pool": weekly
    raises a typed per-slot error the service turns into a sentence, and the
    structured path returns None.
    """
    from services import intent_facets, plan_parameters
    from services.candidates_client import CANDIDATES, effective_diet, screening_allergens
    from services.plan_client import PLANNER

    cuisines, _ = CANDIDATES.split_cuisines(profile.get("food_likes") or [])
    request_spec = spec if days is None else _spec_for_days(spec, days)

    try:
        envelope = PLANNER.plan_meals(
            spec=request_spec,
            count_per_slot=max(1, int(per_plate)),
            allergens=screening_allergens(profile),
            diet=effective_diet(profile),
            **intent_facets.facet_kwargs(profile, cuisines),
            exclude_ingredients=profile.get("food_dislikes") or [],
            exclude_recipe_ids=list(exclude_recipe_ids or []),
            favorite_recipe_ids=list(boost_ids or []),
            max_minutes=plan_parameters.max_duration_minutes(
                profile.get("plan_parameters") or {}
            ),
            min_nutri_score=profile.get("min_nutri_score"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Role-scoped pool fetch failed: %s", exc)
        return {}, []

    notes = PLANNER.describe_relaxations(envelope)
    if notes:
        logger.info("Pool relaxations: %s", "; ".join(notes))
    pools = PLANNER.to_role_pools(
        envelope, request_spec, allergens=screening_allergens(profile),
    )
    return pools, notes


def _spec_for_days(spec, days: int):
    """The same shape over a different horizon.

    Weekly fetches a day at a time — its pools exclude everything already
    committed, which is what stops a week repeating a recipe — so it needs the
    spec's plate structure with `num_days` of one.
    """
    from dataclasses import replace

    return replace(spec, num_days=max(1, int(days)))


# ── composing ───────────────────────────────────────────────────────────────

def compose(
    slot: str,
    roles: tuple[str, ...],
    pools: dict[tuple[str, str], list],
    *,
    kcal_split: Optional[dict[str, float]] = None,
    meal_kcal_target: Optional[float] = None,
    exclude_ids: Optional[set] = None,
    enrichment: Optional[dict] = None,
    limit: int = JUDGE_OPTIONS,
) -> list[Composition]:
    """The best ways to fill this meal's plates, best first.

    Returns a list so a caller can hand the top few to the judge; the first
    entry is already a usable answer and is what gets used when the judge does
    not run.

    Empty when any plate has no candidates. A meal short a plate is reported by
    the caller, never quietly served as a smaller meal — the member asked for a
    main AND a salad.
    """
    exclude = set(exclude_ids or ())
    per_role: list[list] = []
    for role in roles:
        options = [
            c for c in (pools.get((slot, role)) or [])
            if str(getattr(c, "recipe_id", "") or "") not in exclude
        ]
        if not options:
            logger.info("No candidates for %s %s", slot, role)
            return []
        per_role.append(options)

    combos = itertools.islice(itertools.product(*per_role), _COMBO_SCAN_LIMIT)
    scored: list[Composition] = []
    for combo in combos:
        ids = [str(getattr(c, "recipe_id", "") or "") for c in combo]
        # The same dish twice is not a meal with two plates.
        if len(set(ids)) != len(ids):
            continue
        composition = Composition(
            slot=slot,
            plates=[Plate(role=role, candidate=c) for role, c in zip(roles, combo)],
        )
        _score(composition, kcal_split, meal_kcal_target, enrichment or {})
        scored.append(composition)

    if not scored:
        return []
    # Stable: ties keep RecipeWrangler's own order, which is planning tier then
    # Nutri-Score then curated source — a perfectly reasonable tiebreak and one
    # this module has no business overriding with a coin flip.
    scored.sort(key=lambda c: -c.score)
    return scored[:max(1, int(limit))]


def _score(
    composition: Composition,
    kcal_split: Optional[dict],
    meal_kcal_target: Optional[float],
    enrichment: dict,
) -> None:
    """Fill in `score` and `findings`. Only what can be measured."""
    plates = composition.plates
    score = 0.0

    # 1. Ingredient overlap between plates.
    if len(plates) > 1:
        from services.plan_quality import extract_ingredient_names

        sets = [
            {
                name for name in extract_ingredient_names(
                    str(getattr(p.candidate, "ingredients", "") or "")
                )
                if name not in STAPLES
            }
            for p in plates
        ]
        shared: set = set()
        for left, right in itertools.combinations(range(len(sets)), 2):
            shared |= sets[left] & sets[right]
        if shared:
            score -= OVERLAP_PENALTY * len(shared)
            composition.findings.append(
                "plates share " + ", ".join(sorted(shared)[:4])
            )

    # 2. Calorie fit, per plate, against the share the spec assigns its role.
    #
    # Skipped entirely without a target — a plate judged against a budget
    # nobody set would be penalised for a number the member never chose.
    if meal_kcal_target and kcal_split:
        misses = []
        for plate in plates:
            kcal = _kcal(plate.candidate, enrichment)
            share = kcal_split.get(plate.role)
            if kcal is None or not share:
                continue
            want = meal_kcal_target * share
            if want <= 0:
                continue
            miss = abs(kcal - want) / want
            score -= KCAL_MISS_WEIGHT * miss
            if miss > 0.5:
                misses.append(
                    f"{plate.role} is {int(kcal)} kcal against a "
                    f"{int(want)} kcal share"
                )
        composition.findings.extend(misses)

    # 3. Does each plate look like the course it is filling?
    from models.plan_spec import ROLE_COURSE_TYPES

    for plate in plates:
        wanted = set(ROLE_COURSE_TYPES.get(plate.role, ()))
        rich = enrichment.get(plate.recipe_id)
        types = {str(t).lower() for t in (getattr(rich, "dish_types", None) or [])}
        if not wanted or not types:
            # No annotation to check against. Not a finding: silence in the
            # corpus is not evidence of a mismatch, and reporting it as one
            # would fill the ledger with rows about missing data.
            continue
        if not (types & wanted):
            score -= ROLE_MISMATCH_PENALTY
            plate.notes.append(f"not annotated as a {plate.role}")
            composition.findings.append(
                f"{plate.title} is not annotated as a {plate.role}"
            )

    composition.score = score


def _kcal(candidate, enrichment: dict) -> Optional[float]:
    """Per-serving kcal from the candidate, or the enrichment, or None."""
    nutrition = getattr(candidate, "nutrition", None) or {}
    for key in ("kcal", "calories"):
        value = nutrition.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    rich = enrichment.get(str(getattr(candidate, "recipe_id", "") or ""))
    kcal = getattr(rich, "kcal", None)
    return float(kcal) if isinstance(kcal, (int, float)) else None


# ── judging ─────────────────────────────────────────────────────────────────

class _Judge:
    """Lazily built and held on the class, so importing costs no client.

    Same shape as the other lazy agents in this package, and for the same
    reason: an instance built without `__init__` — which the test suite does
    routinely — must not be the difference between routing and an
    `AttributeError`.
    """

    _agent = None

    @classmethod
    def get(cls):
        if cls._agent is None:
            from agents import MealJudge

            cls._agent = MealJudge()
        return cls._agent


def offer_text(composition: Composition) -> str:
    """One composition as a line the judge can read.

    Titles and the first of each plate's ingredients — enough to tell whether a
    slaw cuts a rich pork and no more. The whole ingredient blob for three
    options across seven meals is a prompt nobody reads carefully, model
    included.
    """
    parts = []
    for plate in composition.plates:
        raw = str(getattr(plate.candidate, "ingredients", "") or "")
        from services.plan_quality import extract_ingredient_names

        items = [n for n in extract_ingredient_names(raw) if n not in STAPLES][:5]
        head = f"{plate.role}: {plate.title}"
        parts.append(f"{head} ({', '.join(items)})" if items else head)
    return " | ".join(parts)


def judge(
    query: str,
    options: dict[str, list[Composition]],
    *,
    agent=None,
) -> dict[str, Composition]:
    """Pick one composition per meal. Never raises.

    `options` is `{label: [Composition, …]}` best-measured-first. The returned
    mapping always has an entry for every label — the measured winner when the
    judge did not rule, so a caller never has to ask whether it ran.

    Skipped without spending anything when the turn is running late. Composition
    is a judgement about polish: a plan whose side was chosen by arithmetic is a
    good plan, and one that never arrives is not.
    """
    chosen = {label: entries[0] for label, entries in options.items() if entries}
    # Only meals that are actually served as more than one plate, and only
    # where there is something to choose between.
    #
    # A single-plate meal inside a multi-plate spec has a pool too, but picking
    # among four mains is RANKING, not composition — the prompt asks whether
    # these dishes belong on a table together, and for one dish that is not a
    # question. Offering them made the judge answer "the slaw cuts it" about a
    # lunch with no slaw in it.
    offerable = {
        label: entries for label, entries in options.items()
        if len(entries) > 1 and len(entries[0].plates) > 1
    }
    if not offerable:
        return chosen

    from services import turn_budget

    if turn_budget.skip("meal composition", turn_budget.COST_METRICS):
        return chosen

    offers = [
        {"meal": label, "options": [offer_text(c) for c in entries]}
        for label, entries in offerable.items()
    ]
    try:
        verdicts = (agent or _Judge.get()).choose(query, offers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Meal judging failed, keeping the measured order: %s", exc)
        return chosen

    for label, (index, reason) in (verdicts or {}).items():
        entries = offerable.get(label)
        if not entries or not 0 <= index < len(entries):
            continue
        picked = entries[index]
        if reason:
            # Recorded on the composition, so the ledger can say a model chose
            # this and why — rather than presenting a judgement as a measurement.
            picked.findings.append(f"chosen for the table: {reason}")
        chosen[label] = picked
    return chosen


def label_for(day: Optional[int], slot: str) -> str:
    """The label a meal is offered and matched under.

    Includes the day because a week has seven dinners and they are different
    questions. Matching is by this string, so it has to be unique within one
    call and stable across the two places it is built.
    """
    return f"day {int(day)} {slot}" if day else str(slot)
