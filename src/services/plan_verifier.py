"""
Check the plan that came back against what was asked for.

Every constraint FoodChat reports has, until now, been reported from the
REQUEST: the ledger walks `profile["diet"]`, `profile["allergies"]` and the
plan parameters, and renders each one as satisfied because it was sent. That is
a statement about what was asked, dressed as a statement about what arrived.

The gap is not theoretical. Three cases already found:

* 26 of 37 gateway dietary groups were dropped before the fetch and reported as
  `hard / satisfied` anyway — a member selecting `peanut_free` saw a peanut-free
  guarantee with no filter behind it.
* `low-carb` / `high-protein` / `low-fat` were sent as diet filters and appear
  on ZERO recipes as diet tags, so those searches were guaranteed-empty and the
  fallback plan was still reported as honouring them.
* the weekly tracker's invented defaults (2000 kcal, meat_limit 3) were rendered
  with `source: "dietary preference"` for a member who set neither.

So: measure. This module takes a produced plan plus the constraints that were
requested, and reports what it can actually SEE on the plates that came back.

**What it can measure, and what it cannot, is the whole design.** The recipe
details endpoint returns macros, duration, tags, allergens and — as of the
matching RecipeWrangler change — the recipe's own `diet_tags`. It does NOT
return the four facet families (cuisines, moods, flavours, food groups): those
are Elasticsearch-only annotations that the Neo4j-backed details path never
sees. A facet is therefore reported as `applied by the search, not
independently verified`, which is true, rather than as `satisfied`, which would
be the same lie in a new place.

    verify(plan, requested, enrichment) -> VerificationReport

Nothing here calls a model. A verifier that asks an LLM whether the plan is
vegetarian has replaced a measurement with a second opinion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Statuses, in the order a reader should worry about them. The first three are
# measurements; `unverified` is the honest fourth, and it is deliberately not a
# synonym for either pass or fail.
PASSED = "passed"
FAILED = "failed"
PARTIAL = "partial"        # measured, and true of some plates but not all
UNVERIFIED = "unverified"  # requested and sent, but nothing here can check it
UNKNOWN = "unknown"        # checkable in principle, but the data is missing

# How far a day's calories may land from the target before it is a miss.
# Generous on purpose: per-serving macros are estimates over a corpus with
# uneven coverage, and a verifier that fails a plan for being 6% off is a
# verifier whose failures get ignored.
KCAL_TOLERANCE = 0.20

# Nutri-Score labels, best first, so "at least C" is a comparison and not a
# lookup table of every pair.
_NUTRI_ORDER = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


@dataclass
class Check:
    """One thing that was asked for, and what the plates actually show."""

    name: str
    status: str
    # Member-facing. Says what was measured, not what was requested.
    detail: str
    # Plates that fail this check, by recipe id — the input to a repair pass.
    offenders: tuple[str, ...] = ()
    # How many plates the check could actually see. A check over 2 of 21 plates
    # is not the same evidence as a check over 21, and a reader deserves both.
    observed: int = 0
    of: int = 0

    @property
    def is_blocking(self) -> bool:
        """Whether this failure should stop the plan being offered as-is.

        Only safety. A calorie target that lands 30% high is worth saying and
        worth repairing; it is not worth withholding a plan over. An allergen
        is.
        """
        return self.status == FAILED and self.name in {"allergens", "diet"}


@dataclass
class VerificationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAILED]

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.is_blocking]

    @property
    def offenders(self) -> list[str]:
        """Every plate that failed something, deduplicated, order preserved."""
        seen: list[str] = []
        for check in self.checks:
            for rid in check.offenders:
                if rid not in seen:
                    seen.append(rid)
        return seen

    def get(self, name: str) -> Optional[Check]:
        return next((c for c in self.checks if c.name == name), None)

    def as_ledger_rows(self) -> list[dict]:
        """The report in the shape the UI's constraint ledger already renders.

        Statuses map onto the four the UI knows. `unverified` becomes the
        `unsupported` chip — grey, informational — because the honest reading
        of both is "we did not prove this", and inventing a fifth chip state
        for a distinction the member does not have a use for is worse than
        reusing the one that already says it.
        """
        mapping = {
            PASSED: "satisfied",
            FAILED: "violated",
            PARTIAL: "relaxed",
            UNVERIFIED: "unsupported",
            UNKNOWN: "unsupported",
        }
        return [
            {
                "constraint": check.name,
                "type": "hard" if check.name in {"allergens", "diet"} else "soft",
                "status": mapping.get(check.status, "unsupported"),
                "source": "measured on the plan",
                "detail": check.detail,
            }
            for check in self.checks
        ]


def _plates(plan) -> list:
    """Every plate on every day, which is the only complete view of a plan."""
    return [
        plate
        for day in plan.day_plans
        for meal in day.meals
        for plate in meal.plates
        if getattr(plate, "recipe_id", None)
    ]


def _text(plate) -> str:
    return f"{getattr(plate, 'title', '')} {getattr(plate, 'ingredients', '')}".lower()


def verify(plan, requested: dict, enrichment: Optional[dict] = None) -> VerificationReport:
    """What the plan actually shows, against what was asked for.

    `requested` is the shape the fetch was built from — allergens, diet tags,
    claim tags, facets, kcal target, max minutes, min Nutri-Score, pantry,
    anchors. `enrichment` is `{recipe_id: RecipeEnrichment}`; without it most
    checks report `unknown` rather than passing, because a check that passes on
    absent data is worse than no check.
    """
    enrichment = enrichment or {}
    plates = _plates(plan)
    report = VerificationReport()

    for check in (
        _check_allergens(plates, requested, enrichment),
        _check_diet(plates, requested, enrichment),
        _check_claim_tags(plates, requested, enrichment),
        _check_kcal(plan, requested, enrichment),
        _check_time(plates, requested, enrichment),
        _check_nutri_score(plates, requested, enrichment),
        _check_pantry(plates, requested),
        _check_anchors(plates, requested),
        _check_facets(requested),
    ):
        if check is not None:
            report.checks.append(check)
    return report


# ── safety ────────────────────────────────────────────────────────────────

def _check_allergens(plates, requested, enrichment) -> Optional[Check]:
    """Both signals, because they fail in opposite directions.

    The corpus allergen list is authoritative when present and absent for
    plenty of recipes; the ingredient text is always present and matches
    loosely. Using either alone means either missing a labelled allergen or
    trusting a recipe nobody labelled.
    """
    allergens = [str(a).strip().lower() for a in (requested.get("allergens") or []) if a]
    if not allergens:
        return None

    offenders: list[str] = []
    hits: list[str] = []
    observed = 0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        if rich is not None:
            observed += 1
        labelled = {str(a).lower() for a in (getattr(rich, "allergens", None) or [])}
        text = _text(plate)
        for allergen in allergens:
            if allergen in labelled or allergen in text:
                offenders.append(plate.recipe_id)
                hits.append(f"{plate.title} ({allergen})")
                break

    if offenders:
        return Check(
            name="allergens", status=FAILED,
            detail="Found " + ", ".join(hits[:3]) + (
                f" and {len(hits) - 3} more" if len(hits) > 3 else ""),
            offenders=tuple(offenders), observed=observed, of=len(plates),
        )
    return Check(
        name="allergens", status=PASSED,
        detail=f"No {', '.join(allergens)} on any of the {len(plates)} dishes",
        observed=observed, of=len(plates),
    )


def _check_diet(plates, requested, enrichment) -> Optional[Check]:
    """Does every plate actually carry the diet tag that was filtered on.

    This is the check the vegetarian failure needed and nobody could run: the
    plan said `vegetarian: satisfied` because the word had been sent, and there
    was no way to look at what came back.

    A plate with no diet tags at all is not counted as a violation. The corpus
    annotates unevenly, and treating "unlabelled" as "not vegetarian" would
    condemn most of it — so the honest report is PARTIAL, naming how many
    plates could actually be checked.
    """
    diet = [str(d).strip().lower() for d in (requested.get("diet") or []) if d]
    if not diet:
        return None

    offenders: list[str] = []
    checked = 0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        tags = {str(t).lower() for t in (getattr(rich, "diet_tags", None) or [])}
        if not tags:
            continue
        checked += 1
        # `vegetarian_or_vegan` and `pescatarian_safe` satisfy their base tag:
        # the corpus uses both forms and a member asking for vegetarian is
        # served by either.
        if not any(
            want in tags or any(t.startswith(want) or want in t for t in tags)
            for want in diet
        ):
            offenders.append(plate.recipe_id)

    if offenders:
        return Check(
            name="diet", status=FAILED,
            detail=(
                f"{len(offenders)} of the {checked} labelled dishes are not "
                f"{', '.join(diet)}"
            ),
            offenders=tuple(offenders), observed=checked, of=len(plates),
        )
    if checked == 0:
        return Check(
            name="diet", status=UNKNOWN,
            detail=(
                f"Filtered for {', '.join(diet)}, but none of the {len(plates)} "
                "dishes carry dietary labels to check against"
            ),
            observed=0, of=len(plates),
        )
    if checked < len(plates):
        return Check(
            name="diet", status=PARTIAL,
            detail=(
                f"{checked} of {len(plates)} dishes are labelled "
                f"{', '.join(diet)}; the rest carry no dietary labels"
            ),
            observed=checked, of=len(plates),
        )
    return Check(
        name="diet", status=PASSED,
        detail=f"All {len(plates)} dishes are labelled {', '.join(diet)}",
        observed=checked, of=len(plates),
    )


# ── what was asked for ────────────────────────────────────────────────────

def _check_claim_tags(plates, requested, enrichment) -> Optional[Check]:
    """Claim tags ride the `tags` field, which the card does return.

    Reported per plate rather than as all-or-nothing: "high protein" over a
    week means the plan leans that way, not that every single dish qualifies,
    and a binary verdict here would fail almost every real plan.
    """
    wanted = [str(t).strip().lower() for t in (requested.get("tags") or []) if t]
    if not wanted:
        return None

    matching = 0
    observed = 0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        if rich is None:
            continue
        observed += 1
        tags = {str(t).lower() for t in (getattr(rich, "tags", None) or [])}
        if tags & set(wanted):
            matching += 1

    label = ", ".join(t.replace("_", " ") for t in wanted)
    if observed == 0:
        return Check(name="claims", status=UNKNOWN,
                     detail=f"Asked for {label}; no dish details available to check",
                     of=len(plates))
    if matching == 0:
        return Check(name="claims", status=FAILED,
                     detail=f"None of the {observed} dishes are tagged {label}",
                     observed=observed, of=len(plates))
    if matching < observed:
        return Check(name="claims", status=PARTIAL,
                     detail=f"{matching} of {observed} dishes are tagged {label}",
                     observed=observed, of=len(plates))
    return Check(name="claims", status=PASSED,
                 detail=f"All {observed} dishes are tagged {label}",
                 observed=observed, of=len(plates))


def _check_kcal(plan, requested, enrichment) -> Optional[Check]:
    """Per DAY, because that is the unit a calorie target is set in.

    A day whose plates carry no macros is not scored — reporting "0 kcal, 100%
    under target" for missing data would be a measurement of nothing presented
    as a measurement of the plan.
    """
    target = requested.get("kcal_target")
    if not target:
        return None
    try:
        target = float(target)
    except (TypeError, ValueError):
        return None
    if target <= 0:
        return None

    per_day: list[tuple[int, float, int, int]] = []
    for day in plan.day_plans:
        total = 0.0
        counted = 0
        plates = [p for meal in day.meals for p in meal.plates if p.recipe_id]
        for plate in plates:
            rich = enrichment.get(plate.recipe_id)
            kcal = getattr(rich, "kcal", None)
            if kcal is None:
                continue
            total += float(kcal)
            counted += 1
        if counted:
            per_day.append((day.day, total, counted, len(plates)))

    if not per_day:
        return Check(name="calories", status=UNKNOWN,
                     detail=f"Target {int(target)} kcal a day; no dish has "
                            "nutrition data to add up")

    low, high = target * (1 - KCAL_TOLERANCE), target * (1 + KCAL_TOLERANCE)
    misses = [(d, total) for d, total, _, _ in per_day if not (low <= total <= high)]
    complete = all(counted == total_plates for _, _, counted, total_plates in per_day)
    caveat = "" if complete else " (some dishes have no nutrition data, so this is a floor)"

    if not misses:
        average = sum(t for _, t, _, _ in per_day) / len(per_day)
        return Check(
            name="calories", status=PASSED,
            detail=f"Averaging {int(average)} kcal a day against a {int(target)} "
                   f"target{caveat}",
            observed=len(per_day), of=len(plan.day_plans),
        )
    worst = max(misses, key=lambda item: abs(item[1] - target))
    direction = "over" if worst[1] > target else "under"
    return Check(
        name="calories", status=FAILED,
        detail=(
            f"{len(misses)} of {len(per_day)} days miss the {int(target)} kcal "
            f"target — day {worst[0]} is {int(abs(worst[1] - target))} kcal "
            f"{direction}{caveat}"
        ),
        observed=len(per_day), of=len(plan.day_plans),
    )


def _check_time(plates, requested, enrichment) -> Optional[Check]:
    limit = requested.get("max_minutes")
    if not limit:
        return None
    try:
        limit = float(limit)
    except (TypeError, ValueError):
        return None

    offenders: list[str] = []
    observed = 0
    worst = 0.0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        duration = getattr(rich, "duration", None)
        if duration is None:
            continue
        observed += 1
        if float(duration) > limit:
            offenders.append(plate.recipe_id)
            worst = max(worst, float(duration))

    if observed == 0:
        return Check(name="cooking time", status=UNKNOWN,
                     detail=f"Asked for {int(limit)} minutes or less; no dish "
                            "records a cooking time", of=len(plates))
    if offenders:
        return Check(
            name="cooking time", status=FAILED,
            detail=f"{len(offenders)} of {observed} dishes run over "
                   f"{int(limit)} minutes — the longest is {int(worst)}",
            offenders=tuple(offenders), observed=observed, of=len(plates),
        )
    return Check(name="cooking time", status=PASSED,
                 detail=f"All {observed} timed dishes are {int(limit)} minutes or less",
                 observed=observed, of=len(plates))


def _check_nutri_score(plates, requested, enrichment) -> Optional[Check]:
    minimum = str(requested.get("min_nutri_score") or "").strip().lower()
    if minimum not in _NUTRI_ORDER:
        return None
    ceiling = _NUTRI_ORDER[minimum]

    offenders: list[str] = []
    observed = 0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        label = str(getattr(rich, "nutri_score_label", None) or "").strip().lower()
        if label not in _NUTRI_ORDER:
            continue
        observed += 1
        if _NUTRI_ORDER[label] > ceiling:
            offenders.append(plate.recipe_id)

    if observed == 0:
        return Check(name="nutri-score", status=UNKNOWN,
                     detail=f"Asked for {minimum.upper()} or better; no dish "
                            "has a Nutri-Score", of=len(plates))
    if offenders:
        return Check(
            name="nutri-score", status=FAILED,
            detail=f"{len(offenders)} of {observed} scored dishes are below "
                   f"{minimum.upper()}",
            offenders=tuple(offenders), observed=observed, of=len(plates),
        )
    return Check(name="nutri-score", status=PASSED,
                 detail=f"All {observed} scored dishes are {minimum.upper()} or better",
                 observed=observed, of=len(plates))


def _check_pantry(plates, requested) -> Optional[Check]:
    """Measured on the ingredient text, which is always present.

    Partial is the expected outcome and is not a failure: a pantry is a
    preference for using things up, not a requirement that every dish use them.
    """
    pantry = [str(p).strip().lower() for p in (requested.get("pantry") or []) if p]
    if not pantry:
        return None

    from services.pantry_service import matched_items

    used: list[str] = []
    plates_using = 0
    for plate in plates:
        matches = matched_items(_text(plate), pantry)
        if matches:
            plates_using += 1
            for item in matches:
                if item not in used:
                    used.append(item)

    unused = [item for item in pantry if item not in used]
    if not used:
        return Check(name="pantry", status=FAILED,
                     detail="None of " + ", ".join(pantry) + " are used by this plan",
                     of=len(plates))
    if unused:
        return Check(
            name="pantry", status=PARTIAL,
            detail=f"Uses {', '.join(used)} across {plates_using} dishes; "
                   f"{', '.join(unused)} went unused",
            observed=plates_using, of=len(plates),
        )
    return Check(name="pantry", status=PASSED,
                 detail=f"Uses all of {', '.join(used)} across {plates_using} dishes",
                 observed=plates_using, of=len(plates))


def _check_anchors(plates, requested) -> Optional[Check]:
    """A dish the member named must actually be on the plan.

    The apple-pie failure in a check: a named dish was served and then
    regenerated away a turn later, with nothing reporting that it had gone.
    """
    anchors = {
        str(slot): str(rid)
        for slot, rid in (requested.get("anchors") or {}).items()
        if rid
    }
    if not anchors:
        return None

    present = {p.recipe_id for p in plates}
    missing = [slot for slot, rid in anchors.items() if rid not in present]
    if missing:
        return Check(
            name="your picks", status=FAILED,
            detail=f"{len(missing)} dish you asked for is missing: "
                   + ", ".join(sorted(missing)),
            observed=len(anchors) - len(missing), of=len(anchors),
        )
    return Check(name="your picks", status=PASSED,
                 detail=f"All {len(anchors)} dishes you asked for are on the plan",
                 observed=len(anchors), of=len(anchors))


def _check_facets(requested) -> Optional[Check]:
    """Reported honestly as unverified, because it cannot be measured here.

    The four facet families are Elasticsearch-only annotations written by an
    annotation pass; the recipe details endpoint reads Neo4j and never sees
    them. So the search filtered on them and the result carries no evidence
    either way.

    Saying so is the point. `satisfied` here would be the same mistake the
    dietary-group ledger made — a statement about the request wearing the
    clothes of a statement about the result.
    """
    families = ("cuisines", "moods", "flavor_profiles", "food_groups")
    values = [
        value
        for family in families
        for value in (requested.get(family) or [])
        if value
    ]
    if not values:
        return None
    return Check(
        name="taste preferences", status=UNVERIFIED,
        detail=(
            "Searched for " + ", ".join(str(v).replace("_", " ") for v in values)
            + " — the recipe search filtered on these, but the dish details "
              "carry no annotation to check them against"
        ),
    )


def describe(report: VerificationReport) -> str:
    """One line for the log and for a grounded reply. Never empty."""
    if not report.checks:
        return "nothing to verify"
    return " · ".join(f"{c.name}: {c.status}" for c in report.checks)
