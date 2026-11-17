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

from services.candidates_client import allergen_conflict, allergen_label_match, display_allergen

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
    # The same outcome as a key and its numbers, for a reader that says it in
    # the member's language. `detail` is English and stays the fallback: the
    # Hungarian and Slovenian living labs read every one of these sentences.
    code: str = ""
    params: dict = field(default_factory=dict)
    # Every finding, when `params` can only carry the first few (dislikes).
    found: tuple = ()

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
        rows = []
        for check in self.checks:
            row = {
                "constraint": check.name,
                "type": "hard" if check.name in {"allergens", "diet"} else "soft",
                "status": mapping.get(check.status, "unsupported"),
                "source": "measured on the plan",
                "detail": check.detail,
            }
            if check.code:
                row["code"] = check.code
                row["params"] = dict(check.params)
            rows.append(row)
        return rows


#: Marks the dislike rows `merge` changed or added, so a second merge (after a
#: repair changed the plates) starts from the profile's rows again.
_MEASURED_DISLIKE = "dislikes"


def merge(rows, report: "VerificationReport") -> list[dict]:
    """The plan's rule list with this report's measurements in it.

    Measured rows are appended, as before. Dislikes are folded in rather than
    appended: the profile's "avoiding X" rows (household attribution, a name
    code) said "satisfied" without looking; now one that a dish breaks says
    "violated" and names the dish, a disliked food not among the rows shown
    gets a row of its own, and the summary row stays only when nothing was
    found — otherwise the member would read the same finding twice.

    Safe to call again on its own output: rows it touched are reset first.
    """
    out: list[dict] = []
    for row in rows or []:
        if row.get("measured") == _MEASURED_DISLIKE:
            if row.get("added"):
                continue
            row = {k: v for k, v in row.items() if k not in ("measured", "code", "params", "detail")}
            row["status"] = "satisfied"
        out.append(dict(row))

    check = report.get("dislikes") if report else None
    measured = report.as_ledger_rows() if report else []
    if check is None:
        return out + measured

    dishes_for: dict[str, list[str]] = {}
    written: dict[str, str] = {}
    for hit in check.found or ():
        key = str(hit.get("dislike") or "").strip().lower()
        if not key:
            continue
        written.setdefault(key, str(hit.get("dislike")))
        dishes = dishes_for.setdefault(key, [])
        if hit.get("dish") and hit["dish"] not in dishes:
            dishes.append(hit["dish"])

    def found_on(row: dict, dishes: list[str]) -> None:
        row.update(
            status="violated", measured=_MEASURED_DISLIKE,
            detail="on the plan: " + ", ".join(dishes),
            code="dislike.found", params={"dishes": list(dishes)},
        )

    shown = set()
    for row in out:
        if row.get("name_code") != "dislike":
            continue
        key = str((row.get("name_params") or {}).get("item") or "").strip().lower()
        shown.add(key)
        if key in dishes_for:
            found_on(row, dishes_for[key])
    for key, dishes in dishes_for.items():
        if key in shown:
            continue
        row = {
            "constraint": f"avoiding {written[key]}", "type": "soft",
            "source": "preferences", "members": [],
            "name_code": "dislike", "name_params": {"item": written[key]},
            "added": True,
        }
        found_on(row, dishes)
        out.append(row)

    if check.status == FAILED:
        measured = [r for r in measured if r.get("constraint") != "dislikes"]
    return out + measured


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
        _check_dislikes(plates, requested),
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
    found: list[dict] = []
    observed = 0
    for plate in plates:
        rich = enrichment.get(plate.recipe_id)
        if rich is not None:
            observed += 1
        labelled = {str(a).lower() for a in (getattr(rich, "allergens", None) or [])}
        text = _text(plate)
        for allergen in allergens:
            # Through the alias table on both sides: the corpus labels the
            # canonical identifier (`tree_nut`) while a profile may still say
            # "nuts", and the ingredient text names almonds, not either.
            if allergen_label_match(allergen, labelled) or allergen_conflict(text, [allergen]):
                offenders.append(plate.recipe_id)
                hits.append(f"{plate.title} ({display_allergen(allergen)})")
                found.append({"dish": plate.title, "allergen": allergen})
                break

    if offenders:
        return Check(
            name="allergens", status=FAILED,
            detail="Found " + ", ".join(hits[:3]) + (
                f" and {len(hits) - 3} more" if len(hits) > 3 else ""),
            offenders=tuple(offenders), observed=observed, of=len(plates),
            code="allergens.found",
            params={"found": found[:3], "more": max(len(found) - 3, 0)},
        )
    return Check(
        name="allergens", status=PASSED,
        detail=f"No {', '.join(display_allergen(a) for a in allergens)} on any of the {len(plates)} dishes",
        observed=observed, of=len(plates),
        code="allergens.clear", params={"allergens": allergens, "dishes": len(plates)},
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
            code="diet.offenders",
            params={"count": len(offenders), "checked": checked, "diets": diet},
        )
    if checked == 0:
        return Check(
            name="diet", status=UNKNOWN,
            detail=(
                f"Filtered for {', '.join(diet)}, but none of the {len(plates)} "
                "dishes carry dietary labels to check against"
            ),
            observed=0, of=len(plates),
            code="diet.unlabelled", params={"diets": diet, "dishes": len(plates)},
        )
    if checked < len(plates):
        return Check(
            name="diet", status=PARTIAL,
            detail=(
                f"{checked} of {len(plates)} dishes are labelled "
                f"{', '.join(diet)}; the rest carry no dietary labels"
            ),
            observed=checked, of=len(plates),
            code="diet.partial",
            params={"checked": checked, "dishes": len(plates), "diets": diet},
        )
    return Check(
        name="diet", status=PASSED,
        detail=f"All {len(plates)} dishes are labelled {', '.join(diet)}",
        observed=checked, of=len(plates),
        code="diet.all", params={"dishes": len(plates), "diets": diet},
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
                     of=len(plates),
                     code="claims.unknown", params={"tags": wanted})
    if matching == 0:
        return Check(name="claims", status=FAILED,
                     detail=f"None of the {observed} dishes are tagged {label}",
                     observed=observed, of=len(plates),
                     code="claims.none", params={"observed": observed, "tags": wanted})
    if matching < observed:
        return Check(name="claims", status=PARTIAL,
                     detail=f"{matching} of {observed} dishes are tagged {label}",
                     observed=observed, of=len(plates),
                     code="claims.some",
                     params={"matching": matching, "observed": observed, "tags": wanted})
    return Check(name="claims", status=PASSED,
                 detail=f"All {observed} dishes are tagged {label}",
                 observed=observed, of=len(plates),
                 code="claims.all", params={"observed": observed, "tags": wanted})


def _measured(plate, enrichment, attribute: str, nutrition_key: str):
    """A per-recipe figure, from the fresh details call or from the plate itself.

    Enrichment first: it is a live fetch, so it is the more current of the two.
    The plate's own `nutrition` is the fallback, and it exists more often than
    it used to — the planning envelope carries macros with every card, and the
    composer now keeps them on the plate rather than throwing them away and
    re-fetching the same numbers.

    Without this the verifier reported `calories: unknown` for a plan that was
    carrying its calories, which is the exact failure the ledger exists to
    prevent: a measurement is missing, so nothing is said, so the member reads
    silence as agreement.
    """
    value = getattr(enrichment.get(plate.recipe_id), attribute, None)
    if value is not None:
        return value
    return (getattr(plate, "nutrition", None) or {}).get(nutrition_key)


def _heaviest_plate(plan, enrichment, day_number: int):
    """`(title, kcal)` of the biggest single plate on a day, or None.

    What makes an overshoot actionable. "Day 1 is 470 kcal over" is a fact the
    member can do nothing with; naming the plate carrying most of it is the
    difference between a report and a suggestion.
    """
    best = None
    for day in plan.day_plans:
        if int(getattr(day, "day", 1) or 1) != day_number:
            continue
        for meal in day.meals:
            for plate in meal.plates:
                if not plate.recipe_id:
                    continue
                kcal = _measured(plate, enrichment, "kcal", "kcal")
                if not isinstance(kcal, (int, float)):
                    continue
                if best is None or float(kcal) > best[1]:
                    best = (str(getattr(plate, "title", "") or ""), float(kcal))
    return best


def _check_kcal(plan, requested, enrichment) -> Optional[Check]:
    """Per DAY, because that is the unit a calorie target is set in.

    A day whose plates carry no macros is not scored — reporting "0 kcal, 100%
    under target" for missing data would be a measurement of nothing presented
    as a measurement of the plan.

    Two numbers can be measured against, and which one it is changes what this
    check is allowed to say:

    * **the member's own target** — a miss is a FAILURE, because they asked to
      be planned against it;
    * **a population reference**, used only when they set no target. A day over
      it is worth telling them ("three snacks and a dinner add up"), and it is
      NOT a broken rule — nobody set one. So it reports as PARTIAL under its
      own name, and every sentence says whose number it is.

    That distinction is the weekly meat limit's lesson applied here in advance:
    FoodChat's own default rendered as the member's "dietary preference", with
    an apology addressed to somebody who never asked for it.
    """
    target = requested.get("kcal_target")
    reference = requested.get("kcal_reference")
    chosen = bool(target)
    if not chosen:
        target = reference
    if not target:
        return None
    try:
        target = float(target)
    except (TypeError, ValueError):
        return None
    if target <= 0:
        return None

    # The name is what `transparency.split_ledger` hands the reply, on its own
    # and without the detail — so it has to read as a sentence in both "was
    # honoured" and "could not be honoured". "calorie reference" does not; "a
    # 2000 kcal daily reference" does.
    name = "calories" if chosen else f"a {int(target)} kcal daily reference"
    basis = "" if chosen else str(requested.get("kcal_reference_basis") or "")
    against = (
        f"your {int(target)} kcal target" if chosen
        else f"a {int(target)} kcal reference"
    )

    per_day: list[tuple[int, float, int, int]] = []
    for day in plan.day_plans:
        total = 0.0
        counted = 0
        plates = [p for meal in day.meals for p in meal.plates if p.recipe_id]
        for plate in plates:
            kcal = _measured(plate, enrichment, "kcal", "kcal")
            if not isinstance(kcal, (int, float)):
                continue
            total += float(kcal)
            counted += 1
        if counted:
            per_day.append((day.day, total, counted, len(plates)))

    if not per_day:
        return Check(name=name, status=UNKNOWN,
                     code="kcal.unknown",
                     params={"target": int(target), "own": chosen},
                     detail=f"Nothing to compare against {against}: no dish has "
                            "nutrition data to add up")

    low, high = target * (1 - KCAL_TOLERANCE), target * (1 + KCAL_TOLERANCE)
    misses = [(d, total) for d, total, _, _ in per_day if not (low <= total <= high)]
    complete = all(counted == total_plates for _, _, counted, total_plates in per_day)
    caveat = "" if complete else " (some dishes have no nutrition data, so this is a floor)"

    if not misses:
        average = sum(t for _, t, _, _ in per_day) / len(per_day)
        detail = f"Averaging {int(average)} kcal a day against {against}{caveat}"
        if basis:
            detail += f" — {basis}"
        return Check(
            name=name, status=PASSED, detail=detail,
            observed=len(per_day), of=len(plan.day_plans),
            code="kcal.ok",
            params={"average": int(average), "target": int(target), "own": chosen,
                    "partial": not complete},
        )

    worst = max(misses, key=lambda item: abs(item[1] - target))
    direction = "over" if worst[1] > target else "under"
    detail = (
        f"{len(misses)} of {len(per_day)} days miss {against} — day {worst[0]} "
        f"is {int(abs(worst[1] - target))} kcal {direction}{caveat}"
    )
    params = {
        "misses": len(misses), "days": len(per_day), "day": worst[0],
        "delta": int(abs(worst[1] - target)), "direction": direction,
        "target": int(target), "own": chosen, "partial": not complete,
    }
    # Only when the day is OVER, and only the plate that would move it most.
    # Under-target is a different conversation and naming its lightest dish
    # would be telling somebody to eat more of something.
    if direction == "over":
        heaviest = _heaviest_plate(plan, enrichment, worst[0])
        if heaviest and heaviest[0]:
            params["heaviest"] = {"dish": heaviest[0], "kcal": int(heaviest[1])}
            detail += (
                f". The heaviest plate is “{heaviest[0]}” at {int(heaviest[1])} "
                "kcal — ask me for a lighter one there and I will re-plan it"
            )
    if basis:
        detail += f". {basis[0].upper()}{basis[1:]}"
    return Check(
        name=name,
        # A population reference is not a rule the member broke. It reports as
        # partial — worth reading, not a red mark for a number nobody set.
        status=FAILED if chosen else PARTIAL,
        detail=detail,
        observed=len(per_day), of=len(plan.day_plans),
        code="kcal.miss", params=params,
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
                            "records a cooking time", of=len(plates),
                     code="time.unknown", params={"limit": int(limit)})
    if offenders:
        return Check(
            name="cooking time", status=FAILED,
            detail=f"{len(offenders)} of {observed} dishes run over "
                   f"{int(limit)} minutes — the longest is {int(worst)}",
            offenders=tuple(offenders), observed=observed, of=len(plates),
            code="time.over",
            params={"count": len(offenders), "observed": observed,
                    "limit": int(limit), "longest": int(worst)},
        )
    return Check(name="cooking time", status=PASSED,
                 detail=f"All {observed} timed dishes are {int(limit)} minutes or less",
                 observed=observed, of=len(plates),
                 code="time.ok", params={"observed": observed, "limit": int(limit)})


def _check_nutri_score(plates, requested, enrichment) -> Optional[Check]:
    minimum = str(requested.get("min_nutri_score") or "").strip().lower()
    if minimum not in _NUTRI_ORDER:
        return None
    ceiling = _NUTRI_ORDER[minimum]

    offenders: list[str] = []
    observed = 0
    for plate in plates:
        label = str(
            _measured(plate, enrichment, "nutri_score_label", "nutri_score_label") or ""
        ).strip().lower()
        if label not in _NUTRI_ORDER:
            continue
        observed += 1
        if _NUTRI_ORDER[label] > ceiling:
            offenders.append(plate.recipe_id)

    if observed == 0:
        return Check(name="nutri-score", status=UNKNOWN,
                     detail=f"Asked for {minimum.upper()} or better; no dish "
                            "has a Nutri-Score", of=len(plates),
                     code="nutri.unknown", params={"minimum": minimum.upper()})
    if offenders:
        return Check(
            name="nutri-score", status=FAILED,
            detail=f"{len(offenders)} of {observed} scored dishes are below "
                   f"{minimum.upper()}",
            offenders=tuple(offenders), observed=observed, of=len(plates),
            code="nutri.below",
            params={"count": len(offenders), "observed": observed, "minimum": minimum.upper()},
        )
    return Check(name="nutri-score", status=PASSED,
                 detail=f"All {observed} scored dishes are {minimum.upper()} or better",
                 observed=observed, of=len(plates),
                 code="nutri.ok", params={"observed": observed, "minimum": minimum.upper()})


def _check_dislikes(plates, requested) -> Optional[Check]:
    """Whether a food the member dislikes made it onto a plate.

    Measured, not restated: the rule list used to repeat the first five
    dislikes and mark each "satisfied" without looking, and a member who
    dislikes fish was served Tuna filo pies under it. Soft — a dish that slips
    through is named, not withheld; the search already excluded what it could.
    """
    from services.dislikes import dislike_hits

    dislikes = [str(d) for d in (requested.get("dislikes") or []) if str(d).strip()]
    if not dislikes or not plates:
        return None
    found: list[dict] = []
    offenders: list[str] = []
    for plate in plates:
        for dislike in dislike_hits(_text(plate), dislikes):
            found.append({"dish": plate.title, "dislike": dislike})
            rid = getattr(plate, "recipe_id", None)
            if rid and rid not in offenders:
                offenders.append(rid)
    if found:
        shown = found[:6]
        more = len(found) - len(shown)
        detail = "; ".join(f"{hit['dish']} has {hit['dislike']}" for hit in shown)
        if more:
            detail += f" (and {more} more)"
        return Check(
            name="dislikes", status=FAILED, detail=detail,
            offenders=tuple(offenders), observed=len(plates), of=len(plates),
            code="dislikes.found", params={"hits": shown, "more": more},
            found=tuple(found),
        )
    count = len(dislikes)
    return Check(
        name="dislikes", status=PASSED,
        detail=f"None of the {count} food{'s' if count != 1 else ''} you dislike on any of the {len(plates)} dishes",
        observed=len(plates), of=len(plates),
        code="dislikes.clear", params={"count": count, "dishes": len(plates)},
    )


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
                     of=len(plates),
                     code="pantry.none", params={"items": pantry})
    if unused:
        return Check(
            name="pantry", status=PARTIAL,
            detail=f"Uses {', '.join(used)} across {plates_using} dishes; "
                   f"{', '.join(unused)} went unused",
            observed=plates_using, of=len(plates),
            code="pantry.some",
            params={"used": used, "dishes": plates_using, "unused": unused},
        )
    return Check(name="pantry", status=PASSED,
                 detail=f"Uses all of {', '.join(used)} across {plates_using} dishes",
                 observed=plates_using, of=len(plates),
                 code="pantry.all", params={"used": used, "dishes": plates_using})


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
            code="picks.missing", params={"count": len(missing), "slots": sorted(missing)},
        )
    return Check(name="your picks", status=PASSED,
                 detail=f"All {len(anchors)} dishes you asked for are on the plan",
                 observed=len(anchors), of=len(anchors),
                 code="picks.all", params={"count": len(anchors)})


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
        code="tastes.unverified", params={"values": [str(v) for v in values]},
    )


def describe(report: VerificationReport) -> str:
    """One line for the log and for a grounded reply. Never empty."""
    if not report.checks:
        return "nothing to verify"
    return " · ".join(f"{c.name}: {c.status}" for c in report.checks)
