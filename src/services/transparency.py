"""
Transparency (M4a) — the *reasons* behind a plan as structured data.

Pure functions (no LLM, no I/O) that compute what the UI renders:

    match_reasons(course, ...)      → per-course chips ("you liked chickpeas",
                                      "no peanuts — Tom's allergy", "requested by you")
    constraints_ledger(profile, …)  → plan-level hard/soft constraint rows
    personalization_summary(...)    → counts linking to the memory panel

Reason kinds (shared contract with the UI):
    pinned | favorite | memory | profile | feedback | diner | guideline
"""

from typing import Optional


def match_reasons(
    course_recipe_id: str,
    ingredients_text: str,
    profile: dict,
    pinned_recipe_ids: set[str],
) -> list[dict]:
    """Why this recipe is in the plan (order: strongest signal first)."""
    reasons: list[dict] = []

    if course_recipe_id in pinned_recipe_ids:
        reasons.append({"kind": "pinned", "label": "requested by you"})

    if course_recipe_id in (profile.get("favorite_recipe_ids") or []):
        reasons.append({"kind": "favorite", "label": "one of your favorites"})

    text = (ingredients_text or "").lower()
    memory_values = {
        str(e.get("value", "")).lower()
        for e in (profile.get("memory_log") or [])
        if e.get("kind") in ("like", "cuisine")
    }
    for like in profile.get("food_likes") or []:
        like_l = str(like).lower()
        if like_l and like_l in text:
            kind = "memory" if like_l in memory_values else "profile"
            label = f"you like {like_l}"
            reasons.append({"kind": kind, "label": label})
            break  # one like-chip per course keeps cards readable

    return reasons


def _diet_row(diet, members: list[str]) -> dict:
    """One ledger row per diet value, saying what actually happened to it — or
    ``None`` for a value that is not a constraint at all.

    Every value used to render as ``hard`` / ``satisfied``, which is only true
    of the ones RecipeWrangler has a filter for. 26 of the gateway's 37 dietary
    groups have none, so a member who selected ``peanut_free`` was shown a
    peanut-free guarantee with nothing behind it.

    Three things can happen to a value, and the row says which:

    * **filtered** — the original claim, now made only when it is true;
    * **unsupported** — a real restriction the member chose that nothing
      upstream can filter on (``peanut_free``, ``halal``, and the nutrition
      claims, which travel as claim tags instead). Never ``satisfied``, and
      never silently dropped either: dropping it would trade a false claim for
      a silent one, which is the same failure in its other direction. A
      free-from slug names the allergen backstop that DOES cover it;
    * **not a restriction** — ``omnivore``, ``mediterranean``, ``flexitarian``:
      reported as the soft description it is, rather than as an enforced rule.
      EVERY value keeps a row, which is the invariant worth more than the one
      it costs: a member can see each diet value they set and what became of
      it, and none of them disappears silently.
    """
    from services.candidates_client import (  # local import; avoids a cycle
        DIET_FILTER,
        DIET_NOT_RESTRICTIVE,
        FREE_FROM_TO_ALLERGEN,
        diet_tag_status,
    )

    status, _tag = diet_tag_status(diet)
    if status == DIET_NOT_RESTRICTIVE:
        return {
            "constraint": str(diet), "type": "soft", "status": "satisfied",
            "source": "dietary group",
            "detail": "a description of how you eat, not a recipe filter — "
                      "no dishes were excluded for it",
            "members": members,
        }
    if status == DIET_FILTER:
        return {
            "constraint": str(diet), "type": "hard", "status": "satisfied",
            "source": "dietary group", "members": members,
        }

    detail = (
        "The recipe catalogue has no filter for this, so it did not narrow "
        "the search."
    )
    backstop = FREE_FROM_TO_ALLERGEN.get(str(diet).strip().lower())
    if backstop:
        detail = (
            "The recipe catalogue has no filter for this, so it did not narrow "
            f"the search — but {backstop} are screened out of every plate by "
            "ingredient name."
        )
    row = {
        "constraint": str(diet), "type": "hard", "status": "unsupported",
        "source": "dietary group", "detail": detail, "members": members,
    }
    if backstop:
        # Read by `split_ledger`: an unsupported value that something ELSE
        # covers must not be reported to the member as a failure.
        row["covered_by"] = backstop
    return row


def constraints_ledger(profile: dict, downvoted_count: int = 0) -> list[dict]:
    """Hard/soft constraint rows for the plan header.

    Sources: the merged session profile (whose ``constraint_origins`` records
    which diner each constraint came from) and feedback exclusions.

    Rows carry ``members`` so a household plan can attribute a constraint to
    the diner it protects. Listing every diner on every row — which is what
    this did before — reads as if the whole table were allergic, and hides the
    one member the row is actually there for.

    Goals get rows too, including the ones reconciliation demoted to a soft
    signal. A goal that steered the plan without appearing anywhere is the
    "silently dominates" failure; a goal that was dropped without a trace is
    its mirror image.
    """
    ledger: list[dict] = []
    diners = profile.get("cooking_for_names") or []
    household = len(diners) > 1
    origins = profile.get("constraint_origins") or {}

    def members_for(key: str, value: str) -> list[str]:
        # Attribution only means something when several people are eating
        if not household:
            return []
        return list((origins.get(key) or {}).get(value) or [])

    for allergen in profile.get("allergies") or []:
        ledger.append({
            "constraint": f"no {allergen}",
            "type": "hard", "status": "satisfied",
            "source": "allergy",
            "members": members_for("allergies", allergen),
        })
    for diet in profile.get("diet") or []:
        ledger.append(_diet_row(diet, members_for("diet", diet)))
    for dislike in (profile.get("food_dislikes") or [])[:5]:
        ledger.append({
            "constraint": f"avoiding {dislike}",
            "type": "soft", "status": "satisfied",
            "source": "preferences",
            "members": members_for("food_dislikes", dislike),
        })

    ledger.extend(_goal_rows(profile, household))

    if downvoted_count:
        ledger.append({
            "constraint": f"excluding {downvoted_count} recipe(s) you disliked",
            "type": "soft", "status": "satisfied",
            "source": "your feedback",
            "members": [],
        })
    return ledger


def _goal_rows(profile: dict, household: bool) -> list[dict]:
    """One row per goal, saying whether it became a target or only a signal."""
    reconciliation = profile.get("goal_reconciliation") or []
    if not reconciliation:
        # Solo plans have no reconciliation record; the member's own goals are
        # all targets, so report them as such rather than not at all.
        return [
            {
                "constraint": _goal_label(slug),
                "type": "soft", "status": "satisfied",
                "source": "your goal", "members": [],
            }
            for slug in (profile.get("dietary_goals") or [])
        ]

    # Collapse to one row per goal, collecting who asked for it
    rows: dict[str, dict] = {}
    for record in reconciliation:
        slug = str(record.get("slug") or "")
        if not slug:
            continue
        applied = record.get("applied") == "target"
        member = str(record.get("member") or "")
        row = rows.setdefault(slug, {
            "constraint": _goal_label(slug),
            "type": "soft",
            "status": "satisfied" if applied else "relaxed",
            "source": "goal",
            "members": [],
        })
        if member and member not in row["members"]:
            row["members"].append(member)
        # Any member holding it as a target makes the goal a target
        if applied:
            row["status"] = "satisfied"

    for row in rows.values():
        if row["status"] == "relaxed":
            row["detail"] = (
                "Applied as a preference, not a target — another diner's goal "
                "sets this plan's numeric targets."
            )
        elif household:
            row["detail"] = "Sets this plan's numeric targets."

    return list(rows.values())


def _goal_label(slug: str) -> str:
    return str(slug).replace("_", " ").strip() or "goal"


def split_ledger(ledger: list, limit: int = 4) -> tuple[list[str], list[str]]:
    """Split a ledger into what was actually honoured and what was not.

    ``constraints_applied`` mixes statuses — "satisfied", "relaxed" and, on
    weekly plans, "violated". Handing the first N rows to the response writer
    under ``constraints_honored`` let it announce a relaxed goal as honoured
    while the ledger rendered beside it said otherwise; the writer is told to
    mention "an honored request", and it believed the key.

    Returns ``(honored, not_honored)`` constraint strings. A row with no
    recognised status lands in NEITHER list: plans stored before the status
    field existed would otherwise be claimed as honoured (a false positive) or
    apologised for (a false negative), and silence is the only honest option
    when the ledger does not say.
    """
    honored: list[str] = []
    not_honored: list[str] = []
    for row in ledger or []:
        text = row.get("constraint")
        if not text:
            continue
        status = row.get("status")
        if status == "satisfied":
            honored.append(text)
        elif status in ("relaxed", "violated"):
            not_honored.append(text)
        elif status == "unsupported" and not row.get("covered_by"):
            # Nothing filtered on it and nothing else covers it, so the member
            # should hear that from the reply and not only from a chip.
            not_honored.append(text)
        # An `unsupported` value WITH a backstop is deliberately in neither
        # list. Claiming it honoured would be the lie this status exists to
        # stop; reporting "couldn't honour peanut_free" would over-alarm a
        # member whose peanuts ARE screened out of every plate by ingredient
        # name. The row carries that nuance in its detail; prose should not
        # flatten it either way.
    return honored[:limit], not_honored[:limit]


def personalization_summary(profile: dict, feedback_lines: int = 0) -> dict:
    """Counts for the "Personalized with …" line (links to the memory panel)."""
    return {
        "memories_used": len(profile.get("memory_log") or []),
        "favorites_used": len(profile.get("favorite_recipe_ids") or []),
        "feedback_signals": feedback_lines,
        "diners": len(profile.get("cooking_for_names") or []) or 1,
    }


def apply_transparency(
    meal_plan,
    profile: dict,
    pinned_recipe_ids: set[str],
    enrichment: dict,
    downvoted_count: int = 0,
    feedback_lines: int = 0,
) -> None:
    """Attach enrichment + transparency to a freshly built MealPlan in place.

    Iterates `day_plans`, the uniform days→meals→plates view, rather than the
    three legacy scalar accessors. Those only ever reach day 1's main plates —
    a compatibility projection — so on a multi-plate or multi-day plan every
    side, dessert and every day after the first got no nutrition, no image and
    no reason chip. The same iteration `pantry_service.annotate_daily_plan`
    already uses, and it reads a legacy plan identically.
    """
    for day in meal_plan.day_plans:
        for meal in day.meals:
            for plate in meal.plates:
                if not getattr(plate, "recipe_id", ""):
                    continue  # `from_days` inserts blanks for absent legacy slots
                rich = enrichment.get(plate.recipe_id)
                if rich:
                    plate.nutrition = rich.nutrition_dict()
                    plate.image_url = rich.image_url
                plate.match_reasons = match_reasons(
                    plate.recipe_id, plate.ingredients, profile, pinned_recipe_ids,
                )
    meal_plan.constraints_applied = constraints_ledger(profile, downvoted_count)
    meal_plan.personalization_summary = personalization_summary(profile, feedback_lines)

# --------------------------------------------------------------------------- #
# What is GOOD about this plan                                                  #
# --------------------------------------------------------------------------- #
#
# The ledger above answers "was every constraint respected". That is a real
# question and it has an honest answer, and it is not what a member came for.
#
# FoodChat is not a constraint solver. It is meant to help a household shape
# meals that are better for them and to say WHY — and the facts handed to the
# reply were five parts constraint bookkeeping (`constraints_honored`,
# `constraints_not_honored`, `verified_problems`, `repair`, `pantry`) to zero
# parts health. So every plan was explained as a compliance result: what was
# permitted, what was swapped, what fell short. Meanwhile the system had
# measured food variety, guideline adherence, meal diversity, Nutri-Score and
# the day's calories, and handed the member a collapsed panel of scores out of
# five instead of a sentence.
#
# This builds the other half. Every value here is MEASURED — a count, a total,
# or a judge's own sentence — because a reply is only allowed to phrase what
# the facts contain, and "healthy" is not a measurement.


def plan_value(
    meal_plan,
    profile: dict,
    metrics: Optional[dict] = None,
    kcal_target: Optional[float] = None,
    pantry_facts: Optional[dict] = None,
) -> dict:
    """The reasons this plan is worth eating, for the reply to draw on.

    Empty keys are omitted rather than sent as zero: a reply that says "0
    unique foods" or "no guidance" because a metric was skipped is worse than
    one that talks about the food.
    """
    metrics = metrics or {}
    value: dict = {}

    diners = [str(d) for d in (profile.get("cooking_for_names") or []) if d]
    if len(diners) > 1:
        # Named, because a plan for a household is a different thing from a
        # plan for one person, and the reply should sound like it knows.
        value["cooking_for"] = diners

    totals = _day_totals(meal_plan)
    if totals:
        nutrition = {
            "kcal_per_day": round(totals["kcal"] / max(1, totals["days"])),
            "protein_g_per_day": round(totals["protein"] / max(1, totals["days"])),
        }
        if not totals["complete"]:
            # Said, so the reply cannot present a partial sum as the day.
            nutrition["partial"] = "some dishes carry no nutrition data"
        if kcal_target:
            nutrition["kcal_target"] = int(kcal_target)
        value["nutrition"] = nutrition

    if metrics.get("fvs_count"):
        # Food variety: distinct ingredients across the plan. A real number
        # about a real thing — a varied plate is the least controversial
        # nutrition advice there is.
        value["distinct_foods"] = int(metrics["fvs_count"])

    for key, name in (
        ("guideline_adherence_reasoning", "guidance"),
        ("diversity_llm_reasoning", "balance"),
    ):
        sentence = str(metrics.get(key) or "").strip()
        if sentence:
            # The judge's own sentence, not its score. "3 out of 5" is not
            # something to tell someone about their dinner.
            value[name] = sentence[:300]

    grades = _nutri_grades(meal_plan)
    if grades:
        value["nutri_score"] = grades

    pairings = [str(x) for x in (getattr(meal_plan, "pairings", None) or []) if x]
    if pairings:
        # Why the dishes of a meal go together, in the judge's own words. The
        # one part of composition worth saying out loud — and it reaches the
        # reply as a fact to phrase rather than as a string rendered on the
        # canvas, which is how "chosen for the table:" ended up on a plan.
        value["pairings"] = pairings[:3]

    if pantry_facts and pantry_facts.get("used"):
        # The sustainability half, and the only one the member asked for
        # directly: food they already had, now going into a meal instead of a
        # bin.
        value["using_up"] = list(pantry_facts["used"])

    return value


def _day_totals(meal_plan) -> Optional[dict]:
    """Summed macros across every plate, and how many it could not see."""
    kcal = protein = 0.0
    counted = plates = 0
    days = 0
    for day in getattr(meal_plan, "day_plans", None) or []:
        days += 1
        for meal in getattr(day, "meals", None) or []:
            for plate in getattr(meal, "plates", None) or []:
                if not getattr(plate, "recipe_id", ""):
                    continue
                plates += 1
                nutrition = getattr(plate, "nutrition", None) or {}
                value = nutrition.get("kcal")
                if not isinstance(value, (int, float)):
                    continue
                counted += 1
                kcal += float(value)
                protein += float(nutrition.get("protein_g") or 0)
    if not counted:
        return None
    return {
        "kcal": kcal, "protein": protein, "days": max(1, days),
        "complete": counted == plates,
    }


def _nutri_grades(meal_plan) -> Optional[str]:
    """"4 of 5 dishes are Nutri-Score A or B", or None when none are graded."""
    good = graded = 0
    for day in getattr(meal_plan, "day_plans", None) or []:
        for meal in getattr(day, "meals", None) or []:
            for plate in getattr(meal, "plates", None) or []:
                label = str(
                    (getattr(plate, "nutrition", None) or {}).get("nutri_score_label")
                    or ""
                ).strip().upper()
                if not label:
                    continue
                graded += 1
                if label[-1] in ("A", "B"):
                    good += 1
    if not graded:
        return None
    return f"{good} of {graded} dishes are Nutri-Score A or B"
