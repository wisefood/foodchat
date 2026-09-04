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

from models.session import MealCourse


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


def _diet_row(diet, members: list[str]) -> dict:
    """One row per diet value, saying what actually happened to it.

    Every value used to render as ``hard`` / ``satisfied``, which is only true
    of the ones RecipeWrangler has a filter for. A profile saying
    ``flexitarian`` — a word this service does not know, and which reaches
    neither the recipe query nor the weekly meat limit — was reported as a
    satisfied hard constraint. The row was the only place a member could have
    learned otherwise, and it said the opposite.

    Three outcomes, because there are three things that can happen:

    - forwarded as a filter — the original claim, and now only made when true;
    - a non-restrictive label ("omnivore", "mediterranean"): nothing was
      excluded and nothing was meant to be, so it is reported as the soft
      description it is rather than as an enforced rule;
    - unknown: ``relaxed``, which puts it in ``constraints_not_honored`` and
      obliges the reply to say so. Dropping the row instead would trade a
      false claim for a silent one — the failure this module exists to
      prevent, in its other direction.
    """
    from services.candidates_client import (  # local import; avoids a cycle
        DIET_FILTER,
        DIET_NOT_RESTRICTIVE,
        diet_tag_status,
    )

    status, _tag = diet_tag_status(diet)
    if status == DIET_FILTER:
        return {
            "constraint": str(diet), "type": "hard", "status": "satisfied",
            "source": "dietary group", "members": members,
        }
    if status == DIET_NOT_RESTRICTIVE:
        return {
            "constraint": str(diet), "type": "soft", "status": "satisfied",
            "source": "dietary group",
            "detail": "a description of how you eat, not a recipe filter — "
                      "no dishes were excluded for it",
            "members": members,
        }
    return {
        "constraint": str(diet), "type": "hard", "status": "relaxed",
        "source": "dietary group",
        "detail": "the recipe service has no filter for this, so no dishes "
                  "were excluded for it",
        "members": members,
    }


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
    """Attach enrichment + transparency to a freshly built MealPlan in place."""
    for course in (meal_plan.breakfast, meal_plan.lunch, meal_plan.dinner):
        rich = enrichment.get(course.recipe_id)
        if rich:
            course.nutrition = rich.nutrition_dict()
            course.image_url = rich.image_url
        course.match_reasons = match_reasons(
            course.recipe_id, course.ingredients, profile, pinned_recipe_ids,
        )
    meal_plan.constraints_applied = constraints_ledger(profile, downvoted_count)
    meal_plan.personalization_summary = personalization_summary(profile, feedback_lines)
