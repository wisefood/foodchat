"""
What this member has just eaten, so a fresh plan is a fresh plan.

Ask for a daily plan twice and you get the same plan. Not similar — identical.
Three things line up to guarantee it:

* RecipeWrangler returns a **deterministic** order (planning tier, then
  Nutri-Score, then curated source). Same filters, same eight candidates, same
  order.
* The grader runs at **temperature 0.0**. Same pool and same prompt produce the
  same ranking, by design.
* Nothing carried forward. `exclude_recipe_ids` held downvotes and dishes the
  member had explicitly rejected — and **not one recipe the member had just
  been served.**

So the pipeline was working exactly as built and the member got courgette
omelette every morning. The determinism is not the bug; every part of it is
worth keeping. The bug is that "plan my day" was asked and answered as though
no day had ever been planned.

    avoid = plan_history.recently_served(session)

Deliberately not randomness. A random pick makes two plans differ and makes
neither explainable — and it throws away the pool order, which encodes planning
tier and Nutri-Score. Excluding what was just served keeps every ranking intact
and moves the window along: the second plan is the next best plan, not a
shuffle of the first.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# How many past plans count as "recent".
#
# Three, not all of them: the member should stop seeing this week's repeats,
# not be barred from a dish they liked a month ago. It also bounds the
# exclusion list, which is what keeps the pool from being emptied by its own
# history on a corpus of a few thousand recipes.
RECENT_PLANS = 3

# Hard ceiling on the ids sent. A weekly plan is 21 recipes and three of them
# is 63 — enough to strain a slot's pool on a narrow diet. Newest first, so the
# cut falls on the oldest.
RECENT_CAP = 45


def plan_recipe_ids(plan) -> list[str]:
    """Every recipe on a plan, in plan order, whatever shape it is.

    Walks `day_plans` for a daily/structured plan and `entries` for a weekly
    one. The existing helper on the orchestrator reads `plan.breakfast`,
    `plan.lunch`, `plan.dinner` — which silently misses every extra day and
    every second plate, so a multi-plate plan's sides would keep coming back.
    """
    out: list[str] = []
    for day in getattr(plan, "day_plans", None) or []:
        for meal in getattr(day, "meals", None) or []:
            for plate in getattr(meal, "plates", None) or []:
                recipe_id = str(getattr(plate, "recipe_id", "") or "")
                if recipe_id and recipe_id not in out:
                    out.append(recipe_id)
    for entry in getattr(plan, "entries", None) or []:
        if not isinstance(entry, dict):
            continue
        recipe_id = str((entry.get("recipe") or {}).get("recipe_id") or "")
        if recipe_id and recipe_id not in out:
            out.append(recipe_id)
    return out


def recently_served(session, *, plans: int = RECENT_PLANS,
                    cap: int = RECENT_CAP) -> list[str]:
    """Recipe ids from this session's last few plans, newest first.

    Both canvases, because a member who had salmon in yesterday's week does not
    want it as today's dinner either — the question is what they have been
    served, not which button produced it.

    A soft signal, always: the caller passes it as a preference the fetch drops
    the moment it would empty a slot. A member on a narrow diet must not be
    told "no meals exist" because they asked twice.
    """
    if session is None:
        return []
    recent: list[str] = []
    for attribute in ("meal_plans", "weekly_meal_plans"):
        # Newest last in both lists (they are appended), so take the tail and
        # reverse it — the cap should fall on the oldest ids, not the newest.
        stored = list(getattr(session, attribute, None) or [])
        for plan in reversed(stored[-plans:]):
            for recipe_id in plan_recipe_ids(plan):
                if recipe_id not in recent:
                    recent.append(recipe_id)
    if len(recent) > cap:
        recent = recent[:cap]
    if recent:
        logger.info(
            "Avoiding %d recently served recipe(s) so this plan is a new one",
            len(recent),
        )
    return recent
