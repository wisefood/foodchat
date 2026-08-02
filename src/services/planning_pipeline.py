"""
Daily meal-plan generation pipeline.

Replaces the pre-M0 ``FoodChat`` class and its LangChain runnable chains
(``create_pre_clarification_chain`` / ``create_post_clarification_chain``),
which existed to serve a local Chroma/BM25 RAG stack that has been removed.
The pipeline is now three explicit, typed steps:

    generate(query, profile)    → candidates (RecipeWrangler) → LLM-graded plans

Consumers: ``services.chat_service`` (the only caller). The reconciled
profile is produced upstream by ``services.clarification`` — this pipeline
only fetches and grades.

No data files, vector stores, or embeddings are required — the service boots
with Groq + RecipeWrangler + WiseFood credentials alone.
"""

import logging
import os

from agents import DocumentGrader
from models.plan_spec import PlanSpec
from models.recipe import CandidateRecipe, ScoredPlan
from models.session import MealPlan
from services import plan_parameters
from services.candidates_client import CANDIDATES, normalize_diet_tags

logger = logging.getLogger(__name__)

# Candidates requested per meal slot from RecipeWrangler. 8 per slot bounds the
# grading space (8³ = 512 combos, sampled down by DocumentGrader) while keeping
# enough variety for refinements to find alternatives.
CANDIDATE_LIMIT = int(os.getenv("FOODCHAT_CANDIDATE_LIMIT", "8"))


class PlanningPipeline:
    """Reconcile → fetch candidates → grade combinations → ranked plans."""

    def __init__(self):
        self.grader = DocumentGrader()

    def generate(
        self,
        query: str,
        profile: dict,
        pinned: dict[str, "CandidateRecipe"] | None = None,
        exclude_recipe_ids: list[str] | None = None,
        feedback_history: str = "",
    ) -> list[ScoredPlan]:
        """Produce ranked daily-plan combinations for the reformulated query.

        Hard constraints (allergies, diet, dislikes) are enforced server-side
        by RecipeWrangler; the LLM grader ranks the surviving combinations
        against the query and soft preferences. Profile favorites are passed
        as a server-side ranking boost.

        ``pinned`` maps slot names ("breakfast"/"lunch"/"dinner") to anchor
        recipes the user explicitly requested (seeded planning, M2): a pinned
        slot has exactly one candidate — its anchor — so every graded
        combination contains it, and the other slots are ranked around it.
        """
        pinned = pinned or {}

        # `food_likes` mixes cuisines and ingredients, because a "cuisine"
        # memory is folded into the same list as a "like". Sent whole they all
        # became ingredient searches, so a member who told us they love Greek
        # food had that turned into a hunt for an ingredient named "greek" —
        # which matches nothing. Splitting lets the cuisine drive the filter it
        # was always meant to.
        likes = list(set(
            (profile.get("food_likes") or []) + (profile.get("include_ingredients") or [])
        ))
        cuisines, liked_ingredients = CANDIDATES.split_cuisines(likes)
        if cuisines:
            logger.info("Applying cuisine preferences from profile: %s", cuisines)

        # The cooking-time slider, as a constraint rather than a phrase for the
        # grader. `plan_parameters.describe` still renders it into the
        # refinement text; this makes it narrow the candidate set as well.
        max_minutes = plan_parameters.max_duration_minutes(
            profile.get("plan_parameters") or {}
        )

        # Candidates come from `/api/v2/tools/plan_meals`.
        #
        # It is the planning surface: it knows the corpus's cuisine, mood,
        # flavour and food-group annotations, it honours `planning_tier` so a
        # recipe withdrawn from automated planning is never offered, and it
        # relaxes soft preferences rather than returning an empty slot. The v1
        # candidate endpoint knows none of that — it queries a store that has
        # never held an annotation — so every preference this pipeline resolves
        # above was being computed and then thrown away.
        #
        # Asking for `CANDIDATE_LIMIT` per slot turns it into a candidate
        # source: the grader still ranks the combinations, it just ranks a pool
        # that already respects what the member asked for.
        candidates = _fetch_candidate_pool(
            profile=profile,
            cuisines=cuisines,
            liked_ingredients=liked_ingredients,
            max_minutes=max_minutes,
            exclude_recipe_ids=(
                [r.recipe_id for r in pinned.values()] + list(exclude_recipe_ids or [])
            ),
            limit_per_slot=CANDIDATE_LIMIT,
        )
        for slot, anchor in pinned.items():
            candidates[slot] = [anchor]
            logger.info("Slot %s pinned to %r", slot, anchor.title)

        empty_slots = [slot for slot, recipes in candidates.items() if not recipes]
        if empty_slots:
            # No second source to try. `plan_meals` already relaxed every soft
            # preference it was allowed to before returning an empty slot, so a
            # gap here means the hard constraints — allergens, diet, exclusions
            # — genuinely admit nothing.
            logger.warning(
                "No candidates for slot(s) %s even after relaxation", empty_slots
            )
            return []

        try:
            scored = self.grader.grade_daily_plans(
                query, candidates, profile, feedback_history
            )
        except Exception as exc:  # noqa: BLE001
            # The grader is a model call. Losing it should cost the *ranking*,
            # not the plan: the pool is already constraint-correct and already
            # ordered by planning tier and Nutri-Score, so serving its top pick
            # beats an apology.
            logger.error("Grader failed (%s) — serving the unranked pool", exc)
            return self._assemble_from_pool(candidates, "not ranked — grader unavailable")

        if not scored:
            logger.warning("Grader returned no plans — serving the unranked pool")
            fallback = self._assemble_from_pool(candidates)
            if fallback:
                return fallback

        logger.info(
            "Pipeline produced %d scored plan(s); best score=%s",
            len(scored), scored[0].score if scored else "n/a",
        )
        return scored

    def plan_structured(
        self,
        profile: dict,
        spec: "PlanSpec",
        exclude_recipe_ids: list[str] | None = None,
        pinned: dict | None = None,
    ) -> "MealPlan | None":
        """Generate a plan of any shape — N days, N meals, multi-plate meals.

        This is Phase 2 of DYNAMIC_MEALS_PLAN.md. The core model landed in
        Phase 1; what blocked generation was the plan's own open question —
        whether RecipeWrangler could answer role-scoped queries. It can:
        `/api/v2/tools/plan_meals` takes a `course_types` override per slot, so
        a `side` plate becomes a query for salads and soups rather than a
        keyword-biased guess.

        Returns a real `MealPlan` with `days` populated, not a bag of dicts —
        so `day_plans`, the serializer, the session store and the UI all read it
        through the same accessors they already use for a legacy plan.

        `None` when the service could not produce a plan. The caller decides
        whether that is an apology or a retry; inventing a partial plan here
        would hand the user a day with meals silently missing.
        """
        from models.session import DayPlan, Meal, MealCourse, MealPlan
        from services.plan_client import PLANNER, _meal_nutrition

        # Anchors the member named — "I want apple pie for breakfast".
        #
        # Their ids are excluded from the request so the service does not offer
        # the same recipe a second time, and the anchor is substituted into its
        # slot's *main* plate afterwards. Only day 1 is anchored: "apple pie for
        # breakfast" is a request about today, not a week of apple pie.
        pinned = pinned or {}
        exclude_recipe_ids = list(exclude_recipe_ids or []) + [
            anchor.recipe_id for anchor in pinned.values() if anchor.recipe_id
        ]

        cuisines, _ = CANDIDATES.split_cuisines(profile.get("food_likes") or [])
        max_minutes = plan_parameters.max_duration_minutes(
            profile.get("plan_parameters") or {}
        )

        try:
            envelope = PLANNER.plan_meals(
                spec=spec,
                allergens=profile.get("allergies") or [],
                # Normalised, never raw.
            #
            # RecipeWrangler ANDs diet tags and never relaxes them, so a single
            # value it does not know — "balanced", "mediterranean", "omnivore",
            # a goal slug that leaked into the diet list — returns zero
            # candidates for every slot and the member gets an apology. The old
            # client did this and its docstring said why; dropping it when the
            # candidate source moved is what emptied the plan.
            diet=normalize_diet_tags(profile.get("diet")),
                cuisines=cuisines,
                exclude_ingredients=profile.get("food_dislikes") or [],
                exclude_recipe_ids=list(exclude_recipe_ids or []),
                max_minutes=max_minutes,
                min_nutri_score=profile.get("min_nutri_score"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Structured planning failed: %s", exc)
            return None

        notes = PLANNER.describe_relaxations(envelope)
        if notes:
            logger.info("Structured plan relaxations: %s", notes)

        # The response carries the slot but not the role — roles are FoodChat's
        # vocabulary. `role_sequence()` regenerates the same (slot, role) order
        # the request was built from, so the two zip back together by position.
        sequence = spec.role_sequence()
        days: list[DayPlan] = []

        for day_payload in envelope.get("days") or []:
            returned = [
                (slot.get("slot"), recipe)
                for slot in (day_payload.get("slots") or [])
                for recipe in (slot.get("recipes") or [])
            ]
            by_slot: dict[str, list[MealCourse]] = {}
            for (expected_slot, role), (slot_name, recipe) in zip(sequence, returned):
                recipe_id = str(recipe.get("recipe_id") or "").strip()
                if not recipe_id:
                    continue
                slot = str(slot_name or expected_slot)
                by_slot.setdefault(slot, []).append(
                    MealCourse(
                        recipe_id=recipe_id,
                        title=str(recipe.get("title") or ""),
                        ingredients="",
                        directions="",
                        nutrition=_meal_nutrition(recipe),
                        image_url=recipe.get("image_url"),
                        role=role,
                    )
                )

            meals = [
                Meal(meal_type=slot, plates=plates)
                for slot, plates in by_slot.items()
                if plates
            ]

            if not days:  # day 1 only
                for meal in meals:
                    anchor = pinned.get(meal.meal_type)
                    if anchor is None:
                        continue
                    # Replace the main plate, keeping any side or dessert the
                    # spec asked for — someone who wants apple pie for breakfast
                    # and a side with it wants both.
                    for index, plate in enumerate(meal.plates):
                        if plate.role == "main":
                            meal.plates[index] = MealCourse(
                                recipe_id=anchor.recipe_id,
                                title=anchor.title,
                                ingredients=anchor.ingredients,
                                directions=anchor.directions,
                                nutrition=getattr(anchor, "nutrition", None),
                                role="main",
                            )
                            logger.info("Anchored %s to %r", meal.meal_type, anchor.title)
                            break
            if meals:
                days.append(DayPlan(day=int(day_payload.get("day") or len(days) + 1),
                                    meals=meals))

        if not days:
            logger.warning("plan_meals returned no usable days for %s", spec.describe())
            return None

        # Say what could not be done, in the plan itself.
        #
        # Three separate things the member is entitled to hear, and none of them
        # is an error: a preference that had to be dropped to fill a slot, a
        # plate the corpus simply could not supply, and a shape that is not a
        # balanced day. Silently returning a plan that is short a course, or is
        # three desserts, is how an assistant becomes something you cannot
        # trust to tell you when it fell short.
        parts = [f"Planned {spec.describe()} from your preferences"]

        expected = spec.total_plates
        produced = sum(len(m.plates) for d in days for m in d.meals)
        if produced < expected:
            parts.append(
                f"{expected - produced} of {expected} plates could not be "
                "filled from the recipes that match your requirements"
            )
        if notes:
            parts.append("; ".join(notes))

        concerns = spec.concerns()
        reasoning = ". ".join(parts) + "."
        if concerns:
            reasoning += " " + " ".join(concerns)

        return MealPlan.from_days(days, reasoning=reasoning)

    @staticmethod
    def _assemble_from_pool(
        candidates: dict, note: str = ""
    ) -> list[ScoredPlan]:
        """Take the top candidate per slot, unranked.

        Used when the grader cannot rank — it raised, or returned nothing. The
        pool it would have ranked is already there and already respects every
        constraint, so there is nothing to re-fetch: the deterministic order
        `plan_meals` returned (planning tier, then Nutri-Score, then curated
        source) is a perfectly reasonable plan on its own.

        This used to make a second call to `plan_meals` for exactly the pool it
        already had in hand.

        Scored 0 with an explicit reason rather than a fabricated rating: the
        score is the grader's output, and inventing one makes an unranked plan
        indistinguishable from a well-rated one everywhere downstream.
        """
        missing = [s for s in ("breakfast", "lunch", "dinner") if not candidates.get(s)]
        if missing:
            logger.warning("cannot assemble a day — no candidates for %s", missing)
            return []

        reasoning = "Assembled directly from your constraints"
        if note:
            reasoning += f" ({note})"
        return [
            ScoredPlan(
                breakfast=candidates["breakfast"][0],
                lunch=candidates["lunch"][0],
                dinner=candidates["dinner"][0],
                score=0,
                reasoning=reasoning + ".",
            )
        ]


def _fetch_candidate_pool(
    *,
    profile: dict,
    cuisines: list[str],
    liked_ingredients: list[str],
    max_minutes: int | None,
    exclude_recipe_ids: list[str],
    favorite_recipe_ids: list[str] | None = None,
    limit_per_slot: int,
    slots: tuple[str, ...] = ("breakfast", "lunch", "dinner"),
) -> dict[str, list[CandidateRecipe]]:
    """A pool of candidates per slot, from `/api/v2/tools/plan_meals`.

    `plan_meals` returns an assembled plan, but asking it for N recipes per slot
    makes it a candidate source — and a better one than the v1 endpoint it
    replaced, because the pool has already had the member's cuisine, mood,
    flavour, time and Nutri-Score preferences applied, and has already excluded
    anything withdrawn from automated planning. None of that was reachable from
    Neo4j, which holds no annotations at all.

    Liked ingredients are *not* forwarded. The two endpoints disagree about
    `include_ingredients`: v1 ranked by it, `plan_meals` requires it. Sending a
    member's likes would demand chickpeas in every breakfast, and there is no
    such recipe — the slot would come back empty.

    Returns `{}` on failure. The caller treats an empty slot as "no plan", which
    is honest: a pool this path could not build is not one the grader can rank.
    """
    from services.plan_client import PLANNER

    try:
        envelope = PLANNER.plan_meals(
            days=1,
            slots=slots,
            count_per_slot=limit_per_slot,
            allergens=profile.get("allergies") or [],
            diet=normalize_diet_tags(profile.get("diet")),
            cuisines=cuisines,
            exclude_ingredients=profile.get("food_dislikes") or [],
            exclude_recipe_ids=exclude_recipe_ids,
            favorite_recipe_ids=(
                favorite_recipe_ids
                if favorite_recipe_ids is not None
                else profile.get("favorite_recipe_ids") or []
            ),
            max_minutes=max_minutes,
            min_nutri_score=profile.get("min_nutri_score"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("plan_meals candidate fetch failed: %s", exc)
        return {}

    notes = PLANNER.describe_relaxations(envelope)
    if notes:
        logger.info("Candidate pool relaxations: %s", notes)

    pool = PLANNER.to_candidates(envelope, allergens=profile.get("allergies") or [])
    # A slot the endpoint could not fill is absent from the envelope; the caller
    # checks for empties, so make the absence explicit.
    return {slot: pool.get(slot, []) for slot in slots}
