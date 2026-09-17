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
from services import (
    intent_facets, meal_composer, pantry_service, plan_parameters, plate_critic,
)
from services.candidates_client import CANDIDATES, effective_diet, screening_allergens
from services import turn_budget

logger = logging.getLogger(__name__)

# Candidates requested per meal slot from RecipeWrangler. 8 per slot bounds the
# grading space (8³ = 512 combos, sampled down by DocumentGrader) while keeping
# enough variety for refinements to find alternatives.
CANDIDATE_LIMIT = int(os.getenv("FOODCHAT_CANDIDATE_LIMIT", "8"))


def _day_kcal_target(profile: dict) -> float | None:
    """The member's daily calorie budget, or None when nobody set one.

    Delegated to `PlanBrief`, which already owns this rule — including the part
    that matters: a target nobody set is None, not a default, because a plate
    scored against 2000 kcal the member never chose is being marked down for
    someone else's number.
    """
    from models.plan_brief import PlanBrief

    return PlanBrief.build(profile).kcal_target



# What each food-waste setting asks the grader for. Deliberately phrased as a
# trade rather than a rule: reuse pulls against variety, and the member chose
# which side of that they want.
_WASTE_PREFERENCE = {
    "reuse": (
        "The user wants to reduce food waste. Prefer combinations whose meals "
        "share fresh ingredients with each other, so fewer things are bought "
        "and left to spoil — but not at the cost of the day being repetitive."
    ),
    "strict": (
        "The user wants the smallest possible shopping list. Strongly prefer "
        "combinations whose meals share ingredients, even where that makes the "
        "day less varied. They have asked for this trade explicitly."
    ),
}


def _repeat_note(slots: list[str]) -> str:
    """One clause naming the meals that had to reuse a recent dish, or ""."""
    if not slots:
        return ""
    where = ", ".join(slots)
    return (
        f"Your collection had nothing new left for {where}, so that "
        f"{'meal repeats' if len(slots) == 1 else 'those meals repeat'} "
        "something you have had recently"
    )


def _with_note(plans: list[ScoredPlan], note: str) -> list[ScoredPlan]:
    """The same plans, each carrying `note`. Unchanged when there is none."""
    if not note or not plans:
        return plans
    return [
        ScoredPlan(
            score=plan.score,
            reasoning=f"{plan.reasoning} {note}." if plan.reasoning else f"{note}.",
            slots=dict(plan.slots),
        )
        for plan in plans
    ]



def _every_plate_has_a_candidate(pools_by_day: dict, spec) -> bool:
    """Whether every requested plate of every day came back with something."""
    if not pools_by_day:
        return False
    for pools in pools_by_day.values():
        for slot in spec.meals:
            for role in spec.roles_for(slot):
                if not pools.get((slot, role)):
                    return False
    return True


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
        avoid_recent: list[str] | None = None,
        window_offset: int = 0,
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

        ``profile["_pantry"]`` (transient, stashed by chat_service like the
        other underscore keys) lists on-hand ingredients to use up: matching
        recipes are folded into the pool coverage-first and the grader is told
        to prefer combinations that use them. A boost, never a filter — the
        pool stays constraint-correct and full-sized either way.

        ``avoid_recent`` is what the member was just served. A SOFT exclusion,
        and the difference matters: `exclude_recipe_ids` is a decision (a
        downvote, "not that one") and is never relaxed, while this is only a
        preference for something new. It is dropped the moment it would empty a
        slot, because a member on a narrow diet must not be told no meals exist
        for the crime of asking twice.
        """
        pinned = pinned or {}
        pantry = pantry_service.normalize_items(profile.pop("_pantry", None) or [])

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
        hard_exclusions = (
            [r.recipe_id for r in pinned.values()] + list(exclude_recipe_ids or [])
        )
        # Recently served dishes are asked for last and given up first.
        recent = [r for r in (avoid_recent or []) if r not in hard_exclusions]
        candidates = _fetch_candidate_pool(
            profile=profile,
            cuisines=cuisines,
            liked_ingredients=liked_ingredients,
            max_minutes=max_minutes,
            exclude_recipe_ids=hard_exclusions + recent,
            limit_per_slot=CANDIDATE_LIMIT,
            offset=window_offset,
        )
        repeated_slots: list[str] = []
        if recent:
            # A slot the history emptied is refetched without it. Only that
            # slot: the other slots keep their fresh dishes, so asking twice
            # costs the repeat of one meal rather than of the whole day.
            thin = [slot for slot, pool in (candidates or {}).items() if not pool]
            if not candidates or thin:
                refetched = _fetch_candidate_pool(
                    profile=profile,
                    cuisines=cuisines,
                    liked_ingredients=liked_ingredients,
                    max_minutes=max_minutes,
                    exclude_recipe_ids=hard_exclusions,
                    limit_per_slot=CANDIDATE_LIMIT,
                    offset=window_offset,
                )
                if not candidates:
                    candidates = refetched
                    repeated_slots = sorted(refetched)
                else:
                    for slot in thin:
                        if refetched.get(slot):
                            candidates[slot] = refetched[slot]
                            repeated_slots.append(slot)
                if repeated_slots:
                    logger.info(
                        "Nothing new left for %s — reusing recent dishes there",
                        ", ".join(repeated_slots),
                    )

        # Pantry (food waste): fold in recipes that use the member's on-hand
        # ingredients, coverage-first. The per-item fan-out exists because
        # `plan_meals` ANDs `include_ingredients` — the whole pantry at once
        # would demand every item in every recipe and empty the slots.
        if pantry and candidates:
            pantry_pools = pantry_service.fetch_pantry_candidates(
                profile, pantry,
                exclude_recipe_ids=(
                    [r.recipe_id for r in pinned.values()]
                    + list(exclude_recipe_ids or [])
                ),
                # The same cuisine and cooking-time constraints the base pool
                # was fetched with. Omitting them let pantry candidates that
                # break the time slider rank ABOVE the compliant ones, since
                # the merge sorts coverage-first.
                cuisines=cuisines,
                max_minutes=max_minutes,
            )
            candidates = pantry_service.merge_pantry_pool(
                candidates, pantry_pools, pantry, CANDIDATE_LIMIT,
            )
            logger.info("Pantry boost applied for: %s", ", ".join(pantry))

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

        # Reason about the picks BEFORE anything selects one.
        #
        # `plan_meals` returns a deterministic order — planning tier, then
        # Nutri-Score, then curated source. That is a good tiebreak and a poor
        # decision, and taking its head had two consequences a member saw: an
        # 11%-of-the-day breakfast beside a 45% lunch, every constraint
        # honoured; and the same three dishes every time, because a fixed order
        # plus "take the first" is a function with one output.
        #
        # Reordering here rather than overriding a pick later means the grader
        # ranks a pool whose head already fits the slot, and the unranked
        # fallback's "first" is a reasoned first. Nothing is dropped: a pool
        # this emptied would turn a quality opinion into "no meals exist".
        candidates, critic_findings = plate_critic.rank_pool(
            candidates, kcal_target=_day_kcal_target(profile),
        )
        if critic_findings:
            logger.info("Plate critic: %s", "; ".join(critic_findings[:4]))

        # The grader hears about the pantry as a preference in the query text;
        # every user-facing coverage CLAIM still comes from the deterministic
        # matcher (pantry_service), never from the model.
        grader_query = query
        if pantry:
            grader_query = (
                f"{query}\n\nThe user has these ingredients at home to use up "
                f"(reduce food waste): {', '.join(pantry)}. Prefer combinations "
                "that together use as many of them as possible."
            )
        # The food-waste setting, on the path where it did nothing.
        #
        # `waste_mode` had exactly one reader — the weekly planner's scorer —
        # and its own docstring said the daily path "hears the same setting as
        # prose via describe". It does not: `describe` builds the canonical
        # message for a slider APPLY, so a member with reuse standing got it
        # once, on the turn they set it, and never again. Every later "plan my
        # day" ignored it.
        #
        # It belongs in the grader's query rather than in a filter, because it
        # is a property of a COMBINATION — whether three meals share a bunch of
        # coriander — and the grader is the only thing here that sees all three
        # at once. No recipe is excluded for it.
        waste = plan_parameters.waste_mode(profile.get("plan_parameters") or {})
        if waste != "off":
            grader_query += (
                "\n\n" + _WASTE_PREFERENCE[waste]
            )
            logger.info("Food-waste preference reaching the grader: %s", waste)

        # Running late? Take the same exit a failed grader takes. The pool is
        # already constraint-correct and ordered by planning tier and
        # Nutri-Score, so its top pick is a real plan — just not a ranked one.
        # An unranked plan the member receives beats a ranked one the gateway
        # cuts off before it arrives.
        # A repeat the corpus forced is said, not hidden. The member asked for
        # something new and is getting a dish they have just had; that is a fact
        # about their collection, and finding out by recognising the photo is
        # worse than being told.
        note = _repeat_note(repeated_slots)

        if turn_budget.skip("plan grading", turn_budget.COST_GRADING):
            return _with_note(self._assemble_from_pool(
                candidates, "not ranked — the plan was taking too long",
            ), note)

        try:
            scored = self.grader.grade_daily_plans(
                grader_query, candidates, profile, feedback_history,
                # Not only in the prose above: the batch now contains a day
                # built to use these up, so the instruction has something to
                # act on rather than whatever the sampler drew.
                prefer_items=pantry,
            )
        except Exception as exc:  # noqa: BLE001
            # The grader is a model call. Losing it should cost the *ranking*,
            # not the plan: the pool is already constraint-correct and already
            # ordered by planning tier and Nutri-Score, so serving its top pick
            # beats an apology.
            logger.error("Grader failed (%s) — serving the unranked pool", exc)
            return _with_note(
                self._assemble_from_pool(candidates, "not ranked — grader unavailable"),
                note,
            )

        if not scored:
            logger.warning("Grader returned no plans — serving the unranked pool")
            fallback = self._assemble_from_pool(candidates)
            if fallback:
                return _with_note(fallback, note)

        logger.info(
            "Pipeline produced %d scored plan(s); best score=%s",
            len(scored), scored[0].score if scored else "n/a",
        )
        return _with_note(scored, note)

    def plan_structured(
        self,
        profile: dict,
        spec: "PlanSpec",
        exclude_recipe_ids: list[str] | None = None,
        avoid_recent: list[str] | None = None,
        window_offset: int = 0,
        pinned: dict | None = None,
        query: str = "",
    ) -> "MealPlan | None":
        """Generate a plan of any shape — N days, N meals, multi-plate meals.

        ``query`` is the member's request. This method had no such parameter at
        all: the shape was honoured and the words were not, so "three days of
        Italian dinners with a salad on the side" produced a correctly-shaped
        plan built from profile fields only — and the reply was then phrased
        around a request that had reached nothing. Facets and claim tags are
        extracted from it upstream; it is carried here so the pool can be
        ranked against what was actually asked for.

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

        # Pantry boost for the structured path. Fetching is `meal_composer`'s
        # job now — including the cuisine split, the diet, the allergens and
        # the time ceiling, which it applies for both callers so the weekly and
        # structured pools cannot disagree about the member's constraints.
        # What stays here is the boost list: `plan_meals`'s only soft rank
        # signal is the favourites float, so pantry-matching ids ride it. Hard
        # filters still decide eligibility; this reorders, never widens.
        pantry = pantry_service.normalize_items(profile.pop("_pantry", None) or [])
        # The member's OWN favourites, plus any pantry-matching ids. This path
        # sent only the pantry boost, so a member who had said yes to the
        # favourites offer had their favourites ignored on every plan whose
        # shape was not the default three meals — and with no pantry the field
        # went out empty.
        boost_ids = list(profile.get("favorite_recipe_ids") or []) \
            if profile.get("use_favorites") is not False else []
        if pantry:
            for recipe_id in pantry_service.pantry_boost_ids(profile, pantry):
                if recipe_id not in boost_ids:
                    boost_ids.append(recipe_id)

        # A pool per PLATE, and more than one candidate in it when the meal has
        # more than one plate.
        #
        # This path used to take RecipeWrangler's first recipe for each plate —
        # assembly "delegated wholly to plan_meals", in this module's own
        # earlier words. The request was already one entry per plate with its
        # own course types, so the role-scoped fetch was never the blocker;
        # what was missing was asking for a CHOICE and then making it. Nothing
        # was in a position to notice that a lasagne main and a macaroni salad
        # side are two plates of pasta.
        #
        # Single-plate specs still ask for one candidate each, so the common
        # three-meal plan sends exactly the request it always did.
        multi_plate = any(len(spec.roles_for(slot)) > 1 for slot in spec.meals)
        per_plate = meal_composer.POOL_PER_PLATE if multi_plate else 1

        # Recently served dishes are a SOFT exclusion here too: asked for, and
        # given up the moment they would leave a plate unfillable. A shaped
        # plan has more plates to fill than a plain day, so it runs out of new
        # dishes sooner — and a member who asked twice must not be told their
        # shape is impossible.
        recent = [
            r for r in (avoid_recent or [])
            if r not in (exclude_recipe_ids or [])
        ]
        pools_by_day, relaxations = meal_composer.role_pools(
            profile, spec,
            exclude_recipe_ids=list(exclude_recipe_ids or []) + recent,
            boost_ids=boost_ids,
            per_plate=per_plate,
            offset=window_offset,
        )
        if recent and not _every_plate_has_a_candidate(pools_by_day, spec):
            logger.info(
                "Nothing new left to fill every plate — planning without the "
                "recently-served exclusion"
            )
            pools_by_day, relaxations = meal_composer.role_pools(
                profile, spec,
                exclude_recipe_ids=list(exclude_recipe_ids or []),
                boost_ids=boost_ids,
                per_plate=per_plate,
                offset=window_offset,
            )
        if not pools_by_day:
            return None

        kcal_target = _day_kcal_target(profile)
        # Compose every meal first, then judge them all in one call. Judging as
        # each meal is built would mean one round trip per meal, which for a
        # week with a side at dinner is seven inside one turn budget.
        options: dict[str, list] = {}
        placement: dict[str, tuple[int, str]] = {}
        unfilled: list[str] = []
        used: set = set()
        for day in sorted(pools_by_day):
            for slot in spec.meals:
                roles = spec.roles_for(slot)
                composed = meal_composer.compose(
                    slot, roles, pools_by_day[day],
                    kcal_split=spec.kcal_split(slot),
                    meal_kcal_target=(
                        kcal_target / max(1, len(spec.meals)) if kcal_target else None
                    ),
                    exclude_ids=used,
                    enrichment={},
                )
                if not composed:
                    # NOT a silent `continue`.
                    #
                    # It was, and a member asking to reuse ingredients got back
                    # a plan containing only lunch: dinner's plates could not be
                    # filled from what was left after the exclusions, so dinner
                    # was dropped and the reply described the lunch as though
                    # that were the day. A plan short a whole meal has to say so
                    # — losing a meal is not a detail.
                    logger.warning(
                        "day %s %s could not be filled — reporting it, not "
                        "dropping it", day, slot,
                    )
                    unfilled.append(slot)
                    continue
                label = meal_composer.label_for(day, slot)
                options[label] = composed
                placement[label] = (day, slot)
                # Reserved against the measured winner so two meals cannot
                # claim the same dish while the judge is still deciding. The
                # judge only ever swaps within one meal's own options, all of
                # which were drawn from that plate's pool, so a swap cannot
                # introduce a duplicate this reservation missed.
                used.update(composed[0].recipe_ids)

        # The food-waste setting, on the path where a multi-plate plan is
        # built. It reached the classic grader's query and the weekly scorer,
        # and nothing here — so a member who asked to reuse ingredients and had
        # a shaped plan got the setting applied to nothing at all.
        #
        # It rides the judge's query, which is where a preference about how
        # dishes go TOGETHER belongs: sharing a bunch of coriander is a property
        # of the combination, not of any one dish.
        waste = plan_parameters.waste_mode(profile.get("plan_parameters") or {})
        judge_query = query
        if waste != "off":
            judge_query = f"{query}\n\n{_WASTE_PREFERENCE[waste]}"
            logger.info("Food-waste preference reaching the meal judge: %s", waste)

        chosen = meal_composer.judge(judge_query, options)

        # A judge that moves a meal off its measured winner can, in principle,
        # land on a dish another meal already took: the reservation above was
        # made against the winner, and a plate's pool is not guaranteed
        # disjoint from another plate's. So the result is swept, and a collision
        # falls back to that meal's measured winner — which WAS reserved and is
        # therefore collision-free by construction. Cheaper and more certain
        # than re-running the whole reservation pass, and it degrades toward
        # the answer arithmetic already produced.
        committed: set = set()
        for label in sorted(chosen):
            composition = chosen[label]
            if committed & set(composition.recipe_ids):
                measured = options[label][0]
                if composition is not measured:
                    logger.info(
                        "%s: the judged pick repeats a dish already on the "
                        "plan — keeping the measured one", label,
                    )
                    chosen[label] = measured
                    composition = measured
            committed.update(composition.recipe_ids)

        days: list[DayPlan] = []
        by_day_meals: dict[int, list] = {}
        for label, composition in chosen.items():
            day, slot = placement[label]
            plates = [
                MealCourse(
                    recipe_id=plate.recipe_id,
                    title=plate.title,
                    # The envelope carries the text; the pantry coverage
                    # matcher and the UI read it downstream.
                    ingredients=str(getattr(plate.candidate, "ingredients", "") or ""),
                    directions=str(getattr(plate.candidate, "directions", "") or ""),
                    nutrition=getattr(plate.candidate, "nutrition", None),
                    image_url=getattr(plate.candidate, "image_url", None),
                    role=plate.role,
                )
                for plate in composition.plates
            ]
            if plates:
                by_day_meals.setdefault(day, []).append(Meal(meal_type=slot, plates=plates))

        findings = [
            f"{label}: {finding}"
            for label, composition in sorted(chosen.items())
            for finding in composition.findings
        ]
        if findings:
            logger.info("Composition findings: %s", "; ".join(findings[:6]))

        for day in sorted(by_day_meals):
            meals = by_day_meals[day]

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
                days.append(DayPlan(day=int(day), meals=meals))

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

        # A missing MEAL first, and in those words.
        #
        # "3 of 8 plates could not be filled" is technically the same fact and
        # is not the same sentence: a member reads it as a plate short, not as
        # "you have no dinner". Naming the meals is what makes it legible.
        missing = sorted(set(unfilled))
        if missing:
            parts.append(
                f"there is no {', no '.join(missing)} in this plan — nothing in "
                "your collection could fill it under these requirements"
            )

        expected = spec.total_plates
        produced = sum(len(m.plates) for d in days for m in d.meals)
        if produced < expected and not missing:
            parts.append(
                f"{expected - produced} of {expected} plates could not be "
                "filled from the recipes that match your requirements"
            )
        if relaxations:
            parts.append("; ".join(relaxations))
        # Composition findings deliberately do NOT go in here.
        #
        # They did, and the member read this on their plan:
        #
        #   "day 1 dinner: chosen for the table: The hearty burgers are
        #    balanced by the bright ginger-dressed beans; day 1 lunch: side is
        #    52 kcal against a 300 kcal share"
        #
        # "chosen for the table" is a marker this code invented for its own
        # bookkeeping, "against a 300 kcal share" is internal accounting, and
        # "day 1" is noise on a one-day plan. All of it is true and none of it
        # is a sentence anybody wants about their dinner.
        #
        # The findings stay where they belong: the log, and `plan_value` for the
        # response writer, which phrases them. This field is the plan's own
        # short factual account — shape, relaxations, plates that could not be
        # filled — and it is rendered verbatim on the canvas.

        concerns = spec.concerns()
        reasoning = ". ".join(parts) + "."
        if concerns:
            reasoning += " " + " ".join(concerns)

        plan = MealPlan.from_days(days, reasoning=reasoning)
        # Why the dishes of each meal go together, for the reply to phrase.
        # Attached rather than folded into `reasoning`, which is rendered
        # verbatim on the canvas — the same reason the metrics are attached.
        plan.pairings = [
            f"{label.replace('day 1 ', '')}: {composition.pairing}"
            for label, composition in sorted(chosen.items())
            if composition.pairing
        ]
        return plan

    @staticmethod
    def _assemble_from_pool(
        candidates: dict, note: str = ""
    ) -> list[ScoredPlan]:
        """Take the top candidate per slot — now a REASONED top.

        Used when the grader cannot rank — it raised, or returned nothing. The
        pool it would have ranked is already there and already respects every
        constraint, so there is nothing to re-fetch: the deterministic order
        `plan_meals` returned (planning tier, then Nutri-Score, then curated
        source) is a reasonable tiebreak — but it is not a decision, which is
        why `plate_critic` has already reordered the pool by then. "First" here
        means the best-fitting candidate for the slot, not the first one the
        service happened to return.

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
        # The critic's arithmetic is NOT appended here. It is real and it is
        # internal: "226 kcal is 54% under what breakfast should carry" belongs
        # in the log, where someone debugging a pick will look for it, and not
        # on a member's plan. What they are owed is already in `note` — that
        # this plan was not ranked.
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
    offset: int = 0,
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
            allergens=screening_allergens(profile),
            diet=effective_diet(profile),
            **intent_facets.facet_kwargs(profile, cuisines),
            exclude_ingredients=profile.get("food_dislikes") or [],
            exclude_recipe_ids=exclude_recipe_ids,
            favorite_recipe_ids=(
                favorite_recipe_ids
                if favorite_recipe_ids is not None
                else profile.get("favorite_recipe_ids") or []
            ),
            max_minutes=max_minutes,
            min_nutri_score=profile.get("min_nutri_score"),
            # Where each slot's window starts. Recipes come back in pages, and
            # without this every request is page one.
            offset=offset,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("plan_meals candidate fetch failed: %s", exc)
        return {}

    notes = PLANNER.describe_relaxations(envelope)
    if notes:
        logger.info("Candidate pool relaxations: %s", notes)

    pool = PLANNER.to_candidates(envelope, allergens=screening_allergens(profile))
    # A slot the endpoint could not fill is absent from the envelope; the caller
    # checks for empties, so make the absence explicit.
    return {slot: pool.get(slot, []) for slot in slots}
