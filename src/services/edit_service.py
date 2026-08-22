"""
Verified slot editing (M4b) — "swap Tuesday's dinner for something lighter,"
and the result is PROVABLY lighter.

Flow: EditCommandExtractor parses the request → the target slot is resolved
on the active canvas (one conversational follow-up when ambiguous, via the
persisted clarification state kind="edit_slot") → replacement candidates are
fetched for that slot only (hard filters + current plan excluded) → the
directive becomes a measurable predicate checked against RecipeWrangler
nutrition BEFORE selection → the plan is PATCHED (only the target slot
changes; version+1, lineage preserved) → the response carries the
before/after proof in ``changed_slots``.

Honest failure: when no candidate passes both the hard constraints and the
predicate, we say so and offer the nearest miss instead of pretending.

Directive predicates (quantitative when nutrition data exists):
    lighter / lower calorie   → kcal_new ≤ 0.85 × kcal_old
    more protein / high prot. → protein_new > protein_old
    quicker / faster          → duration_new < duration_old
    vegetarian/vegan/…        → diet tag present on the replacement
    anything else             → no predicate (best-effort pick, noted as
                                unverified in the response facts)
"""

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agents import EditCommandExtractor
from models.recipe import CandidateRecipe, RecipeEnrichment
from models.session import MealCourse, MealPlan
from services import plan_parameters
from services.candidates_client import CANDIDATES, effective_diet, screening_allergens
from .session_service import SessionService
from .weekly_planner.day_summary import build_day_summaries
from .weekly_planner.explainability import build_weekly_explainability

logger = logging.getLogger(__name__)

MEAL_IDX = {"breakfast": 0, "lunch": 1, "dinner": 2}
IDX_MEAL = {v: k for k, v in MEAL_IDX.items()}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CANDIDATE_POOL = 8
LIGHTER_RATIO = 0.85

DIET_TAG_DIRECTIVES = {
    "vegetarian": "vegetarian", "vegan": "vegan",
    "gluten free": "gluten_free", "gluten-free": "gluten_free",
    "dairy free": "dairy_free", "dairy-free": "dairy_free",
    "pescatarian": "pescatarian",
}


@dataclass
class EditOutcome:
    """Result of an edit turn — mirrors the shapes ChatTurn carries."""
    text: str
    needs_clarification: bool = False
    meal_plan: Optional[object] = None          # MealPlan on daily edits
    weekly_meal_plan: Optional[object] = None   # WeeklyMealPlan on weekly edits
    changed_slots: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)   # for the ResponseWriter
    # True when a clarification reply didn't answer the slot question at all:
    # the state is cleared, NOTHING was logged, and the orchestrator should
    # route the message as a fresh turn instead of re-interrogating.
    unresolved: bool = False


class DirectivePredicate:
    """A measurable check the replacement must pass, given old/new enrichment."""

    def __init__(self, directive: str):
        self.directive = (directive or "different").lower().strip()
        self.kind, self.tag = self._classify(self.directive)

    # "something with zucchini", "one using the leftover chicken", "use up my
    # spinach" — an ingredient requirement, verifiable by text match.
    #
    # The capture is at most TWO words ("ground beef") and never crosses into
    # the rest of the sentence. A greedy [a-z\s-]{1,40} run swallowed whatever
    # followed — "zucchini please", "chicken instead", "zucchini and spinach" —
    # and a term like that hard-fails: the include fetch finds nothing, the
    # text match finds nothing, and the member is told no candidate satisfies a
    # request that used to produce an ordinary swap.
    _USES_RE = re.compile(
        r"(?:\busing\b|\bwith\b|\bthat\s+uses\b|\buses?\s+up\b)\s+"
        r"(?:(?:the|my|our|some|a|an|leftover|left-over|remaining)\s+)*"
        r"([a-z][a-z-]*(?:\s+[a-z][a-z-]*)?)"
    )
    # Words that can follow an ingredient without belonging to it.
    _TRAILING_FILLER = frozenset({
        "please", "instead", "too", "also", "again", "thanks", "tonight",
        "today", "tomorrow", "now", "and", "or", "but", "for", "in", "on",
        "with", "this", "that", "time", "one", "ones",
    })
    # An opener that means the phrase never named an ingredient. Comparatives
    # matter most: "with less salt" is a directive about a quantity, and
    # searching for an ingredient called "less salt" finds nothing at all.
    _NOT_AN_INGREDIENT = frozenset({
        "something", "anything", "nothing", "it", "them", "one",
        "more", "less", "fewer", "lower", "higher", "extra", "much", "many",
        "other", "another", "different", "new", "same", "better",
    })

    @staticmethod
    def _ingredient_term(directive: str) -> Optional[str]:
        """The ingredient a directive requires, or None when it names none.

        Returning None is the safe direction: the directive falls through to
        "unverified", which still produces a swap. Claiming a term that is not
        an ingredient produces a dead end instead.
        """
        match = DirectivePredicate._USES_RE.search(directive)
        if not match:
            return None
        words = [w.strip(".!?,;:") for w in match.group(1).split()]
        words = [w for w in words if w]
        while words and words[-1] in DirectivePredicate._TRAILING_FILLER:
            words.pop()
        if not words or words[0] in DirectivePredicate._NOT_AN_INGREDIENT:
            return None
        return " ".join(words[:2])

    @staticmethod
    def _classify(d: str) -> tuple[str, Optional[str]]:
        if any(w in d for w in ("lighter", "lower calorie", "less calorie", "fewer calorie", "light ")) or d == "light":
            return "lighter", None
        if "protein" in d:
            return "more_protein", None
        if any(w in d for w in ("quicker", "faster", "less time", "quick ")) or d == "quick":
            return "quicker", None
        for phrase, tag in DIET_TAG_DIRECTIVES.items():
            if phrase in d:
                return "diet_tag", tag
        # After the diet tags so "with something vegetarian" stays a diet
        # directive rather than a hunt for an ingredient named "something".
        term = DirectivePredicate._ingredient_term(d)
        if term:
            return "uses_ingredient", term
        return "unverified", None

    @property
    def verifiable(self) -> bool:
        return self.kind != "unverified"

    def passes(self, old: Optional[RecipeEnrichment], new: Optional[RecipeEnrichment]) -> bool:
        """True when `new` provably satisfies the directive vs `old`.

        Missing measurements fail closed (except unverified directives, which
        always pass) — a swap we cannot verify must not claim compliance.
        """
        if self.kind == "unverified":
            return True
        if new is None:
            return False
        if self.kind == "lighter":
            if old is None or old.kcal is None or new.kcal is None:
                return False
            return new.kcal <= LIGHTER_RATIO * old.kcal
        if self.kind == "more_protein":
            if old is None or old.protein_g is None or new.protein_g is None:
                return False
            return new.protein_g > old.protein_g
        if self.kind == "quicker":
            if old is None or old.duration is None or new.duration is None:
                return False
            return new.duration < old.duration
        if self.kind == "diet_tag":
            return self.tag in (new.tags or [])
        if self.kind == "uses_ingredient":
            # Enrichment carries no ingredient text; the check runs against
            # the candidate's own text in `_find_replacement`. From here
            # alone this fails closed, per the rule above.
            return False
        return False

    def nearest(self, old: Optional[RecipeEnrichment], candidates: dict) -> Optional[str]:
        """recipe_id of the nearest miss for quantitative predicates."""
        if self.kind == "lighter" and old and old.kcal is not None:
            measurable = {rid: e for rid, e in candidates.items() if e.kcal is not None}
            if measurable:
                return min(measurable, key=lambda rid: measurable[rid].kcal)
        if self.kind == "more_protein":
            measurable = {rid: e for rid, e in candidates.items() if e.protein_g is not None}
            if measurable:
                return max(measurable, key=lambda rid: measurable[rid].protein_g)
        return None


# Unverified directives that mean "change it" rather than naming a dish.
_GENERIC_DIRECTIVES = frozenset({
    "different", "something else", "something different", "anything",
    "another", "another one", "surprise me", "change it", "swap it",
    "new", "something new", "other", "else",
})


# Ordinal words for "the second day". Only as far as a plan can go.
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7,
}
_DAY_N_RE = re.compile(r"\bday\s*([1-7])\b")
_ORDINAL_RE = re.compile(
    r"\b(" + "|".join(_ORDINALS) + r")\s+day\b"
)


def _named_day(message: str, *, weekdays: bool) -> Optional[int]:
    """The day the member named, read from their own words. None when none.

    A deterministic fallback for when the extractor returns no day — which it
    routinely does on a multi-day DAILY plan, because the prompt telling it
    how to answer says the day field is "only for weekly plans". That prompt
    is Langfuse-managed: editing the text here would ship dead (`sync_prompts`
    skips prompts that already exist), and a new prompt name is a bigger
    change than this needs. So the day is read here instead, from the same
    message, with no model call.

    `weekdays` is False for a multi-day daily plan on purpose: such a plan
    carries no calendar anchoring — its day 1 is "the first day", not Monday —
    so mapping "Wednesday" onto index 3 would be a guess presented as a fact.
    """
    text = (message or "").lower()
    match = _DAY_N_RE.search(text)
    if match:
        return int(match.group(1))
    match = _ORDINAL_RE.search(text)
    if match:
        return _ORDINALS[match.group(1)]
    if weekdays:
        for idx, name in enumerate(DAY_NAMES, start=1):
            if re.search(rf"\b{name.lower()}\b", text):
                return idx
    return None


def _names_a_dish(directive: str) -> bool:
    """Whether an unverified directive reads as a dish name.

    Errs toward yes: a false positive costs one name search that returns
    nothing and falls back to the old behaviour; a false negative silently
    hands the member the slot's default instead of what they asked for.
    """
    d = (directive or "").strip().lower()
    return bool(d) and d not in _GENERIC_DIRECTIVES and len(d) > 2


class EditService:
    """Targeted single-slot plan edits with verified directives."""

    def __init__(self, session_service: SessionService, client=None, extractor=None):
        self.session_service = session_service
        self.client = client or CANDIDATES
        self.extractor = extractor or EditCommandExtractor()

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #

    def process(self, session_id: str, message: str) -> EditOutcome:
        session = self._get_session(session_id)

        canvas = session.active_canvas
        if canvas is None:
            text = ("There's no plan on the canvas yet — ask me for a daily or "
                    "weekly plan first, then I can swap meals in it.")
            self.session_service.add_message(session_id, "user", message)
            self.session_service.add_message(session_id, "assistant", text)
            return EditOutcome(text=text)

        command = self.extractor.extract(message, plan_type=canvas.plan_type)
        if command is None:
            # Not parseable as a slot edit — but the classifier heard a
            # request to change the plan, and "could you rephrase?" answers
            # that request with homework. Nothing is logged here; the
            # orchestrator reroutes the message through the refinement path,
            # which understands free text and stores the turn itself.
            return EditOutcome(text="", unresolved=True)

        self.session_service.add_message(session_id, "user", message)

        # Every edit needs a meal type. A day is needed whenever the plan HAS
        # days — weekly always, and a daily canvas holding a multi-day plan,
        # which used to silently edit day 1 whatever the member named.
        needs_day = self._needs_day(session, canvas)
        if needs_day and command.get("day") is None:
            command["day"] = _named_day(message, weekdays=canvas.plan_type == "weekly")

        missing_slot = (
            command.get("needs_slot_clarification")
            or command.get("meal_type") is None
            or (needs_day and command.get("day") is None)
        )
        if missing_slot:
            question = command.get("question") or (
                "Which meal should I swap — breakfast, lunch, or dinner"
                + (", and on which day?" if needs_day else "?")
            )
            self.session_service.set_clarification_state(session_id, {
                "kind": "edit_slot",
                "original_message": message,
                "command": command,
                "plan_type": canvas.plan_type,
                "needs_day": needs_day,
            })
            self.session_service.add_message(session_id, "assistant", question)
            return EditOutcome(text=question, needs_clarification=True)

        return self._execute(session, command, message)

    def continue_clarification(self, session_id: str, message: str) -> EditOutcome:
        """Resume after the slot question — re-extract over combined context.

        A reply that still doesn't resolve the slot usually isn't an answer
        at all (a preference, a new question, a topic change), so instead of
        re-asking we return ``unresolved=True`` with no messages logged and
        let the orchestrator classify the turn fresh.
        """
        session = self._get_session(session_id)
        pending = session.clarification or {}
        self.session_service.clear_clarification_state(session_id)

        combined = f"{pending.get('original_message', '')} — {message}"
        plan_type = pending.get("plan_type", "daily")
        command = self.extractor.extract(combined, plan_type=plan_type)
        # `needs_day` is carried in the state rather than re-derived: the rule
        # must be the same one that asked the question, even if the canvas
        # moved on between turns.
        needs_day = bool(pending.get("needs_day", plan_type == "weekly"))
        if command is not None and needs_day and command.get("day") is None:
            command["day"] = _named_day(combined, weekdays=plan_type == "weekly")
        if command is None or command.get("meal_type") is None or (
            needs_day and command.get("day") is None
        ):
            return EditOutcome(text="", unresolved=True)

        self.session_service.add_message(session_id, "user", message)
        return self._execute(session, command, pending.get("original_message", message))

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def _execute(self, session, command: dict, original_message: str) -> EditOutcome:
        canvas = session.active_canvas
        meal_type = command["meal_type"]
        predicate = DirectivePredicate(command.get("directive", "different"))

        if canvas.plan_type == "daily":
            return self._edit_daily(
                session, meal_type, predicate, original_message,
                day=command.get("day"),
            )
        return self._edit_weekly(session, command.get("day"), meal_type, predicate, original_message)

    def _needs_day(self, session, canvas) -> bool:
        """Whether an edit on this canvas has to name a day.

        Weekly always does. A daily canvas does when the plan on it spans more
        than one day — the shape `plan_structured` produces and that the daily
        edit path used to flatten.
        """
        if canvas is None:
            return False
        if canvas.plan_type == "weekly":
            return True
        try:
            plan = session.get_current_daily_plan()
        except Exception:  # noqa: BLE001
            return False
        return plan is not None and len(plan.day_plans) > 1

    def _find_replacement(
        self, session, meal_type: str, predicate: DirectivePredicate,
        old_recipe_id: str, exclude_ids: list[str],
    ) -> tuple[Optional[CandidateRecipe], Optional[RecipeEnrichment], Optional[RecipeEnrichment], dict]:
        """Fetch slot candidates, verify the predicate, pick the replacement.

        Returns (choice, old_enrichment, new_enrichment, facts).
        """
        # The standing constraints, applied to the profile this fetch uses. An
        # edit is a fetch like any other: "swap the dinner, but keep it under
        # 20 minutes" has to narrow the replacement pool, and the pool is
        # narrowed by `plan_parameters.max_duration_minutes(profile[...])`.
        from services import plan_parameters, turn_intake

        profile = plan_parameters.apply_state(
            dict(session.user_profile or {}),
            turn_intake.current(session.session_id, session_service=self.session_service),
        )

        # A directive that names a dish resolves BY NAME, before any slot
        # candidates. "i want apple pie for breakfast" used to classify as an
        # unverified predicate that every breakfast candidate trivially
        # passes — so the member got the slot's top-ranked muffins while the
        # reply claimed a "best match for apple pie". An explicit name beats
        # the slot's course taxonomy: someone asking for pie at breakfast has
        # already decided pie is breakfast food. Hard constraints still hold —
        # the name search runs with the member's allergens, diet and dislikes.
        if predicate.kind == "unverified" and _names_a_dish(predicate.directive):
            named = self._resolve_named_dish(predicate.directive, profile, exclude_ids)
            if named is not None:
                enrichment = self.client.fetch_details([old_recipe_id, named.recipe_id])
                facts = {
                    "directive": predicate.directive, "verified": False,
                    "named_dish": named.title,
                }
                return named, enrichment.get(old_recipe_id), enrichment.get(named.recipe_id), facts
            # Fall through to slot candidates, but say the truth about it.
            # (facts merged below via named_miss.)

        # Replacements come from the planning endpoint, like every other
        # candidate in the service. A swap that pulled from a source with no
        # annotations and no `planning_tier` could hand the user a recipe the
        # original plan was not allowed to contain — "make it lighter" is not
        # licence to reach outside the constraints.
        if predicate.kind == "uses_ingredient":
            # "Something with zucchini": fetch this slot with a single-item
            # hard include (see pantry_service — one item is exactly "must
            # use this"), then verify by text match. slot_candidates cannot
            # express the include, so it is only the fallback pool here.
            from services.candidates_client import CANDIDATES as _CANDS
            from services.pantry_service import fetch_pantry_candidates

            # Same cuisine filter `slot_candidates` applies, so the two
            # branches of this if/else cannot offer differently-constrained
            # pools — "make it lighter is not licence to reach outside the
            # constraints" has to hold for "something with zucchini" too.
            # (Neither branch applies the cooking-time slider; that gap is
            # older than the pantry work and belongs to slot_candidates.)
            _cuisines, _ = _CANDS.split_cuisines(profile.get("food_likes") or [])
            candidates = fetch_pantry_candidates(
                profile, [predicate.tag], slots=(meal_type,),
                exclude_recipe_ids=exclude_ids, per_item=CANDIDATE_POOL,
                cuisines=_cuisines,
            ).get(meal_type, [])
            if not candidates:
                candidates = self.client.slot_candidates(profile, meal_type, exclude_ids)
        else:
            candidates = self.client.slot_candidates(profile, meal_type, exclude_ids)

        if not candidates:
            return None, None, None, {"failure": "no candidates for this slot"}

        # One batch details call covers the predicate for old + all candidates.
        enrichment = self.client.fetch_details(
            [old_recipe_id] + [c.recipe_id for c in candidates]
        )
        old_rich = enrichment.get(old_recipe_id)

        if predicate.kind == "uses_ingredient":
            # Verified against the candidate's own ingredient text — the one
            # source that can prove "uses zucchini". Deterministic, no model.
            from services.pantry_service import matched_items

            passing = [
                c for c in candidates
                if matched_items(f"{c.title} {c.ingredients}", [predicate.tag])
            ]
        else:
            passing = [
                c for c in candidates
                if predicate.passes(old_rich, enrichment.get(c.recipe_id))
            ]

        facts: dict = {"directive": predicate.directive, "verified": predicate.verifiable}
        if predicate.kind == "unverified" and _names_a_dish(predicate.directive):
            # The name found nothing above; the reply must not pretend the
            # slot's top candidate matched it.
            facts["named_miss"] = predicate.directive
        if passing:
            favorites = set(session.user_profile.get("favorite_recipe_ids") or [])
            choice = next((c for c in passing if c.recipe_id in favorites), passing[0])
            return choice, old_rich, enrichment.get(choice.recipe_id), facts

        # Honest failure — offer the nearest miss when one is measurable.
        nearest_id = predicate.nearest(old_rich, {
            c.recipe_id: enrichment[c.recipe_id]
            for c in candidates if c.recipe_id in enrichment
        })
        if nearest_id:
            nearest = next(c for c in candidates if c.recipe_id == nearest_id)
            facts["nearest_miss"] = {
                "title": nearest.title,
                "kcal": enrichment[nearest_id].kcal,
                "protein_g": enrichment[nearest_id].protein_g,
            }
        return None, old_rich, None, facts

    def _resolve_named_dish(
        self, name: str, profile: dict, exclude_ids: list[str]
    ) -> Optional[CandidateRecipe]:
        """The member's named dish, via full-text search under hard constraints.

        No course-type filter on purpose — the name is the member overriding
        the taxonomy. Allergens, diet and dislikes still apply; a named dish
        that violates them returns nothing rather than something unsafe.
        """
        from services.plan_client import PLANNER

        try:
            hits = PLANNER.find_recipes(
                name,
                allergens=screening_allergens(profile),
                diet=effective_diet(profile),
                exclude_ingredients=profile.get("food_dislikes") or [],
                max_minutes=plan_parameters.max_duration_minutes(
                    profile.get("plan_parameters") or {}
                ),
                min_nutri_score=profile.get("min_nutri_score"),
                favorite_recipe_ids=profile.get("favorite_recipe_ids") or [],
                limit=3,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("named-dish search failed for %r: %s", name, exc)
            return None
        for hit in hits:
            if hit.recipe_id not in exclude_ids:
                return hit
        return None

    def _edit_daily(self, session, meal_type: str, predicate, original_message: str,
                    day: Optional[int] = None) -> EditOutcome:
        plan = session.get_current_daily_plan()
        if plan is None:
            text = "I couldn't find the current daily plan to edit."
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text)

        # A plan with `days` is the source of truth for itself. Rebuilding it
        # from three scalar courses — which is all the path below can do —
        # keeps day 1's mains and throws away every other day and every side,
        # dessert and drink. So a plan that has days is patched in place.
        if plan.days is not None:
            return self._edit_structured(
                session, plan, day, meal_type, predicate, original_message,
            )

        old_course = getattr(plan, meal_type)
        current_ids = [plan.breakfast.recipe_id, plan.lunch.recipe_id, plan.dinner.recipe_id]
        choice, old_rich, new_rich, facts = self._find_replacement(
            session, meal_type, predicate, old_course.recipe_id,
            current_ids + self._standing_exclusions(session),
        )

        if choice is None:
            text = self._failure_text(meal_type, predicate, facts)
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text, facts=facts)

        # PATCH: unchanged slots carry over; only the target slot is replaced.
        courses = [
            choice if slot == meal_type else getattr(plan, slot).to_candidate()
            for slot in ("breakfast", "lunch", "dinner")
        ]
        metrics = {  # carry scores forward; a slot swap doesn't re-grade the plan
            "llm_score": plan.llm_score, "llm_reasoning": plan.llm_reasoning,
            "fvs_count": plan.fvs_count, "fvs_reasoning": plan.fvs_reasoning,
            "diversity_llm_score": plan.diversity_llm_score,
            "diversity_llm_reasoning": plan.diversity_llm_reasoning,
            "guideline_adherence_score": plan.guideline_adherence_score,
            "guideline_adherence_reasoning": plan.guideline_adherence_reasoning,
        }
        new_plan = self.session_service.refine_meal_plan(
            session.session_id, courses,
            reasoning=f"Swapped {meal_type}: {predicate.directive}", metrics=metrics,
        )
        # Preserve prior transparency/enrichment on unchanged courses
        for slot in ("breakfast", "lunch", "dinner"):
            if slot != meal_type:
                old_slot = getattr(plan, slot)
                new_slot = getattr(new_plan, slot)
                new_slot.nutrition = old_slot.nutrition
                new_slot.image_url = old_slot.image_url
                new_slot.match_reasons = old_slot.match_reasons
        new_course = getattr(new_plan, meal_type)
        if new_rich:
            new_course.nutrition = new_rich.nutrition_dict()
            new_course.image_url = new_rich.image_url
        new_course.match_reasons = [{"kind": "pinned", "label": "swapped at your request"}]
        new_plan.constraints_applied = plan.constraints_applied
        new_plan.personalization_summary = plan.personalization_summary

        changed = [self._changed_slot(meal_type, None, old_course.title, old_rich,
                                      choice.title, new_rich, predicate)]
        for key in ("named_dish", "named_miss"):
            if key in facts:
                changed[0][key] = facts[key]
        facts.update({"changed": changed[0]})
        text = self._success_text(changed[0], predicate)

        # A dish the member named and got is a standing choice, not a
        # property of this turn: pin it so the next refinement re-anchors it
        # instead of regenerating it away — which is exactly how the apple
        # pie vanished one turn after being served.
        if facts.get("named_dish"):
            from models.planning_state import PlanningStateDelta
            state = self.session_service.get_planning_state(session.session_id)
            self.session_service.set_planning_state(
                session.session_id,
                state.merge(PlanningStateDelta(anchors={meal_type: choice.recipe_id})),
            )
        self.session_service.add_message(session.session_id, "assistant", text)
        return EditOutcome(
            text=text, meal_plan=new_plan, changed_slots=changed, facts=facts,
        )

    def _edit_structured(
        self, session, plan, day: Optional[int], meal_type: str, predicate,
        original_message: str,
    ) -> EditOutcome:
        """Swap ONE plate of a multi-day / multi-plate plan, in place.

        The plan is the shape `plan_structured` builds: N days, each with its
        own meals, each meal one or more plates. Every plate outside the target
        comes through byte-identical — the plan is deep-copied and exactly one
        `MealCourse` is replaced, so nothing depends on a rebuild getting the
        rest right.

        The copy matters twice: the parent version is a live object in
        `session.meal_plans`, so patching `plan.days` in place would rewrite
        history the member can scroll back to.
        """
        days = plan.day_plans
        target_day = int(day) if day else (int(days[0].day) if len(days) == 1 else 0)

        coords = None
        for di, dp in enumerate(days):
            if int(getattr(dp, "day", di + 1)) != target_day:
                continue
            for mi, meal in enumerate(dp.meals):
                if meal.meal_type != meal_type:
                    continue
                # The same plate `Meal.main` resolves to, by index so the copy
                # below can be addressed identically.
                mains = [
                    pi for pi, plate in enumerate(meal.plates)
                    if getattr(plate, "role", "main") == "main"
                ]
                coords = (di, mi, (mains or [0])[0])
                break
            break

        if coords is None:
            text = self._structured_miss_text(days, target_day, meal_type)
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text)

        di, mi, pi = coords
        old_course = days[di].meals[mi].plates[pi]

        # Every recipe already in the plan is excluded, not just three slots —
        # otherwise a swap on day 2 can hand back day 5's dinner.
        in_plan = [
            plate.recipe_id
            for dp in days for meal in dp.meals for plate in meal.plates
            if plate.recipe_id
        ]
        choice, old_rich, new_rich, facts = self._find_replacement(
            session, meal_type, predicate, old_course.recipe_id,
            in_plan + self._standing_exclusions(session),
        )
        if choice is None:
            text = self._failure_text(meal_type, predicate, facts)
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text, facts=facts)

        new_days = copy.deepcopy(days)
        replacement = MealCourse.from_candidate(choice)
        # The plate keeps its place in the meal: a main stays a main, a side
        # stays a side, and the order the UI renders is untouched.
        replacement.role = getattr(old_course, "role", "main")
        if new_rich:
            replacement.nutrition = new_rich.nutrition_dict()
            replacement.image_url = new_rich.image_url
        replacement.match_reasons = [
            {"kind": "pinned", "label": "swapped at your request"}
        ]
        new_days[di].meals[mi].plates[pi] = replacement

        metrics = {  # a slot swap doesn't re-grade the plan
            "llm_score": plan.llm_score, "llm_reasoning": plan.llm_reasoning,
            "fvs_count": plan.fvs_count, "fvs_reasoning": plan.fvs_reasoning,
            "diversity_llm_score": plan.diversity_llm_score,
            "diversity_llm_reasoning": plan.diversity_llm_reasoning,
            "guideline_adherence_score": plan.guideline_adherence_score,
            "guideline_adherence_reasoning": plan.guideline_adherence_reasoning,
        }
        new_plan = MealPlan.from_days(
            new_days,
            reasoning=f"Swapped day {target_day} {meal_type}: {predicate.directive}",
            metrics=metrics,
        )
        # Set before storing: `refine_prepared_meal_plan` serializes to the
        # database inside the call, so anything attached afterwards would be
        # missing from the row the next reload reads.
        new_plan.constraints_applied = plan.constraints_applied
        new_plan.personalization_summary = plan.personalization_summary
        new_plan = self.session_service.refine_prepared_meal_plan(
            session.session_id, new_plan,
        )

        changed = [self._changed_slot(
            meal_type, target_day, old_course.title, old_rich,
            choice.title, new_rich, predicate,
        )]
        for key in ("named_dish", "named_miss"):
            if key in facts:
                changed[0][key] = facts[key]
        facts.update({"changed": changed[0]})
        text = self._success_text(changed[0], predicate)

        if facts.get("named_dish"):
            from models.planning_state import PlanningStateDelta
            state = self.session_service.get_planning_state(session.session_id)
            self.session_service.set_planning_state(
                session.session_id,
                state.merge(PlanningStateDelta(anchors={meal_type: choice.recipe_id})),
            )
        self.session_service.add_message(session.session_id, "assistant", text)
        return EditOutcome(
            text=text, meal_plan=new_plan, changed_slots=changed, facts=facts,
        )

    @staticmethod
    def _structured_miss_text(days: list, day: int, meal_type: str) -> str:
        """Say what the plan actually covers rather than editing the wrong slot."""
        numbers = [int(getattr(dp, "day", i + 1)) for i, dp in enumerate(days)]
        matching = [dp for i, dp in enumerate(days) if numbers[i] == day]
        if not matching:
            span = ", ".join(f"day {n}" for n in numbers)
            return (
                f"This plan covers {span} — I don't have a day {day} to change. "
                "Tell me which of those you meant and I'll swap it."
            )
        have = [m.meal_type for m in matching[0].meals]
        return (
            f"Day {day} doesn't have a {meal_type} in this plan — it has "
            f"{', '.join(have) or 'no meals'}. Which of those should I change?"
        )

    def _edit_weekly(self, session, day: int, meal_type: str, predicate, original_message: str) -> EditOutcome:
        plan = session.get_current_weekly_plan()
        if plan is None:
            text = "I couldn't find the current weekly plan to edit."
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text)

        meal_idx = MEAL_IDX[meal_type]
        target = next(
            (e for e in plan.entries if e.get("day") == day and e.get("meal_idx") == meal_idx),
            None,
        )
        if target is None:
            text = f"I couldn't find {meal_type} on {DAY_NAMES[day - 1]} in the current plan."
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text)

        old = target.get("recipe", {})
        all_ids = [str(e.get("recipe", {}).get("recipe_id", "")) for e in plan.entries]
        choice, old_rich, new_rich, facts = self._find_replacement(
            session, meal_type, predicate, str(old.get("recipe_id", "")),
            all_ids + self._standing_exclusions(session),
        )

        if choice is None:
            text = self._failure_text(meal_type, predicate, facts, day=day)
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text, facts=facts)

        # PATCH: copy entries, replace only the target slot. No 21-meal regen.
        new_entries = []
        for entry in plan.entries:
            if entry is target:
                replacement = {
                    "recipe_id": choice.recipe_id,
                    "recipe_title": choice.title,
                    "recipe_ingredients": choice.ingredients,
                    "recipe_directions": choice.directions,
                    "pinned": True,
                }
                if new_rich:
                    nutrition = new_rich.nutrition_dict()
                    if nutrition:
                        replacement["nutrition"] = nutrition
                    replacement["image_url"] = new_rich.image_url
                    replacement["tags"] = new_rich.tags or []
                    replacement["dish_types"] = new_rich.dish_types or []
                new_entries.append({
                    **entry,
                    "recipe": replacement,
                    "reward": entry.get("reward", 0.0),
                })
            else:
                new_entries.append(dict(entry))
        # Day summaries + explainability reflect the patched week (M6/M7).
        # No planner ran, so there are no selection events; ledger statuses
        # come from the final counts alone, and the feedback rows stay out
        # (a patch doesn't consult feedback exclusions — claiming them
        # here would be unverified).
        day_summaries = build_day_summaries(new_entries)
        explainability = build_weekly_explainability(
            new_entries, session.user_profile,
            selection_events=[], day_summaries=day_summaries,
        )
        new_plan = self.session_service.refine_weekly_meal_plan(
            session.session_id, new_entries,
            day_summaries=day_summaries, explainability=explainability,
        )

        changed = [self._changed_slot(
            meal_type, day, old.get("recipe_title", ""), old_rich,
            choice.title, new_rich, predicate,
        )]
        facts.update({"changed": changed[0]})
        text = self._success_text(changed[0], predicate)
        self.session_service.add_message(session.session_id, "assistant", text)
        return EditOutcome(
            text=text, weekly_meal_plan=new_plan, changed_slots=changed, facts=facts,
        )

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _changed_slot(meal_type, day, old_title, old_rich, new_title, new_rich, predicate) -> dict:
        return {
            "meal_type": meal_type,
            "day": day,
            "old": {"title": old_title, "kcal": old_rich.kcal if old_rich else None},
            "new": {"title": new_title, "kcal": new_rich.kcal if new_rich else None},
            "directive": predicate.directive,
            "verified": predicate.verifiable,
        }

    @staticmethod
    def _success_text(changed: dict, predicate) -> str:
        where = f"{DAY_NAMES[changed['day'] - 1]}'s {changed['meal_type']}" if changed["day"] else f"the {changed['meal_type']}"
        text = f"Done — I swapped {where}: “{changed['old']['title']}” → “{changed['new']['title']}”."
        old_kcal, new_kcal = changed["old"]["kcal"], changed["new"]["kcal"]
        if predicate.kind == "lighter" and old_kcal and new_kcal:
            text += f" That takes it from {old_kcal:.0f} to {new_kcal:.0f} kcal per serving."
        elif predicate.kind == "uses_ingredient":
            text += f" Verified: it uses your {predicate.tag} — less food waste."
        elif predicate.verifiable and changed["verified"]:
            text += f" Verified: the new pick satisfies “{changed['directive']}”."
        elif changed.get("named_dish"):
            # The member named this dish and we found exactly it — no hedging.
            pass
        elif changed.get("named_miss"):
            text += (
                f" I couldn't find “{changed['named_miss']}” in our recipes, "
                "so I picked a fitting alternative — name another dish if "
                "you had one in mind."
            )
        elif not predicate.verifiable:
            text += f" I picked the best match for “{changed['directive']}” — tell me if it's not quite right."
        return text

    @staticmethod
    def _failure_text(meal_type, predicate, facts, day=None) -> str:
        where = f"{DAY_NAMES[day - 1]}'s {meal_type}" if day else f"the {meal_type}"
        text = (f"I looked for a replacement for {where} that's provably "
                f"“{predicate.directive}”, but nothing in the matching recipes passes the bar")
        nearest = facts.get("nearest_miss")
        if nearest and nearest.get("kcal") is not None:
            text += (f" — the closest is “{nearest['title']}” at {nearest['kcal']:.0f} kcal "
                     "per serving. Want that one, or should I relax something else?")
        else:
            text += ". Want me to relax the requirement or another constraint?"
        return text

    def _standing_exclusions(self, session) -> list[str]:
        """Recipes the member already rejected (downvote, "not that one").

        A swap that could resurrect one of them reads as not listening —
        the same rule regeneration already follows. Best-effort: an
        unreadable state costs the exclusion, never the swap.
        """
        try:
            state = self.session_service.get_planning_state(session.session_id)
            return [r for r in state.excluded_recipe_ids if r]
        except Exception:  # noqa: BLE001
            return []

    def _get_session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
