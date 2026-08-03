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

import logging
from dataclasses import dataclass, field
from typing import Optional

from agents import EditCommandExtractor
from models.recipe import CandidateRecipe, RecipeEnrichment
from services.candidates_client import CANDIDATES
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

        # Weekly edits need a day; daily edits need a meal type. Ask once.
        missing_slot = (
            command.get("needs_slot_clarification")
            or command.get("meal_type") is None
            or (canvas.plan_type == "weekly" and command.get("day") is None)
        )
        if missing_slot:
            question = command.get("question") or (
                "Which meal should I swap — breakfast, lunch, or dinner"
                + (", and on which day?" if canvas.plan_type == "weekly" else "?")
            )
            self.session_service.set_clarification_state(session_id, {
                "kind": "edit_slot",
                "original_message": message,
                "command": command,
                "plan_type": canvas.plan_type,
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
        command = self.extractor.extract(combined, plan_type=pending.get("plan_type", "daily"))
        if command is None or command.get("meal_type") is None or (
            pending.get("plan_type") == "weekly" and command.get("day") is None
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
            return self._edit_daily(session, meal_type, predicate, original_message)
        return self._edit_weekly(session, command.get("day"), meal_type, predicate, original_message)

    def _find_replacement(
        self, session, meal_type: str, predicate: DirectivePredicate,
        old_recipe_id: str, exclude_ids: list[str],
    ) -> tuple[Optional[CandidateRecipe], Optional[RecipeEnrichment], Optional[RecipeEnrichment], dict]:
        """Fetch slot candidates, verify the predicate, pick the replacement.

        Returns (choice, old_enrichment, new_enrichment, facts).
        """
        profile = session.user_profile

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
        candidates = self.client.slot_candidates(profile, meal_type, exclude_ids)

        if not candidates:
            return None, None, None, {"failure": "no candidates for this slot"}

        # One batch details call covers the predicate for old + all candidates.
        enrichment = self.client.fetch_details(
            [old_recipe_id] + [c.recipe_id for c in candidates]
        )
        old_rich = enrichment.get(old_recipe_id)

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
        from services.candidates_client import normalize_diet_tags
        from services.plan_client import PLANNER

        try:
            hits = PLANNER.find_recipes(
                name,
                allergens=profile.get("allergies") or [],
                diet=normalize_diet_tags(profile.get("diet")),
                exclude_ingredients=profile.get("food_dislikes") or [],
                limit=3,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("named-dish search failed for %r: %s", name, exc)
            return None
        for hit in hits:
            if hit.recipe_id not in exclude_ids:
                return hit
        return None

    def _edit_daily(self, session, meal_type: str, predicate, original_message: str) -> EditOutcome:
        plan = session.get_current_daily_plan()
        if plan is None:
            text = "I couldn't find the current daily plan to edit."
            self.session_service.add_message(session.session_id, "assistant", text)
            return EditOutcome(text=text)

        old_course = getattr(plan, meal_type)
        current_ids = [plan.breakfast.recipe_id, plan.lunch.recipe_id, plan.dinner.recipe_id]
        choice, old_rich, new_rich, facts = self._find_replacement(
            session, meal_type, predicate, old_course.recipe_id, current_ids,
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
            session, meal_type, predicate, str(old.get("recipe_id", "")), all_ids,
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

    def _get_session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
