"""
PlanScorerService — the ``score_plan`` turn (IDEAS.md "Plan scorer").

A member pastes a meal plan they wrote. Two ways in, one service:

    chat      the orchestrator routes the turn here, from the classifier or
              from an explicit "rate this: <listing>"
    endpoint  ``POST /sessions/{id}/score-plan`` — the text box — which skips
              classification and may say what the plan is (``plan_type``)
              and what the member is aiming for (``context``)

    process(session_id, text)                  parse → ground → build → score → reply
    continue_clarification(session_id, reply)  resume a ``score_plan`` question

Nothing is written to a canvas and no plan version is created: a pasted plan
is scored, not adopted, so refine and edit turns keep targeting the member's
own plan. The member's text and FoodChat's reply are persisted as ordinary
messages (intent ``score_plan``), and the whole score payload is stored with
the reply (``messages.plan_score``) so the card survives a reload.

Clarification (``sessions.clarification_state``, restart-safe like every other
kind)::

    {"kind": "score_plan", "reason": "no_meals", "pasted_text", "plan_type", "context"}
    {"kind": "score_plan", "reason": "shape",    "pasted_text", "plan_type", "context", "plan"}

``no_meals``: nothing in the text could be read as a dish. ``shape``: one
block with no day names in which several meals repeat — one long day or
several short ones is the member's call. The pasted text and the member's aim
are kept, so nobody re-pastes. A reply that answers neither question comes back
``unresolved`` with the state cleared and nothing logged, so the orchestrator
routes it as a fresh turn — and never asks a score question twice in a row.

The reply is written by ``ResponseWriter`` from facts (scores, broken
constraints, dishes not found) with a deterministic fallback. An allergen found
in a dish is always named: if the writer leaves one out, the deterministic
sentence is appended.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from models.pasted_plan import (
    APPROXIMATE,
    CLOSEST_RECIPE,
    NUTRITION_MODEL_ESTIMATE,
    NUTRITION_TYPICAL,
    MAIN_SLOTS,
    MATCHED,
    PLAN_TYPES,
    SLOT_INDEX,
    UNRESOLVED,
    GroundedMeal,
    PastedPlan,
)

from .building import (
    DailyScoringInput,
    ScoringInput,
    WeeklyScoringInput,
    build_scoring_input,
    entry_dicts,
)
from .grounding import DishGrounder
from .parsing import (
    days_from_answer,
    needs_shape_question,
    parse_plan_text,
    split_into_days,
)
from .scoring import LIKERT, ScoreResult

logger = logging.getLogger(__name__)

CLARIFICATION_KIND = "score_plan"
REASON_NO_MEALS = "no_meals"
REASON_SHAPE = "shape"

NO_MEALS_QUESTION = (
    "I couldn't find any meals in that. Could you list them one per line, like "
    "“breakfast: oats” and “lunch: lentil soup”? For more than one day, put the "
    "day's name above its meals."
)
SPLIT_WARNING = "No day names were given, so a new day starts wherever a meal repeats."

_PLURAL = {"breakfast": "breakfasts", "lunch": "lunches", "dinner": "dinners"}
_SLOT_WORD = re.compile(r"\b(?:breakfast|brunch|lunch|dinner|supper|snacks?)\b", re.IGNORECASE)


@dataclass
class ScoreTurn:
    text: str
    needs_clarification: bool = False
    plan_score: Optional[dict] = None
    # The step-3 objects the metrics were computed from. Not serialised.
    scoring_input: Optional[ScoringInput] = None
    # The reply was not an answer to the pending question (nothing logged).
    unresolved: bool = False


class PlanScorerService:
    CLARIFICATION_KIND = CLARIFICATION_KIND

    def __init__(self, session_service, parser=None, grounder=None, scorer=None, writer=None):
        self.session_service = session_service
        if parser is None:
            from agents import PlanTextParser

            parser = PlanTextParser()
        if scorer is None:
            from .scoring import PastedPlanScorer

            scorer = PastedPlanScorer()
        if writer is None:
            from agents import ResponseWriter

            writer = ResponseWriter()
        self.parser = parser
        if grounder is None:
            from agents import DishIngredientEstimator

            grounder = DishGrounder(estimator=DishIngredientEstimator())
        self.grounder = grounder
        self.scorer = scorer
        self.writer = writer

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #

    def process(
        self,
        session_id: str,
        message: str,
        *,
        plan_type: str = "auto",
        context: Optional[str] = None,
        may_ask: bool = True,
    ) -> ScoreTurn:
        """Score-turn for a message that carries a plan.

        ``may_ask=False`` is set by the orchestrator when this turn is the
        fall-through from a score question the member did not answer — the
        turn then ends with what could be read instead of asking again.
        """
        session = self._session(session_id)
        self.session_service.add_message(session_id, "user", message)
        plan_type = plan_type if plan_type in PLAN_TYPES else "auto"
        plan = parse_plan_text(message, self.parser, plan_type)
        carried = {"pasted_text": message, "plan_type": plan_type, "context": context}

        if plan.is_empty:
            if may_ask:
                return self._ask(
                    session_id, {"reason": REASON_NO_MEALS, **carried}, NO_MEALS_QUESTION,
                )
            return self._reply(session_id, nothing_read_text(plan))

        if plan_type == "auto" and needs_shape_question(plan):
            if may_ask:
                return self._ask(
                    session_id,
                    {"reason": REASON_SHAPE, **carried, "plan": plan.to_dict()},
                    shape_question(plan),
                )
            plan = split_into_days(plan, 0)
            plan.warnings.append(SPLIT_WARNING)

        return self._finish(session, plan, message, context)

    def score_canvas(
        self, session_id: str, plan_type: str = "", *, context: Optional[str] = None,
    ) -> Optional[ScoreTurn]:
        """Score the plan already on the member's canvas. ``None`` if there is none.

        The scorer was built for text a member pasted, and every entry point
        into it required that text — so "score my plan", with a plan open on
        the screen, had no route at all. It fell through to the tool selector,
        which answered with the only refusal it had.

        Nothing about the metrics needed changing. A canvas plan is *better*
        grounded than a pasted one: its dishes are catalogue recipes, so the
        lookup, the similarity threshold and the estimated servings that the
        pasted path needs are all skipped. `canvas.from_canvas` is the whole
        adapter.

        No canvas is touched and no version created, exactly as for a pasted
        plan — scoring is a reading, not an edit.
        """
        from .canvas import from_canvas

        session = self._session(session_id)
        wanted = str(plan_type or "").strip().lower()
        weekly = session.get_current_weekly_plan()
        daily = session.get_current_daily_plan()
        has_weekly = weekly is not None and bool(getattr(weekly, "entries", None))

        # The canvas the member is looking at wins; with nothing to go on,
        # whichever one holds a plan, weekly first — a session with both was
        # most recently planning a week. Same rule as `plan_tools._day_plan`,
        # for the same reason: two readers disagreeing about which plan "my
        # plan" means is worse than either answer.
        if wanted == "weekly" and has_weekly:
            plan, kind = weekly, "weekly"
        elif wanted == "daily" and daily is not None:
            plan, kind = daily, "daily"
        elif has_weekly:
            plan, kind = weekly, "weekly"
        elif daily is not None:
            plan, kind = daily, "daily"
        else:
            return None

        pasted, grounded = from_canvas(plan, kind)
        if not grounded:
            return None
        return self._finish(
            session, pasted, "", context, grounded=grounded, own_plan=True,
        )

    def continue_clarification(self, session_id: str, message: str) -> ScoreTurn:
        """Resume a pending ``score_plan`` question with the member's reply."""
        session = self._session(session_id)
        pending = dict(session.clarification or {})
        self.session_service.clear_clarification_state(session_id)
        reason = pending.get("reason")
        context = pending.get("context")
        pasted = pending.get("pasted_text") or message
        plan_type = pending.get("plan_type") if pending.get("plan_type") in PLAN_TYPES else "auto"

        if reason == REASON_SHAPE:
            days = days_from_answer(message)
            stored = PastedPlan.from_dict(pending.get("plan") or {})
            if days is not None and not stored.is_empty:
                self.session_service.add_message(session_id, "user", message)
                return self._finish(session, split_into_days(stored, days), pasted, context)
            if not _SLOT_WORD.search(message or ""):
                return ScoreTurn(text="", unresolved=True)

        # A reply carrying meals answers either question: it is the listing.
        plan = parse_plan_text(message, self.parser, plan_type)
        if plan.is_empty:
            return ScoreTurn(text="", unresolved=True)
        self.session_service.add_message(session_id, "user", message)
        if needs_shape_question(plan):
            plan = split_into_days(plan, 0)
            plan.warnings.append(SPLIT_WARNING)
        return self._finish(session, plan, message, context)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session

    def _ask(self, session_id: str, fields: dict, question: str) -> ScoreTurn:
        self.session_service.set_clarification_state(
            session_id, {"kind": CLARIFICATION_KIND, **fields}
        )
        self.session_service.add_message(session_id, "assistant", question, intent=CLARIFICATION_KIND)
        return ScoreTurn(text=question, needs_clarification=True)

    def _reply(self, session_id: str, text: str) -> ScoreTurn:
        self.session_service.add_message(session_id, "assistant", text, intent=CLARIFICATION_KIND)
        return ScoreTurn(text=text)

    def _finish(
        self,
        session,
        plan: PastedPlan,
        message: str,
        context: Optional[str],
        *,
        grounded: Optional[list] = None,
        own_plan: bool = False,
    ) -> ScoreTurn:
        """Ground (unless already grounded), score, write the reply, store it.

        `grounded` is passed by `score_canvas`, where the dishes are catalogue
        recipes and there is nothing to look up. `own_plan` says whose plan
        this is, which changes only what the reply is allowed to offer: the
        pasted path must not offer to replace a plan the member wrote, and the
        canvas path is looking at one FoodChat built, where an offer to adjust
        it is the obvious next thing.
        """
        profile = session.user_profile or {}
        if grounded is None:
            grounded = self.grounder.ground(plan, profile)
        built = build_scoring_input(plan, grounded)
        result = self.scorer.score(
            plan.plan_type, grounded, built, profile, context=aim_text(context, plan),
        )
        payload = score_payload(plan, grounded, built, result, context)
        payload["source"] = "canvas" if own_plan else "pasted"
        text = self.writer.write(
            summary_facts(plan, grounded, result, own_plan=own_plan), message,
            fallback=fallback_summary(plan, grounded, result),
        )
        text = ensure_calorie_caveat(text, grounded)
        text = ensure_allergen_warnings(text, grounded)
        logger.info(
            "[%s] Pasted %s plan scored: %d dish(es) over %d day(s); grounding %s; scores %s",
            session.session_id, plan.plan_type, len(grounded), len(plan.days),
            {state: sum(1 for g in grounded if g.state == state)
             for state in (MATCHED, APPROXIMATE, UNRESOLVED)},
            {m["key"]: m["score"] for m in result.metrics},
        )
        self.session_service.add_message(
            session.session_id, "assistant", text, intent=CLARIFICATION_KIND, plan_score=payload,
        )
        return ScoreTurn(text=text, plan_score=payload, scoring_input=built)


# --------------------------------------------------------------------- #
# Payload (pure)                                                          #
# --------------------------------------------------------------------- #

def aim_text(context: Optional[str], plan: PastedPlan) -> Optional[str]:
    """What the member said they are aiming for: the text box's ``context``
    and whatever they wrote around the listing."""
    parts = [str(context).strip()] if context and str(context).strip() else []
    parts.extend(note for note in plan.notes if note.strip())
    return "\n".join(parts) or None


def score_payload(
    plan: PastedPlan,
    grounded: list[GroundedMeal],
    built: ScoringInput,
    result: ScoreResult,
    context: Optional[str] = None,
) -> dict:
    """The ``plan_score`` payload returned on the turn and stored with the reply."""
    warnings = list(plan.warnings)
    if isinstance(built, DailyScoringInput) and built.missing_slots:
        warnings.append(
            "No " + " or ".join(built.missing_slots)
            + " is listed, so the day is scored as written."
        )
    unresolved = sum(
        1 for g in grounded
        if g.state == UNRESOLVED or (g.state == APPROXIMATE and not g.borrows_nutrition)
    )
    if unresolved:
        warnings.append(
            f"{unresolved} dish(es) have no close recipe in the catalogue, so only what "
            "you wrote about them is known."
        )
    no_nutrition = sum(1 for g in grounded if not g.nutrition)
    if no_nutrition:
        warnings.append(f"{no_nutrition} of {len(grounded)} dish(es) have no nutrition data.")
    typical = sum(1 for g in grounded if g.nutrition_source == NUTRITION_TYPICAL)
    if typical:
        warnings.append(
            f"{typical} dish(es) have no recipe in the catalogue, so their calories are "
            "estimated from typical ingredients."
        )
    guessed = sum(1 for g in grounded if g.nutrition_source == NUTRITION_MODEL_ESTIMATE)
    if guessed:
        warnings.append(
            f"{guessed} dish(es) use a rough calorie guess, because their typical ingredients "
            "could not be profiled."
        )

    return {
        "plan_type": plan.plan_type,
        "days_scored": len(plan.days),
        "meals_scored": len(grounded),
        "metrics": list(result.metrics),
        "constraints_applied": list(result.constraints),
        "grounding": [g.to_row() for g in grounded],
        "unparsed": list(plan.unparsed),
        "warnings": warnings,
        "scored_plan": plan_view(plan.plan_type, built, result),
        "context": (str(context).strip() or None) if context else None,
    }


def _plate(recipe: dict) -> dict:
    return {
        "recipe_id": str(recipe.get("recipe_id") or ""),
        "title": str(recipe.get("title") or ""),
        "ingredients": str(recipe.get("ingredients") or ""),
        "directions": "",
        "nutrition": recipe.get("nutrition"),
        "image_url": recipe.get("image_url"),
        "match_reasons": [
            {"kind": str(r.get("kind")), "label": str(r.get("label") or "")}
            for r in recipe.get("match_reasons") or [] if r.get("kind")
        ],
        "role": "main",
    }


def plan_view(plan_type: str, built: ScoringInput, result: ScoreResult) -> dict:
    """The pasted plan in the plan-card shapes the UI already renders.

    ``days`` is the plates-as-list view (``DayPlanResponse``), which can show a
    partial or two-plate day; ``entries`` is the weekly card's flat list.
    ``origin: "pasted"`` tells the UI to hide refine and edit controls.
    """
    weekly = isinstance(built, WeeklyScoringInput)
    if weekly:
        entries = sorted(built.entries + built.extras, key=lambda e: (e["day"], e["meal_idx"]))
    else:
        entries = list(result.plan_entries) or entry_dicts(built.meals)

    grouped: dict[int, dict[str, list]] = {}
    for entry in entries:
        slots = grouped.setdefault(int(entry["day"]), {})
        slots.setdefault(str(entry["meal_type"]), []).append(_plate(entry["recipe"]))

    return {
        "origin": "pasted",
        "plan_type": plan_type,
        "days": [
            {
                "day": day,
                "meals": [
                    {"meal_type": slot, "plates": plates}
                    for slot, plates in sorted(
                        slots.items(), key=lambda kv: SLOT_INDEX.get(kv[0], len(SLOT_INDEX)),
                    )
                ],
            }
            for day, slots in sorted(grouped.items())
        ],
        "entries": [dict(entry) for entry in entries] if weekly else [],
    }


# --------------------------------------------------------------------- #
# Prose (pure)                                                            #
# --------------------------------------------------------------------- #

def _distinct(meals: list[GroundedMeal]) -> list[GroundedMeal]:
    seen: set[str] = set()
    out = []
    for meal in meals:
        key = meal.title_given.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(meal)
    return out


def _is_are(n: int) -> str:
    return "is" if n == 1 else "are"


def _titles(titles: list[str]) -> str:
    quoted = [f"“{t}”" for t in titles]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def allergen_sentences(grounded: list[GroundedMeal]) -> list[tuple[list[str], str]]:
    """One (allergens, sentence) per certainty and set of allergens.

    Dishes that share both are named together — "“Greek yogurt with walnuts”
    and “Porridge with almond butter” contain tree nuts" — and one dish's
    related allergies are one warning — "may contain lactose and dairy".
    Live, a three-day week repeated the same sentence three times.
    """
    per_dish: dict[tuple[str, str], dict] = {}
    for meal in grounded:
        title = meal.title_given.strip()
        for conflict in meal.allergen_conflicts:
            possible = conflict["evidence"] == CLOSEST_RECIPE
            entry = per_dish.setdefault((title.lower(), possible), {"title": title, "allergens": []})
            if conflict["allergen"] not in entry["allergens"]:
                entry["allergens"].append(conflict["allergen"])

    grouped: dict[tuple, list[str]] = {}
    for (_title, possible), entry in per_dish.items():
        grouped.setdefault((possible, tuple(entry["allergens"])), []).append(entry["title"])

    sentences = []
    for (possible, allergens), titles in grouped.items():
        several = len(titles) > 1
        if possible:
            verb = "may contain"
        else:
            verb = "contain" if several else "contains"
        which = "which are" if len(allergens) > 1 else "which is"
        sentences.append((
            list(allergens),
            f"Heads up: {_titles(titles)} {verb} {_and(list(allergens))}, {which} on your allergy list.",
        ))
    return sentences


def _plain(text: str) -> str:
    """Lower case with every dash and odd space as a plain space, so
    "tree‑nut" (a non-breaking hyphen) names tree nuts."""
    return re.sub(r"[\s\u2010-\u2015\u00ad\-]+", " ", (text or "").lower())


def ensure_allergen_warnings(text: str, grounded: list[GroundedMeal]) -> str:
    """Append the deterministic warning for any allergen the reply does not
    name. A summary may be short; it may not be silent about this."""
    lowered = _plain(text)
    missing = [
        sentence for allergens, sentence in allergen_sentences(grounded)
        if any(_plain(allergen).rstrip("s") not in lowered for allergen in allergens)
    ]
    return " ".join([text.strip(), *missing]).strip() if missing else text


_CALORIE_MENTION = re.compile(r"\bkcal\b|\bcalori", re.IGNORECASE)
_CALORIE_CAVEAT = re.compile(
    r"estimat|rough|guess|partial|approximat|incomplete|only some|not all|typical", re.IGNORECASE,
)


def calorie_caveat(grounded: list[GroundedMeal]) -> str:
    """The sentence a reply quoting calories owes the member, or ""."""
    total = len(grounded)
    known = sum(1 for g in grounded if g.nutrition)
    estimated = sum(
        1 for g in grounded if g.nutrition_source in (NUTRITION_TYPICAL, NUTRITION_MODEL_ESTIMATE)
    )
    if known < total:
        return f"Calories are known for only {known} of {total} dishes, so any total is incomplete."
    if estimated:
        return f"Calories for {estimated} of {total} dishes are estimates, not recipe figures."
    return ""


def ensure_calorie_caveat(text: str, grounded: list[GroundedMeal]) -> str:
    """Append the calorie caveat when the reply quotes calories without one.

    The writer is told to qualify partly known or estimated calories; live, it
    quoted "about 1,030 kcal, far below your target" for a day two of whose
    three dishes were estimates.
    """
    caveat = calorie_caveat(grounded)
    if not caveat or not _CALORIE_MENTION.search(text or "") or _CALORIE_CAVEAT.search(text):
        return text
    return f"{text.strip()} {caveat}"


def describe_reading(plan: PastedPlan, grounded: list[GroundedMeal]) -> str:
    """What was read, in plain sentences. Deterministic — no model writes it."""
    shape = "a one-day plan" if plan.plan_type == "daily" else f"a {len(plan.days)}-day plan"
    parts = [f"I read this as {shape} with {len(grounded)} dish(es)."]

    matched = [g for g in grounded if g.state == MATCHED]
    approximate = [g for g in grounded if g.state == APPROXIMATE]
    unresolved = [g for g in grounded if g.state == UNRESOLVED]
    if matched:
        parts.append(f"{len(matched)} of them {_is_are(len(matched))} in the recipe catalogue.")
    close = [g for g in approximate if g.borrows_nutrition]
    loose = [g for g in approximate if not g.borrows_nutrition]
    if close:
        named = ", ".join(f"“{g.title_given}” as “{g.title_matched}”" for g in _distinct(close)[:3])
        verb = "has" if len(close) == 1 else "have"
        parts.append(
            f"{len(close)} only {verb} a close match, so the calories come from reading {named}."
        )
    if loose:
        named = ", ".join(
            f"“{g.title_given}” (nearest: “{g.title_matched}”)" for g in _distinct(loose)[:3]
        )
        parts.append(
            f"The nearest catalogue recipes for {named} are different dishes, so only "
            "what you wrote about them is used."
        )
    if unresolved:
        named = ", ".join(f"“{g.title_given}”" for g in _distinct(unresolved)[:3])
        parts.append(
            f"I couldn't find {len(unresolved)} in the catalogue ({named}), so only "
            "what you wrote about them is known."
        )
    parts.extend(sentence for _allergen, sentence in allergen_sentences(grounded))
    if plan.unparsed:
        lines = "; ".join(f"“{line}”" for line in plan.unparsed[:5])
        parts.append(f"I couldn't place these lines: {lines}.")
    return " ".join(parts)


def fallback_summary(plan: PastedPlan, grounded: list[GroundedMeal], result: ScoreResult) -> str:
    """The reply when the writer is unavailable: the reading plus the scores."""
    parts = [describe_reading(plan, grounded)]
    graded = [
        f"{m['label'].lower()} {m['score']} out of 5"
        for m in result.metrics if m["kind"] == LIKERT and m["score"] is not None
    ]
    if graded:
        parts.append("Scores: " + ", ".join(graded) + ".")
    broken = [r["constraint"] for r in result.constraints if r.get("status") == "violated"]
    if broken:
        parts.append("It doesn't meet: " + ", ".join(broken[:4]) + ".")
    ungraded = [
        m["label"].lower() for m in result.metrics if m["kind"] == LIKERT and m["score"] is None
    ]
    if ungraded:
        parts.append("I couldn't grade " + " or ".join(ungraded) + " just now.")
    return " ".join(parts)


def summary_facts(
    plan: PastedPlan,
    grounded: list[GroundedMeal],
    result: ScoreResult,
    *,
    own_plan: bool = False,
) -> dict:
    """What the ResponseWriter may say — and only this.

    `own_plan` is the one thing that differs between scoring a plan FoodChat
    built and one the member wrote. "Do not offer a new plan" is right for a
    pasted plan — replacing what somebody wrote is not what they asked for —
    and wrong for the plan on the canvas, where adjusting it is the obvious
    next step and refusing to mention it makes the score a dead end.
    """
    fit = next((m for m in result.metrics if m["key"] == "fit"), None)
    facts = {
        "action": "scored_own_plan" if own_plan else "scored_pasted_plan",
        "instruction": (
            (
                "This is the plan FoodChat built and the user is looking at. "
                "Report how it scored and the most important reason, then offer "
                "to adjust it."
            )
            if own_plan else
            (
                "The user wrote this plan themselves and asked for it to be scored. "
                "Report how it scored and the most important reason. Do not offer a new plan."
            )
        ),
        "plan": "one day" if plan.plan_type == "daily" else f"{len(plan.days)} days",
        "dishes": len(grounded),
        "scores": [
            {"metric": m["label"], "score": m["score"], "out_of": 5}
            for m in result.metrics if m["kind"] == LIKERT and m["score"] is not None
        ],
        "measured": [
            {"metric": m["label"], "result": m["reasoning"]}
            for m in result.metrics if m["kind"] != LIKERT
        ],
        "constraints_broken": [
            f"{r['constraint']} ({r.get('detail', '')})"
            for r in result.constraints if r.get("status") == "violated"
        ][:4],
        "fit_reasoning": (fit or {}).get("reasoning", "")[:400],
        "not_found_in_catalogue": [
            g.title_given for g in _distinct([g for g in grounded if g.state == UNRESOLVED])
        ][:4],
        "close_matches": [
            f"{g.title_given} read as {g.title_matched}"
            for g in _distinct([g for g in grounded if g.state == APPROXIMATE])
        ][:3],
    }
    # Live, a figure covering one dish of three came back as "the day falls
    # far short of energy needs". The metric's own sentence said "counting only
    # the 1 of 3 dishes"; the writer dropped it. The limit is now a fact of its
    # own, stated as an instruction.
    known = sum(1 for g in grounded if g.nutrition)
    estimated = sum(
        1 for g in grounded if g.nutrition_source in (NUTRITION_TYPICAL, NUTRITION_MODEL_ESTIMATE)
    )
    if known < len(grounded):
        facts["calories"] = {
            "known_for": f"{known} of {len(grounded)} dishes",
            "instruction": (
                "Calories are known for only some dishes. Do not state or compare the plan's "
                "total calories, and do not say it is short of or over a target."
            ),
        }
    elif estimated:
        facts["calories"] = {
            "estimated_for": f"{estimated} of {len(grounded)} dishes",
            "instruction": "If you mention calories, say they are partly estimated.",
        }
    return facts


def shape_question(plan: PastedPlan) -> str:
    counts = {slot: 0 for slot in MAIN_SLOTS}
    for meal in plan.meals:
        if meal.slot in counts:
            counts[meal.slot] += 1
    listed = [
        f"{n} {_PLURAL[slot] if n != 1 else slot}" for slot, n in counts.items() if n
    ]
    if len(listed) > 1:
        seen = ", ".join(listed[:-1]) + " and " + listed[-1]
    else:
        seen = listed[0] if listed else "several meals"
    return (
        f"Does this cover one day or several? I can see {seen}, with no day names. "
        "Tell me how many days it is, for example “3 days” or “just one day”."
    )


def nothing_read_text(plan: PastedPlan) -> str:
    text = "I still couldn't find any meals in that, so there is nothing to score."
    if plan.unparsed:
        text += " These lines didn't read as meals: " + "; ".join(
            f"“{line}”" for line in plan.unparsed[:5]
        ) + "."
    return text
