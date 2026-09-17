"""
LLM agents — every Groq-backed reasoning step in the FoodChat pipeline.

Each agent wraps one pooled ChatGroq client (``backend.groq.GROQ_CHAT``) with
its prompt(s) and, where applicable, a structured-output schema. Agents hold
NO per-user state — anything conversational is passed in per call, so a single
instance is safe to share across sessions and replicas.

Agents and their consumers:
    OrchestratorAgent        — intent routing            (orchestrator_service)
    DocumentGrader           — daily-plan combo scoring  (planning_pipeline)
    MealDiversityGrader      — plan diversity metric     (chat_service)
    GuidelineAdherenceGrader — guideline metric          (chat_service)
    PlanJudge                — pasted plan: diversity + guidelines + fit
                               in one call                (plan_scorer.scoring)
    QueryReconciler          — query/profile conflicts   (planning_pipeline, clarification)
    DietaryIntentExtractor   — diet tags from a query    (weekly_plan_service)
    PlanTextParser           — pasted plan → days/slots  (plan_scorer.parsing)
    DishIngredientEstimator  — typical servings for dishes no recipe matched
                                                          (plan_scorer.grounding)
    SimpleChatBot            — small-talk fallback       (chat_service)

Removed in M0 (see CHANGES.md): QueryClassifier (superseded by
OrchestratorAgent — one classification per turn), and the offline-evaluation
agents (FoodChatResponseEvaluator, QueryRewriter, FeedBackRewriter) that were
only used by the deleted Ollama-era eval scripts.
"""

import itertools
import json
import logging
import os
import random
from typing import TYPE_CHECKING, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.groq import GROQ_CHAT
from backend.observability import build_trace_config
from models.recipe import CandidatesBySlot, ScoredPlan, slot_sort_key
if TYPE_CHECKING:  # import-time cycle: models.plan_spec reaches back here
    from models.plan_spec import PlanSpec

from prompts import (
    PLAN_GRADER_SYSTEM,
    TOOL_SELECTOR_SYSTEM,
    TOOL_SELECTOR_USER,
    PLAN_GRADER_USER,
    PLAN_STRATEGIST_SYSTEM,
    PLAN_STRATEGIST_USER,
    PLAN_ANALYST_SYSTEM,
    PLAN_JUDGE_DAILY_SYSTEM,
    PLAN_JUDGE_USER,
    PLAN_JUDGE_WEEKLY_SYSTEM,
    MEAL_DIVERSITY_SYSTEM,
    GUIDELINE_ADHERENCE_SYSTEM,
    QUERY_RECONCILER_SYSTEM,
    QUERY_RECONCILER_USER,
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_USER,
    DIETARY_INTENT_EXTRACTOR_SYSTEM,
    PLAN_SPEC_EXTRACTOR_SYSTEM,
    PLAN_SPEC_EXTRACTOR_USER,
    PLAN_TEXT_PARSER_SYSTEM,
    DISH_ESTIMATOR_SYSTEM,
    DISH_ESTIMATOR_USER,
    PLAN_TEXT_PARSER_USER,
    SCORE_PLAN_INTENT_ADDENDUM,
    DIETARY_INTENT_EXTRACTOR_USER,
    SEED_EXTRACTOR_SYSTEM,
    SEED_EXTRACTOR_USER,
    PANTRY_EXTRACTOR_SYSTEM,
    PANTRY_EXTRACTOR_USER,
    PREFERENCE_EXTRACTOR_SYSTEM,
    PREFERENCE_EXTRACTOR_USER,
    EDIT_COMMAND_EXTRACTOR_SYSTEM,
    EDIT_COMMAND_EXTRACTOR_USER,
    RESPONSE_WRITER_SYSTEM,
    RESPONSE_WRITER_USER,
    CHATBOT_SYSTEM,
    SESSION_TITLE_SYSTEM,
    SESSION_TITLE_USER,
    PLAN_INTENT_EXTRACTOR_SYSTEM,
    PLAN_INTENT_EXTRACTOR_USER,
    MEAL_COMPOSER_SYSTEM,
    MEAL_COMPOSER_USER,
)
from schemas import (
    MealCompositionSchema,
    PlanStrategySchema,
    ToolChoiceSchema,
    BatchScoringSchema,
    ScoringSchema,
    QueryReconcilerSchema,
    OrchestratorSchema,
    DietaryTagsSchema,
    PlanSpecSchema,
    ParsedPlanSchema,
    DishEstimatesSchema,
    PlanJudgementSchema,
    SeedExtractionSchema,
    PantryExtractionSchema,
    PreferenceExtractionSchema,
    EditCommandSchema,
    PlanIntentSchema,
)

logger = logging.getLogger(__name__)


def _opt_float(name: str) -> Optional[float]:
    """An env float, or None when unset/unparseable so the pool default wins.

    None is meaningful here, not merely absent: every call site below reads
    ``x if x is not None else DEFAULT``, and the pool in turn reads
    ``temperature if temperature is not None else GROQ_DEFAULT_TEMPERATURE``.
    Carrying a literal at this layer would break that chain (see groq.py).
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using the pool default.", name, raw)
        return None


# Deliberately WITHOUT a literal fallback: None means "inherit
# GROQ_DEFAULT_MODEL", which is the only place the default model name lives.
# With a literal here, GROQ_DEFAULT_MODEL was dead config — it could never be
# reached, so operators who set it saw no change at all.
DEFAULT_MODEL = os.getenv("FOODCHAT_LLM_MODEL")
# The cheap structured-output extractors (dietary tags, plan spec, seeds,
# preferences, edit commands) do span-picking, not judgment, and every planning
# turn hits several of them. A small fast model is the right tool and keeps
# per-turn latency off the reasoning model's price.
FAST_MODEL = os.getenv("FOODCHAT_FAST_MODEL", "openai/gpt-oss-20b")
DEFAULT_TEMPERATURE = _opt_float("FOODCHAT_LLM_TEMPERATURE")
# Prose, not extraction — a distinct setting rather than a shadowed default, so
# it keeps its own literal. Governs every agent that writes to the member:
# SimpleChatBot, ResponseWriter and PlanAnalyst.
CHATBOT_TEMPERATURE = float(os.getenv("FOODCHAT_CHATBOT_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("FOODCHAT_MAX_RETRIES", "3"))
MAX_PLANS_TO_SCORE = int(os.getenv("FOODCHAT_MAX_PLANS_TO_SCORE", "10"))

# How far into the combination space the grader will walk before it stops
# enumerating and samples instead.
#
# `itertools.product` over three slots of eight candidates is 512 — cheap to
# materialise, which is why the old three-slot grader could. Over seven slots
# it is two million, and a day with a main and two sides is worse. The scan is
# bounded so a generalised grader cannot hang the turn on exactly the plans it
# exists to support; the batch itself is still `MAX_PLANS_TO_SCORE`.
_COMBO_SCAN_LIMIT = 512
CHATBOT_HISTORY_TURNS = int(os.getenv("FOODCHAT_CHATBOT_HISTORY_TURNS", "12"))



def as_json_messages(messages: list) -> list:
    """The same messages, guaranteed to contain the word Groq requires.

    Groq rejects any `json_object` request whose messages do not contain the
    string "json" — a 400 on every call. Each agent's `except` turns that into
    a silent fallback ("no shape extracted", "no diet found"), so the symptom is
    a feature that quietly does nothing.

    It has happened. The in-code prompt said "json"; the managed copy in
    Langfuse did not, and prompts are served from Langfuse at runtime while
    existing copies are never overwritten by a deploy — so multi-plate planning
    was disabled in production while every local test passed.

    That is why the guarantee cannot live in the prompt text. It lives here, at
    the last point before the call, applied to whatever the messages turned out
    to be. `tests/test_prompt_contracts.py` fails the build if a schema agent
    invokes without it, so a new agent cannot inherit the bug by forgetting.
    """
    if not messages:
        return messages
    if "json" in " ".join(
        str(getattr(m, "content", "") or "") for m in messages
    ).lower():
        return messages
    head, *rest = messages
    logger.info(
        "Prompt reached the client without the word 'json' — adding the "
        "instruction Groq requires rather than taking a 400."
    )
    return [
        SystemMessage(
            content=f"{getattr(head, 'content', '')}\nReturn the result as a JSON object."
        ),
        *rest,
    ]


class DocumentGrader:
    """Scores candidate days against the query + profile.

    `grade_daily_plans` is the three-slot entry point every existing caller
    uses, and it now delegates to `grade_plans`, which takes whatever slots it
    is given. The three-slot version was not a simplification — it was the
    reason a four-meal day, or a dinner served as a main and a side, could not
    be ranked at all, which is why the structured planning path shipped with no
    grading and no quality metrics.
    """

    def __init__(self, model: str = None, temperature: float = None, max_plans_to_score: int = None):
        self.grader = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=BatchScoringSchema.model_json_schema(),
        )
        self.max_plans_to_score = max_plans_to_score or MAX_PLANS_TO_SCORE

    def grade_daily_plans(
        self, query: str, candidates: CandidatesBySlot, user_profile: dict,
        feedback_history: str = "", prefer_items: Sequence[str] = (),
    ) -> list[ScoredPlan]:
        """The three-slot entry point. Kept because every caller uses it."""
        return self.grade_plans(
            query, candidates, user_profile, feedback_history,
            slots=("breakfast", "lunch", "dinner"),
            prefer_items=prefer_items,
        )

    def grade_plans(
        self, query: str, candidates: CandidatesBySlot, user_profile: dict,
        feedback_history: str = "", slots: Optional[Sequence[str]] = None,
        prefer_items: Sequence[str] = (),
    ) -> list[ScoredPlan]:
        """Return the top-scored days, best first (at most 3).

        One LLM call for the whole batch. One call *per combination* made
        grading the latency floor of every plan request — ten sequential
        Groq round-trips before the member saw anything — and scored each
        day in isolation, when the actual task is comparative: pick the best
        day of the batch.

        Sampling is rank-aware, not uniform. RecipeWrangler returns each
        slot's candidates best-first (planning tier, Nutri-Score, curated
        source); `random.sample` over the full product ignored that order,
        so the strongest combination could simply never be graded. The
        top-of-ranking combo is always in the batch; the rest of the space
        still gets sampled so the judge sees variety.
        """
        # Which slots this day has. Given explicitly by the three-slot entry
        # point; otherwise whatever the pool actually filled, in eating order.
        if slots:
            # An explicitly requested shape fails CLOSED. The three-slot entry
            # point's caller stores the result through `MealPlan.from_courses`,
            # which requires exactly three — so quietly grading a two-slot day
            # because lunch came back empty would turn a "no candidates"
            # warning into a 500 two frames later. The caller degrades to the
            # unranked pool on `[]`, which is the right answer here.
            names = [str(n) for n in slots]
            empty = [n for n in names if not candidates.get(n)]
            if empty:
                logger.warning(
                    "No candidates for %s — cannot grade the requested shape",
                    ", ".join(empty),
                )
                return []
        else:
            # An inferred shape takes whatever the pool actually filled: there
            # is no downstream contract to break, and a day of two real meals
            # beats no ranking at all.
            names = [
                n for n in sorted(candidates, key=slot_sort_key) if candidates.get(n)
            ]
        if not names:
            logger.warning("No slot has candidates — nothing to grade")
            return []

        # The product is bounded before it is built, not after.
        #
        # Three slots of eight candidates is 512 combinations; the old code
        # materialised all of them and sampled ten. Seven slots of eight is
        # two million, and a day with a main and two sides is worse — so a
        # generalised grader that kept `list(itertools.product(...))` would
        # hang the turn on exactly the plans this change exists to support.
        #
        # `islice` walks the product lazily and stops. Because `product`
        # iterates its LAST argument fastest, walking a prefix would vary only
        # the final slot — so the prefix is taken for its guaranteed
        # top-of-ranking first element, and the variety comes from independent
        # per-slot sampling below.
        pools = [candidates[n] for n in names]
        best_combo = tuple(pool[0] for pool in pools)
        head = list(itertools.islice(itertools.product(*pools), _COMBO_SCAN_LIMIT))
        logger.info(
            "Grading %d-slot days (%s) — scanned %d combination(s)",
            len(names), ", ".join(names), len(head),
        )

        # The top-of-ranking day is always graded; the rest of the batch is
        # sampled per slot so the judge sees variety across every slot rather
        # than across the last one only.
        # Deduplicated by RECIPE ID, not by hashing the candidates.
        #
        # `CandidateRecipe` is a frozen dataclass, so it looks hashable — until
        # one of its fields is the `nutrition` dict, and then `hash()` raises
        # `TypeError: unhashable type: 'dict'`. Every candidate from
        # `plan_meals` carries nutrition, so putting a combination in a set
        # threw on EVERY real plan: the pipeline caught it, served the unranked
        # pool, and told the member "not ranked — grader unavailable". Not
        # flaky — every single time, and invisible here because the test
        # fixtures build candidates with no macros at all.
        wanted = max(0, self.max_plans_to_score - 1)
        rest: list[tuple] = []

        def combo_key(combo) -> tuple:
            return tuple(str(c.recipe_id) for c in combo)

        seen = {combo_key(best_combo)}

        # The day that uses up the most of what the member already has.
        #
        # Put in the batch rather than left to the sampler: asking the grader
        # in prose to "prefer combinations that use as many as possible" can
        # only work on the combinations it is shown, and those were drawn at
        # random. A member with tomatoes, feta and basil could watch all three
        # go unused because no sampled day happened to cover them.
        if prefer_items:
            from services.pantry_service import best_covering_combo

            covering = best_covering_combo(pools, prefer_items)
            if covering is not None and combo_key(covering) not in seen:
                seen.add(combo_key(covering))
                rest.append(covering)

        for _ in range(wanted * 4):          # bounded attempts, not a while-true
            if len(rest) >= wanted:
                break
            combo = tuple(random.choice(pool) for pool in pools)
            key = combo_key(combo)
            if key in seen:
                continue
            seen.add(key)
            rest.append(combo)
        # A pool small enough to enumerate gets exhaustive coverage instead of
        # sampling, which is the old behaviour for three short slots.
        if len(head) <= self.max_plans_to_score:
            rest = [
                c for c in head if combo_key(c) != combo_key(best_combo)
            ][:wanted]
        sampled = [best_combo] + rest

        def course_text(slot: str, course) -> str:
            lines = [f"{slot}: {course.title}"]
            nutrition = getattr(course, "nutrition", None) or {}
            kcal = nutrition.get("kcal") or nutrition.get("calories")
            protein = nutrition.get("protein_g")
            if kcal is not None:
                macro = f"  ~{round(float(kcal))} kcal"
                if protein is not None:
                    macro += f", {round(float(protein))}g protein"
                lines.append(macro)
            lines.append(f"  Ingredients: {str(course.ingredients)[:400]}")
            return "\n".join(lines)

        plans_text = "\n\n".join(
            f"PLAN {i}\n" + "\n".join(
                course_text(slot, course)
                for slot, course in zip(names, combo)
            )
            for i, combo in enumerate(sampled)
        )

        try:
            result = self.grader.invoke(as_json_messages([
                SystemMessage(content=PLAN_GRADER_SYSTEM.compile()),
                HumanMessage(content=PLAN_GRADER_USER.compile(
                    plan_count=len(sampled),
                    query=query,
                    plans=plans_text,
                    preferences=",".join(user_profile.get("preferences", [])),
                    feedback_history=feedback_history or "No prior feedback.",
                )),
            ]), config=build_trace_config(run_name="plan_grade_batch", tags=["planning"]))
            grades = json.loads(result.content).get("grades", [])
        except Exception as exc:  # noqa: BLE001
            # The caller already degrades to the unranked pool on [].
            logger.warning("Batch grading failed: %s", exc)
            return []

        scored: list[ScoredPlan] = []
        for grade in grades:
            try:
                index = int(grade.get("plan_index"))
                combo = sampled[index]
            except (TypeError, ValueError, IndexError):
                continue
            scored.append(ScoredPlan(
                slots=dict(zip(names, combo)),
                score=int(grade.get("score", 0)),
                reasoning=str(grade.get("reasoning", "")),
            ))

        scored.sort(key=lambda p: p.score, reverse=True)
        top = scored[:3]
        logger.info("Top plan scores: %s", [p.score for p in top])
        return top


class MealDiversityGrader:
    """LLM judge for the ingredient/cuisine diversity of a rendered plan."""

    def __init__(self, model: str = None, temperature: float = None):
        self.client = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=ScoringSchema.model_json_schema(),
        )

    def score(self, plan_text: str) -> dict:
        result = self.client.invoke(as_json_messages([
            SystemMessage(content=MEAL_DIVERSITY_SYSTEM.compile()),
            HumanMessage(content=plan_text),
        ]), config=build_trace_config(run_name="meal_diversity", tags=["metrics"]))
        try:
            return json.loads(result.content)
        except Exception:
            return {"reasoning": "Could not parse diversity score", "score": 0}


class GuidelineAdherenceGrader:
    """LLM judge for adherence to national dietary guidelines."""

    def __init__(self, model: str = None, temperature: float = None):
        self.client = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=ScoringSchema.model_json_schema(),
        )

    def score(self, plan_text: str, guidelines_text: str) -> dict:
        result = self.client.invoke(as_json_messages([
            SystemMessage(content=GUIDELINE_ADHERENCE_SYSTEM.compile()),
            HumanMessage(content=f"GUIDELINES:\n{guidelines_text}\n\nMEAL PLAN:\n{plan_text}"),
        ]), config=build_trace_config(run_name="guideline_adherence", tags=["metrics"]))
        try:
            return json.loads(result.content)
        except Exception:
            return {"reasoning": "Could not parse guideline adherence score", "score": 0}


class PlanJudge:
    """The plan scorer's judge: diversity, guideline adherence and fit in ONE call.

    The planner's own judges above are untouched and still grade generated
    daily plans one metric at a time. This one exists because a pasted plan
    needs three judgements over the SAME text: sending that text (and the
    member's profile) three times cost three prompts and three reasoning
    passes, which is most of a minute's token allowance on the on-demand tier.

    The fit half shares ``prompts.PLAN_SCORING_RUBRIC`` with the planner's
    batch grader, and ``plan_scorer.scoring`` still caps the fit score in code,
    so a model that under-weights an allergen cannot out-vote the check.

    Raises on a failed call or unparseable content; the scorer retries once and
    then reports every judged metric as ungraded.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.client = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=PlanJudgementSchema.model_json_schema(),
        )

    def judge(
        self, *, weekly: bool, plan_text: str, plan_shape: str, hard_constraints: str,
        conflicts: str, preferences: str, aim: str, guidelines: str, facts: str,
    ) -> dict:
        system = PLAN_JUDGE_WEEKLY_SYSTEM if weekly else PLAN_JUDGE_DAILY_SYSTEM
        result = self.client.invoke(as_json_messages([
            SystemMessage(content=system.compile()),
            HumanMessage(content=PLAN_JUDGE_USER.compile(
                hard_constraints=hard_constraints,
                conflicts=conflicts,
                preferences=preferences,
                aim=aim,
                guidelines=guidelines or "(none available)",
                facts=facts or "(none)",
                plan_shape=plan_shape,
                plan=plan_text,
            )),
        ]), config=build_trace_config(
            run_name="plan_judge_weekly" if weekly else "plan_judge",
            tags=["metrics", "scoring"],
        ))
        return json.loads(result.content)


class SimpleChatBot:
    """Small-talk / out-of-scope fallback.

    Stateless: conversation history is passed per call as (role, content)
    tuples taken from the session's persisted conversation. Pre-M0 this class
    held one process-global ConversationBufferMemory shared by ALL sessions —
    a cross-user context leak. Do not reintroduce instance-level history.
    """

    # Nutrition-science questions never reach this prompt — the orchestrator
    # routes them to FoodScholar (nutrition_question intent). This bot only
    # sees greetings, small talk, and the leftovers; it must never claim it
    # "can't answer" something — it redirects warmly instead. The persona text
    # lives in the prompt registry (prompts.CHATBOT_SYSTEM) so it is Langfuse-
    # managed like every other prompt.

    def __init__(self, model: str = None, temperature: float = None):
        self.chatbot = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else CHATBOT_TEMPERATURE,
        )

    def chat(self, query: str, history: list[tuple[str, str]] = None) -> str:
        """Respond to ``query`` given recent (role, content) history pairs."""
        messages = [SystemMessage(content=CHATBOT_SYSTEM.compile())]
        for role, content in (history or [])[-CHATBOT_HISTORY_TURNS:]:
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=query))
        return self.chatbot.invoke(
            messages,
            config=build_trace_config(run_name="smalltalk", tags=["chat"]),
        ).content


class QueryReconciler:
    """Detects dietary conflicts / missing info between a query and the profile."""

    def __init__(self, model: str = None, temperature: float = None):
        self.query_reconciler = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=QueryReconcilerSchema.model_json_schema(),
        )

    def reconcile(self, query: str, user_profile: dict) -> dict:
        # Never-ask-twice (M4c): everything the profile already answers is
        # surfaced to the reconciler so it cannot mark it "missing".
        known_facts = "; ".join(filter(None, [
            ", ".join(user_profile.get("preferences") or []),
            user_profile.get("history") or "",
            ", ".join(f"likes {like}" for like in (user_profile.get("food_likes") or [])[:5]),
        ])) or "(nothing on file)"

        result = self.query_reconciler.invoke(as_json_messages([
            SystemMessage(content=QUERY_RECONCILER_SYSTEM.compile()),
            HumanMessage(content=QUERY_RECONCILER_USER.compile(
                query=query,
                diet=user_profile.get("diet", []),
                allergies=user_profile.get("allergies", []),
                known_facts=known_facts,
            )),
        ]), config=build_trace_config(run_name="query_reconcile", tags=["clarify"]))
        return json.loads(result.content)


class DietaryIntentExtractor:
    """Extracts dietary requirement tags (vegan, low-carb, …) from a user query."""

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=DietaryTagsSchema.model_json_schema(),
        )

    def extract(self, query: str) -> list[str]:
        try:
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=DIETARY_INTENT_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=DIETARY_INTENT_EXTRACTOR_USER.compile(query=query)),
            ]), config=build_trace_config(run_name="dietary_intent", tags=["extract"]))
            return json.loads(result.content).get("dietary_tags", [])
        except Exception as e:
            logger.warning("DietaryIntentExtractor failed: %s", e)
            return []


class PlanSpecExtractor:
    """Works out the SHAPE of plan a user asked for — days, meals, plates.

    Separate from the dietary and seed extractors because it answers a
    different kind of question: not *what* to cook but *how many things, when,
    and served as how many plates*. The seed extractor is already told
    explicitly to ignore everything this one looks for.

    Abstention is the important behaviour. Most messages say nothing about
    shape, and a shape invented from a vague message changes what the user is
    actually served — so `mentioned: false` falls straight through to the
    default three meals.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=PlanSpecSchema.model_json_schema(),
        )

    def extract(self, query: str) -> "PlanSpec":
        """Return a `PlanSpec`; the default shape when nothing was asked.

        Never raises. A model outage must not stop a plan being made — it just
        means the plan has the shape it has always had.
        """
        from models.plan_spec import PlanSpec

        try:
            system_text = PLAN_SPEC_EXTRACTOR_SYSTEM.compile()
            user_text = PLAN_SPEC_EXTRACTOR_USER.compile(query=query)
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="plan_spec", tags=["extract"]))
            payload = json.loads(result.content)
        except Exception as e:
            logger.warning("PlanSpecExtractor failed: %s", e)
            return PlanSpec.default()

        if not payload.get("mentioned"):
            return PlanSpec.default()

        # The schema carries plates as a list of {slot, roles}; PlanSpec keys
        # them by slot. Converted here rather than in the model so the model
        # stays independent of whatever shape a given extractor emits.
        spec = PlanSpec.from_spec({
            "num_days": payload.get("num_days"),
            "meals": payload.get("meals"),
            "plates": {
                entry.get("slot"): entry.get("roles")
                for entry in (payload.get("plates") or [])
                if entry.get("slot")
            },
        })
        logger.info("Plan shape requested: %s", spec.describe())
        return spec


class PlanTextParser:
    """Reads a meal plan the MEMBER wrote into days, slots and dishes (plan scorer).

    Span-picking, not judgment, so it runs on the fast model like the other
    extractors. Only called when the deterministic line scanner in
    ``services.plan_scorer.parsing`` could not read the text from structure
    alone. Returns the raw ``ParsedPlanSchema`` payload, or None on any
    failure — the caller checks every title, ingredient list and unparsed
    line against the pasted text, so an invented ingredient never survives a
    prompt that drifted in Langfuse.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=ParsedPlanSchema.model_json_schema(),
        )

    def parse(self, text: str, structure_hint: str = "") -> Optional[dict]:
        try:
            system_text = PLAN_TEXT_PARSER_SYSTEM.compile()
            user_text = PLAN_TEXT_PARSER_USER.compile(
                plan_text=text, structure_hint=structure_hint or "(none)",
            )
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="plan_text_parse", tags=["extract", "scoring"]))
            payload = json.loads(result.content)
        except Exception as e:
            logger.warning("PlanTextParser failed: %s", e)
            return None
        return payload if isinstance(payload, dict) else None


class DishIngredientEstimator:
    """Typical single-serving ingredients for dishes no recipe matched (plan scorer).

    One batched call per pasted plan, on the fast model: ingredient lines with
    quantities, which ``plan_scorer.grounding`` hands to RecipeWrangler's
    profiler so the nutrition comes from composition tables rather than from
    the model. The model's own calorie guess rides along and is used only when
    the profiler cannot give reliable figures.

    ``estimate`` takes ``[{title, ingredients?, quantity?}]`` and returns
    ``{index: {"ingredients": [(quantity, name)], "kcal": float | None}}`` —
    ``{}`` on any failure. A wrong or missing entry for one dish never
    affects another.
    """

    MAX_INGREDIENTS = 15

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=DishEstimatesSchema.model_json_schema(),
        )

    def estimate(self, dishes: list[dict]) -> dict[int, dict]:
        if not dishes:
            return {}
        listing = []
        for index, dish in enumerate(dishes):
            line = f"{index}. {dish.get('title', '')}"
            if dish.get("ingredients"):
                line += f" (the user's ingredients: {dish['ingredients']})"
            if dish.get("quantity"):
                line += f" [amount: {dish['quantity']}]"
            listing.append(line)
        try:
            system_text = DISH_ESTIMATOR_SYSTEM.compile()
            user_text = DISH_ESTIMATOR_USER.compile(dishes="\n".join(listing))
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="dish_estimate", tags=["extract", "scoring"]))
            payload = json.loads(result.content)
        except Exception as e:
            logger.warning("DishIngredientEstimator failed: %s", e)
            return {}

        estimates: dict[int, dict] = {}
        items = payload.get("dishes") if isinstance(payload, dict) else None
        for item in items or []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(dishes):
                continue
            ingredients, seen = [], set()
            for line in item.get("ingredients") or []:
                if not isinstance(line, dict):
                    continue
                name = str(line.get("name") or "").strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    ingredients.append((str(line.get("quantity") or "").strip(), name))
            kcal = item.get("kcal_per_serving")
            estimates[index] = {
                "ingredients": ingredients[:self.MAX_INGREDIENTS],
                "kcal": float(kcal) if isinstance(kcal, (int, float)) and not isinstance(kcal, bool) else None,
            }
        return estimates


class PreferenceExtractor:
    """Detects durable preference candidates in a user turn (M3 memory).

    Output feeds ``services.memory_service`` which applies the consent
    policy — this agent only detects; it NEVER writes memory itself.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=PreferenceExtractionSchema.model_json_schema(),
        )

    def extract(self, message: str) -> list[dict]:
        try:
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=PREFERENCE_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=PREFERENCE_EXTRACTOR_USER.compile(message=message)),
            ]), config=build_trace_config(run_name="preference_extract", tags=["memory"]))
            memories = json.loads(result.content).get("memories", [])
            return [m for m in memories if isinstance(m, dict) and m.get("value") and m.get("kind")]
        except Exception as e:
            logger.warning("PreferenceExtractor failed: %s", e)
            return []


class SeedExtractor:
    """Extracts named anchor dishes ("pastitsio", "fakes") from a plan request.

    Returns [{"name", "meal_type"|None, "day"|None}] — empty when the user
    named no specific dish. Consumers: seed_service (resolution + pinning)
    and the favorites-offer gate (an explicit dish request suppresses the offer).
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=SeedExtractionSchema.model_json_schema(),
        )

    def extract(self, query: str) -> list[dict]:
        try:
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=SEED_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=SEED_EXTRACTOR_USER.compile(query=query)),
            ]), config=build_trace_config(run_name="seed_extract", tags=["planning"]))
            seeds = json.loads(result.content).get("seeds", [])
            return [s for s in seeds if isinstance(s, dict) and s.get("name")]
        except Exception as e:
            logger.warning("SeedExtractor failed: %s", e)
            return []


class PantryExtractor:
    """Extracts on-hand ingredients ("I have zucchini and spinach") for
    pantry-driven, food-waste-reducing planning.

    Deliberately NOT folded into SeedExtractor: that agent's managed prompt
    explicitly refuses ingredients, and existing Langfuse prompt copies are
    never overwritten by a deploy — extending it would work locally and stay
    silently disabled in production (the PlanSpecExtractor "json" incident).
    A new agent with a new prompt name syncs cleanly. Callers gate this behind
    a cheap regex (services.pantry_service) so most turns never pay the call.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=PantryExtractionSchema.model_json_schema(),
        )

    def extract(self, message: str) -> dict:
        """{"have": [...], "used_up": [...]} — both empty when nothing stated."""
        empty = {"have": [], "used_up": []}
        try:
            system_text = PANTRY_EXTRACTOR_SYSTEM.compile()
            user_text = PANTRY_EXTRACTOR_USER.compile(message=message)
            # Same structural guard as PlanSpecExtractor: Groq 400s any
            # json_object request whose messages omit the word "json", and a
            # managed prompt edit can strip it silently.
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="pantry_extract", tags=["planning"]))
            payload = json.loads(result.content)
        except Exception as e:
            logger.warning("PantryExtractor failed: %s", e)
            return empty
        if not payload.get("mentioned"):
            return empty
        return {
            "have": [str(i) for i in (payload.get("have") or []) if i],
            "used_up": [str(i) for i in (payload.get("used_up") or []) if i],
        }


class EditCommandExtractor:
    """Parses a targeted slot-edit request into a structured command (M4b)."""

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=EditCommandSchema.model_json_schema(),
        )

    def extract(self, message: str, plan_type: str) -> Optional[dict]:
        """Returns the command dict, or None when parsing fails (caller
        degrades to a whole-plan refinement)."""
        try:
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=EDIT_COMMAND_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=EDIT_COMMAND_EXTRACTOR_USER.compile(
                    plan_type=plan_type, message=message,
                )),
            ]), config=build_trace_config(run_name="edit_command", tags=["edit"]))
            command = json.loads(result.content)
            if not command.get("directive"):
                command["directive"] = "different"
            return command
        except Exception as e:
            logger.warning("EditCommandExtractor failed: %s", e)
            return None


class ResponseWriter:
    """Grounded persona voice (M4c) — writes chat prose from structured facts.

    It can phrase, emphasize, and echo the user's wording, but every concrete
    claim must come from the facts dict. Callers keep a canned fallback for
    LLM failures — a broken writer must never block a plan response.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else CHATBOT_TEMPERATURE,
        )

    def write(self, facts: dict, user_message: str, fallback: str) -> str:
        try:
            result = self.llm.invoke([
                SystemMessage(content=RESPONSE_WRITER_SYSTEM.compile()),
                HumanMessage(content=RESPONSE_WRITER_USER.compile(
                    facts=json.dumps(facts, ensure_ascii=False),
                    user_message=user_message[:400],
                )),
            ], config=build_trace_config(run_name="response_write", tags=["chat"]))
            text = (result.content or "").strip()
            # Guard against runaway or empty generations.
            if not text or len(text) > 900:
                return fallback
            return text
        except Exception as e:
            logger.warning("ResponseWriter failed, using fallback: %s", e)
            return fallback


class PlanAnalyst:
    """Answers questions ABOUT the active plan ("does it adhere to that?").

    Grounded in a serialized plan summary (titles + per-meal nutrition) and
    recent conversation (so "that" resolves to e.g. the protein guidance the
    user just read). Read-only by design: it never generates or edits a plan.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else CHATBOT_TEMPERATURE,
        )

    def answer(self, question: str, plan_summary: str,
               history: list[tuple[str, str]] = None) -> str:
        messages = [SystemMessage(content=PLAN_ANALYST_SYSTEM.compile())]
        for role, content in (history or [])[-CHATBOT_HISTORY_TURNS:]:
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(
            content=f"CURRENT PLAN:\n{plan_summary}\n\nQUESTION: {question}"
        ))
        return self.llm.invoke(
            messages,
            config=build_trace_config(run_name="plan_analyst", tags=["plan_qa"]),
        ).content


class PlanIntentExtractor:
    """Names the recipe qualities a message asks for, as RecipeWrangler facets.

    A separate agent from `DietaryIntentExtractor` rather than an extension of
    it: that one's prompt is Langfuse-managed and a deploy never overwrites an
    existing copy, so adding facets there would work locally and ship dead.

    The live vocabulary is injected into the prompt at call time. It is not
    decoration — RecipeWrangler ANDs facet values and an unlisted one matches no
    recipe, so a hallucinated "energising" mood would empty every slot and the
    member would be told no meals exist. Post-validated against the same
    vocabulary anyway, because a prompt instruction is not a guarantee.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=PlanIntentSchema.model_json_schema(),
        )

    def extract(self, message: str, vocabularies: dict) -> dict:
        """{"cuisines": [...], "moods": [...], ...} — only listed values."""
        families = ("cuisines", "moods", "flavor_profiles", "food_groups")
        allowed = {
            family: [str(v).lower() for v in (vocabularies.get(family) or [])]
            for family in families
        }
        if not any(allowed.values()):
            # No vocabulary means no safe value to send. Sending nothing is the
            # behaviour that existed before facets were wired at all.
            return {family: [] for family in families}

        empty = {family: [] for family in families}
        try:
            system_text = PLAN_INTENT_EXTRACTOR_SYSTEM.compile(
                **{f: ", ".join(allowed[f]) or "(none)" for f in families}
            )
            user_text = PLAN_INTENT_EXTRACTOR_USER.compile(message=message)
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="plan_intent", tags=["planning"]))
            payload = json.loads(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PlanIntentExtractor failed: %s", exc)
            return empty

        out = {}
        for family in families:
            values = payload.get(family) or []
            permitted = set(allowed[family])
            kept = []
            for value in values:
                slug = str(value).strip().lower().replace("-", "_").replace(" ", "_")
                if slug in permitted and slug not in kept:
                    kept.append(slug)
                elif slug:
                    logger.info(
                        "Dropping %s=%r — not in the live vocabulary", family, value
                    )
            out[family] = kept
        return out


class ToolSelector:
    """Chooses one of FoodChat's own capabilities, or none.

    The tool registry has been complete and unreachable from chat: `manifest()`
    and `describe_tools()` are generated from it, and neither reached a prompt.
    The only in-chat tool call was one hardcoded `plan_totals` to stop the
    analyst doing arithmetic in prose. So "summarise my week" and "redo
    Thursday" had no path — the closest available action was a full refinement,
    which regenerates every slot and throws away a swap the member already
    approved.

    Fast tier, and a small prompt. This is a routing decision over a handful of
    named capabilities, not a judgement about food — and it runs before the
    intent classifier, so it must be cheap enough that a turn which selects
    NOTHING has barely paid for the question.

    Returns `{}` for "no tool", which is the expected answer for most messages.
    A failure is also `{}`: the turn then routes exactly as it did before this
    existed.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=ToolChoiceSchema.model_json_schema(),
        )

    def choose(self, message: str, *, plan_type: str, plan_shape: str,
               manifest: str, allowed: set) -> dict:
        """`{"tool": name, "day": int|None, ...}` or `{}`. Never raises."""
        if not message.strip() or not allowed:
            return {}
        try:
            system_text = TOOL_SELECTOR_SYSTEM.compile(tools=manifest)
            user_text = TOOL_SELECTOR_USER.compile(
                message=message[:400], plan_type=plan_type, plan_shape=plan_shape,
            )
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="tool_select", tags=["tools"]))
            payload = json.loads(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ToolSelector failed, routing normally: %s", exc)
            return {}

        if not isinstance(payload, dict):
            return {}
        name = str(payload.get("tool") or "").strip()
        if not name:
            return {}
        if name not in allowed:
            # A tool that does not exist, or one this canvas cannot serve.
            # Dropped rather than attempted: the registry would reject it, but
            # a 400 is a worse answer than routing the message normally.
            logger.info("ToolSelector chose %r, which is not available here", name)
            return {}
        logger.info("Tool selected: %s — %s", name, payload.get("reason") or "")
        return payload


class PlanStrategist:
    """Decides HOW to search, before a recipe is fetched.

    The reasoning half of the hybrid. The pipeline stays the executor and the
    verifier stays deterministic; what this adds is a step that reads what the
    member actually meant and shapes the search accordingly, instead of a fixed
    chain that maps the same words to the same filters every time.

    It runs on the REASONING tier, not the fast one, and that is the point:
    "something light after the gym" becoming high protein with a light mood is
    a judgement about food, not a span to pick out of a sentence.

    Three things keep it safe:

    * It cannot touch allergens or diet. Those are not in its schema, they are
      derived deterministically, and the verifier checks them on the way back.
      A reasoning step may decide how to search; it may not decide to drop a
      safety constraint.
    * Every value it proposes is validated against the LIVE vocabulary by
      `PlanBrief.with_strategy` before it reaches a search — because the search
      ANDs facet values and never relaxes an unknown one, so an invented mood
      empties the result set rather than narrowing it.
    * A failure returns `{}`, leaving the deterministic brief exactly as it
      was. The plan that used to be built is the floor, never the casualty.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=PlanStrategySchema.model_json_schema(),
        )

    def propose(self, message: str, brief, vocabularies: dict) -> dict:
        """A proposal dict for `PlanBrief.with_strategy`. Never raises.

        Returns `{}` — meaning "no adjustment" — whenever the vocabulary is
        unavailable or the call fails. With no live list there is no value that
        is safe to add, and the deterministic brief is a working plan on its
        own.
        """
        families = ("cuisines", "moods", "flavor_profiles", "food_groups")
        allowed = {
            family: [str(v).lower() for v in (vocabularies.get(family) or [])]
            for family in families
        }
        if not any(allowed.values()):
            return {}

        try:
            vocab_text = "\n".join(
                f"{family}: {', '.join(allowed[family]) or '(none)'}"
                for family in families
            )
            system_text = PLAN_STRATEGIST_SYSTEM.compile()
            user_text = PLAN_STRATEGIST_USER.compile(
                message=(message or "")[:600],
                standing=brief.describe(),
                vocabularies=vocab_text,
            )
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="plan_strategy", tags=["planning"]))
            payload = json.loads(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PlanStrategist failed, using the plain brief: %s", exc)
            return {}

        if not isinstance(payload, dict):
            return {}
        logger.info("Plan strategy: %s", payload.get("rationale") or payload)
        return payload


class MealJudge:
    """Chooses between complete meals the arithmetic could not separate.

    The last step of composition, and deliberately the only part of it that
    costs a model call. Whether two plates are the same dish, repeat an
    ingredient, or add up to the meal's share of the day is measurable, and
    `meal_composer` measures it. Whether a dish SUITS another one is not: the
    corpus's cuisine annotation does not survive into the planning envelope in
    any shape this service reads, and texture and richness are not annotated
    at all. So that judgement goes to a judgement.

    **One call for the whole plan.** A week with a side at dinner is seven of
    these, and seven round trips inside one turn budget is how a plan stops
    arriving. The options are batched, labelled, and matched back by label
    rather than by position — a model that answers in a different order is
    common and is not a reason to lose the answer.

    Reasoning tier, like the strategist: this is a judgement about food, not a
    span to pick out of a sentence.

    Returns `{}` on any failure. The deterministic winner is already the first
    option, so a judge that is down costs the plan its polish and never its
    existence.
    """

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
            format=MealCompositionSchema.model_json_schema(),
        )

    def choose(self, message: str, offers: list[dict]) -> dict:
        """`{meal label: (index, reason)}` for the meals it ruled on.

        `offers` is `[{"meal": label, "options": [text, …]}]`. A label the
        caller did not ask about, or an index outside the options offered, is
        dropped rather than corrected — the caller's fallback is the option the
        arithmetic already chose, which is a good answer, so a confused reply
        should cost nothing rather than land somewhere unintended.
        """
        usable = [o for o in offers if o.get("meal") and len(o.get("options") or []) > 1]
        if not usable:
            return {}

        blocks = []
        for offer in usable:
            lines = [f"MEAL: {offer['meal']}"]
            for index, text in enumerate(offer["options"]):
                lines.append(f"  [{index}] {text}")
            blocks.append("\n".join(lines))

        try:
            system_text = MEAL_COMPOSER_SYSTEM.compile()
            user_text = MEAL_COMPOSER_USER.compile(
                message=(message or "a meal plan")[:400],
                meals="\n\n".join(blocks),
            )
            result = self.llm.invoke(as_json_messages([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]), config=build_trace_config(run_name="meal_compose", tags=["planning"]))
            payload = json.loads(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MealJudge failed, keeping the measured order: %s", exc)
            return {}

        if not isinstance(payload, dict):
            return {}
        sizes = {o["meal"]: len(o["options"]) for o in usable}
        chosen: dict[str, tuple[int, str]] = {}
        for entry in payload.get("choices") or []:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("meal") or "").strip()
            if label not in sizes:
                continue
            try:
                pick = int(entry.get("pick") or 0)
            except (TypeError, ValueError):
                continue
            if not 0 <= pick < sizes[label]:
                logger.info(
                    "MealJudge chose option %d for %r, which was not offered",
                    pick, label,
                )
                continue
            chosen[label] = (pick, str(entry.get("reason") or ""))
        if chosen:
            logger.info(
                "Meal judge: %s",
                "; ".join(f"{k}->{v[0]}" for k, v in sorted(chosen.items())),
            )
        return chosen


class SessionTitler:
    """Names a planning conversation from its opening message.

    Sessions were only ever named by an explicit rename, which almost nobody
    does — so the picker showed a wall of timestamps, and a saved plan inherited
    `undefined` as its name because the save path borrows the session title.

    Plain text, not JSON: the whole answer IS the title, and a schema would only
    add a wrapper to unwrap. Runs on the fast tier — naming a conversation is
    not a reasoning task, and it happens once per session.
    """

    # Longer than any name this prompt should produce; the column allows 120.
    _MAX_LEN = 60

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or FAST_MODEL,
            temperature=(
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            ),
        )

    @staticmethod
    def _clean(raw: str) -> Optional[str]:
        """The title, or None when the model declined or rambled.

        None is the safe direction: the caller leaves the session untitled and
        the client falls back to its timestamp, which is worse than a good name
        but better than a wrong one — and a member rename still wins either way.
        """
        text = (raw or "").strip()
        # A reasoning model with reasoning hidden still occasionally prefixes a
        # line; the title is the last non-empty line in that case.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        title = lines[-1].strip().strip('"').strip("'").rstrip(".").strip()
        if not title or title.upper() == "NONE":
            return None
        if len(title) > SessionTitler._MAX_LEN:
            return None  # a rambling answer is not a name
        return title

    def title(self, message: str) -> Optional[str]:
        try:
            result = self.llm.invoke([
                SystemMessage(content=SESSION_TITLE_SYSTEM.compile()),
                HumanMessage(content=SESSION_TITLE_USER.compile(message=message)),
            ], config=build_trace_config(run_name="session_title", tags=["session"]))
            return self._clean(result.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SessionTitler failed: %s", exc)
            return None


class OrchestratorAgent:
    """Single intent classifier per turn — the ONLY router in the pipeline.

    Valid intents: daily_plan | weekly_plan | refine_plan | edit_plan_slot
    | switch_plan_type | nutrition_question | plan_question
    | preference_update | score_plan | chat.
    ``target_plan_type`` is populated only for switch_plan_type;
    nutrition_question turns are delegated to FoodScholar; plan_question
    turns are answered by the PlanAnalyst grounded in the active canvas;
    edit_plan_slot targets ONE slot of the active canvas with a verified
    directive; preference_update is a stated durable preference — it is
    acknowledged, never interrogated (the write stays behind the M3 nudge);
    score_plan is a plan the member wrote and pasted, handed to the plan
    scorer — it never touches a canvas.
    """

    VALID_INTENTS = {
        "daily_plan", "weekly_plan", "refine_plan", "edit_plan_slot",
        "switch_plan_type", "nutrition_question", "plan_question",
        "preference_update", "score_plan", "chat",
    }

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=OrchestratorSchema.model_json_schema(),
        )

    @staticmethod
    def system_prompt() -> str:
        """The router prompt, guaranteed to know every routed intent.

        A managed Langfuse copy that predates score_plan is never overwritten
        by a deploy, so the intent is appended when the compiled text lacks it
        — otherwise production could never classify a pasted plan.
        """
        text = ORCHESTRATOR_SYSTEM.compile()
        if "score_plan" not in text:
            text += SCORE_PLAN_INTENT_ADDENDUM
        return text

    def classify(self, message: str, history: list[dict]) -> dict:
        """Classify intent given the last turns ({"role", "content"} dicts, recent last)."""
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content'][:300]}"
            for turn in history[-6:]
        ) or "(no prior conversation)"

        messages = [
            SystemMessage(content=self.system_prompt()),
            HumanMessage(content=ORCHESTRATOR_USER.compile(
                history=history_text, message=message,
            )),
        ]

        config = build_trace_config(run_name="orchestrate", tags=["router"])
        for attempt in range(MAX_RETRIES):
            try:
                result = self.llm.invoke(as_json_messages(messages), config=config)
                parsed = json.loads(result.content)
                intent = parsed.get("intent", "chat")
                if intent in self.VALID_INTENTS:
                    target = parsed.get("target_plan_type") if intent == "switch_plan_type" else None
                    logger.info(
                        "Orchestrator intent: %s target=%s — %s",
                        intent, target, parsed.get("reasoning", ""),
                    )
                    return {"intent": intent, "target_plan_type": target}
            except Exception as e:
                logger.warning("Orchestrator attempt %d failed: %s", attempt + 1, e)

        # `failed` is the difference between "the member is chatting" and "we
        # could not ask": on a rate-limited key every turn looked like small
        # talk, including pasted plans. The caller decides what to do about it.
        logger.error("Orchestrator failed after retries, defaulting to chat")
        return {"intent": "chat", "target_plan_type": None, "failed": True}
