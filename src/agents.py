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
    QueryReconciler          — query/profile conflicts   (planning_pipeline, clarification)
    DietaryIntentExtractor   — diet tags from a query    (weekly_plan_service)
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
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.groq import GROQ_CHAT
from backend.observability import build_trace_config
from models.recipe import CandidatesBySlot, ScoredPlan
from prompts import (
    BATCH_GRADER_USER,
    GRADER_SYSTEM,
    GRADER_USER,
    PLAN_ANALYST_SYSTEM,
    MEAL_DIVERSITY_SYSTEM,
    GUIDELINE_ADHERENCE_SYSTEM,
    QUERY_RECONCILER_SYSTEM,
    QUERY_RECONCILER_USER,
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_USER,
    DIETARY_INTENT_EXTRACTOR_SYSTEM,
    PLAN_SPEC_EXTRACTOR_SYSTEM,
    PLAN_SPEC_EXTRACTOR_USER,
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
)
from schemas import (
    BatchScoringSchema,
    ScoringSchema,
    QueryReconcilerSchema,
    OrchestratorSchema,
    DietaryTagsSchema,
    PlanSpecSchema,
    SeedExtractionSchema,
    PantryExtractionSchema,
    PreferenceExtractionSchema,
    EditCommandSchema,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("FOODCHAT_LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("FOODCHAT_LLM_TEMPERATURE", "0.0"))
CHATBOT_TEMPERATURE = float(os.getenv("FOODCHAT_CHATBOT_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("FOODCHAT_MAX_RETRIES", "3"))
MAX_PLANS_TO_SCORE = int(os.getenv("FOODCHAT_MAX_PLANS_TO_SCORE", "10"))
CHATBOT_HISTORY_TURNS = int(os.getenv("FOODCHAT_CHATBOT_HISTORY_TURNS", "12"))


class DocumentGrader:
    """Scores breakfast×lunch×dinner combinations against the query + profile."""

    def __init__(self, model: str = None, temperature: float = None, max_plans_to_score: int = None):
        self.grader = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=BatchScoringSchema.model_json_schema(),
        )
        self.max_plans_to_score = max_plans_to_score or MAX_PLANS_TO_SCORE

    def grade_daily_plans(
        self, query: str, candidates: CandidatesBySlot, user_profile: dict,
        feedback_history: str = "",
    ) -> list[ScoredPlan]:
        """Return the top-scored combinations, best first (at most 3).

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
        combos = list(itertools.product(
            candidates.get("breakfast", []),
            candidates.get("lunch", []),
            candidates.get("dinner", []),
        ))
        logger.info("Grading daily plans — %d possible combinations", len(combos))
        if not combos:
            logger.warning("No possible daily plans — at least one slot has no candidates")
            return []

        # combos[0] is top-of-ranking in every slot by construction of
        # itertools.product over best-first lists.
        rest = random.sample(combos[1:], min(len(combos) - 1, self.max_plans_to_score - 1))
        sampled = [combos[0]] + rest

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
                for slot, course in (("breakfast", b), ("lunch", l), ("dinner", d))
            )
            for i, (b, l, d) in enumerate(sampled)
        )

        try:
            result = self.grader.invoke([
                SystemMessage(content=GRADER_SYSTEM.compile()),
                HumanMessage(content=BATCH_GRADER_USER.compile(
                    plan_count=len(sampled),
                    query=query,
                    plans=plans_text,
                    preferences=",".join(user_profile.get("preferences", [])),
                    feedback_history=feedback_history or "No prior feedback.",
                )),
            ], config=build_trace_config(run_name="plan_grade_batch", tags=["planning"]))
            grades = json.loads(result.content).get("grades", [])
        except Exception as exc:  # noqa: BLE001
            # The caller already degrades to the unranked pool on [].
            logger.warning("Batch grading failed: %s", exc)
            return []

        scored: list[ScoredPlan] = []
        for grade in grades:
            try:
                index = int(grade.get("plan_index"))
                breakfast, lunch, dinner = sampled[index]
            except (TypeError, ValueError, IndexError):
                continue
            scored.append(ScoredPlan(
                breakfast=breakfast, lunch=lunch, dinner=dinner,
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
        result = self.client.invoke([
            SystemMessage(content=MEAL_DIVERSITY_SYSTEM.compile()),
            HumanMessage(content=plan_text),
        ], config=build_trace_config(run_name="meal_diversity", tags=["metrics"]))
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
        result = self.client.invoke([
            SystemMessage(content=GUIDELINE_ADHERENCE_SYSTEM.compile()),
            HumanMessage(content=f"GUIDELINES:\n{guidelines_text}\n\nMEAL PLAN:\n{plan_text}"),
        ], config=build_trace_config(run_name="guideline_adherence", tags=["metrics"]))
        try:
            return json.loads(result.content)
        except Exception:
            return {"reasoning": "Could not parse guideline adherence score", "score": 0}


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
            ", ".join(f"likes {l}" for l in (user_profile.get("food_likes") or [])[:5]),
        ])) or "(nothing on file)"

        result = self.query_reconciler.invoke([
            SystemMessage(content=QUERY_RECONCILER_SYSTEM.compile()),
            HumanMessage(content=QUERY_RECONCILER_USER.compile(
                query=query,
                diet=user_profile.get("diet", []),
                allergies=user_profile.get("allergies", []),
                known_facts=known_facts,
            )),
        ], config=build_trace_config(run_name="query_reconcile", tags=["clarify"]))
        return json.loads(result.content)


class DietaryIntentExtractor:
    """Extracts dietary requirement tags (vegan, low-carb, …) from a user query."""

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            format=DietaryTagsSchema.model_json_schema(),
        )

    def extract(self, query: str) -> list[str]:
        try:
            result = self.llm.invoke([
                SystemMessage(content=DIETARY_INTENT_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=DIETARY_INTENT_EXTRACTOR_USER.compile(query=query)),
            ], config=build_trace_config(run_name="dietary_intent", tags=["extract"]))
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

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
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
            # Groq rejects any json_object request whose messages don't
            # contain the word "json" — a 400 on every call, which this
            # except swallows into "no shape extracted", silently. The
            # in-code prompt says it, but prompts are served from Langfuse
            # at runtime and existing managed copies are never overwritten
            # by a deploy — so a stale managed prompt disabled multi-plate
            # planning in production while every local test passed. This
            # guard makes the requirement structural instead of editorial.
            if "json" not in f"{system_text} {user_text}".lower():
                system_text += "\nReturn the result as a JSON object."
            result = self.llm.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ], config=build_trace_config(run_name="plan_spec", tags=["extract"]))
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


class PreferenceExtractor:
    """Detects durable preference candidates in a user turn (M3 memory).

    Output feeds ``services.memory_service`` which applies the consent
    policy — this agent only detects; it NEVER writes memory itself.
    """

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            format=PreferenceExtractionSchema.model_json_schema(),
        )

    def extract(self, message: str) -> list[dict]:
        try:
            result = self.llm.invoke([
                SystemMessage(content=PREFERENCE_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=PREFERENCE_EXTRACTOR_USER.compile(message=message)),
            ], config=build_trace_config(run_name="preference_extract", tags=["memory"]))
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

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            format=SeedExtractionSchema.model_json_schema(),
        )

    def extract(self, query: str) -> list[dict]:
        try:
            result = self.llm.invoke([
                SystemMessage(content=SEED_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=SEED_EXTRACTOR_USER.compile(query=query)),
            ], config=build_trace_config(run_name="seed_extract", tags=["planning"]))
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

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
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
            if "json" not in f"{system_text} {user_text}".lower():
                system_text += "\nReturn the result as a JSON object."
            result = self.llm.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ], config=build_trace_config(run_name="pantry_extract", tags=["planning"]))
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

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            format=EditCommandSchema.model_json_schema(),
        )

    def extract(self, message: str, plan_type: str) -> Optional[dict]:
        """Returns the command dict, or None when parsing fails (caller
        degrades to a whole-plan refinement)."""
        try:
            result = self.llm.invoke([
                SystemMessage(content=EDIT_COMMAND_EXTRACTOR_SYSTEM.compile()),
                HumanMessage(content=EDIT_COMMAND_EXTRACTOR_USER.compile(
                    plan_type=plan_type, message=message,
                )),
            ], config=build_trace_config(run_name="edit_command", tags=["edit"]))
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


class OrchestratorAgent:
    """Single intent classifier per turn — the ONLY router in the pipeline.

    Valid intents: daily_plan | weekly_plan | refine_plan | edit_plan_slot
    | switch_plan_type | nutrition_question | plan_question
    | preference_update | chat.
    ``target_plan_type`` is populated only for switch_plan_type;
    nutrition_question turns are delegated to FoodScholar; plan_question
    turns are answered by the PlanAnalyst grounded in the active canvas;
    edit_plan_slot targets ONE slot of the active canvas with a verified
    directive; preference_update is a stated durable preference — it is
    acknowledged, never interrogated (the write stays behind the M3 nudge).
    """

    VALID_INTENTS = {
        "daily_plan", "weekly_plan", "refine_plan", "edit_plan_slot",
        "switch_plan_type", "nutrition_question", "plan_question",
        "preference_update", "chat",
    }

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=OrchestratorSchema.model_json_schema(),
        )

    def classify(self, message: str, history: list[dict]) -> dict:
        """Classify intent given the last turns ({"role", "content"} dicts, recent last)."""
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content'][:300]}"
            for turn in history[-6:]
        ) or "(no prior conversation)"

        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM.compile()),
            HumanMessage(content=ORCHESTRATOR_USER.compile(
                history=history_text, message=message,
            )),
        ]

        config = build_trace_config(run_name="orchestrate", tags=["router"])
        for attempt in range(MAX_RETRIES):
            try:
                result = self.llm.invoke(messages, config=config)
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

        logger.error("Orchestrator failed after retries, defaulting to chat")
        return {"intent": "chat", "target_plan_type": None}
