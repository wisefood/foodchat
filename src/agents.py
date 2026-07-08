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
from models.recipe import CandidatesBySlot, ScoredPlan
from prompts import (
    GRADER_SYSTEM_INSTRUCTIONS,
    PLAN_ANALYST_SYSTEM_INSTRUCTIONS,
    GRADER_USER_INSTRUCTIONS,
    MEAL_DIVERSITY_SYSTEM_INSTRUCTIONS,
    GUIDELINE_ADHERENCE_SYSTEM_INSTRUCTIONS,
    QUERY_RECONCILER_SYSTEM_INSTRUCTIONS,
    QUERY_RECONCILER_USER_INSTRUCTIONS,
    ORCHESTRATOR_SYSTEM_INSTRUCTIONS,
    ORCHESTRATOR_USER_INSTRUCTIONS,
    DIETARY_INTENT_EXTRACTOR_SYSTEM_INSTRUCTIONS,
    DIETARY_INTENT_EXTRACTOR_USER_INSTRUCTIONS,
    SEED_EXTRACTOR_SYSTEM_INSTRUCTIONS,
    SEED_EXTRACTOR_USER_INSTRUCTIONS,
    PREFERENCE_EXTRACTOR_SYSTEM_INSTRUCTIONS,
    PREFERENCE_EXTRACTOR_USER_INSTRUCTIONS,
    EDIT_COMMAND_EXTRACTOR_SYSTEM_INSTRUCTIONS,
    EDIT_COMMAND_EXTRACTOR_USER_INSTRUCTIONS,
    RESPONSE_WRITER_SYSTEM_INSTRUCTIONS,
    RESPONSE_WRITER_USER_INSTRUCTIONS,
)
from schemas import (
    ScoringSchema,
    QueryReconcilerSchema,
    OrchestratorSchema,
    DietaryTagsSchema,
    SeedExtractionSchema,
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
            format=ScoringSchema.model_json_schema(),
        )
        self.max_plans_to_score = max_plans_to_score or MAX_PLANS_TO_SCORE

    def grade_daily_plans(
        self, query: str, candidates: CandidatesBySlot, user_profile: dict,
        feedback_history: str = "",
    ) -> list[ScoredPlan]:
        """Return the top-scored combinations, best first (at most 3).

        The combination space (limit³) is sampled down to
        ``max_plans_to_score`` before grading — each grade is one LLM call, so
        this bounds latency/cost per plan request.
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

        sampled = random.sample(combos, min(len(combos), self.max_plans_to_score))
        scored: list[ScoredPlan] = []
        for breakfast, lunch, dinner in sampled:
            plan_text = "".join(
                f"{slot}: {course.title}\nIngredients: {course.ingredients}\n"
                f"Directions: {course.directions}\n\n"
                for slot, course in (("breakfast", breakfast), ("lunch", lunch), ("dinner", dinner))
            )
            result = self.grader.invoke([
                SystemMessage(content=GRADER_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=GRADER_USER_INSTRUCTIONS.format(
                    query=query,
                    daily_plan=plan_text,
                    preferences=",".join(user_profile.get("preferences", [])),
                    feedback_history=feedback_history or "No prior feedback.",
                )),
            ])
            grade = json.loads(result.content)
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
            SystemMessage(content=MEAL_DIVERSITY_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=plan_text),
        ])
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
            SystemMessage(content=GUIDELINE_ADHERENCE_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=f"GUIDELINES:\n{guidelines_text}\n\nMEAL PLAN:\n{plan_text}"),
        ])
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
    # "can't answer" something — it redirects warmly instead.
    SYSTEM_PROMPT = (
        "You are FoodChat, the friendly meal-planning assistant of the WiseFood platform. "
        "You help people plan what to eat: daily meal plans, weekly meal plans, and "
        "refinements to plans you've already made ('swap the dinner', 'make it lighter'). "
        "Nutrition-science questions are answered for you by FoodScholar, WiseFood's "
        "evidence-based Q&A service, so never tell the user a question can't be answered here. "
        "For this conversation: respond warmly and briefly, stay food-related where natural, "
        "and when the user seems unsure what to do next, suggest something concrete like "
        "'want a plan for today?' or 'shall we plan your week around something you love?'. "
        "Never invent recipes or plans in this mode — offer to create one instead."
    )

    def __init__(self, model: str = None, temperature: float = None):
        self.chatbot = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else CHATBOT_TEMPERATURE,
        )

    def chat(self, query: str, history: list[tuple[str, str]] = None) -> str:
        """Respond to ``query`` given recent (role, content) history pairs."""
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]
        for role, content in (history or [])[-CHATBOT_HISTORY_TURNS:]:
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=query))
        return self.chatbot.invoke(messages).content


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
            SystemMessage(content=QUERY_RECONCILER_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=QUERY_RECONCILER_USER_INSTRUCTIONS.format(
                query=query,
                diet=user_profile.get("diet", []),
                allergies=user_profile.get("allergies", []),
                known_facts=known_facts,
            )),
        ])
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
                SystemMessage(content=DIETARY_INTENT_EXTRACTOR_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=DIETARY_INTENT_EXTRACTOR_USER_INSTRUCTIONS.format(query=query)),
            ])
            return json.loads(result.content).get("dietary_tags", [])
        except Exception as e:
            logger.warning("DietaryIntentExtractor failed: %s", e)
            return []


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
                SystemMessage(content=PREFERENCE_EXTRACTOR_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=PREFERENCE_EXTRACTOR_USER_INSTRUCTIONS.format(message=message)),
            ])
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
                SystemMessage(content=SEED_EXTRACTOR_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=SEED_EXTRACTOR_USER_INSTRUCTIONS.format(query=query)),
            ])
            seeds = json.loads(result.content).get("seeds", [])
            return [s for s in seeds if isinstance(s, dict) and s.get("name")]
        except Exception as e:
            logger.warning("SeedExtractor failed: %s", e)
            return []


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
                SystemMessage(content=EDIT_COMMAND_EXTRACTOR_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=EDIT_COMMAND_EXTRACTOR_USER_INSTRUCTIONS.format(
                    plan_type=plan_type, message=message,
                )),
            ])
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
                SystemMessage(content=RESPONSE_WRITER_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=RESPONSE_WRITER_USER_INSTRUCTIONS.format(
                    facts=json.dumps(facts, ensure_ascii=False),
                    user_message=user_message[:400],
                )),
            ])
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
        messages = [SystemMessage(content=PLAN_ANALYST_SYSTEM_INSTRUCTIONS)]
        for role, content in (history or [])[-CHATBOT_HISTORY_TURNS:]:
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(
            content=f"CURRENT PLAN:\n{plan_summary}\n\nQUESTION: {question}"
        ))
        return self.llm.invoke(messages).content


class OrchestratorAgent:
    """Single intent classifier per turn — the ONLY router in the pipeline.

    Valid intents: daily_plan | weekly_plan | refine_plan | edit_plan_slot
    | switch_plan_type | nutrition_question | plan_question | chat.
    ``target_plan_type`` is populated only for switch_plan_type;
    nutrition_question turns are delegated to FoodScholar; plan_question
    turns are answered by the PlanAnalyst grounded in the active canvas;
    edit_plan_slot targets ONE slot of the active canvas with a verified
    directive.
    """

    VALID_INTENTS = {
        "daily_plan", "weekly_plan", "refine_plan", "edit_plan_slot",
        "switch_plan_type", "nutrition_question", "chat",
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
            SystemMessage(content=ORCHESTRATOR_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=ORCHESTRATOR_USER_INSTRUCTIONS.format(
                history=history_text, message=message,
            )),
        ]

        for attempt in range(MAX_RETRIES):
            try:
                result = self.llm.invoke(messages)
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
