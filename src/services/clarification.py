"""
Clarification engine — collects missing information before plan generation.

Pre-M0 this was a live Python generator held only in process memory, which
meant a restart (or a second replica) stranded any session whose DB state said
``clarifying``. This module replaces it with an explicit, JSON-serializable
state machine:

    ClarificationManager.start(query, profile)
        → ClarificationOutcome(final_query=...)            # nothing to ask
        → ClarificationOutcome(state=..., question=...)    # ask the user

    ClarificationManager.step(state, user_answer)
        → next question (state advances)  or  final reformulated query

``ClarificationState`` round-trips through ``to_dict``/``from_dict`` and is
persisted on the session row (``sessions.clarification_state``), so any
replica can resume the flow after any restart.

The conversational behaviour is unchanged from the generator version:
 1. Reconcile the query against the profile (dietary conflicts, missing info).
 2. Warn about dietary conflicts first and record the user's reply.
 3. Ask one LLM-generated question per missing-info topic, each conditioned
    on the answers so far.
 4. If the query was merely too vague (no conflict/missing info), run the
    specificity check and, when the profile lacks signal, collect it.
 5. Reformulate everything into a single retrieval-ready query.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.groq import GROQ_CHAT
from agents import QueryReconciler
from prompts import (
    QUERY_CHECKER_SYSTEM_INSTRUCTIONS,
    QUERY_CHECKER_USER_INSTRUCTIONS,
    USER_PROFILE_CHECKER_SYSTEM_INSTRUCTIONS,
    USER_PROFILE_CHECKER_USER_INSTRUCTIONS,
    USER_INFO_COLLECTOR_SYSTEM_INSTRUCTIONS,
    USER_INFO_COLLECTOR_USER_INSTRUCTIONS,
    QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS,
    QUERY_REFORMULATOR_USER_INSTRUCTIONS,
)
from schemas import (
    QueryCheckerSchema,
    UserProfileCheckerSchema,
    QueryReformulatorSchema,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("FOODCHAT_LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("FOODCHAT_LLM_TEMPERATURE", "0.0"))


@dataclass
class ClarificationState:
    """Serializable snapshot of an in-progress clarification dialogue.

    Invariant: while a question is outstanding, ``current_question`` holds it
    and ``pending_topics[0]`` is the topic it belongs to (except in the
    ``conflict`` phase, where the question is the conflict warning itself).
    """

    original_query: str
    profile: dict                       # reconciled profile used for generation
    origin_intent: str                  # "daily_plan" | "refine_plan" — restored on completion
    phase: str                          # "conflict" | "collect"
    pending_topics: list = field(default_factory=list)
    current_question: Optional[str] = None
    transcript: list = field(default_factory=list)   # [{"question": ..., "answer": ...}]
    conflict_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "profile": self.profile,
            "origin_intent": self.origin_intent,
            "phase": self.phase,
            "pending_topics": self.pending_topics,
            "current_question": self.current_question,
            "transcript": self.transcript,
            "conflict_note": self.conflict_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClarificationState":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "ClarificationState":
        return cls.from_dict(json.loads(raw))


@dataclass
class ClarificationOutcome:
    """Result of start()/step(): exactly one of ``question`` or ``final_query`` is set.

    ``profile`` is always populated with the reconciled profile (stored profile
    merged with query-level reconciliation) so the caller can generate the plan
    without re-running reconciliation.

    ``collected_facts`` holds the Q/A pairs gathered on the way to a final
    query — the caller appends them to the session profile's history so the
    same things are never asked again within the session.
    """

    profile: dict
    state: Optional[ClarificationState] = None
    question: Optional[str] = None
    final_query: Optional[str] = None
    collected_facts: list = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.question is not None


class ClarificationManager:
    """Stateless orchestrator of the clarification state machine.

    All conversational state lives in ``ClarificationState`` (persisted by the
    caller); this class only holds LLM clients and is safe to share.
    """

    def __init__(self, model: str = None, temperature: float = None):
        model = model or DEFAULT_MODEL
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.reconciler = QueryReconciler(model=model, temperature=temperature)
        self.query_checker = GROQ_CHAT.get_client(
            model=model, temperature=temperature,
            format=QueryCheckerSchema.model_json_schema(),
        )
        self.profile_checker = GROQ_CHAT.get_client(
            model=model, temperature=temperature,
            format=UserProfileCheckerSchema.model_json_schema(),
        )
        self.question_generator = GROQ_CHAT.get_client(model=model, temperature=temperature)
        self.reformulator = GROQ_CHAT.get_client(
            model=model, temperature=temperature,
            format=QueryReformulatorSchema.model_json_schema(),
        )

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #

    def start(self, query: str, profile: dict, origin_intent: str) -> ClarificationOutcome:
        """Evaluate a fresh plan request; ask the first question or finalize."""
        reconciliation = self.reconciler.reconcile(query, profile)
        logger.info("Reconciliation result: %s", reconciliation)
        # Session-scoped merge only — never written back to the durable
        # profile (the M3 consent flow owns durable writes).
        merged = {**profile, **reconciliation}

        if reconciliation.get("needs_clarification"):
            # Cap at ONE question per request — interrogation kills flow, and
            # the slider card owns every parameter-style topic now; the only
            # askable thing left is food direction.
            missing = list(reconciliation.get("missing_info") or [])[:1]

            if reconciliation.get("has_dietary_conflict"):
                warning = reconciliation.get("conflict_explanation") or (
                    "Your request seems to conflict with your dietary profile — "
                    "should I follow the request anyway?"
                )
                state = ClarificationState(
                    original_query=query, profile=merged, origin_intent=origin_intent,
                    phase="conflict", pending_topics=missing, current_question=warning,
                )
                return ClarificationOutcome(profile=merged, state=state, question=warning)

            if missing:
                state = ClarificationState(
                    original_query=query, profile=merged, origin_intent=origin_intent,
                    phase="collect", pending_topics=missing,
                )
                question = self._next_question(state)
                return ClarificationOutcome(profile=merged, state=state, question=question)
            # needs_clarification without conflict or topics → fall through to
            # the specificity check, mirroring the pre-M0 generator behaviour.

        return self._specificity_path(query, merged, origin_intent)

    def step(self, state: ClarificationState, answer: str) -> ClarificationOutcome:
        """Consume the user's answer; ask the next question or finalize."""
        if state.phase == "conflict":
            state.conflict_note = f"User response to dietary conflict warning: {answer}"
            state.current_question = None
            if state.pending_topics:
                state.phase = "collect"
                question = self._next_question(state)
                return ClarificationOutcome(profile=state.profile, state=state, question=question)
            return ClarificationOutcome(
                profile=state.profile, final_query=self._reformulate(state),
                collected_facts=self._collected(state),
            )

        # phase == "collect": record the answer for the topic just asked
        state.transcript.append({"question": state.current_question, "answer": answer})
        if state.pending_topics:
            state.pending_topics.pop(0)
        state.current_question = None

        if state.pending_topics:
            question = self._next_question(state)
            return ClarificationOutcome(profile=state.profile, state=state, question=question)
        return ClarificationOutcome(
            profile=state.profile, final_query=self._reformulate(state),
            collected_facts=self._collected(state),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _specificity_path(self, query: str, profile: dict, origin_intent: str) -> ClarificationOutcome:
        """Vague-query fallback: check specificity, then profile signal."""
        response = self.query_checker.invoke([
            SystemMessage(content=QUERY_CHECKER_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=QUERY_CHECKER_USER_INSTRUCTIONS.format(user_query=query)),
        ])
        check = json.loads(response.content).get("response", "YES")
        if check != "NO":
            return ClarificationOutcome(profile=profile, final_query=query)

        logger.info("Query too general — checking profile signal before asking")
        preferences = profile.get("preferences", [])
        suggestions: list[str] = []
        if preferences:
            pc = self.profile_checker.invoke([
                SystemMessage(content=USER_PROFILE_CHECKER_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=USER_PROFILE_CHECKER_USER_INSTRUCTIONS.format(
                    preferences=preferences, feedback_history=[],
                )),
            ])
            pc_json = json.loads(pc.content)
            if pc_json.get("response") != "NO":
                # Profile carries enough signal — reformulate without asking.
                state = ClarificationState(
                    original_query=query, profile=profile,
                    origin_intent=origin_intent, phase="collect",
                )
                return ClarificationOutcome(profile=profile, final_query=self._reformulate(state))
            suggestions = list(pc_json.get("suggestions", []))[:1]
        else:
            suggestions = ["food preferences or cravings for this plan"]

        state = ClarificationState(
            original_query=query, profile=profile, origin_intent=origin_intent,
            phase="collect", pending_topics=suggestions,
        )
        question = self._next_question(state)
        return ClarificationOutcome(profile=profile, state=state, question=question)

    def _next_question(self, state: ClarificationState) -> str:
        """Generate the question for pending_topics[0], conditioned on the transcript."""
        topic = state.pending_topics[0]
        history: list = []
        for qa in state.transcript:
            history.append(AIMessage(content=qa["question"] or ""))
            history.append(HumanMessage(content=qa["answer"] or ""))

        response = self.question_generator.invoke(
            [
                SystemMessage(content=USER_INFO_COLLECTOR_SYSTEM_INSTRUCTIONS),
                HumanMessage(content=USER_INFO_COLLECTOR_USER_INSTRUCTIONS.format(
                    user_query=state.original_query,
                    preferences=state.profile.get("preferences", []),
                    feedback_history=[],
                    suggestions=topic,
                )),
            ]
            + history
        )
        state.current_question = response.content
        return response.content

    @staticmethod
    def _collected(state: ClarificationState) -> list[str]:
        """The gathered facts, as short lines usable for profile history."""
        collected: list[str] = []
        if state.conflict_note:
            collected.append(state.conflict_note)
        for qa in state.transcript:
            collected.append(f"Q: {qa['question']}\nA: {qa['answer']}")
        return collected

    def _reformulate(self, state: ClarificationState) -> str:
        """Fold the collected answers into one retrieval-ready query."""
        collected = self._collected(state)

        response = self.reformulator.invoke([
            SystemMessage(content=QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=QUERY_REFORMULATOR_USER_INSTRUCTIONS.format(
                original_query=state.original_query,
                preferences=state.profile.get("preferences", []),
                feedback_history=[],
                collected_info=collected or None,
            )),
        ])
        final_query = json.loads(response.content)["reformulated_query"]
        logger.info("Reformulated query: %s", final_query)
        return final_query
