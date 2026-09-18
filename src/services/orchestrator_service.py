"""
OrchestratorService — unified conversational entry point.

The ONLY intent classification in the pipeline happens here (one
``OrchestratorAgent.classify`` call per turn), then the turn is routed:

  daily_plan         → ChatService.process_plan_request (fresh canvas)
  refine_plan        → ChatService / WeeklyPlanService with is_refinement=True,
                       targeting whichever canvas was most recently updated
  weekly_plan        → WeeklyPlanService (fresh canvas)
  switch_plan_type   → acknowledge, then fresh canvas of the target type
  edit_plan_slot     → EditService (one verified slot swap on the canvas)
  nutrition_question → FoodScholarService (evidence-based answer + attribution)
  plan_question      → PlanAnalyst (question ABOUT the active canvas — answered
                       from its nutrition data, never modifies the plan;
                       no canvas → falls through to FoodScholar)
  preference_update  → acknowledge a stated durable preference ("remember I
                       don't like chicken") — the durable write stays
                       consent-gated behind the M3 memory nudge
  score_plan         → PlanScorerService (a plan the member WROTE and pasted:
                       parsed, grounded against RecipeWrangler, built into the
                       planners' objects; never touches a canvas)
  chat               → ChatService.process_smalltalk

An explicit score request — a scoring word ("rate", "score", "how does it
look") together with a meal listing in at least two slots (prose counts), and
no request to make or change a plan — bypasses the classifier, like an explicit
FoodScholar consult, and supersedes any pending clarification: the member has
moved on to a different question. ``score_plan()`` is the same turn for the
``/score-plan`` endpoint (the text box), with no classification at all. When
the classifier itself fails — its default is "chat" — a message that lists
meals is scored rather than answered as small talk.

While a session is mid-clarification (session.state == "clarifying"), the
classifier is normally skipped — the user is answering our question. The
persisted clarification dict routes the turn: ``kind == "foodscholar"`` goes
back to FoodScholarService, ``kind == "score_plan"`` to PlanScorerService,
anything else to ChatService (plan flow, whose
state carries the original intent). Both are restart-safe (data, not objects).
Edit-slot and score-plan clarifications are the exception: when the reply
doesn't answer the question it usually isn't an answer at all, so the turn
falls back to normal classification instead of re-interrogating (and a
score_plan classification on that fall-through never asks again).

Memory nudges (M3) run on EVERY turn — including clarification turns — so a
preference stated while answering a question is never silently dropped.

Returns a unified ChatTurn so the router needs one response model.
"""

import contextvars
import json
import logging
import re
import threading
import time
import uuid as _uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from agents import DishIngredientEstimator, OrchestratorAgent, PlanAnalyst, ToolSelector
from backend.observability import trace_context
from models.attribution import Attribution
from models.session import MealPlan, WeeklyMealPlan
from . import plan_parameters, turn_budget, turn_intake
from .edit_service import EditService
from .foodscholar_service import FoodScholarService
from .plan_scorer import CLARIFICATION_KIND as SCORE_CLARIFICATION_KIND
from .plan_scorer import PlanScorerService
from .plan_scorer.grounding import DishGrounder
from .plan_scorer.parsing import looks_like_plan_listing
from .seed_service import SeedService
from .session_service import SessionService

logger = logging.getLogger(__name__)

#: What a turn handler learned, for the guard to attach when it reports. A
#: ContextVar rather than an attribute: two turns on two sessions run
#: concurrently in the same process, and an attribute would let one describe
#: the other.
_turn_detail: contextvars.ContextVar = contextvars.ContextVar(
    "foodchat_turn_detail", default=None
)

# Words that count as accepting the favorites offer. Kept deliberately simple
# for M2 — anything else is treated as a decline and the original request
# proceeds unchanged, so a misread costs nothing but the boost.
_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|sure|ok(ay)?|please|sounds good|do it|why not|go for it)\b",
    re.IGNORECASE,
)
# Favorites shown by title in the offer message (each needs a detail fetch).
MAX_OFFERED_FAVORITES = 3


# A turn is abandoned this long after it started. The turn budget is 70s, so
# this is that plus room for storage and serialisation: long enough that a slow
# turn is never treated as dead, short enough that a crash frees the session
# before the member gives up on it.
_TURN_GUARD_TTL = 120.0


class SessionAccessError(ValueError):
    """Session missing or owned by another member.

    Subclasses ValueError so existing ``except ValueError`` handlers keep
    working, but lets routers map ONLY lookup failures to 404 — a ValueError
    raised deep in the planning stack is a 500, not "session not found".
    """


@dataclass
class ChatTurn:
    role: str
    content: str
    intent: str
    needs_clarification: bool = False
    meal_plan: Optional[MealPlan] = None
    weekly_meal_plan: Optional[WeeklyMealPlan] = None
    at_message_limit: bool = False
    # Version metadata surfaced to the caller (canvas tracking in the UI)
    plan_version: Optional[int] = None
    plan_parent_id: Optional[str] = None
    # Provenance when the answer came from another WiseFood app (FoodScholar)
    attribution: Optional[Attribution] = None
    # Consent nudges ("remember this?") detected in the user's turn (M3)
    memory_suggestions: Optional[list] = None
    # Slot-edit proof (M4b): [{meal_type, day, old{title,kcal}, new{...}, directive, verified}]
    changed_slots: Optional[list] = None
    # Optional slider card (time/difficulty/goal) attached to fresh daily
    # plans; answered via POST /sessions/{id}/plan-parameters
    plan_parameters: Optional[dict] = None
    # What the plan scorer read from a pasted plan (score_plan turns only):
    # {plan_type, days_scored, meals_scored, metrics, constraints_applied,
    #  grounding[], unparsed[], warnings[], scored_plan, context}
    plan_score: Optional[dict] = None


class OrchestratorService:

    # Sessions with a turn currently running, and when it started. Guarded by
    # its own lock so checking and claiming is one step — two threads checking
    # an unguarded dict would both find it free.
    #
    # Class-level, not per instance, for two reasons. The damage it prevents is
    # per PROCESS — the in-memory session cache and the canvas write both live
    # there — so two service instances in one process must still not run two
    # turns on one session. And an instance built without `__init__` (which the
    # test suite does routinely) would otherwise have no guard at all, which is
    # a failure mode that only shows up under load.
    _turns_in_flight: dict[str, float] = {}
    _turn_lock = threading.Lock()

    def __init__(
        self,
        session_service: SessionService,
        chat_service: Any,
        weekly_plan_service: Any,
        foodscholar_service: Optional[FoodScholarService] = None,
        memory_service: Any = None,
        plan_scorer: Optional[PlanScorerService] = None,
    ):
        self.session_service = session_service
        self.chat_service = chat_service
        self.weekly_plan_service = weekly_plan_service
        self.foodscholar_service = foodscholar_service or FoodScholarService(session_service)
        self.seed_service = SeedService()
        self.plan_scorer = plan_scorer or PlanScorerService(
            session_service,
            grounder=DishGrounder(self.seed_service, estimator=DishIngredientEstimator()),
        )
        self.edit_service = EditService(session_service)
        self.memory_service = memory_service
        self.orchestrator = OrchestratorAgent()
        self.plan_analyst = PlanAnalyst()

    # ------------------------------------------------------------------ #
    # Entry-point guards (shared by every public entry point)              #
    # ------------------------------------------------------------------ #

    def _owned_session(self, session_id: str, member_id: str):
        """Load a session, proving ownership. Raises SessionAccessError."""
        session = self.session_service.get_session(session_id, member_id=member_id)
        if not session:
            raise SessionAccessError(f"Session {session_id} not found or access denied")
        return session

    @staticmethod
    def _limit_turn(session) -> Optional[ChatTurn]:
        """The refusal turn when the session hit its message cap, else None."""
        if not session.is_at_message_limit:
            return None
        return ChatTurn(
            role="assistant",
            content=(
                f"This conversation has reached the {session.max_messages}-message limit. "
                "Please start a new session to continue."
            ),
            intent="chat",
            at_message_limit=True,
        )

    @contextmanager
    def _one_turn_at_a_time(self, session_id: str):
        """Refuse a second turn on a session while the first is still running.

        Two turns on one session both load it, both plan, and both write the
        canvas pointer. Last write wins and the loser's plan is orphaned —
        stored, paid for, and unreachable. A single tab cannot do this (the
        composer disables while sending), but a second tab, or a slider apply
        landing on top of a chat turn, can.

        Refused rather than queued. Queueing would hold a worker for the length
        of the first turn and then run a plan the member asked for a minute
        ago; saying "still working" is both cheaper and truer.

        The guard is per process, which matches where the damage is: the
        in-memory session cache and the canvas write both live here. Across
        replicas the same member would need two tabs on two pods, and the
        database write is still atomic.
        """
        now = time.monotonic()
        with self._turn_lock:
            started = self._turns_in_flight.get(session_id)
            # A stale entry means a turn died without unwinding. Bounded by the
            # budget plus slack, so a crash cannot lock a session out for good.
            busy = started is not None and (now - started) < _TURN_GUARD_TTL
            if not busy:
                self._turns_in_flight[session_id] = now
        try:
            # A boolean rather than an exception: `@contextmanager` runs this
            # body at `__enter__`, so a `try/except` around the CALL would never
            # fire and the exception would escape to the router as a 500. The
            # check and the claim still happen together under the lock, so this
            # is not check-then-act.
            yield not busy
            outcome = "ok" if not busy else "busy"
        except BaseException:
            outcome = "error"
            raise
        finally:
            if not busy:
                with self._turn_lock:
                    self._turns_in_flight.pop(session_id, None)
            # Reported here, not at the four entry points, because this is the
            # one thing all four share — and because it is the only place that
            # also sees a refused turn, a turn that hit the message cap, and a
            # turn that raised. An earlier version reported from `process()`
            # alone, so three of the four entry points and every failed turn
            # were missing from the record.
            self._report_turn(session_id, outcome, now)

    @staticmethod
    def _busy_turn() -> ChatTurn:
        """What the member sees when they send twice."""
        return ChatTurn(
            role="assistant",
            content=(
                "I'm still working on your last message — give me a moment and "
                "then try again."
            ),
            intent="chat",
        )

    def _attach_memory_suggestions(self, session, turn: ChatTurn, message: str) -> ChatTurn:
        """Consent nudges (M3): detect durable preferences in the user's turn
        and ATTACH suggestions — durable writes happen only when the user
        answers via POST /sessions/{id}/memory.

        Runs on EVERY turn carrying the member's own words, including
        clarification answers and manual-compose messages: a preference
        stated while doing something else still counts. Best-effort by
        design — a nudge failure must never break an answered turn.
        """
        if self.memory_service is None or turn.at_message_limit or not (message or "").strip():
            return turn
        try:
            suggestions = self.memory_service.suggest(session, message)
            diet_nudge = self._diet_memory_nudge(session, message)
            if diet_nudge:
                # Ahead of the extractor's own candidates: a stated diet is the
                # highest-value thing to remember, and the per-turn cap would
                # otherwise let two likes crowd it out.
                suggestions = [diet_nudge] + [
                    s for s in suggestions if s.get("kind") != "diet"
                ]
            if suggestions:
                turn.memory_suggestions = suggestions
        except Exception as e:
            logger.warning("[%s] Memory suggestion failed: %s", session.session_id, e)
        return turn

    def _diet_memory_nudge(self, session, message: str):
        """A consent nudge for a diet stated in chat but not on the profile.

        Built deterministically from the standing planning state rather than by
        the preference extractor. Two reasons, and the second is the binding
        one: it costs no extra LLM call, and it needs no edit to the
        `preference_extractor` managed prompt — a deploy never overwrites an
        existing Langfuse copy, so adding a kind there would work locally and
        ship dead to production.

        The evidence is the member's own sentence, so the memory panel can
        answer "why am I seeing this?" with something true.
        """
        try:
            from services import diet_intent

            state = self.session_service.get_planning_state(session.session_id)
            if not state.diet_tags:
                return None
            nudge = diet_intent.suggest_diet_memory(
                state.diet_tags, message, session.user_profile
            )
            if not nudge:
                return None
            # Respect the same never-ask-twice ledger as every other nudge.
            optouts = {
                str(v).strip().lower()
                for v in (session.user_profile.get("memory_optouts") or [])
            }
            if nudge["value"] in optouts:
                return None
            nudge["id"] = str(_uuid.uuid4())
            return nudge
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Diet nudge failed: %s", session.session_id, exc
            )
            return None

    def process(self, session_id: str, member_id: str, message: str) -> ChatTurn:
        """Validate ownership, check the message cap, classify, and route.

        The whole turn runs inside ``trace_context`` so every downstream LLM
        call (classify, clarify, plan grading, response writing, …) groups
        under one Langfuse Session (session_id) and User (member_id).
        """
        # One budget and one in-flight claim per turn, at the only four
        # places a turn begins. Nested calls keep the outermost deadline, so a
        # slider apply that routes into the chat handlers does not hand the
        # inner stage a fresh full allowance.
        with self._one_turn_at_a_time(session_id) as claimed, \
                trace_context(session_id=session_id, user_id=member_id), \
                turn_budget.start():
            if not claimed:
                return self._busy_turn()
            # A new turn hears the message fresh. The intake memo exists to stop
            # one turn extracting twice, not to carry an answer into the next.
            turn_intake.forget()
            session = self._owned_session(session_id, member_id)
            limit_turn = self._limit_turn(session)
            if limit_turn is not None:
                return limit_turn

            # Read BEFORE routing: the handlers append this message, after
            # which "no user messages yet" is no longer true and the
            # opening-turn signal is gone.
            opening_turn = session.title is None and not any(
                m.role == "user" for m in session.conversation
            )

            # An explicit "rate this: <listing>" is unambiguous — no classifier
            # call, and it supersedes a pending question the member has moved on
            # from (the same courtesy compose extends).
            if self.is_explicit_score_request(message):
                if session.state == "clarifying":
                    logger.info("[%s] Explicit plan score supersedes a pending clarification.", session_id)
                    self.session_service.clear_clarification_state(session_id)
                turn = self._handle_score_plan(session_id, message)
            # Mid-clarification turns usually bypass classification — the user
            # is answering our question. Handlers may still bounce the turn
            # back to normal routing when the reply clearly isn't an answer.
            elif session.state == "clarifying":
                turn = self._handle_clarification_turn(session_id, message)
            else:
                turn = self._classify_and_route(session, session_id, message)

            result = self._attach_memory_suggestions(session, turn, message)
            if opening_turn:
                self._autotitle_session(session_id, member_id, message)
            # What only this path knows. The turn itself is reported by the
            # guard above, which also catches the paths that never get here.
            _turn_detail.set({
                "intent": getattr(result, "intent", None),
                "plan_id": getattr(result, "plan_id", None),
                "opening_turn": bool(opening_turn),
                "has_attribution": bool(getattr(result, "attribution", None)),
            })
            return result

    @staticmethod
    def _report_turn(session_id: str, outcome: str, started: float) -> None:
        """Report one turn. Never raises: analytics must not cost an answer."""
        try:
            import activity

            detail = _turn_detail.get() or {}
            activity.report_turn(
                session_id=session_id,
                intent=detail.get("intent"),
                plan_id=detail.get("plan_id"),
                latency_ms=(time.monotonic() - started) * 1000.0,
                extra={
                    "outcome": outcome,
                    "opening_turn": bool(detail.get("opening_turn")),
                    "has_attribution": bool(detail.get("has_attribution")),
                },
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Turn reporting failed", exc_info=True)
        finally:
            _turn_detail.set(None)

    def _autotitle_session(self, session_id: str, member_id: str, message: str) -> None:
        """Name a session from its opening message. Best-effort, never fatal.

        Sessions were only ever named by an explicit rename, which almost nobody
        does — so the picker showed a wall of timestamps, and a saved plan
        inherited no name at all, because the save path borrows the session
        title. Same idea as foodscholar's SESSION_TITLE_MODEL, on the fast tier.

        Runs AFTER the turn so a title can never delay or break the answer, and
        only when the session has no title — a member rename always wins, and
        this never fires again once one exists.
        """
        try:
            from agents import SessionTitler

            title = SessionTitler().title(message)
            if not title:
                return
            self.session_service.rename_session(session_id, member_id, title)
            logger.info("[%s] Auto-titled session: %r", session_id, title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Auto-title failed: %s", session_id, exc)

    # Explicit FoodScholar consults bypass classification entirely: "can you
    # check with food scholar?" is a request to ask the expert, and the
    # classifier reliably filed it as plan_question — after which the
    # PlanAnalyst ROLE-PLAYED the consult ("I've checked with the Food
    # Scholar...") without the bridge ever running.
    _SCHOLAR_CONSULT_RE = re.compile(r"\bfood\s*scholar\b", re.IGNORECASE)

    # A scoring word. Only half of the explicit-score test: the message must
    # ALSO carry a meal listing, or "rate my week" about the canvas would skip
    # the classifier it needs.
    _SCORE_REQUEST_RE = re.compile(
        r"\b(?:score|rate|grade|evaluate|assess|judge)\b"
        r"|\bhow\s+(?:does|do|would)\s+(?:this|it|these|that|my)\b[^.?!]*\blook"
        r"|\bwhat\s+do\s+you\s+think\s+of\b",
        re.IGNORECASE,
    )

    # Words that ask FoodChat to make or change something. A message carrying
    # one is not a plan the member is presenting, even when it lists meals
    # ("make my week like this: breakfast: oats…") — the classifier decides it.
    _PLAN_REQUEST_RE = re.compile(
        r"\b(?:make|create|generate|build|swap|change|replace|suggest|recommend"
        r"|add|adding|added|include|including|put|remove|instead)\b"
        r"|\bgive\s+me\b|\bplan\s+(?:my|me|a|the|for|out)\b",
        re.IGNORECASE,
    )

    @classmethod
    def looks_like_a_pasted_plan(cls, message: str) -> bool:
        """Meals in at least two slots, and nothing asking FoodChat to plan.

        Prose counts: "fried eggs for breakfast, pasta with zucchini for lunch
        and chicken noodle soup for dinner" is how a plan arrives in a chat
        box. The request-verb guard is what keeps "adding salmon for dinner"
        out of it.

        And the dishes have to BE dishes. "Greek for lunch, lighter for
        breakfast" parses identically — two slots, two titles — and is a member
        asking for two changes. It came back scored: 2/5 for guidelines, with a
        note that the day lacks a dinner. They asked for a swap and got a report
        card.

        The discriminator is not prose versus structure, because a plan written
        in prose is still a plan. It is that "greek" and "lighter" are single
        words describing HOW, while "fried eggs" and "chicken noodle soup" name
        WHAT. A prose listing whose every dish is one word is a request; the
        written-out form is exempt, because "breakfast: eggs" is a real line in
        a real plan.
        """
        if not cls._lists_meals(message):
            return False
        return not cls._is_prose_of_single_words(message or "")

    @classmethod
    def _lists_meals(cls, message: str) -> bool:
        """Two slots with dishes, and no verb asking FoodChat to do something."""
        text = message or ""
        if cls._PLAN_REQUEST_RE.search(text):
            return False
        return looks_like_plan_listing(text, structured_only=False)

    @staticmethod
    def _is_prose_of_single_words(text: str) -> bool:
        """A prose-only listing in which no dish title is more than one word."""
        from services.plan_scorer.parsing import scan

        try:
            result = scan(text)
        except Exception:  # noqa: BLE001 — the parser is the scorer's, not ours
            return False
        if not result.used_prose or result.structured_meals:
            return False
        titles = [str(getattr(meal, "title", "") or "") for meal in result.plan.meals]
        return bool(titles) and all(len(title.split()) < 2 for title in titles)

    @classmethod
    def is_explicit_score_request(cls, message: str) -> bool:
        """A scoring word plus a meal listing the member is presenting.

        Skips the classifier, so a miss only costs a classifier call while a
        false hit would score a message that asked for a plan. An explicit
        FoodScholar consult keeps its own bypass: a member who names the
        scholar asked the scholar.
        """
        text = message or ""
        if cls._SCHOLAR_CONSULT_RE.search(text) or not cls._SCORE_REQUEST_RE.search(text):
            return False
        # The listing check WITHOUT the single-word guard. That guard exists to
        # separate a plan from a request when nothing else can, and "rate this"
        # already did: a member who says it has told us which one this is, even
        # if their dishes are one word each ("rate this: oats for breakfast,
        # soup for lunch").
        return cls._lists_meals(text)

    def _compose_scholar_question(self, session, message: str) -> str:
        """The question FoodScholar should answer for an explicit consult.

        A bare consult ("check with food scholar?") carries no question of its
        own — reuse the member's previous question. Either way, attach the
        active plan's meals as context so the scholar answers about THESE
        dishes, not in the abstract.
        """
        stripped = self._SCHOLAR_CONSULT_RE.sub("", message)
        stripped = re.sub(
            r"\b(can|could|would|will)\s+you\s+(check|ask|consult|verify)\s*(with|the)?\b",
            "", stripped, flags=re.IGNORECASE,
        ).strip(" ?,.!-")
        if len(stripped.split()) >= 4:
            return message
        prior = next(
            (m.content for m in reversed(session.conversation) if m.role == "user"),
            None,
        )
        return prior or message

    def _with_plan_context(self, session, question: str) -> str:
        """Attach the active plan's meals so the scholar answers about THESE
        dishes rather than in the abstract. No plan → question unchanged."""
        summary = self._summarize_active_plan(session)
        if summary:
            return f"{question}\n\nContext — the meals under discussion:\n{summary}"
        return question

    def _classify_and_route(self, session, session_id: str, message: str,
                            score_may_ask: bool = True) -> ChatTurn:
        """One classifier call, then dispatch — the only intent decision per turn."""
        # Hear the member BEFORE deciding what kind of turn this is.
        #
        # The four extractors used to live inside the handlers, so what got
        # heard depended on where the turn was routed: the daily path ran all
        # four, weekly ran two, and an edit, a question, a tool call or plain
        # conversation ran none. "Swap Tuesday's dinner — by the way I'm coeliac
        # now" recorded the swap and lost the coeliac. Standing here, in front
        # of the routing, the statement lands whatever the turn turns out to be.
        #
        # Memoised per turn, so the handlers below still call intake where they
        # always extracted and neither pays twice.
        turn_intake.intake(session_id, message, session_service=self.session_service)

        # Going back to a version, or asking for an order we cannot give.
        #
        # Deterministic and ahead of the classifier, like the FoodScholar
        # bypass below: both of these classify as slot edits, and an edit can
        # only replace the dish on a slot. "Go back to the first version"
        # produced a new plan — the one thing the member was asking not to
        # happen — and "better before lunch" swapped an 82 kcal breakfast for a
        # 1,907 kcal recipe called "Sunday Lunch".
        navigated = self._maybe_navigate(session, session_id, message)
        if navigated is not None:
            return navigated

        if self._SCHOLAR_CONSULT_RE.search(message):
            logger.info("Explicit FoodScholar consult — routing to the M1 bridge")
            return self._handle_nutrition_question(
                session_id, message, session=session,
                question=self._compose_scholar_question(session, message),
            )

        # The agent's own capabilities, offered to the model on the same
        # pre-classification seam. `manifest()` and `describe_tools()` have been
        # generated from the registry since the tools were written and neither
        # reached a prompt, so "summarise my week" and "redo Thursday" had no
        # path at all — the closest available action was a full refinement,
        # which regenerates every slot and throws away a swap the member had
        # already approved.
        # "Score my plan" — about the plan on the canvas, not a pasted one.
        #
        # Ahead of the tool selector, because that is what answered it: the
        # message names no dishes, so `is_explicit_score_request` said no, and
        # the selector reached for a weekly reader, which replied "there's no
        # weekly plan in this conversation yet" to a member with a daily plan
        # open in front of them. Two correct refusals, one nonsense answer.
        scored = self._maybe_score_canvas(session, session_id, message)
        if scored is not None:
            return scored

        tool_turn = self._maybe_use_tool(session, session_id, message)
        if tool_turn is not None:
            return tool_turn

        history = [
            {"role": m.role, "content": m.content}
            for m in session.conversation[-12:]
        ]
        classification = self.orchestrator.classify(message, history)
        intent = classification["intent"]
        target_plan_type = classification.get("target_plan_type")
        if classification.get("failed") and self.looks_like_a_pasted_plan(message):
            # The classifier could not answer (an outage, or a spent API
            # budget) and its default is "chat". A message listing meals in
            # two slots is not small talk, and answering it as such is how a
            # pasted plan came back as chatter on a rate-limited key.

            logger.warning(
                "[%s] Classification unavailable — the message lists meals, scoring it.", session_id,
            )
            intent, target_plan_type = "score_plan", None
        logger.info("[%s] intent=%s target=%s", session_id, intent, target_plan_type)
        return self._route(session, session_id, message, intent, target_plan_type,
                           score_may_ask=score_may_ask)

    def _maybe_score_canvas(self, session, session_id: str, message: str):
        """Score the plan on the canvas, or `None` to route the turn normally.

        Three conditions, all deterministic, because this runs before any
        model is asked anything:

        * a **scoring word** — the same one `is_explicit_score_request` uses;
        * **no meal listing**, so a pasted plan still goes to the pasted path,
          which is the one that can read it;
        * **no request verb**. "Make it score better" is an instruction, not a
          question about the current plan, and scoring it would answer
          something nobody asked.

        A session with no plan returns `None` and the turn routes on — the
        member is talking about a plan they have not made yet, and the
        classifier is better placed to work out which.
        """
        if not self._SCORE_REQUEST_RE.search(message or ""):
            return None
        if self._PLAN_REQUEST_RE.search(message or ""):
            return None
        if self._lists_meals(message):
            return None      # a pasted plan; `_handle_score_plan` owns it
        try:
            outcome = self.plan_scorer.score_canvas(
                session_id, self._canvas_kind(session),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Could not score the canvas plan: %s", session_id, exc)
            return None
        if outcome is None:
            return None
        logger.info("[%s] Scoring the plan on the canvas.", session_id)
        return self._turn_from_score(outcome)

    @staticmethod
    def _canvas_kind(session) -> str:
        """Which canvas the member is looking at, or "" when there is none."""
        canvas = getattr(session, "active_canvas", None)
        return str(getattr(canvas, "plan_type", "") or "")

    # Chooses one of FoodChat's own capabilities, or none. Fast tier: routing
    # over a handful of named tools, running before the intent classifier on
    # every eligible turn.
    #
    # Lazily built and held on the CLASS, not assigned in `__init__`. An
    # instance constructed without `__init__` — which this test suite does
    # routinely — would otherwise have no selector at all, and the first turn
    # through it would raise `AttributeError` where it used to route fine. Same
    # reason the in-flight guard is class-level.
    _tool_selector = None

    @property
    def tool_selector(self) -> ToolSelector:
        if OrchestratorService._tool_selector is None:
            OrchestratorService._tool_selector = ToolSelector()
        return OrchestratorService._tool_selector

    def _maybe_use_tool(self, session, session_id: str, message: str):
        """Run one of FoodChat's capabilities, or return None to route normally.

        Gated before the model is asked anything, because the selector runs on
        every eligible turn and a question nobody needed is still a question
        that was paid for:

        * **A plan must be on the canvas.** Every tool acts on one, so with no
          plan there is nothing to summarise, total or replace, and the answer
          is known without asking.
        * **The turn must not be mid-clarification.** Handled earlier, but
          stated here because a tool firing on "yes please" would answer a
          question the member was not asking.

        Returns None on anything unexpected. A tool surface that can break an
        ordinary turn is worse than no tool surface.
        """
        canvas = session.active_canvas
        if canvas is None:
            return None
        plan_type = canvas.plan_type
        plan = (
            session.get_current_weekly_plan() if plan_type == "weekly"
            else session.get_current_daily_plan()
        )
        if plan is None:
            return None

        try:
            import tools

            # Only what this canvas can actually serve — which is what this
            # comment always claimed and the line below did not do: every tool
            # was offered on every canvas, so a member on a daily plan could
            # have `replace_day` chosen for them and be told there is no weekly
            # plan. The registry declares which canvases each tool works on.
            available = {t.name for t in tools.for_canvas(plan_type)}
            manifest = tools.describe_tools(plan_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool registry unavailable: %s", exc)
            return None

        shape = self._describe_canvas_shape(plan, plan_type)
        choice = self.tool_selector.choose(
            message, plan_type=plan_type, plan_shape=shape,
            manifest=manifest, allowed=available,
        )
        # Re-checked here, not just inside the selector. `_maybe_use_tool` is
        # the thing that spends the tool call, so it verifies its own
        # precondition rather than trusting a caller's discipline: a choice of
        # `{"tool": ""}` is a truthy dict, and an unchecked one would reach
        # `tools.invoke("")` and turn an ordinary message into a 400-flavoured
        # reply instead of routing it normally.
        name = str((choice or {}).get("tool") or "").strip()
        if not name or name not in available:
            if name:
                logger.info(
                    "[%s] Ignoring tool choice %r — not available here",
                    session_id, name,
                )
            return None
        spec = tools.get(name)
        # Only the arguments the chosen tool actually declares. The selector
        # answers one schema for every tool, so it can name a title for a tool
        # that has no title — and the registry rejects an unknown key as a
        # member-facing error, which would turn a routable message into a
        # complaint about a field the member never mentioned.
        props = (spec.parameters.get("properties") or {}) if spec is not None else {}
        arguments: dict = {"session_id": session_id}
        if "day" in props and choice.get("day") is not None:
            arguments["day"] = int(choice["day"])
        if "plan_type" in props:
            # Defaulted to the canvas rather than left out: a tool that takes a
            # plan type should act on what the member is looking at.
            arguments["plan_type"] = choice.get("plan_type") or plan_type
        if "title" in props and str(choice.get("title") or "").strip():
            arguments["title"] = str(choice["title"]).strip()
        if "saved" in props and choice.get("saved") is not None:
            # The registry's enum is a string, so the boolean the selector
            # answers is converted here rather than widening the tool's schema.
            arguments["saved"] = "true" if choice["saved"] else "false"

        try:
            result = tools.invoke(name, arguments)
        except tools.ToolError as exc:
            # Member-facing prose from the registry — "this plan covers Monday
            # to Wednesday". Worth saying verbatim rather than routing on and
            # answering a different question.
            logger.info("[%s] Tool %s declined: %s", session_id, name, exc)
            self.session_service.add_message(session_id, "user", message)
            self.session_service.add_message(session_id, "assistant", str(exc))
            return ChatTurn(role="assistant", content=str(exc), intent="chat")
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Tool %s failed: %s", session_id, name, exc, exc_info=True)
            return None

        return self._answer_from_tool(
            session, session_id, message, name, spec, result, choice,
        )

    def _maybe_navigate(self, session, session_id: str, message: str):
        """Restore a version, or decline an order we cannot serve. None to route on."""
        from models.planning_state import PlanningStateDelta
        from services import plan_navigation

        # A plan the member PASTED is not a navigation request, whatever words
        # it happens to contain. "Snack before lunch" is a line in somebody's
        # own plan far more often than it is an instruction about ours, and
        # this bypass runs ahead of the scorer that the paste is for.
        if self.looks_like_a_pasted_plan(message):
            return None

        canvas = session.active_canvas
        wanted = plan_navigation.restore_request(message)

        if wanted is not None:
            if canvas is None:
                return None          # nothing to go back to; route normally
            versions = self.session_service.plan_versions(session_id, canvas.plan_type)
            if not versions:
                return None
            current = next((v for v, _id, is_current in versions if is_current), None)
            target = (current - 1) if wanted == "previous" else int(wanted)

            if target is not None and current is not None and target == current:
                text = f"You're already on version {current}."
                self._say(session_id, message, text)
                return ChatTurn(role="assistant", content=text, intent="chat")

            plan = (self.session_service.restore_plan_version(
                session_id, target, canvas.plan_type,
            ) if target and target >= 1 else None)
            if plan is None:
                have = ", ".join(str(v) for v, _i, _c in versions)
                text = (
                    f"I don't have a version {target} of this plan — "
                    f"there {'is' if len(versions) == 1 else 'are'} {have}."
                )
                self._say(session_id, message, text)
                return ChatTurn(role="assistant", content=text, intent="chat")

            text = f"Back to version {target}."
            self._say(session_id, message, text)
            turn = ChatTurn(role="assistant", content=text, intent="chat")
            # Nothing was regenerated, so the canvas the member is looking at
            # has to be the one that comes back with this turn.
            if canvas.plan_type == "weekly":
                turn.weekly_meal_plan = plan
            else:
                turn.meal_plan = plan
            turn.plan_version = getattr(plan, "version", None)
            turn.plan_parent_id = getattr(plan, "parent_id", None)
            return turn

        # A reorder is only handled when nothing else in the sentence can be
        # served: "add a salad before lunch" is an addition that happens to
        # mention an order, and re-planning it is a better answer.
        if canvas is None or turn_intake.added_shape():
            return None

        state = self.session_service.get_planning_state(session_id)
        move = plan_navigation.reorder_request(message)
        if move is not None:
            slot, where, anchor = move
            spec = state.spec.reorder(slot, **{where: anchor})
            if spec is state.spec:
                text = (
                    f"I can't put the {slot} {where} the {anchor} — "
                    f"this plan has {', '.join(state.spec.meals)}."
                )
                self._say(session_id, message, text)
                return ChatTurn(role="assistant", content=text, intent="chat")

            # The standing shape, so the NEXT plan keeps the arrangement too.
            self.session_service.set_planning_state(
                session_id, state.merge(PlanningStateDelta(spec=spec)),
            )
            plan = self.session_service.reorder_current_plan(
                session_id, list(spec.meals), canvas.plan_type,
            )
            text = f"Moved the {slot} {where} the {anchor} — {', '.join(spec.meals)}."
            self._say(session_id, message, text)
            turn = ChatTurn(role="assistant", content=text, intent="chat")
            if plan is not None:
                turn.meal_plan = plan
                turn.plan_version = getattr(plan, "version", None)
                turn.plan_parent_id = getattr(plan, "parent_id", None)
            return turn

        if plan_navigation.asks_to_reorder(message):
            # An order was asked for but not a whole one. Asking beats guessing
            # which meal they meant, and it beats declining something we can do.
            question = plan_navigation.reorder_question(message, state.spec.meals)
            if question:
                self._say(session_id, message, question)
                return ChatTurn(role="assistant", content=question, intent="chat")
        return None

    def _say(self, session_id: str, message: str, text: str) -> None:
        """Record the exchange the way every other handler does."""
        self.session_service.add_message(session_id, "user", message)
        self.session_service.add_message(session_id, "assistant", text)

    @staticmethod
    def _describe_canvas_shape(plan, plan_type: str) -> str:
        """The plan's shape, so the selector knows which days exist."""
        if plan_type == "weekly":
            days = sorted({int(e.get("day") or 0) for e in (plan.entries or [])})
            return f"{len(days)} day(s), {len(plan.entries or [])} meals"
        groups = plan.day_plans
        slots = ", ".join(m.meal_type for m in groups[0].meals) if groups else ""
        return f"{len(groups)} day(s): {slots}"

    def _answer_from_tool(self, session, session_id: str, message: str,
                          name: str, spec, result: dict, choice: dict) -> ChatTurn:
        """Turn a tool result into a reply, and a plan change into a canvas.

        A read is answered from the tool's own numbers — it summed them, so the
        model must not re-add them. A mutation reloads the canvas so the member
        sees the plan the tool produced rather than the one before it.
        """
        self.session_service.add_message(session_id, "user", message)

        canned = self._tool_fallback(name, result)
        if turn_budget.skip("tool reply", turn_budget.COST_WRITER):
            answer = canned
        else:
            # The PlanAnalyst, not the ResponseWriter: this is a question ABOUT
            # the plan, which is exactly what the analyst is for, and it is the
            # agent this service already owns. The tool's own output is the
            # entire grounding — it summed the numbers, so the model is told
            # not to re-add them.
            summary = (
                f"{name} returned:\n{json.dumps(result, default=str)[:2500]}\n\n"
                "These figures are already computed. Report them; do not "
                "recalculate or add anything to them."
            )
            history = [(m.role, m.content) for m in session.conversation[-6:]]
            try:
                answer = self.plan_analyst.answer(message, summary, history)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Tool reply failed: %s", session_id, exc)
                answer = canned
            if not (answer or "").strip():
                answer = canned

        turn = ChatTurn(role="assistant", content=answer, intent="chat")
        if spec is not None and spec.mutates:
            # The tool rewrote part of the plan. Reload the canvas so the
            # response carries what the member is now looking at — whichever
            # canvas it was. Only the weekly one was reloaded, so a swap on a
            # daily plan answered with prose and left the old plan attached.
            canvas = session.active_canvas
            if canvas is not None and canvas.plan_type == "daily":
                refreshed = session.get_current_daily_plan()
                if refreshed is not None:
                    turn.meal_plan = refreshed
                    turn.plan_version = refreshed.version
                    turn.plan_parent_id = refreshed.parent_id
            else:
                refreshed = session.get_current_weekly_plan()
                if refreshed is not None:
                    turn.weekly_meal_plan = refreshed
                    turn.plan_version = refreshed.version
                    turn.plan_parent_id = refreshed.parent_id
        self.session_service.add_message(
            session_id, "assistant", answer, intent="chat",
        )
        return turn

    @staticmethod
    def _tool_fallback(name: str, result: dict) -> str:
        """A usable sentence without the writer, from the tool's own numbers."""
        total = result.get("week_totals") or result.get("total") or result.get("totals")
        if isinstance(total, dict) and total.get("calories"):
            line = f"That comes to about {round(float(total['calories'])):,} kcal"
            if result.get("daily_average_kcal"):
                line += f" — roughly {round(float(result['daily_average_kcal']))} a day"
            if total.get("complete") is False:
                line += (
                    f" (counted {total.get('meals_counted')} of "
                    f"{total.get('meals_total')} meals; the rest carry no "
                    "nutrition data)"
                )
            return line + "."
        if result.get("name") and result.get("meals"):
            dishes = ", ".join(
                str(m.get("title")) for m in result["meals"] if m.get("title")
            )
            return f"{result['name']}: {dishes}." if dishes else f"Here's {result['name']}."
        if result.get("days"):
            days = len(result["days"])
            if result.get("plan_type") == "daily":
                return f"Here's the plan — {days} day(s) on it."
            return f"Here's the week — {days} days on the plan."
        if result.get("items") and result.get("item_count") is not None:
            # A shopping list, and the sentence must not imply amounts: the
            # corpus stores ingredients as text, so the tool reports what each
            # item is FOR rather than how much of it to buy.
            head = ", ".join(str(row["item"]) for row in result["items"][:6])
            return (
                f"{result['item_count']} things to buy across "
                f"{result.get('dishes', 0)} dishes — {head}"
                f"{'…' if result['item_count'] > 6 else ''}. "
                "No quantities: the recipes list ingredients as text."
            )
        if "saved" in result and result.get("plan_type"):
            if not result["saved"]:
                return "Taken off your saved plans."
            name = result.get("title")
            return f"Saved as “{name}”." if name else "Saved to your plans."
        return "Done."

    def _route(self, session, session_id: str, message: str, intent: str,
               target_plan_type: Optional[str], score_may_ask: bool = True) -> ChatTurn:

        if intent == "score_plan":
            # Checked first: a pasted plan is never a request to plan, edit or
            # refine, whatever else the message says.
            return self._handle_score_plan(session_id, message, may_ask=score_may_ask)

        if intent == "switch_plan_type":
            return self._handle_switch(session_id, message, target_plan_type)

        if intent in ("weekly_plan", "daily_plan"):
            # Named anchor dishes are extracted once per plan turn: they feed
            # pinned-slot planning AND gate the favorites offer (an explicit
            # dish request means the user already has a starting point).
            seeds = self.seed_service.extract_seeds(message)

            # No favourites toll-booth. The offer intercepted the first plan
            # of EVERY session with a yes/no question before doing anything —
            # a member who plans daily answered the same question daily, and
            # called it what it was: repetitive. Favourites need no question:
            # they are a soft ranking boost that every hard filter still
            # gates, the response writer names one when it lands, and a
            # member who says "no favourites" has that decline held by
            # PlanningState for the rest of the session.

            if intent == "weekly_plan":
                # A *shaped* week — salads beside dinner, a two-plate lunch —
                # cannot be expressed by the weekly planner, whose loop is
                # fixed at seven days of three single-plate meals. The
                # structured path can, and RecipeWrangler plans N days
                # natively without reusing a recipe. Route the request there;
                # the spec extractor reads "week" as num_days=7, and the plan
                # lands on the plan canvas with every day rendered.
                # The shape intake already read, not a second extraction of
                # the same sentence — this branch used to pay its own
                # `extract_state_delta` call and then throw the result away
                # except for this one boolean.
                spec = turn_intake.intake(
                    session_id, message, session_service=self.session_service,
                ).spec
                if spec is not None and spec.plates:
                    logger.info(
                        "[%s] Weekly request carries a multi-plate shape (%s) — "
                        "routing through the structured path",
                        session_id, spec.describe(),
                    )
                    return self._handle_plan(
                        session_id, message, "daily_plan", is_refinement=False, seeds=seeds
                    )
                return self._handle_weekly(session_id, message, intent, is_refinement=False, seeds=seeds)
            return self._handle_plan(session_id, message, intent, is_refinement=False, seeds=seeds)

        if intent == "refine_plan":
            canvas = session.active_canvas
            if canvas is None:
                # Nothing to refine yet — treat as a fresh daily plan.
                logger.info("[%s] refine_plan with no active canvas — fresh daily plan", session_id)
                return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)
            if canvas.plan_type == "weekly":
                return self._handle_weekly(session_id, message, intent, is_refinement=True)
            return self._handle_plan(session_id, message, intent, is_refinement=True)

        if intent == "edit_plan_slot":
            # A turn that GREW the shape is a re-plan, not a swap.
            #
            # "Add breakfast" and "add a salad as well there for side" classify
            # as slot edits, and an edit can only replace the dish on a slot
            # that exists. So the first got "this plan has lunch, dinner — which
            # of those should I change?" in a loop, and the second had its swap
            # executed and its addition heard by nobody: the member read a
            # confident answer to half of what they asked.
            #
            # The shape reader has already updated the standing spec by the
            # time this runs, so re-planning honours BOTH halves — the new plate
            # and the "lighter" — where the edit path could only ever do one.
            grew = turn_intake.added_shape()
            if grew:
                logger.info(
                    "[%s] %s — re-planning rather than swapping a slot",
                    session_id, "; ".join(grew),
                )
                canvas = session.active_canvas
                if canvas is not None and canvas.plan_type == "weekly":
                    return self._handle_weekly(
                        session_id, message, "refine_plan", is_refinement=True,
                    )
                return self._handle_plan(
                    session_id, message, "refine_plan", is_refinement=True,
                    seeds=self.seed_service.extract_seeds(message),
                )
            if session.active_canvas is None:
                # "get me a side salad as well" with nothing on the canvas is
                # not an error to bounce — it is a plan request wearing edit
                # words. refine_plan already degrades this way; an edit with
                # no plan deserves the same grace, not "ask me for a plan
                # first" (a real member hit exactly that and read it as the
                # assistant refusing to do its one job).
                logger.info("[%s] edit_plan_slot with no active canvas — fresh daily plan", session_id)
                seeds = self.seed_service.extract_seeds(message)
                return self._handle_plan(
                    session_id, message, "daily_plan", is_refinement=False, seeds=seeds
                )
            return self._handle_edit(session_id, message)

        if intent == "nutrition_question":
            return self._handle_nutrition_question(session_id, message, session=session)

        if intent == "plan_question":
            return self._handle_plan_question(session, session_id, message)

        if intent == "preference_update":
            return self._handle_preference_update(session_id, message)

        # "chat" and any unexpected values
        return self._handle_smalltalk(session_id, message)

    # ------------------------------------------------------------------ #
    # Handlers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_favorites(self, favorite_ids: list) -> list[tuple[str, str]]:
        """(recipe_id, title) for each resolvable favorite, junk-tolerant.

        Legacy favorite rows hold titles ("Leftover Turkey Casserole") or dead
        ids; a failed direct fetch retries through the seed path's tolerant
        autocomplete so those still resolve to a real recipe when one exists.
        """
        resolved_pairs: list[tuple[str, str]] = []
        for raw_id in list(favorite_ids)[:MAX_OFFERED_FAVORITES]:
            raw_id = str(raw_id).strip()
            if not raw_id:
                continue
            resolved = self.seed_service.client.fetch_recipe(raw_id)
            if resolved is None:
                suggestions = self.seed_service._autocomplete_tolerant(raw_id)
                if suggestions:
                    candidate_id, _title = suggestions[0]
                    resolved = self.seed_service.client.fetch_recipe(candidate_id)
            if resolved and resolved.recipe.title:
                resolved_pairs.append((resolved.recipe.recipe_id, resolved.recipe.title))
        return resolved_pairs

    def _maybe_offer_favorites(
        self, session, message: str, intent: str, seeds: list[dict]
    ) -> Optional[ChatTurn]:
        """One-time proactive offer to work the member's favorites into the plan.

        Fires only when ALL hold: first plan of the session (no canvases yet),
        the member has favorites, the request names no dishes itself, and no
        offer was made before in this session (offer messages are tagged with
        intent="favorites_offer" — the tag is the dedupe record, persisted
        with the message). Declining is safe: the original request proceeds
        unchanged on the next turn.
        """
        if seeds:
            return None
        if session.meal_plans or session.weekly_meal_plans:
            return None
        favorites = session.user_profile.get("favorite_recipe_ids") or []
        if not favorites:
            return None
        if any(m.intent == "favorites_offer" for m in session.conversation):
            return None

        # Resolve favorites ONCE, here — legacy rows may hold titles or dead
        # ids, so unresolvable direct fetches fall back to the same tolerant
        # autocomplete the seed path uses. The resolved pairs are stored in
        # the clarification state so acceptance anchors EXACTLY what was
        # offered: offer and accept previously read the favorites list
        # independently and could diverge, offering titles it then dropped.
        resolved_favorites = self._resolve_favorites(favorites)
        if not resolved_favorites:
            return None
        named = ", ".join(f"“{title}”" for _rid, title in resolved_favorites)

        offer_text = (
            f"Before I plan — I noticed you've favorited {named}. "
            "Want me to work them into this plan? (yes / no)"
        )
        self.session_service.set_clarification_state(session.session_id, {
            "kind": "favorites_offer",
            "original_message": message,
            "origin_intent": intent,
            "favorites": [
                {"recipe_id": rid, "title": title} for rid, title in resolved_favorites
            ],
        })
        self.session_service.add_message(
            session.session_id, "assistant", offer_text, intent="favorites_offer"
        )
        logger.info("[%s] Favorites offer made (%d favorites).", session.session_id, len(favorites))
        return ChatTurn(
            role="assistant", content=offer_text,
            intent="favorites_offer", needs_clarification=True,
        )

    def _handle_favorites_offer_reply(self, session_id: str, message: str) -> ChatTurn:
        """Consume the yes/no reply to the favorites offer and generate the plan."""
        session = self.session_service.get_session(session_id)
        pending = session.clarification or {}
        self.session_service.clear_clarification_state(session_id)

        original_message = pending.get("original_message", message)
        intent = pending.get("origin_intent", "daily_plan")
        accepted = bool(_AFFIRMATIVE.match(message or ""))
        logger.info("[%s] Favorites offer %s.", session_id, "accepted" if accepted else "declined")

        # Record the answer so it outlives this turn.
        #
        # It was computed and thrown away: the decline suppressed the boost for
        # exactly one request, and the next regeneration read `favorite_recipe_ids`
        # off the profile again and put the favourite straight back. A member who
        # says no and sees their favourite in the plan anyway has been told their
        # answer does not matter.
        from models.planning_state import PlanningStateDelta

        try:
            state = self.session_service.get_planning_state(session_id)
            self.session_service.set_planning_state(
                session_id, state.merge(PlanningStateDelta(use_favorites=accepted))
            )
        except Exception as exc:  # noqa: BLE001
            # Losing the preference is bad; losing the plan is worse.
            logger.warning("[%s] Could not persist favorites answer: %s", session_id, exc)

        seeds: list[dict] = []
        if accepted:
            # Anchor exactly the favorites that were OFFERED (resolved pairs
            # stored with the offer). recipe_id makes resolution a DIRECT
            # fetch — no name round-trip that could land on another recipe.
            # Seed resolution still re-checks allergies, so an unsafe
            # favorite is skipped with a note.
            for fav in pending.get("favorites") or []:
                title = str(fav.get("title") or "").strip()
                if title:
                    seeds.append({"name": title, "recipe_id": fav.get("recipe_id")})
            if not seeds:
                # Offers made before this fix carry no stored pairs.
                for rid, title in self._resolve_favorites(
                    session.user_profile.get("favorite_recipe_ids") or []
                ):
                    seeds.append({"name": title, "recipe_id": rid})

        if intent == "weekly_plan":
            return self._handle_weekly(session_id, original_message, intent, is_refinement=False, seeds=seeds)
        return self._handle_plan(session_id, original_message, intent, is_refinement=False, seeds=seeds)

    def _handle_clarification_turn(self, session_id: str, message: str) -> ChatTurn:
        session = self.session_service.get_session(session_id)
        pending = (session.clarification or {}) if session else {}

        if pending.get("kind") == "favorites_offer":
            # The user is answering the favorites offer, not a plan question.
            self.session_service.add_message(session_id, "user", message)
            return self._handle_favorites_offer_reply(session_id, message)

        if pending.get("kind") == "edit_slot":
            # The user is (probably) telling us WHICH meal to swap (M4b).
            outcome = self.edit_service.continue_clarification(session_id, message)
            if outcome.unresolved:
                # The reply didn't answer the slot question — the trap is
                # cleared and nothing was logged, so route it as a fresh turn.
                session = self.session_service.get_session(session_id)
                return self._classify_and_route(session, session_id, message)
            return self._turn_from_edit(session_id, outcome)

        if pending.get("kind") == SCORE_CLARIFICATION_KIND:
            # The member is telling us what their pasted plan is (its meals,
            # or how many days it covers).
            outcome = self.plan_scorer.continue_clarification(session_id, message)
            if outcome.unresolved:
                # Not an answer — the state is cleared and nothing was logged.
                # Route it as a fresh turn, and never ask a score question twice.
                session = self.session_service.get_session(session_id)
                return self._classify_and_route(session, session_id, message, score_may_ask=False)
            return self._turn_from_score(outcome)

        # FoodScholar clarifications are tagged with kind="foodscholar";
        # plan-flow states (ClarificationState.to_dict) have no "kind" key.
        if pending.get("kind") == FoodScholarService.CLARIFICATION_KIND:
            # Re-attach FRESH plan context: nutrition computed since the
            # thread started (backfill, recipe visits) must reach the scholar.
            fs_turn = self.foodscholar_service.continue_clarification(
                session_id, message,
                contextualize=lambda q: self._with_plan_context(session, q),
            )
            self._tag_last_message(session_id, "nutrition_question", None)
            return ChatTurn(
                role="assistant",
                content=fs_turn.text,
                intent="nutrition_question",
                needs_clarification=fs_turn.needs_clarification,
                attribution=fs_turn.attribution,
            )

        response_text, needs_clarification, meal_plan, origin_intent = (
            self.chat_service.continue_clarification(session_id, message)
        )
        self._tag_last_message(session_id, origin_intent, meal_plan.id if meal_plan else None)
        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=origin_intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
            plan_parameters=self._parameter_card(session_id, origin_intent, meal_plan),
        )

    def _handle_edit(self, session_id: str, message: str) -> ChatTurn:
        """Targeted single-slot edit with verified directive (M4b)."""
        outcome = self.edit_service.process(session_id, message)
        if outcome.unresolved:
            # The classifier heard "change the plan" but no single slot could
            # be parsed. The member is still asking for a change — run it as a
            # refinement of the active plan, which reads free text, instead of
            # bouncing back "could you rephrase?".
            session = self.session_service.get_session(session_id)
            canvas = session.active_canvas if session else None
            if canvas is not None and canvas.plan_type == "weekly":
                return self._handle_weekly(
                    session_id, message, "refine_plan", is_refinement=True
                )
            return self._handle_plan(
                session_id, message, "refine_plan", is_refinement=True
            )
        return self._turn_from_edit(session_id, outcome)

    def _turn_from_edit(self, session_id: str, outcome) -> ChatTurn:
        plan = outcome.meal_plan or outcome.weekly_meal_plan
        self._tag_last_message(session_id, "edit_plan_slot", plan.id if plan else None)
        if plan is not None and outcome.changed_slots:
            # A verified swap of a hand-picked slot means the user changed
            # their mind — the old pick must not resurrect on the next refine.
            self._drop_manual_picks_for_slots(session_id, outcome.changed_slots)
        return ChatTurn(
            role="assistant",
            content=outcome.text,
            intent="edit_plan_slot",
            needs_clarification=outcome.needs_clarification,
            meal_plan=outcome.meal_plan,
            weekly_meal_plan=outcome.weekly_meal_plan,
            changed_slots=outcome.changed_slots or None,
            plan_version=plan.version if plan else None,
            plan_parent_id=plan.parent_id if plan else None,
        )

    def _handle_score_plan(
        self, session_id: str, message: str, may_ask: bool = True,
        plan_type: str = "auto", context: Optional[str] = None,
    ) -> ChatTurn:
        """Score a plan the member wrote: parse, ground, build, score, reply.

        No canvas is touched and no plan version created — refine and edit
        turns keep targeting the member's own plan.
        """
        return self._turn_from_score(
            self.plan_scorer.process(
                session_id, message, plan_type=plan_type, context=context, may_ask=may_ask,
            )
        )

    @staticmethod
    def _turn_from_score(outcome) -> ChatTurn:
        return ChatTurn(
            role="assistant",
            content=outcome.text,
            intent="score_plan",
            needs_clarification=outcome.needs_clarification,
            plan_score=outcome.plan_score,
        )

    def _handle_nutrition_question(self, session_id: str, message: str,
                                   session=None, question: Optional[str] = None) -> ChatTurn:
        """Delegate a nutrition-science question to FoodScholar (M1 bridge).

        When the session has an active plan, its meals ride along as context —
        "is this good for heart health?" should be answered about the actual
        dishes on the member's plan. The transcript keeps the member's own
        words (``message``); only the question sent to FoodScholar is enriched.
        """
        base = question or message
        ask = self._with_plan_context(session, base) if session is not None else base
        fs_turn = self.foodscholar_service.process_question(
            session_id, message, question=ask, raw_question=base,
        )
        self._tag_last_message(session_id, "nutrition_question", None)
        return ChatTurn(
            role="assistant",
            content=fs_turn.text,
            intent="nutrition_question",
            needs_clarification=fs_turn.needs_clarification,
            attribution=fs_turn.attribution,
        )

    def _handle_plan_question(self, session, session_id: str, message: str) -> ChatTurn:
        """Answer a question ABOUT the active plan without touching it.

        Grounded in the serialized canvas (titles + nutrition enrichment) and
        recent conversation so references like "that" resolve to the guidance
        just discussed. No canvas → the question is really a nutrition
        question, so it falls through to FoodScholar.
        """
        summary = self._summarize_active_plan(session)
        if summary is None:
            return self._handle_nutrition_question(session_id, message)

        self.session_service.add_message(session_id, "user", message)
        history = [(m.role, m.content) for m in session.conversation[-8:]]
        try:
            answer = self.plan_analyst.answer(message, summary, history)
        except Exception as e:
            logger.warning("[%s] PlanAnalyst failed: %s", session_id, e)
            answer = (
                "I couldn't analyze the plan just now — try asking again, or "
                "tell me what you'd like changed and I'll take it from there."
            )
        self.session_service.add_message(
            session_id, "assistant", answer, intent="plan_question"
        )
        return ChatTurn(role="assistant", content=answer, intent="plan_question")

    def _summarize_active_plan(self, session) -> Optional[str]:
        """Serialize the active canvas for the analyst (None if no plan yet).

        Includes per-meal nutrition where M4 enrichment succeeded; missing
        data is marked so the analyst can be explicit about gaps.
        """

        def fmt_nutrition(n: Optional[dict]) -> str:
            if not n:
                return "(no nutrition data)"
            parts = []
            if n.get("kcal") is not None:
                parts.append(f"{n['kcal']} kcal")
            for key, unit in (("protein_g", "g protein"), ("carbs_g", "g carbs"), ("fat_g", "g fat")):
                if n.get(key) is not None:
                    parts.append(f"{n[key]}{unit}")
            return "(" + ", ".join(parts) + ")" if parts else "(no nutrition data)"

        canvas = session.active_canvas
        if canvas is None:
            return None

        # Recipes may have been profiled AFTER the plan was stored (backfill,
        # recipe-page visits) — pull any now-available nutrition in before
        # serializing, so analysts never reason over stale "(no data)" gaps.
        try:
            from .plan_nutrition import refresh_plan_nutrition
            refresh_plan_nutrition(session, self.session_service)
        except Exception:  # noqa: BLE001
            logger.warning("Plan nutrition refresh failed", exc_info=True)

        if canvas.plan_type == "weekly":
            plan = session.get_current_weekly_plan()
            if plan is None:
                return None
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            by_day: dict[int, list] = {}
            for entry in plan.entries:
                by_day.setdefault(entry.get("day", 0), []).append(entry)
            lines = [f"7-day plan (version {plan.version}):"]
            for day_idx in sorted(by_day):
                # Entry days are 1-based (1=Monday) everywhere in the weekly
                # stack; indexing day_names directly labeled every day one
                # late, so the analyst answered about the wrong day.
                label = (
                    day_names[day_idx - 1] if 1 <= day_idx <= len(day_names)
                    else f"Day {day_idx}"
                )
                lines.append(f"{label}:")
                for entry in sorted(by_day[day_idx], key=lambda e: e.get("meal_idx", 0)):
                    recipe = entry.get("recipe", {})
                    lines.append(
                        f"  {entry.get('meal_type', 'meal')}: {recipe.get('title', '?')} "
                        f"{fmt_nutrition(recipe.get('nutrition'))}"
                    )
            totals = self._totals_line(session.session_id, "weekly")
            if totals:
                lines.append(totals)
            return "\n".join(lines)

        plan = session.get_current_daily_plan()
        if plan is None:
            return None
        lines = [f"Daily plan (version {plan.version}):"]
        for slot in ("breakfast", "lunch", "dinner"):
            course = getattr(plan, slot)
            lines.append(f"  {slot}: {course.title} {fmt_nutrition(course.nutrition)}")
        totals = self._totals_line(session.session_id, "daily")
        if totals:
            lines.append(totals)
        return "\n".join(lines)

    @staticmethod
    def _totals_line(session_id: str, plan_type: str) -> str:
        """Summed totals for the analyst, so it never has to add up 21 numbers.

        The analyst was handed per-meal nutrition and nothing else, so any
        question about a whole day or week made it do arithmetic in prose —
        the one thing a language model should not be trusted with here. The
        `plan_totals` tool sums the stored plan and says how many meals it
        could actually see; both facts belong in the context.
        """
        try:
            import tools

            result = tools.invoke(
                "plan_totals", {"session_id": session_id, "plan_type": plan_type}
            )
        except Exception:  # noqa: BLE001
            return ""
        total = result.get("total") or {}
        if not total.get("meals_total"):
            return ""
        bits = [f"{total.get('calories', 0):.0f} kcal total"]
        for key, unit in (("protein_g", "g protein"), ("carbs_g", "g carbs"),
                          ("fat_g", "g fat")):
            if total.get(key):
                bits.append(f"{total[key]:.0f}{unit}")
        line = "COMPUTED TOTALS (already summed — do not re-add): " + ", ".join(bits)
        if not total.get("complete"):
            line += (
                f" — counted {total.get('meals_counted')} of "
                f"{total.get('meals_total')} meals; the rest carry no nutrition data"
            )
        if plan_type == "weekly" and result.get("daily_average_kcal"):
            line += f". Daily average {result['daily_average_kcal']:.0f} kcal"
        return line

    def _handle_smalltalk(self, session_id: str, message: str) -> ChatTurn:
        response_text, _, _ = self.chat_service.process_smalltalk(session_id, message)
        self._tag_last_message(session_id, "chat", None)
        return ChatTurn(role="assistant", content=response_text, intent="chat")

    def _handle_preference_update(self, session_id: str, message: str) -> ChatTurn:
        """Acknowledge a stated durable preference without interrogating (M3).

        The durable write stays consent-gated: process() attaches the memory
        nudge and the profile only changes via POST /sessions/{id}/memory —
        this handler just answers in the persona voice.

        One exception answers deterministically: a pantry statement ("I have
        leftover rice in the fridge") is session planning state, not a durable
        profile fact — it is stored NOW so the next plan uses it, and the
        reply names what was noted and offers to plan. Best-effort: a failed
        capture falls back to the ordinary acknowledgment.
        """
        try:
            from services import pantry_service

            delta = pantry_service.extract_pantry_delta(message)
            if not delta.is_empty:
                state = self.session_service.get_planning_state(session_id)
                state = state.merge(delta)
                self.session_service.set_planning_state(session_id, state)
                if state.pantry:
                    text = (
                        "Noted — I'll plan around your "
                        + ", ".join(state.pantry)
                        + " to keep food waste down. Ask me for a daily or "
                        "weekly plan and I'll work them in."
                    )
                else:
                    text = "Noted — I've cleared those from your use-up list."
                self.session_service.add_message(session_id, "user", message)
                self.session_service.add_message(session_id, "assistant", text)
                self._tag_last_message(session_id, "preference_update", None)
                return ChatTurn(role="assistant", content=text, intent="preference_update")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Pantry capture failed: %s", session_id, exc)

        response_text, _, _ = self.chat_service.process_smalltalk(session_id, message)
        self._tag_last_message(session_id, "preference_update", None)
        return ChatTurn(role="assistant", content=response_text, intent="preference_update")

    def _handle_plan(
        self, session_id: str, message: str, intent: str, is_refinement: bool,
        seeds: Optional[list[dict]] = None, skip_clarification: bool = False,
        persist_seeds: bool = False,
    ) -> ChatTurn:
        if is_refinement:
            seeds = self._seeds_for_refinement(session_id, "daily", seeds)
        response_text, needs_clarification, meal_plan = self.chat_service.process_plan_request(
            session_id, message, is_refinement=is_refinement, seeds=seeds,
            skip_clarification=skip_clarification,
        )
        self._settle_manual_picks(
            session_id, "daily", meal_plan, seeds, is_refinement, persist_seeds,
        )
        self._tag_last_message(session_id, intent, meal_plan.id if meal_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
            plan_parameters=self._parameter_card(session_id, intent, meal_plan),
        )

    def _parameter_card(self, session_id: str, intent: str, plan) -> Optional[dict]:
        """The slider card rides along with fresh plans only (daily AND
        weekly) — showing it again on every text refinement would be noise
        (the apply flow re-attaches it explicitly with updated values)."""
        if intent not in ("daily_plan", "weekly_plan") or plan is None:
            return None
        session = self.session_service.get_session(session_id)
        plan_type = "weekly" if intent == "weekly_plan" else "daily"
        return plan_parameters.build_card(
            session.user_profile if session else {}, plan_type,
        )

    # ------------------------------------------------------------------ #
    # Manual picks (compose mode) — survive refinements, die honestly      #
    # ------------------------------------------------------------------ #
    # Hand-picked dishes are a stronger signal than anything inferred: a
    # text refinement must not silently replace them. Lifecycle, all of it
    # driven by what actually landed on the plan (never by intent alone):
    #   - compose stores the picks that REACHED the plan (a pick that failed
    #     id lookup or the allergy gate is dropped, so it is never retried);
    #   - refinements re-inject stored picks (re-resolved and safety-rechecked
    #     each time — diners may have changed), then keep only those still on
    #     the resulting plan, so an explicit seed that displaced one doesn't
    #     let the old pick resurrect next turn;
    #   - a fresh plan request starts a new lineage and clears them — but only
    #     once generation SUCCEEDED, so a failed generation never strands the
    #     still-active plan without its anchors;
    #   - a verified edit unpins the slot it replaced (the user changed their
    #     mind about that pick).
    #
    # Seed precedence, deliberately: explicit per-turn seeds > stored manual
    # picks > weekly standing_seeds (the last applies only to fresh weekly
    # plans, inside weekly_plan_service, and is skipped whenever seeds are
    # already present).

    def _manual_picks(self, session, plan_type: str) -> list:
        store = session.user_profile.get("manual_picks") or {}
        return list(store.get(plan_type) or [])

    def _store_manual_picks(self, session_id: str, plan_type: str, picks: list,
                            session=None) -> None:
        session = session or self.session_service.get_session(session_id)
        if not session:
            return
        store = dict(session.user_profile.get("manual_picks") or {})
        before = store.get(plan_type) or []
        if picks:
            store[plan_type] = picks
        else:
            store.pop(plan_type, None)
        if (store.get(plan_type) or []) == before:
            return  # no-op — don't touch the DB
        session.user_profile["manual_picks"] = store
        self.session_service.persist_profile(session_id)

    def _seeds_for_refinement(self, session_id: str, plan_type: str,
                              seeds: Optional[list]) -> Optional[list]:
        """Explicit seeds win; otherwise stored manual picks anchor the turn.

        Re-injected picks are marked ``kept`` so the reply says "I kept X"
        rather than "I've worked in X as you asked" — the user didn't ask
        this turn, and claiming they did reads as a misunderstanding.
        """
        if seeds:
            return seeds
        session = self.session_service.get_session(session_id)
        if not session:
            return seeds
        picks = self._manual_picks(session, plan_type)
        if picks:
            logger.info(
                "[%s] Re-injecting %d manual pick(s) into %s refinement.",
                session_id, len(picks), plan_type,
            )
            return [{**pick, "kept": True} for pick in picks]
        return seeds

    @staticmethod
    def _plan_recipe_ids(plan, plan_type: str) -> set:
        """Every recipe id present on a generated plan."""
        ids: set = set()
        if plan is None:
            return ids
        if plan_type == "weekly":
            for entry in getattr(plan, "entries", None) or []:
                recipe = entry.get("recipe") if isinstance(entry, dict) else None
                rid = (recipe or {}).get("recipe_id")
                if rid:
                    ids.add(str(rid))
            return ids
        for slot in ("breakfast", "lunch", "dinner"):
            course = getattr(plan, slot, None)
            if course is not None and getattr(course, "recipe_id", None):
                ids.add(str(course.recipe_id))
        return ids

    def _picks_on_plan(self, plan, plan_type: str, picks: list) -> list:
        """The subset of picks that actually made it onto the plan.

        Drops picks the pipeline refused — unresolvable ids, allergy/diet
        conflicts, slot collisions — so they are never retried, never
        re-apologized for, and never anchor a future turn.
        """
        present = self._plan_recipe_ids(plan, plan_type)
        return [
            {k: v for k, v in pick.items() if k != "kept"}
            for pick in picks or []
            if str(pick.get("recipe_id") or "") in present
        ]

    def _settle_manual_picks(self, session_id: str, plan_type: str, plan,
                             seeds: Optional[list], is_refinement: bool,
                             persist_seeds: bool) -> None:
        """Reconcile the stored picks with the plan that was just produced.

        Never runs when generation produced nothing (failure or a pending
        clarification): the previous picks still protect the plan the member
        is looking at. At most ONE profile write.
        """
        if plan is None:
            return
        session = self.session_service.get_session(session_id)
        if not session:
            return

        if persist_seeds:
            kept = self._picks_on_plan(plan, plan_type, seeds or [])
            dropped = len(seeds or []) - len(kept)
            if dropped:
                logger.info(
                    "[%s] %d manual pick(s) never reached the plan — not stored.",
                    session_id, dropped,
                )
            self._store_manual_picks(session_id, plan_type, kept, session=session)
            return

        if not is_refinement:
            # Fresh lineage — the previous picks belong to a plan the member
            # has moved on from. Cleared only now that a plan exists.
            self._store_manual_picks(session_id, plan_type, [], session=session)
            return

        stored = self._manual_picks(session, plan_type)
        if not stored:
            return
        kept = self._picks_on_plan(plan, plan_type, stored)
        if len(kept) != len(stored):
            logger.info(
                "[%s] %d stored pick(s) no longer on the refined plan — unpinning.",
                session_id, len(stored) - len(kept),
            )
            self._store_manual_picks(session_id, plan_type, kept, session=session)

    def _drop_manual_picks_for_slots(self, session_id: str, changed_slots: list) -> None:
        """Unpin the slots a verified edit replaced — one write per plan type.

        Slots are matched by (day, meal_type); ``day`` is 1-based for weekly
        edits and None for daily ones (edit_service always sets both). A
        day-less weekly pick (spread placement — possible via the API, never
        via the compose UI) can't be slot-matched and survives.
        """
        session = self.session_service.get_session(session_id)
        if not session:
            return
        by_type: dict[str, set] = {}
        for slot in changed_slots or []:
            day = slot.get("day")
            plan_type = "weekly" if day is not None else "daily"
            by_type.setdefault(plan_type, set()).add((day, slot.get("meal_type")))

        for plan_type, edited in by_type.items():
            picks = self._manual_picks(session, plan_type)
            if not picks:
                continue
            remaining = [
                p for p in picks
                if (p.get("day"), p.get("meal_type")) not in edited
            ]
            if len(remaining) != len(picks):
                logger.info(
                    "[%s] Verified edit replaced %d manual pick(s) — unpinning.",
                    session_id, len(picks) - len(remaining),
                )
                self._store_manual_picks(session_id, plan_type, remaining, session=session)

    def apply_plan_parameters(self, session_id: str, member_id: str, values: dict,
                              plan_type: Optional[str] = None) -> ChatTurn:
        """Apply slider-card values (already sanitized by the router).

        Deterministic counterpart of a refinement turn: the values become a
        canonical query (no intent classification, no clarification LLM
        round), are stored on the profile so the card shows current settings,
        and land in the profile history so the reconciler treats them as
        known facts for the rest of the session.

        ``plan_type`` is the card's own address — the plan it was rendered
        with. Without it the target would be whichever canvas happens to be
        newest AT CLICK TIME, so a card sitting above a daily plan could
        regenerate a weekly plan created since. Omitted (older clients) →
        the active canvas, as before.
        """
        # One budget and one in-flight claim per turn, at the only four
        # places a turn begins. Nested calls keep the outermost deadline, so a
        # slider apply that routes into the chat handlers does not hand the
        # inner stage a fresh full allowance.
        with self._one_turn_at_a_time(session_id) as claimed, \
                trace_context(session_id=session_id, user_id=member_id), \
                turn_budget.start():
            if not claimed:
                return self._busy_turn()
            # A new turn hears the message fresh. The intake memo exists to stop
            # one turn extracting twice, not to carry an answer into the next.
            turn_intake.forget()
            session = self._owned_session(session_id, member_id)
            limit_turn = self._limit_turn(session)
            if limit_turn is not None:
                return limit_turn

            applied = dict(session.user_profile.get("plan_parameters") or {})
            applied.update(values)
            session.user_profile["plan_parameters"] = applied
            if values.get("cooking_time") is not None:
                # The slider and a spoken "under 20 minutes" are one
                # constraint, so moving the knob is a NEW statement of it and
                # has to update the standing state. Without this the planning
                # paths would re-apply the older spoken value over the newer
                # slider one and the member would watch their drag undo itself.
                from models.planning_state import PlanningStateDelta

                self.session_service.set_planning_state(
                    session_id,
                    self.session_service.get_planning_state(session_id).merge(
                        PlanningStateDelta(max_minutes=int(values["cooking_time"])),
                    ),
                )
            history = session.user_profile.get("history", "") or ""
            line = plan_parameters.history_line(values)
            session.user_profile["history"] = f"{history}\n{line}" if history else line
            self.session_service.persist_profile(session_id)

            target = plan_type if plan_type in ("daily", "weekly") else None
            if target is None:
                canvas = session.active_canvas
                target = canvas.plan_type if canvas is not None else "daily"

            message = plan_parameters.describe(values)
            if target == "weekly" and session.get_current_weekly_plan() is not None:
                logger.info("[%s] Applying plan parameters %s (weekly refine).", session_id, values)
                turn = self._handle_weekly(
                    session_id, message, "refine_plan", is_refinement=True,
                )
            else:
                is_refinement = session.get_current_daily_plan() is not None
                intent = "refine_plan" if is_refinement else "daily_plan"
                logger.info("[%s] Applying plan parameters %s (%s).", session_id, values, intent)
                turn = self._handle_plan(
                    session_id, message, intent,
                    is_refinement=is_refinement, skip_clarification=True,
                )
            # Always return the card with its new current values so the UI stays
            # in sync, even on the refine paths where the handlers skip it.
            turn.plan_parameters = plan_parameters.build_card(session.user_profile, target)
            return turn

    def regenerate(self, session_id: str, member_id: str,
                   plan_type: Optional[str] = None) -> ChatTurn:
        """Re-plan from the standing state, with no new member statement.

        The deterministic counterpart of `apply_plan_parameters` for state the
        member changed by hand rather than by talking: a facet chip removed, a
        pantry item ticked off. Both write `PlanningState` and then need the
        plan on screen to reflect it — and neither has a sentence to classify,
        so routing them through /chat would mean inventing one and having the
        classifier guess at it.

        The query comes from `PlanningState.as_query()`, so it describes what
        is still wanted rather than what was just taken away.
        """
        # One budget and one in-flight claim per turn, at the only four
        # places a turn begins. Nested calls keep the outermost deadline, so a
        # slider apply that routes into the chat handlers does not hand the
        # inner stage a fresh full allowance.
        with self._one_turn_at_a_time(session_id) as claimed, \
                trace_context(session_id=session_id, user_id=member_id), \
                turn_budget.start():
            if not claimed:
                return self._busy_turn()
            # A new turn hears the message fresh. The intake memo exists to stop
            # one turn extracting twice, not to carry an answer into the next.
            turn_intake.forget()
            session = self._owned_session(session_id, member_id)
            limit_turn = self._limit_turn(session)
            if limit_turn is not None:
                return limit_turn

            state = self.session_service.get_planning_state(session_id)
            message = state.as_query()

            target = plan_type if plan_type in ("daily", "weekly") else None
            if target is None:
                canvas = session.active_canvas
                target = canvas.plan_type if canvas is not None else "daily"

            if target == "weekly" and session.get_current_weekly_plan() is not None:
                logger.info("[%s] Regenerating weekly plan: %s", session_id, message)
                return self._handle_weekly(
                    session_id, message, "refine_plan", is_refinement=True,
                )
            is_refinement = session.get_current_daily_plan() is not None
            logger.info("[%s] Regenerating daily plan: %s", session_id, message)
            return self._handle_plan(
                session_id, message,
                "refine_plan" if is_refinement else "daily_plan",
                is_refinement=is_refinement, skip_clarification=True,
            )

    def score_plan(
        self, session_id: str, member_id: str, plan_text: str,
        plan_type: str = "auto", context: Optional[str] = None,
    ) -> ChatTurn:
        """The text box: score a pasted plan with no intent classification.

        Same guards as every entry point (ownership, message cap), the same
        turn as a ``score_plan`` chat message, and it supersedes a pending
        clarification — pasting a plan into the box is a deliberate action.
        ``plan_type`` "daily"/"weekly" settles the shape instead of asking;
        ``context`` is what the member is aiming for, read by the fit judge.
        """
        # The same guard as every other place a turn begins: one budget and
        # one in-flight claim per turn.
        with self._one_turn_at_a_time(session_id) as claimed, \
                trace_context(session_id=session_id, user_id=member_id), \
                turn_budget.start():
            if not claimed:
                return self._busy_turn()
            turn_intake.forget()
            session = self._owned_session(session_id, member_id)
            limit_turn = self._limit_turn(session)
            if limit_turn is not None:
                return limit_turn
            if session.state == "clarifying":
                logger.info("[%s] Plan scoring supersedes a pending clarification.", session_id)
                self.session_service.clear_clarification_state(session_id)
            turn = self._handle_score_plan(
                session_id, plan_text, plan_type=plan_type, context=context,
            )
            return self._attach_memory_suggestions(session, turn, plan_text)

    def compose_plan(
        self, session_id: str, member_id: str, picks: list[dict],
        plan_type: str = "daily", message: Optional[str] = None,
    ) -> ChatTurn:
        """Manual mode: the user hand-picked recipes on a blank canvas and
        asks FoodChat to fill out the rest (daily or weekly).

        Picks are already validated by the router ({meal_type, recipe_id,
        title?, day?}). They become seed anchors — resolved by id,
        allergy/diet re-checked (an unsafe pick is skipped with a note,
        never silently planned), pinned to their exact slots — and
        generation composes the rest deterministically: no intent
        classification, no clarification round.
        """
        # One budget and one in-flight claim per turn, at the only four
        # places a turn begins. Nested calls keep the outermost deadline, so a
        # slider apply that routes into the chat handlers does not hand the
        # inner stage a fresh full allowance.
        with self._one_turn_at_a_time(session_id) as claimed, \
                trace_context(session_id=session_id, user_id=member_id), \
                turn_budget.start():
            if not claimed:
                return self._busy_turn()
            # A new turn hears the message fresh. The intake memo exists to stop
            # one turn extracting twice, not to carry an answer into the next.
            turn_intake.forget()
            session = self._owned_session(session_id, member_id)
            limit_turn = self._limit_turn(session)
            if limit_turn is not None:
                return limit_turn

            # Composing is a deliberate action that supersedes any pending
            # question — leaving the trap armed would make the member's next
            # message answer a question they've moved on from.
            if session.state == "clarifying":
                logger.info("[%s] Compose supersedes a pending clarification.", session_id)
                self.session_service.clear_clarification_state(session_id)

            seeds = []
            for pick in picks:
                seed = {
                    "recipe_id": pick["recipe_id"],
                    "meal_type": pick["meal_type"],
                    "name": pick.get("title") or "",
                }
                if pick.get("day") is not None:
                    seed["day"] = pick["day"]
                seeds.append(seed)

            logger.info(
                "[%s] Manual compose (%s): %d pick(s) → %s",
                session_id, plan_type, len(seeds),
                [(s.get("day"), s["meal_type"], s["recipe_id"]) for s in seeds],
            )

            # Picks that reach the plan are stored by _settle_manual_picks
            # (persist_seeds=True); ones the pipeline refused are dropped there.
            if plan_type == "weekly":
                text = (message or "").strip() or (
                    "Complete my meal plan for this week around the dishes I picked."
                )
                turn = self._handle_weekly(
                    session_id, text, "weekly_plan", is_refinement=False,
                    seeds=seeds, persist_seeds=True,
                )
            else:
                text = (message or "").strip() or (
                    "Complete my meal plan for today around the dishes I picked."
                )
                turn = self._handle_plan(
                    session_id, text, "daily_plan", is_refinement=False,
                    seeds=seeds, skip_clarification=True, persist_seeds=True,
                )

            # The member's own words reach compose when they type instead of
            # pressing the button — a preference stated there must still nudge.
            return self._attach_memory_suggestions(session, turn, message or "")

    def _handle_weekly(
        self, session_id: str, message: str, intent: str, is_refinement: bool,
        seeds: Optional[list[dict]] = None, persist_seeds: bool = False,
    ) -> ChatTurn:
        if is_refinement:
            seeds = self._seeds_for_refinement(session_id, "weekly", seeds)
        response_text, weekly_plan = self.weekly_plan_service.process_message(
            session_id, message, is_refinement=is_refinement, seeds=seeds
        )
        if weekly_plan is not None:
            # The weekly pipeline has no clarification round of its own, so
            # it never clears a trap the daily flow may have armed — a
            # pending question would otherwise eat the member's next message.
            self.session_service.clear_clarification_state(session_id)
        self._settle_manual_picks(
            session_id, "weekly", weekly_plan, seeds, is_refinement, persist_seeds,
        )
        self._tag_last_message(session_id, intent, weekly_plan.id if weekly_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            weekly_meal_plan=weekly_plan,
            plan_version=weekly_plan.version if weekly_plan else None,
            plan_parent_id=weekly_plan.parent_id if weekly_plan else None,
            plan_parameters=self._parameter_card(session_id, intent, weekly_plan),
        )

    def _handle_switch(self, session_id: str, message: str, target_plan_type: Optional[str]) -> ChatTurn:
        """Freeze the current canvas type and start a fresh one of the target type.

        Both canvases stay retrievable — switching never destroys history.
        """
        target = target_plan_type if target_plan_type in ("daily", "weekly") else "weekly"
        logger.info("[%s] switch_plan_type → %s", session_id, target)

        if target == "weekly":
            ack = "Sure! I'm setting aside the daily plan and starting a fresh weekly plan for you."
            self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
            return self._handle_weekly(session_id, message, "weekly_plan", is_refinement=False)

        ack = "Sure! I'm setting aside the weekly plan and starting a fresh daily plan for you."
        self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
        return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _tag_last_message(self, session_id: str, intent: str, plan_id: Optional[str]) -> None:
        """Back-fill intent and plan_id onto the last assistant message.

        The UI uses these tags to associate chat bubbles with canvas versions.
        """
        session = self.session_service.get_session(session_id)
        if not session or not session.conversation:
            return
        last = session.conversation[-1]
        if last.role == "assistant":
            last.intent = intent
            last.plan_id = plan_id
