import logging
from typing import Optional, Tuple

from src.agents import QueryClassifier, RAGReadyPreparator, SimpleChatBot, MealDiversityGrader, GuidelineAdherenceGrader
from services.session_service import SessionService
from services import profile_service
from models.session import MealPlan

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates chat interactions and RAG chain flow."""

    def __init__(self, foodchat, config, session_service: SessionService):
        self.foodchat = foodchat
        self.config = config
        self.session_service = session_service
        self.rag_preparator = RAGReadyPreparator()
        self.rag_chain_part1 = foodchat.create_pre_clarification_chain(config)
        self.rag_chain_part2 = foodchat.create_post_clarification_chain()
        self.chatbot = SimpleChatBot()
        self.classifier = QueryClassifier()
        logger.info("ChatService initialized.")

    def process_message(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        logger.info("[%s] Incoming message (state=%s): %.120s", session_id, session.state, message)
        self.session_service.add_message(session_id, "user", message)

        if session.state == "clarifying":
            logger.info("[%s] Resuming clarification flow.", session_id)
            return self._handle_clarification(session_id, message)

        routing = self.classifier.router(message)
        logger.info("[%s] Query classified → source=%s", session_id, routing.get("source"))

        if routing["source"] == "vectorstore":
            return self._handle_rag_query(session_id, message)
        else:
            logger.info("[%s] Routing to chatbot (non-RAG).", session_id)
            response = self.chatbot.chat(message)
            self.session_service.add_message(session_id, "assistant", response)
            return response, False, None

    def _handle_rag_query(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        session = self.session_service.get_session(session_id)
        logger.info("[%s] Running pre-clarification RAG chain.", session_id)

        pending_rag_data = self.rag_chain_part1.invoke(
            {
                "question": message,
                "user_id": session.member_id,
                "user_profile_data": session.user_profile,
            }
        )
        logger.debug("[%s] Pre-clarification chain output keys: %s", session_id, list(pending_rag_data.keys()))

        generator = self.rag_preparator.query_check(
            pending_rag_data.get("question", message),
            pending_rag_data.get("modified_user_data", session.user_profile)
        )

        try:
            first_question = next(generator)
            logger.info("[%s] Clarification needed — entering clarification state.", session_id)
            self.session_service.set_clarification_state(session_id, generator, pending_rag_data)
            self.session_service.add_message(session_id, "assistant", first_question)
            return first_question, True, None

        except StopIteration as e:
            final_query = e.value or pending_rag_data.get("question", message)
            logger.info("[%s] No clarification needed — proceeding to post-clarification chain.", session_id)
            return self._run_post_clarification(session_id, final_query, pending_rag_data)

        except Exception as ex:
            logger.error("[%s] Error in clarification generator: %s", session_id, ex, exc_info=True)
            return self._run_post_clarification(session_id, message, pending_rag_data)

    def _handle_clarification(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        session = self.session_service.get_session(session_id)
        generator = session.clarification_generator

        if not generator:
            logger.warning("[%s] Clarification state set but generator is missing — resetting.", session_id)
            self.session_service.clear_clarification_state(session_id)
            return self.process_message(session_id, message)

        try:
            next_question = generator.send(message)
            logger.info("[%s] Clarification continuing — next question sent.", session_id)
            self.session_service.add_message(session_id, "assistant", next_question)
            return next_question, True, None

        except StopIteration as e:
            final_query = e.value or message
            logger.info("[%s] Clarification complete — final query: %.120s", session_id, final_query)
            return self._run_post_clarification(session_id, final_query, session.pending_rag_data)

    def _run_post_clarification(
        self, session_id: str, final_query: str, pending_data: dict
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        logger.info("[%s] Running post-clarification chain (final_query=%.120s).", session_id, final_query)
        self.session_service.clear_clarification_state(session_id)

        chain_input = {**pending_data, "reformulated_query": final_query}
        response = self.rag_chain_part2.invoke(chain_input)

        # Extract meal plan and reasoning from response
        docs = response.get("docs") or []
        logger.info("[%s] Post-clarification chain returned %d candidate plan(s).", session_id, len(docs))
        if not docs:
            logger.warning("[%s] No candidate recipes found — returning empty plan response.", session_id)
            msg = ("I'm sorry, I couldn't find enough recipes to build a "
                   "complete meal plan matching your preferences. "
                   "Could you try adjusting your requirements?")
            self.session_service.add_message(session_id, "assistant", msg)
            return msg, False, None

        raw_plan = docs[0][0]
        llm_reasoning = docs[0][1].get("reasoning", "")
        llm_score = int(docs[0][1].get("score", 0))
        logger.info("[%s] Top plan selected (llm_score=%d).", session_id, llm_score)

        # Build a simple plan text for LLM scoring and FVS
        def _course_to_text(name: str, data) -> str:
            try:
                rid, title, ingredients, directions = data
            except Exception:
                title = ingredients = directions = ""
            return f"{name}: {title}\nIngredients: {ingredients}\nDirections: {directions}\n"

        plan_text = "\n".join([
            _course_to_text("Breakfast", raw_plan[0] if len(raw_plan) > 0 else []),
            _course_to_text("Lunch", raw_plan[1] if len(raw_plan) > 1 else []),
            _course_to_text("Dinner", raw_plan[2] if len(raw_plan) > 2 else []),
        ])

        # Compute Food Variety Score (unique ingredients across meals)
        import re
        def _extract_ingredients(ing_str: str) -> list[str]:
            if not isinstance(ing_str, str):
                return []
            # Split by comma/semicolon/newlines and dashes
            parts = re.split(r"[\n,;•\-]+", ing_str)
            cleaned = []
            for p in parts:
                t = p.strip().lower()
                t = re.sub(r"\([^\)]*\)", "", t)  # remove parentheses
                t = re.sub(r"[^a-zA-Z\s]", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                if t:
                    cleaned.append(t)
            return cleaned

        def _unique_variety_count(raw_plan_tuple: tuple) -> tuple[int, str]:
            items = []
            for idx in range(min(3, len(raw_plan_tuple))):
                try:
                    _, _, ingredients, _ = raw_plan_tuple[idx]
                    items.extend(_extract_ingredients(ingredients))
                except Exception:
                    pass
            unique_items = sorted(set(items))
            reasoning = f"Unique food items across meals: {len(unique_items)} (e.g., {', '.join(unique_items[:8])}{'...' if len(unique_items)>8 else ''})"
            return len(unique_items), reasoning

        fvs_count, fvs_reasoning = _unique_variety_count(raw_plan)
        logger.info("[%s] FVS score: %d unique ingredients.", session_id, fvs_count)

        logger.info("[%s] Running LLM diversity grader.", session_id)
        diversity_grader = MealDiversityGrader()
        diversity_res = diversity_grader.score(plan_text)
        diversity_score = int(diversity_res.get("score", 0))
        diversity_reason = str(diversity_res.get("reasoning", ""))
        logger.info("[%s] Diversity score: %d.", session_id, diversity_score)

        logger.info("[%s] Running guideline adherence grader.", session_id)
        try:
            from pathlib import Path as _Path
            guidelines_path = _Path(__file__).resolve().parents[2] / "belgium_dietary_guidelines_augmentation.cypher"
            with open(guidelines_path, "r", encoding="utf-8") as f:
                guidelines_text = f.read()
        except Exception as e:
            logger.warning("[%s] Could not load guidelines file: %s", session_id, e)
            guidelines_text = ""
        guidelines_grader = GuidelineAdherenceGrader()
        guidelines_res = guidelines_grader.score(plan_text, guidelines_text)
        guideline_score = int(guidelines_res.get("score", 0))
        guideline_reason = str(guidelines_res.get("reasoning", ""))
        logger.info("[%s] Guideline adherence score: %d.", session_id, guideline_score)

        metrics = {
            "llm_score": llm_score,
            "llm_reasoning": llm_reasoning,
            "fvs_count": fvs_count,
            "fvs_reasoning": fvs_reasoning,
            "diversity_llm_score": diversity_score,
            "diversity_llm_reasoning": diversity_reason,
            "guideline_adherence_score": guideline_score,
            "guideline_adherence_reasoning": guideline_reason,
        }

        meal_plan = self.session_service.add_meal_plan(
            session_id, raw_plan, llm_reasoning, metrics
        )
        logger.info("[%s] Meal plan %s stored (scores: llm=%d, fvs=%d, diversity=%d, guidelines=%d).",
                    session_id, meal_plan.id, llm_score, fvs_count, diversity_score, guideline_score)

        # Format a human-readable text response
        course_names = ["Breakfast", "Lunch", "Dinner"]
        courses = [meal_plan.breakfast, meal_plan.lunch, meal_plan.dinner]
        parts = ["Here is your meal plan for today:\n"]
        for name, course in zip(course_names, courses):
            if course.recipe_id:
                parts.append(
                    f"**{name}: {course.title}**\n"
                    f"Ingredients: {course.ingredients}\n"
                    f"Directions: {course.directions}\n"
                )
        parts.append(f"\n{llm_reasoning}")
        formatted = "\n".join(parts)

        self.session_service.add_message(session_id, "assistant", formatted)

        return formatted, False, meal_plan
