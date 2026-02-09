from typing import Optional, Tuple

from src.agents import QueryClassifier, RAGReadyPreparator, SimpleChatBot
from services.session_service import SessionService
from services import profile_service
from models.session import MealPlan


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

    def process_message(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Process a message and return (response, needs_clarification, meal_plan).

        Args:
            session_id: The session ID
            message: The user's message

        Returns:
            Tuple of (response_text, needs_clarification_flag, optional_meal_plan)
        """
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Add user message to history
        self.session_service.add_message(session_id, "user", message)

        # Handle clarification state
        if session.state == "clarifying":
            return self._handle_clarification(session_id, message)

        # Route the query
        routing = self.classifier.router(message)

        if routing["source"] == "vectorstore":
            return self._handle_rag_query(session_id, message)
        else:
            response = self.chatbot.chat(message)
            self.session_service.add_message(session_id, "assistant", response)
            return response, False, None

    def _handle_rag_query(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Handle RAG-based query with potential clarification."""
        session = self.session_service.get_session(session_id)

        # Run pre-clarification chain with profile directly injected
        pending_rag_data = self.rag_chain_part1.invoke(
            {
                "question": message,
                "user_id": session.member_id,
                "user_profile_data": session.user_profile,
            }
        )

        # Start clarification generator
        generator = self.rag_preparator.query_check(
            pending_rag_data["question"], pending_rag_data["modified_user_data"]
        )

        try:
            # Try to get first clarification question
            first_question = next(generator)
            self.session_service.set_clarification_state(
                session_id, generator, pending_rag_data
            )
            self.session_service.add_message(session_id, "assistant", first_question)
            return first_question, True, None

        except StopIteration as e:
            # No clarification needed, run post-clarification directly
            final_query = e.value or message
            return self._run_post_clarification(
                session_id, final_query, pending_rag_data
            )

    def _handle_clarification(
        self, session_id: str, message: str
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Handle user response during clarification flow."""
        session = self.session_service.get_session(session_id)
        generator = session.clarification_generator

        if not generator:
            # No generator, reset state and process as new message
            self.session_service.clear_clarification_state(session_id)
            return self.process_message(session_id, message)

        try:
            # Send user response to generator
            next_question = generator.send(message)
            self.session_service.add_message(session_id, "assistant", next_question)
            return next_question, True, None

        except StopIteration as e:
            # Clarification complete
            final_query = e.value or message
            return self._run_post_clarification(
                session_id, final_query, session.pending_rag_data
            )

    def _run_post_clarification(
        self, session_id: str, final_query: str, pending_data: dict
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Execute post-clarification chain and format response."""
        self.session_service.clear_clarification_state(session_id)

        chain_input = {**pending_data, "reformulated_query": final_query}
        response = self.rag_chain_part2.invoke(chain_input)

        # Extract meal plan and reasoning from response
        docs = response.get("docs") or []
        if not docs:
            msg = ("I'm sorry, I couldn't find enough recipes to build a "
                   "complete meal plan matching your preferences. "
                   "Could you try adjusting your requirements?")
            self.session_service.add_message(session_id, "assistant", msg)
            return msg, False, None

        raw_plan = docs[0][0]
        reasoning = docs[0][1]["reasoning"]

        # Store meal plan in session
        meal_plan = self.session_service.add_meal_plan(
            session_id, raw_plan, reasoning
        )

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
        parts.append(f"\n{reasoning}")
        formatted = "\n".join(parts)

        self.session_service.add_message(session_id, "assistant", formatted)

        return formatted, False, meal_plan
