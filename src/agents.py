import itertools, random
import json
import re
from pathlib import Path
import logging

from colorama import Fore, Style
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os
from backend.groq import GROQ_CHAT
from prompts import (
    ROUTER_SYSTEM_INSTRUCTIONS,
    GRADER_SYSTEM_INSTRUCTIONS,
    GRADER_USER_INSTRUCTIONS,
    MEAL_DIVERSITY_SYSTEM_INSTRUCTIONS,
    GUIDELINE_ADHERENCE_SYSTEM_INSTRUCTIONS,
    SIMPLE_QUESTIONS_GENERATOR_SYSTEM_INSTRUCTIONS,
    SIMPLE_QUESTIONS_GENERATOR_USER_INSTRUCTIONS,
    SIMPLE_QUESTIONS_RESPONDER_SYSTEM_INSTRUCTIONS,
    SIMPLE_QUESTIONS_RESPONDER_USER_INSTRUCTIONS,
    QUERY_REWRITER_SYSTEM_INSTRUCTIONS,
    QUERY_REWRITER_USER_INSTRUCTIONS,
    FEEDBACK_REWRITER_SYSTEM_INSTRUCTIONS,
    FEEDBACK_REWRITER_USER_INSTRUCTIONS,
    QUERY_RECONCILER_SYSTEM_INSTRUCTIONS,
    QUERY_RECONCILER_USER_INSTRUCTIONS,
    QUERY_CHECKER_SYSTEM_INSTRUCTIONS,
    QUERY_CHECKER_USER_INSTRUCTIONS,
    USER_PROFILE_CHECKER_SYSTEM_INSTRUCTIONS,
    USER_PROFILE_CHECKER_USER_INSTRUCTIONS,
    USER_INFO_COLLECTOR_SYSTEM_INSTRUCTIONS,
    USER_INFO_COLLECTOR_USER_INSTRUCTIONS,
    QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS,
    QUERY_REFORMULATOR_USER_INSTRUCTIONS,
    ORCHESTRATOR_SYSTEM_INSTRUCTIONS,
    ORCHESTRATOR_USER_INSTRUCTIONS,
)
from schemas import (
    ScoringSchema,
    QuestionList,
    FoodChatResponseEvaluatorSchema,
    QueryRewriterSchema,
    QueryItemExtractorSchema,
    QueryReconcilerSchema,
    QueryCheckerSchema,
    UserProfileCheckerSchema,
    QueryReformulatorSchema,
    OrchestratorSchema,
)

logger = logging.getLogger(__name__)

PATH = Path(__file__).parent.resolve()

DEFAULT_MODEL = os.getenv("FOODCHAT_LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("FOODCHAT_LLM_TEMPERATURE", "0.0"))
CHATBOT_TEMPERATURE = float(os.getenv("FOODCHAT_CHATBOT_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("FOODCHAT_MAX_RETRIES", "3"))
MAX_PLANS_TO_SCORE = int(os.getenv("FOODCHAT_MAX_PLANS_TO_SCORE", "10"))

class QueryClassifier:

    def __init__(self, model: str = None, temperature: float = None):
        self.routing_llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format="json",
        )

    def router(self, query, max_retries: int = None):
        max_retries = max_retries or MAX_RETRIES
        question = [HumanMessage(content=query)]

        for attempt in range(max_retries):
            try:
                result = self.routing_llm.invoke(
                    [SystemMessage(content=ROUTER_SYSTEM_INSTRUCTIONS)] + question
                )
                routing = json.loads(result.content)
                if (
                    isinstance(routing, dict)
                    and "source" in routing
                    and routing["source"] in ["vectorstore", "chatbot"]
                ):
                    return routing
                else:
                    logger.warning(f"Unexpected routing format: {routing}")
                    return (
                        {"source": "vectorstore"}
                        if "vectorstore" in str(routing)
                        else {"source": "chatbot"}
                    )
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} - Invalid response: {result.content if result else 'No response'}"
                )

        logger.error(
            f"Failed to route query after {max_retries} attempts, defaulting to vectorstore"
        )
        return {"source": "vectorstore"}

class DocumentGrader:
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_plans_to_score: int = None,
    ):
        self.grader = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=ScoringSchema.model_json_schema(),
        )
        self.max_plans_to_score = max_plans_to_score or MAX_PLANS_TO_SCORE
        
    def grading_daily_plans(self, query, documents, user_profile):
        logger.info("Grading daily plans")
        logger.info(f"Len of documents: {len(documents)}")
        logger.info(f"User profile: {user_profile}")


        breakfasts_options = documents["breakfast"]
        lunch_options = documents["lunch"]
        dinner_options = documents["dinner"]

        daily_plans = list(
            itertools.product(breakfasts_options, lunch_options, dinner_options)
        )

        logger.info(f"Number of possible daily plans: {len(daily_plans)}")

        if not daily_plans:
            logger.warning("No possible daily plans — at least one course has no recipes")
            return []

        random_daily_plans = random.sample(daily_plans, min(len(daily_plans), self.max_plans_to_score))
        scored_daily_plans = []
        for daily_plan in random_daily_plans:
            meal_plan = ""
            courses = ["breakfast", "lunch", "dinner"]
            for course in range(len(daily_plan)):
                i, title, ingredients, directions = daily_plan[course]
                c = courses[course]
                meal_plan += f"{c}: {title}\nIngredients: {ingredients}\nDirections: {directions}\n\n"
            #print('Constructed meal plan: ', meal_plan)
            question = [
                HumanMessage(
                    content=GRADER_USER_INSTRUCTIONS.format(
                        query=query,
                        daily_plan=meal_plan,
                        preferences=",".join(user_profile.get("preferences", [])),
                        feedback_history="",
                    )
                )
            ]

            score = self.grader.invoke(
                [SystemMessage(content=GRADER_SYSTEM_INSTRUCTIONS)] + question
            )
            score = json.loads(score.content)
            scored_daily_plans.append([daily_plan, score])

        scored_daily_plans = sorted(
            scored_daily_plans, key=lambda x: x[1]["score"], reverse=True
        )[:3]
        logger.info(f"Top scored daily plans: {scored_daily_plans}")

        return scored_daily_plans[:1]  # return the best daily plan

class MealDiversityGrader:
    def __init__(self, model: str = None, temperature: float = None):
        self.client = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=ScoringSchema.model_json_schema(),
        )

    def score(self, plan_text: str) -> dict:
        messages = [HumanMessage(content=plan_text)]
        result = self.client.invoke([SystemMessage(content=MEAL_DIVERSITY_SYSTEM_INSTRUCTIONS)] + messages)
        try:
            return json.loads(result.content)
        except Exception:
            return {"reasoning": "Could not parse diversity score", "score": 0}

class GuidelineAdherenceGrader:
    def __init__(self, model: str = None, temperature: float = None):
        self.client = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=ScoringSchema.model_json_schema(),
        )

    def score(self, plan_text: str, guidelines_text: str) -> dict:
        content = f"GUIDELINES:\n{guidelines_text}\n\nMEAL PLAN:\n{plan_text}"
        messages = [HumanMessage(content=content)]
        result = self.client.invoke([SystemMessage(content=GUIDELINE_ADHERENCE_SYSTEM_INSTRUCTIONS)] + messages)
        try:
            return json.loads(result.content)
        except Exception:
            return {"reasoning": "Could not parse guideline adherence score", "score": 0}

    # def grading_docs(self, query, documents, user_profile) :
    #     filtered_docs = []
    #     for doc  in documents :
    #         question = [HumanMessage(content=GRADER_USER_INSTRUCTIONS.format(
    #             query=query,
    #             doc=doc,
    #             diet=",".join(user_profile['diet']),
    #             allergies=",".join(user_profile["allergies"]),
    #             preferences=",".join(user_profile["preferences"]),
    #             feedback_history=",".join(user_profile["history"])))]

    #         score = self.grader.invoke([SystemMessage(content=GRADER_SYSTEM_INSTRUCTIONS)] + question)
    #         score = json.loads(score.content)
    #         print(Fore.RED + "SCORING: " + str(score.get("diet")['score']) + " " + str(score.get("allergies")['score']) + Style.RESET_ALL )
    #         if int(score.get("diet")['score']) == 1 and int(score.get("allergies")['score']) == 1:
    #             filtered_docs.append(doc)
    #         else:
    #             print(Fore.RED + "ELIMINATED DOCUMENT: " + str(doc.page_content[:500]) + "..." + Style.RESET_ALL)
    #             print(Fore.YELLOW + "DIET COMMENT: " + str(score["diet"].get('explanation')) + Style.RESET_ALL)
    #             print(Fore.YELLOW + "ALLERGIES COMMENT: " + str(score["allergies"].get('explanation')) + Style.RESET_ALL)

    #         self.record_scoring(doc.page_content, score)
    #         # if score.get('relevance') == "yes" :
    #         #     filtered_docs.append(doc)
    #         # else :
    #         #     print(Fore.RED + "IRRELEVANT DOCUMENT: " + str(doc.page_content[:1000]) + "..." + Style.RESET_ALL)
    #         #     print(Fore.YELLOW + "COMMENT: " + str(score.get('comment')) + Style.RESET_ALL)
    #     return filtered_docs

    # def record_scoring(self, retrieved_doc, scoring):
    #     # print("DEBUG: ", retrieved_doc)
    #     # print("REDEBUG/ ", type(retrieved_doc))
    #     match = re.search(r"Title:\s*(.*?)\s*Ingredients:", retrieved_doc, re.DOTALL)
    #     title = match.group(1).strip() if match else "Unkown Recipe"

    #     with open(self.output_file, "r", encoding='utf-8') as f:
    #         data=json.load(f)

    #     data[title] = scoring

    #     with open(self.output_file, "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)


class SimpleChatBot:
    def __init__(self, model: str = None, temperature: float = None):
        self.chatbot = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else CHATBOT_TEMPERATURE,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def chat(self, query):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        system = SystemMessage(content=(
            "You are FoodChat, a friendly and knowledgeable meal-planning assistant. "
            "You help users plan their meals, discover recipes, understand nutrition, "
            "and make healthy food choices. Keep your answers focused on food, cooking, "
            "nutrition, and meal planning. If asked something completely unrelated to food, "
            "politely steer the conversation back to your area of expertise."
        ))
        messages = [system] + chat_history + [HumanMessage(content=query)]
        chatbot_message = self.chatbot.invoke(messages)
        response = chatbot_message.content
        self.memory.save_context({"input": query}, {"output": response})
        return response



class FoodChatResponseEvaluator:
    def __init__(self, model: str = None, temperature: float = None):
        model = model or DEFAULT_MODEL
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.question_generator = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=QuestionList.model_json_schema(),
        )
        self.question_responder = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=FoodChatResponseEvaluatorSchema.model_json_schema(),
        )

    def generate_questions(self, query):
        question = [
            HumanMessage(
                content=SIMPLE_QUESTIONS_GENERATOR_USER_INSTRUCTIONS.format(query=query)
            )
        ]
        question_list = self.question_generator.invoke(
            [SystemMessage(content=SIMPLE_QUESTIONS_GENERATOR_SYSTEM_INSTRUCTIONS)]
            + question
        )
        question_list = json.loads(question_list.content)["questions"]
        return question_list

    def evaluate(self, query, response):
        question_list = self.generate_questions(query)
        logger.debug(f"Generated questions: {question_list}")
        question = [
            HumanMessage(
                content=SIMPLE_QUESTIONS_RESPONDER_USER_INSTRUCTIONS.format(
                    question_list=question_list, response=response
                )
            )
        ]
        eval_result = self.question_responder.invoke(
            [SystemMessage(content=SIMPLE_QUESTIONS_RESPONDER_SYSTEM_INSTRUCTIONS)]
            + question
        )
        eval_result = json.loads(eval_result.content)
        logger.debug(f"Evaluation result: {eval_result}")

        model_questions = []
        for i in range(len(eval_result["evaluation"])):
            if eval_result["evaluation"][i].get("answer") != question_list[i].get(
                "expected"
            ):
                model_questions.append(eval_result["evaluation"][i].get("question"))

        logger.debug(f"Failed questions: {model_questions}")
        return model_questions


class QueryRewriter:
    def __init__(self, model: str = None, temperature: float = None):
        self.query_rewriter = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=QueryRewriterSchema.model_json_schema(),
        )

    def rewrite(self, query, question_list):
        question = [
            HumanMessage(
                content=QUERY_REWRITER_USER_INSTRUCTIONS.format(
                    query=query, questions=question_list
                )
            )
        ]
        new_query = self.query_rewriter.invoke(
            [SystemMessage(content=QUERY_REWRITER_SYSTEM_INSTRUCTIONS)] + question
        )
        new_query = json.loads(new_query.content)["query"]
        logger.debug(f"Rewritten query: {new_query}")
        return new_query


class FeedBackRewriter:
    def __init__(self, model: str = None, temperature: float = None):
        self.feedback_rewriter = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        )

    def rewrite(self, feedback, response):
        question = [
            HumanMessage(
                content=FEEDBACK_REWRITER_USER_INSTRUCTIONS.format(
                    response=response, feedback=feedback
                )
            )
        ]
        refined_feedback = self.feedback_rewriter.invoke(
            [SystemMessage(content=FEEDBACK_REWRITER_SYSTEM_INSTRUCTIONS)] + question
        )
        logger.debug(f"Refined feedback: {refined_feedback.content}")
        return refined_feedback.content


class QueryReconciler:
    def __init__(self, model: str = None, temperature: float = None):
        model = model or DEFAULT_MODEL
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.query_reconciler = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=QueryReconcilerSchema.model_json_schema(),
        )

    def reconcile(self, query, user_profile):
        question = [
            HumanMessage(
                content=QUERY_RECONCILER_USER_INSTRUCTIONS.format(
                    query=query,
                    diet=user_profile.get("diet", []),
                    allergies=user_profile.get("allergies", []),
                )
            )
        ]
        reconciliation_details = self.query_reconciler.invoke(
            [SystemMessage(content=QUERY_RECONCILER_SYSTEM_INSTRUCTIONS)] + question
        )
        return json.loads(reconciliation_details.content)


class RAGReadyPreparator:

    def __init__(self, model: str = None, temperature: float = None):
        model = model or DEFAULT_MODEL
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.query_checker = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=QueryCheckerSchema.model_json_schema(),
        )
        self.user_profile_checker = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=UserProfileCheckerSchema.model_json_schema(),
        )
        self.user_info_collector = GROQ_CHAT.get_client(
            model=model, temperature=temperature
        )
        self.query_reformulator = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=QueryReformulatorSchema.model_json_schema(),
        )
        self.query_reconciler = QueryReconciler(model=model, temperature=temperature)

    def user_profile_check(self, user_profile):
        preferences = user_profile.get("preferences", [])
        if len(preferences) == 0:
            return {"response": "NO", "suggestions": []}

        question = [
            HumanMessage(
                content=USER_PROFILE_CHECKER_USER_INSTRUCTIONS.format(
                    preferences=preferences, feedback_history=[]
                )
            )
        ]

        response = self.user_profile_checker.invoke(
            [SystemMessage(content=USER_PROFILE_CHECKER_SYSTEM_INSTRUCTIONS)] + question
        )
        response = json.loads(response.content)
        return response

    def collect_user_info(self, query, suggestions, user_info):
        conversation_history = []

        for suggestion in suggestions:
            question = [
                HumanMessage(
                    content=USER_INFO_COLLECTOR_USER_INSTRUCTIONS.format(
                        user_query=query,
                        preferences=user_info.get("preferences", []),
                        feedback_history=[],
                        suggestions=suggestion,
                    )
                )
            ]
            response = self.user_info_collector.invoke(
                [SystemMessage(content=USER_INFO_COLLECTOR_SYSTEM_INSTRUCTIONS)]
                + question
                + conversation_history
            )

            ai_question = response.content
            conversation_history.append(ai_question)

            user_reply = yield ai_question

            if user_reply is None:
                break

            conversation_history.append(user_reply)

        logger.debug(f"Conversation history: {conversation_history}")
        return conversation_history

    def reformulate_query(self, collected_info, user_profile, query):
        question = [
            HumanMessage(
                content=QUERY_REFORMULATOR_USER_INSTRUCTIONS.format(
                    original_query=query,
                    preferences=user_profile.get("preferences", []),
                    feedback_history=[],
                    collected_info=collected_info,
                )
            )
        ]
        response = self.query_reformulator.invoke(
            [SystemMessage(content=QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS)] + question
        )
        reformulated_query = json.loads(response.content)["reformulated_query"]
        logger.info(f"Reformulated query: {reformulated_query}")
        return reformulated_query

    def query_check(self, query, user_profile):
        """Checks if the query is specific enough and reconciles with user profile.
        
        This is a generator function. It yields clarification questions or warnings if needed.
        Returns the final (reformulated) query upon completion.
        """
        # Step 1: Reconciliation check (missing info and dietary conflicts)
        reconciliation = self.query_reconciler.reconcile(query, user_profile)
        logger.info(f"Reconciliation result: {reconciliation}")

        collected_responses = []
        if reconciliation.get("needs_clarification"):
            # Handle dietary conflict first
            if reconciliation.get("has_dietary_conflict"):
                warning = reconciliation.get("conflict_explanation")
                user_reply = yield warning
                collected_responses.append(f"User response to dietary conflict warning: {user_reply}")
            
            # Handle missing info
            missing = reconciliation.get("missing_info", [])
            if missing:
                info_responses = yield from self.collect_user_info(
                    query, missing, user_profile
                )
                collected_responses.extend(info_responses)
            
            if collected_responses:
                return self.reformulate_query(collected_responses, user_profile, query)

        # Step 2: Basic specificity check (fallback)
        question = [
            HumanMessage(
                content=QUERY_CHECKER_USER_INSTRUCTIONS.format(user_query=query)
            )
        ]
        response = self.query_checker.invoke(
            [SystemMessage(content=QUERY_CHECKER_SYSTEM_INSTRUCTIONS)] + question
        )
        response_json = json.loads(response.content)
        check_result = response_json.get("response", "YES")

        if check_result == "NO":
            logger.info("Query is too general for RAG, checking profile and collecting more info")
            user_profile_check = self.user_profile_check(user_profile)

            if user_profile_check.get("response") == "NO":
                suggestions = user_profile_check.get("suggestions", [])
                logger.info(f"Need to collect info on: {suggestions}")
                collected_info = yield from self.collect_user_info(
                    query, suggestions, user_profile
                )
                return self.reformulate_query(collected_info, user_profile, query)
            else:
                return self.reformulate_query(None, user_profile, query)
        else:
            # Mark as generator even if no yield occurred
            if False: yield
            return query

        # return response


class OrchestratorAgent:
    """Routes a user message to the correct sub-service based on conversational intent."""

    def __init__(self, model: str = None, temperature: float = None):
        self.llm = GROQ_CHAT.get_client(
            model=model or DEFAULT_MODEL,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            format=OrchestratorSchema.model_json_schema(),
        )

    def classify(self, message: str, history: list[dict]) -> str:
        """Return one of: 'daily_plan', 'weekly_plan', 'refine_plan', 'chat'.

        history: list of {"role": "user"|"assistant", "content": str} dicts,
                 most recent last, max last 6 turns passed in.
        """
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content'][:300]}"
            for turn in history[-6:]
        ) or "(no prior conversation)"

        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_INSTRUCTIONS),
            HumanMessage(
                content=ORCHESTRATOR_USER_INSTRUCTIONS.format(
                    history=history_text,
                    message=message,
                )
            ),
        ]

        for attempt in range(MAX_RETRIES):
            try:
                result = self.llm.invoke(messages)
                parsed = json.loads(result.content)
                intent = parsed.get("intent", "chat")
                if intent in ("daily_plan", "weekly_plan", "refine_plan", "chat"):
                    logger.info(f"Orchestrator intent: {intent} — {parsed.get('reasoning', '')}")
                    return intent
            except Exception as e:
                logger.warning(f"Orchestrator attempt {attempt + 1} failed: {e}")

        logger.error("Orchestrator failed after retries, defaulting to chat")
        return "chat"


# class QueryEnhancer:
#     def __init__(self):
#         self.enhancer = ChatGroq(model = 'llama3.2:latest', temperature=0.0)
#         self.interrogater = ChatGroq(modle = 'llama3.2:latest', temperature=0.0)

#     # def interrogate(self, query, user_profile):

#     def create_query(self, query, user_profile):

#         question = [HumanMessage(content = QUERY_ENHANCER_USER_INSTRUCTIONS.format(diet = user_profile['diet'],
#                                                                                    preference = user_profile['preference'],
#                                                                                    history = user_profile("feeadback")))]

#         new_query = self.enhancer.invoke([SystemMessage(content = QUERY_ENHANCER_SYSTEM_INSTRUCTIONS)] + question)