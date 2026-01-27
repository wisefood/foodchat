import itertools
import json
import re
from pathlib import Path

from colorama import Fore, Style
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from prompts import *
from schemas import *

PATH = Path(__file__).parent.resolve()
RECIPE_SCORING = PATH / 'retrieved_recipe_scoring.json'


class QueryClassifier: 

    def router(self, query): 
        question = [HumanMessage(content=query)]
        routing_llm = ChatOllama(model="llama3.2", temperature=0, format='json')

        while True : 
            try:
                result = routing_llm.invoke([SystemMessage(content=ROUTER_SYSTEM_INSTRUCTIONS)] + question)
                routing = json.loads(result.content)
                if isinstance(routing, dict) and "source" in routing and routing["source"] in ["vectorstore", "chatbot"]:
                    return routing
                else : 
                    print("Bad Format Handling: ", routing)
                    return {"source" : "vectorstore"} if "vectorstore" in routing else {"source" : "chatbot"}
            except (json.JSONDecodeError, AttributeError, TypeError) as e: 
                print(f"Invalid response format from router: {result.content if result else 'No response'}")

class DocumentGrader:
    def __init__(self, model="llama3.2", temperature=0.0): #mistral:7b-instruct-q8_0, atla/selene-mini:fp_16
        self.grader = ChatOllama(model=model, temperature=temperature, format=ScoringSchema.model_json_schema())
        # self.output_file = RECIPE_SCORING
        # with open(self.output_file, "w", encoding='utf-8') as f: 
        #     json.dump({}, f, indent=4)
    
    def grading_daily_plans(self, query, documents, user_profile):

        print("GRADING DAILY PLANS")

        # breakfasts_data = {'content' : [doc.page_content for doc in documents['breakfast']], 
        #                    'similarity_score' : [doc.metadata.get('score', 'N/A') for doc in documents['breakfast']]}

        breakfasts_options = [doc.page_content for doc in documents['breakfast']]
        lunch_options =  [doc.page_content for doc in documents['lunch']]
        dinner_options = [doc.page_content for doc in documents['dinner']]

        # print("BREAKFAST: ", documents['breakfast'])  # check existing scoring
        # print('LUNCH: ', lunch_options)
        # print("DINNER: ", dinner_options)
        daily_plans = list(itertools.product(breakfasts_options, lunch_options, dinner_options))

        print("Number of possible daily plans: ", len(daily_plans))

        scored_daily_plans = []
        for daily_plan in daily_plans[:5]: #todo: remove 10 for a more powerful machine
            question = [HumanMessage(content=GRADER_USER_INSTRUCTIONS.format(
                query=query, 
                daily_plan =daily_plan,
                preferences=",".join(user_profile['preferences']),
                feedback_history=""
            ))]
            #feedback_history=','.join(user_profile['history'])

            score = self.grader.invoke([SystemMessage(content = GRADER_SYSTEM_INSTRUCTIONS)] + question)
            score = json.loads(score.content)

            scored_daily_plans.append([daily_plan, score])
        
        # print("TOP RESULTS FOR DAILY PLAN: ", scored_daily_plans[:3])
        scored_daily_plans = sorted(scored_daily_plans, key = lambda x: x[1]['score'], reverse=True)[:3]

        print("TOP RESULTS FOR DAILY PLAN: ", scored_daily_plans)

        # for elem in scored_daily_plans: 
        #     for corresponding in documents['breakfast']

        return scored_daily_plans[:1] # return top 3 daily plans

        
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
    def __init__(self, model='llama3.2:latest', temperature=0.7):
        self.chatbot = ChatOllama(model = model, temperature = temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    def chat(self, query) : 

        chat_history = self.memory.load_memory_variables({})["chat_history"]
        messages = chat_history + [HumanMessage(content=query)]
        chatbot_message = self.chatbot.invoke(messages)
        response = chatbot_message.content
        self.memory.save_context({"input" : query}, {"output": response})
        return response



class FoodChatResponseEvaluator: 
    def __init__(self) : 
        self.question_generator = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0,format = QuestionList.model_json_schema())
        self.question_responder = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0,format = FoodChatResponseEvaluatorSchema.model_json_schema())
    
    def generate_questions(self, query) : 
        question = [HumanMessage(content = SIMPLE_QUESTIONS_GENERATOR_USER_INSTRUCTIONS.format(query = query))]
        question_list = self.question_generator.invoke([SystemMessage(content=SIMPLE_QUESTIONS_GENERATOR_SYSTEM_INSTRUCTIONS)] + question) # if the initial query is too simple then just use the initial query directly
        question_list = json.loads(question_list.content)['questions']
        return question_list 
    
    def evaluate(self, query, response):
        question_list = self.generate_questions(query)
        print(Fore.YELLOW + "GENERATED QUESTIONS: " + str(question_list) + Style.RESET_ALL)
        question = [HumanMessage(content= SIMPLE_QUESTIONS_RESPONDER_USER_INSTRUCTIONS.format(question_list = question_list, response = response ))]
        eval_result = self.question_responder.invoke([SystemMessage(content=SIMPLE_QUESTIONS_RESPONDER_SYSTEM_INSTRUCTIONS)] + question)
        eval_result = json.loads(eval_result.content)
        print(Fore.CYAN + "INTERMEDIARY ANSWER: " + str(response) + Style.RESET_ALL)
        print(Fore.RED + "EVAL RESULT: " + str(eval_result) + Style.RESET_ALL)
        model_questions = []

        for i in range(len(eval_result["evaluation"])):
            if eval_result["evaluation"][i].get("answer") != question_list[i].get("expected"):
                model_questions.append(eval_result["evaluation"][i].get('question'))
        print(Fore.YELLOW + "FAILED QUESTIONS" + str(model_questions) + Style.RESET_ALL)
        return model_questions


class QueryRewriter: 
    def __init__(self): 
      self.query_rewriter = ChatOllama(model="llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0, format = QueryRewriterSchema.model_json_schema())
    
    def rewrite(self, query, question_list) :
        question = [HumanMessage(content = QUERY_REWRITER_USER_INSTRUCTIONS.format(query=query, questions = question_list))]
        new_query = self.query_rewriter.invoke([SystemMessage(content=QUERY_REWRITER_SYSTEM_INSTRUCTIONS)] + question)
        new_query = json.loads(new_query.content)["query"]
        print(Fore.BLUE + "REWRITTEN QUERY: " + str(new_query) + Style.RESET_ALL)
        return new_query
    
class FeedBackRewriter: 
    def __init__(self) : 
        self.feedback_rewriter = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0)

    def rewrite(self, feedback, response):
        question = [HumanMessage(content = FEEDBACK_REWRITER_USER_INSTRUCTIONS.format(response = response, feedback = feedback))]
        refined_feedback = self.feedback_rewriter.invoke([SystemMessage(content = FEEDBACK_REWRITER_SYSTEM_INSTRUCTIONS)] + question)
        print(Fore.RED + "DEBUGGING LLM ANSWER : " + str(refined_feedback.content) + Style.RESET_ALL)
        refined_feedback = refined_feedback.content
        return refined_feedback


class QueryReconciler:
    def __init__(self):
        self.query_item_extractor = ChatOllama(model='llama3.2:latest', temperature=0.0, format = QueryItemExtractorSchema.model_json_schema())
        self.query_reconciler = ChatOllama(model="llama3.2", temperature = 0.0, format = QueryReconcilerSchema.model_json_schema())
    
    def extract_items(self, user_query) : 
        question = [HumanMessage(content= QUERY_ITEM_EXTRACTOR_USER_INSTRUCTIONS.format(query=user_query))]

        extracted_items = self.query_item_extractor.invoke([SystemMessage(content=QUERY_ITEM_EXTRACTOR_SYSTEM_INSTRUCTIONS)] + question)
        extracted_items = json.loads(extracted_items.content)
        print(Fore.YELLOW + "Extracted Items: " + str(extracted_items) + Style.RESET_ALL)

        return extracted_items

    def reconcile(self, query, user_profile) :
        extracted_items = self.extract_items(query)
        for item in extracted_items.values() : 
            question = [HumanMessage(content=QUERY_RECONCILER_USER_INSTRUCTIONS.format(item=item,
                                                                                     diet=user_profile['diet'],
                                                                                     allergies=user_profile["allergies"]))]
            conflict_details = self.query_reconciler.invoke([SystemMessage(content = QUERY_RECONCILER_SYSTEM_INSTRUCTIONS)] + question)
            conflict_details = json.loads(conflict_details.content)
            conflicting_elem = conflict_details.get('trigger')
            user_profile["diet"] = [elem for elem in user_profile["diet"] if elem != conflicting_elem]
            user_profile["allergies"] = [elem for elem in user_profile["allergies"] if elem != conflicting_elem]

            print(Fore.GREEN + "NEW USER PROFILE: " + str(user_profile) + Style.RESET_ALL)
        return user_profile


class RAGReadyPreparator:

    def __init__(self): 
        self.query_checker = ChatOllama(model='llama3.2:latest', temperature=0.0, format = QueryCheckerSchema.model_json_schema())
        self.user_profile_checker = ChatOllama(model='llama3.2:latest', temperature=0.0, format = UserProfileCheckerSchema.model_json_schema())
        self.user_info_collector = ChatOllama(model='llama3.2:latest', temperature=0.0)
        self.query_reformulator = ChatOllama(model='llama3.2:latest', temperature=0.0, format = QueryReformulatorSchema.model_json_schema())
    
    def user_profile_check(self, user_profile):
        preferences = user_profile['preferences']
        #feedback_hist = user_profile['history'] #todo: add feedback
        # if there are no preferences at all, then we can dirtectly skip and go to the next step
        if len(preferences) == 0 : 
            return {'responce' : 'NO'}

        question = [HumanMessage(content=USER_PROFILE_CHECKER_USER_INSTRUCTIONS.format(
            preferences = preferences,
            feedback_history = []
        ))]
        # feedback_history = feedback_hist #todo: add feedback to question above

        response = self.user_profile_checker.invoke([SystemMessage(content=USER_PROFILE_CHECKER_SYSTEM_INSTRUCTIONS )] + question)
        response = json.loads(response.content)

        return response
    
    def collect_user_info(self, query,  suggestions, user_info): 
        
        conversation_history = []

        for suggestion in  suggestions: 
            question = [HumanMessage(content=USER_INFO_COLLECTOR_USER_INSTRUCTIONS.format(
                        user_query=query,
                        prefrences=user_info["preferences"],
                        feedback_history=[],
                        suggestions=suggestion
                    ))] #feedback_history=user_info["history"],
            # print(Fore.RED + "HELLO" + Style.RESET_ALL)
            response = self.user_info_collector.invoke([
                SystemMessage(content=USER_INFO_COLLECTOR_SYSTEM_ISNTRUCTIONS)] + question + conversation_history)
            
            ai_question = response.content
            conversation_history.append(ai_question)

            user_reply = yield ai_question

            if user_reply is None:
                break

            # user_reply = get_user_input(prompt_message=response.content)
            conversation_history.append(HumanMessage(content = user_reply).content[0])
        print("Resulting Conversation History: ", conversation_history)
        return conversation_history

    def reformulate_query(self, collected_info, user_profile, query ): 

        question = [HumanMessage(content = QUERY_REFORMULATOR_USER_INSTRUCTIONS.format(
            original_query = query, 
            preferences = user_profile['preferences'],
            feedback_history = [],
            collected_info = collected_info
        ))] # feedback_history = user_profile['history'], #todo: add feedback
        response = self.query_reformulator.invoke([SystemMessage(content = QUERY_REFORMULATOR_SYSTEM_INSTRUCTIONS)] + question)
        reformulated_query = json.loads(response.content)['reformulated_query']
        print(Fore.RED + "New query for RAG: " + Fore.GREEN +  reformulated_query + Style.RESET_ALL)

        return reformulated_query

        

    def query_check(self, query, user_profile) : 
        question = [HumanMessage(content = QUERY_CHECKER_USER_INSTRUCTIONS.format(
            user_query = query
        ))]
        response = self.query_checker.invoke([SystemMessage(content = QUERY_CHECKER_SYSTEM_INSTRUCTIONS)] + question)
        response = json.loads(response.content)['response']

        if response == "NO" :
            print(Fore.RED + "The query is too general for RAG" + Style.RESET_ALL)
            user_profile_check = self.user_profile_check(user_profile)
            
            if user_profile_check['response'] == 'NO' :
                print(Fore.RED + "List of Subject to collect information: " +  str(user_profile_check['suggestions']) + Style.RESET_ALL)
                collected_info = yield from self.collect_user_info(query, user_profile_check['suggestions'], user_profile)
                query = self.reformulate_query(collected_info, user_profile, query )
            else: 
                query = self.reformulate_query(None, user_profile, query)
        else: 
            return query
        # return response
    
    




# class QueryEnhancer: 
#     def __init__(self): 
#         self.enhancer = ChatOllama(model = 'llama3.2:latest', temperature=0.0)
#         self.interrogater = ChatOllama(modle = 'llama3.2:latest', temperature=0.0)

#     # def interrogate(self, query, user_profile):
    
#     def create_query(self, query, user_profile):

#         question = [HumanMessage(content = QUERY_ENHANCER_USER_INSTRUCTIONS.format(diet = user_profile['diet'], 
#                                                                                    preference = user_profile['preference'], 
#                                                                                    history = user_profile("feeadback")))]

#         new_query = self.enhancer.invoke([SystemMessage(content = QUERY_ENHANCER_SYSTEM_INSTRUCTIONS)] + question)