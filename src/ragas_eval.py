import json
from foodchat import PATH, TEST_TRACE_FILE_PATH
import re
from langchain_community.embeddings import OllamaEmbeddings
import torch
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import *
from foodchat import *
from colorama import Fore, Style

RAGAS_TRACE_FILE = PATH / 'ragas_result.json'

#Structured Output with OLLAMA
class StringList(BaseModel): 
    statements: list[str] # list of statements that should be extracted from the response. 

# low top_p => sample from highest probability nucleous 
# top_p : select the smallest number of top words whose combined probability is at least p
# top_k : it is the ability of the model to slect the next k word (ordered by probability). Making it low (1 - 5) makes the modle less random.
class FaithfulnessEvaluator: 
    def __init__(self): 
        self.response_extractor = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0,  format = StringList.model_json_schema())
        self.faithfulness_evaluator = ChatOllama(model = 'llama3.2:latest', temperature=0.0, top_k = 1, top_p = 0, format = StringList.model_json_schema())
    
    def extract_from_response(self, response) : 
        question = [HumanMessage(content = "Here is the text you have to extract 10 claims from: {response}".format(response = response))]
        claims = self.response_extractor.invoke([SystemMessage(content = RESPONSE_EXTRACTOR_INSTRUCTIONS)] + question)
        return claims

    def evaluate(self, response, context):
        claims = json.loads(self.extract_from_response(response).content)['statements']
        print(Fore.GREEN + "Extracted Claims: " + str(claims) + Style.RESET_ALL)
        print(context)
        question = [HumanMessage(content ="Here is a list of claims and the corresponding context. Evaluate whether each claim is explicitly supported by the text.\n"
        "Claims: {claims}\n"
        "Context: {context}".format(claims = claims, context=context) )]
        faithfulness_eval = json.loads(self.faithfulness_evaluator.invoke([SystemMessage(content=FAITHFULNESS_EVALUATOR_INSTRUCTIONS)] + question).content)
        print(Fore.GREEN + "FAITHFULNESS evaluation result: " + str(faithfulness_eval) + Style.RESET_ALL)
        faithfulness_eval = [1 if score == 'yes' else 0 for score in faithfulness_eval["statements"]]
        faithfulness_score = faithfulness_eval.count(1) / len(faithfulness_eval)
        print(Fore.GREEN + "Faithfulness Score: " +  str(faithfulness_score) + Style.RESET_ALL)

        return faithfulness_score
            

class ContextRelevanceEvaluator:
    def __init__(self, embedding_model) : 
        # self.context_sentences_extractor = ChatOllama(model = "llama3.2:latest", temperature = 0.0, format = StringList.model_json_schema())
        # self.context_relevance_evaluator = ChatOllama(model = "llama3.2:latest", temperature = 0.0, format = StringList.model_json_schema())
        self.answer_generator = ChatOllama(model = "llama3.2:latest", top_k = 1, top_p = 0, temperature=0.0)
        self.embeddings = OllamaEmbeddings(model = embedding_model)


    # Classic Implementation (not Working so well)
    # def evaluate(self, query, context) : 
    #     question = [HumanMessage(content = "Here si the provided context: {context} and the question: {question}".format(context=context, question=query))]
    #     relevant_sentences = self.context_relevance_evaluator.invoke([SystemMessage(content=CONTEXT_RELEVANCE_EVALUATOR_INSTRUCTIONS)] + question)
    #     relevant_sentences = json.loads(relevant_sentences.content)['statements']
    #     print(Fore.YELLOW + "Relevant Sentences : " + str(relevant_sentences) + Style.RESET_ALL)
    #     all_sentences = self.context_sentences_extractor.invoke([SystemMessage(content=CONTEXT_EXTRACTOR_INSTRUCTIONS.format(context=context))])
    #     all_sentences = json.loads(all_sentences.content)['statements']
    #     print("All sentences: ", all_sentences)
    #     context_relevance_score = len(relevant_sentences)/len(all_sentences)
    #     return context_relevance_score

    def evaluate(self, query, context) :
        question = [HumanMessage(content = query)]
        generated_answer = self.answer_generator.invoke([SystemMessage(content=ANSWER_GENERATOR_INSTRUCTIONS)] + question)
        generated_answer = generated_answer.content
        print(Fore.YELLOW + "Generated Answer : " + str(generated_answer) + Style.RESET_ALL)
        embedded_generated = self.embeddings.embed_query(generated_answer)
        embedded_context = self.embeddings.embed_documents(context)
        cos_scores = []
        for embed in embedded_context: 
            cos_scores.append(torch.cosine_similarity(torch.tensor(embedded_generated).unsqueeze(0), torch.tensor(embed).unsqueeze(0)))
        context_relevance_score = sum(cos_scores)/len(cos_scores)
        print(Fore.YELLOW + "Context Relevance Score: " + str(context_relevance_score))
        return context_relevance_score.item()

        

class AnswerRelevanceEvaluator: 
    def __init__(self, embedding_model) : 
        self.question_generator = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0,format = StringList.model_json_schema())
        self.answer_relevance_evaluator = ChatOllama(model = "llama3.2:latest", temperature = 0.0, top_k = 1, top_p = 0,format = StringList.model_json_schema())
        self.embeddings = OllamaEmbeddings(model = embedding_model)
    
    def generate_questions(self, response) : 
        # question = [HumanMessage(content = "Can you derive 5 questions from the following response: {response}".format(response = response))]
        question = [HumanMessage(content = "Here is an answer: '{response}'Generate 5 questions that could have led to this response. The questions should be diverse in phrasing and applicable to different contexts.".format(response = response))]
        questions = json.loads(self.question_generator.invoke([SystemMessage(content=QUESTION_GENERATOR_INSTRUCTIONS )]+ question).content)["statements"]
        print(Fore.CYAN + "Derived questions: " + str(questions) + Style.RESET_ALL)
        return questions
    
    def evaluate(self, query, response): 
        questions = self.generate_questions(response)
        # embedded_questions = [self.embeddings.embed_query(question) for question in questions]
        embedded_query = self.embeddings.embed_query(query)
        embedded_questions = self.embeddings.embed_documents(questions)
        cos_scores = []
        for embedded_question in embedded_questions : 
            cos_scores.append(torch.cosine_similarity(torch.tensor(embedded_query).unsqueeze(0), torch.tensor(embedded_question).unsqueeze(0)))
        answer_relevance_score = sum(cos_scores)/len(cos_scores)
        print(Fore.CYAN + "Answer Relevance Score: " + str(answer_relevance_score.item()) + Style.RESET_ALL)
        return answer_relevance_score.item()




class RagasEvaluator: 
    def __init__(self, embedding_model) : 
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.context_relevance_evaluator = ContextRelevanceEvaluator(embedding_model)
        self.answer_relevance_evaluator = AnswerRelevanceEvaluator(embedding_model)

    def evaluate(self, query, response, context):
        return {
            "faithfulness" : self.faithfulness_evaluator.evaluate(response, context), 
            "context_relevance" : self.context_relevance_evaluator.evaluate(query, context), 
            "answer_relevance" : self.answer_relevance_evaluator.evaluate(query, response)
        } 

def ragas(embedding_model) : # TEMPORARY : better to make a flag true false and do the agas evaluation after the usual evaluation if the flag is true for instance. 
    ragas_eval = RagasEvaluator(embedding_model)
    
    with open(TEST_TRACE_FILE_PATH, "r", encoding='utf-8') as test_trace : 
        trace = json.loads(test_trace.read())
        query = trace["User Query"]
        response = trace["FoodChat Response"]
        context = trace["Retrieved Context"]
        context = re.split(r"(?=Title:)", context)[1:] # get a list instead of a str
    result = ragas_eval.evaluate(query, response, context)

    with open(RAGAS_TRACE_FILE, "w", encoding="utf-8") as ragas_res : 
        json.dump(result, ragas_res,  ensure_ascii=False, indent=4)