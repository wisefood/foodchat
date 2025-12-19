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


LLM_EVAL_RES_FILE = PATH / 'llm_eval_res.json'

class EvaluationAspect(BaseModel): 
    verdict: str
    explanation: str

class CorrectenessSchema(BaseModel): 
    answer_to_query: EvaluationAspect
    similar_to_expected: EvaluationAspect
    faitfulness: EvaluationAspect

class CorrectnessEvaluator: 
    def __init__(self) : 
        self.correctness_evaluator = ChatOllama(model = "llama3.2:3b-instruct-q8_0", temperature = 0.0, format = CorrectenessSchema.model_json_schema())
    
    def evaluate(self, query, context, response, expected_response) : 
        question = [HumanMessage(content = CORRECTNESS_EVALUATOR_USER_INSTRUCTIONS.format(query = query, 
                                                                                           context = context, 
                                                                                           response = response, 
                                                                                           expected_response = expected_response))]
        evaluation = self.correctness_evaluator.invoke([SystemMessage(content = CORRECTNESS_EVALUATOR_SYSTEM_INSTRUCTIONS)] + question)
        evaluation = json.loads(evaluation.content)

        return evaluation

class CustomizationSchema(BaseModel) : 
    user_preference: EvaluationAspect
    cultural_awareness: EvaluationAspect

class CustomizationEvaluator: 
    def __init__(self): 
        self.customization_evaluator = ChatOllama(model ="llama3.2:3b-instruct-q8_0", temperature = 0.0, format = CustomizationSchema.model_json_schema())
    
    def evaluate(self, query, response): 
        question = [HumanMessage(content = CUSTOMIZATION_EVALUATOR_USER_INSTRUCTIONS.format(query = query, 
                                                                                            response = response))]
        evaluation = self.customization_evaluator.invoke([SystemMessage(CUSTOMIZATION_EVALUATOR_SYSTEM_INSTRUCTIONS)] + question)
        evaluation = json.loads(evaluation.content)

        return evaluation 
    
class FormatSchema(BaseModel): 
    response_structure : EvaluationAspect
    clarity: EvaluationAspect

class FormatEvaluator:
    def __init__(self): 
        self.format_evaluator = ChatOllama(model ="llama3.2:3b-instruct-q8_0", temperature = 0.0, format = FormatSchema.model_json_schema())

    def evaluate(self, response) : 
        question = [HumanMessage(content = FORMAT_EVALUATOR_USER_INSTRUCTIONS.format(response = response))]
        evaluation = self.format_evaluator.invoke([SystemMessage(FORMAT_EVALUATOR_SYSTEM_INSTRUCTIONS)] + question)
        evaluation = json.loads(evaluation.content)

        return evaluation 

class SafetySchema(BaseModel) : 
    harmfulness : EvaluationAspect
    diet : EvaluationAspect

class SafetyEvaluator: 
    def __init__(self): 
        self.safety_evaluator = ChatOllama(model ="llama3.2:3b-instruct-q8_0", temperature = 0.0, format = SafetySchema.model_json_schema())

    def evaluate(self, query, response) : 
        question = [HumanMessage(content = SAFETY_EVALUATOR_USER_INSTRUCTIONS.format(query = query, 
                                                                                     response = response))]
        evaluation = self.safety_evaluator.invoke([SystemMessage(SAFETY_EVALUATOR_SYSTEM_INSTRUCTIONS)] + question)
        evaluation = json.loads(evaluation.content)

        return evaluation 

class LLMAsJudges(): 
    def __init__(self): 
        self.correctness_evaluator = CorrectnessEvaluator()
        self.customization_evaluator = CustomizationEvaluator()
        self.format_evaluator = FormatEvaluator()
        self.safety_evaluator = SafetyEvaluator()
    
    def evaluate(self, query, context, response, expected_response) : 
        return {
            "Correctness" : self.correctness_evaluator.evaluate(query, context, response, expected_response),
            "Customization" : self.customization_evaluator.evaluate(query, response),
            "Format" : self.format_evaluator.evaluate(response),
            "Safety" : self.safety_evaluator.evaluate(query, response)

        }


def run_evaluation(): 
    llm_eval = LLMAsJudges()
    
    #Temporary
    with open(TEST_TRACE_FILE_PATH, "r", encoding='utf-8') as test_trace : 
        trace = json.loads(test_trace.read())
        query = trace["User Query"]
        response = trace["FoodChat Response"]
        expected_response = trace["Expected Response"]
        context = trace["Retrieved Context"]
        context = re.split(r"(?=Title:)", context)[1:]
    evaluation_result = llm_eval.evaluate(query, context, response, expected_response)

    with open(LLM_EVAL_RES_FILE, "w", encoding="utf-8") as llm_eval_res : 
        json.dump(evaluation_result, llm_eval_res,  ensure_ascii=False, indent=4)