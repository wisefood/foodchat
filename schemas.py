from pydantic import BaseModel, conint
from typing import Literal


class QueryRewriterSchema(BaseModel): 
    query : str

class QASchema(BaseModel): 
    question : str
    expected : str

class QuestionList(BaseModel): 
    questions: list[QASchema]

class EvaluationParameters(BaseModel): 
    question: str
    answer: str
    comment: str

class FoodChatResponseEvaluatorSchema(BaseModel):
    evaluation : list[EvaluationParameters]

class ConstrainingScoringSchema(BaseModel): 
    score: conint(ge=0, le=1)
    explanation: str

class ScoringSchema(BaseModel):
    reasoning: str
    score: conint(ge=1, le=5)

class DocumentGraderSchema(BaseModel): 
    diet : ConstrainingScoringSchema
    allergies : ConstrainingScoringSchema
    preferences : ScoringSchema
    user_feedback : ScoringSchema

class QueryReconcilerSchema(BaseModel): 
    conflict: bool
    trigger: str
    explanation: str

class QueryItemExtractorSchema(BaseModel): 
    item: list[str]

class QueryCheckerSchema(BaseModel): 
    response: Literal['YES', 'NO']

class UserProfileCheckerSchema(BaseModel): 
    response: Literal['YES', 'NO']
    suggestions: list[str]

class QueryReformulatorSchema(BaseModel): 
    reformulated_query: str

# class UserInfoCollectorSchema(BaseModel) : 
#     response : 