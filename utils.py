import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os
import json


PATH = Path(__file__).parent.resolve()

EMBEDDING_MODEL = "davanstrien/autotrain-recipes-2451975973"
TRACE_FILE_PATH = PATH / 'trace.json'


class HuggingFaceEmbeddings(EmbeddingFunction):

    def __init__(self, embedding_model = EMBEDDING_MODEL): 
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)


    def _embed(self, texts): 
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        with torch.no_grad(): # disable gradients because we are not training
            
            inputs = tokenizer(texts, padding = True, truncation=True, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True) #pass okenized inputs to model & return all hidden layers
            #outputs contain logits (for classification) & hidden states (each layer's output)
            hidden_states = outputs.hidden_states[-1] # take last hidden layer output which contains contextualized embeddings for each token. 
            embeddings = hidden_states.mean(dim=1) #mean_pooling => each token has its own embedding, we do mean pooling to have a single vector per sentence/(docs?) 
        return embeddings.tolist()
    
    def embed_documents(self, texts):
        """Embed multiple documents (Required by LangChain)."""
        return self._embed(texts)

    def embed_query(self, text):
        """Embed a single query (Required by LangChain)."""
        return self._embed([text])[0]  # Extract first item from list
    


def get_embeddings(embedding_model): 
    if embedding_model == "fine_tuned-recipes": 
        embeddings = HuggingFaceEmbeddings()
    else: 
        embeddings = OllamaEmbeddings(model = embedding_model)
    return embeddings


def verify_chroma_scores(embedding_model, query, doc): 

    embeddings = get_embeddings(embedding_model)

    query_embed = embeddings.embed_query(query)
    doc_embed = embeddings.embed_query(doc)

    manual_cosine = np.dot(query_embed, doc_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(doc_embed))

    return manual_cosine



def write_to_file(data, user_query=False, context=False, response=False, expected=False) :

    file_path = TRACE_FILE_PATH
    entry = {}

    if os.path.exists(file_path) : 
        try: 
            with open(file_path, "r", encoding="utf-8") as trace_file: 
                sessions = json.load(trace_file)
                if not isinstance(sessions, list): 
                    sessions = []
        except json.JSONDecodeError: 
            sessions = []
    else : 
        sessions=[]

    if sessions and "User Query" in sessions[-1] and not sessions[-1].get("FoodChat Response"): #incomplete session
        entry = sessions[-1]
    else : 
        entry = {}
    
    if user_query : 
        entry["User Query"] = data
        
    elif context : 
        content = "\n".join(doc.page_content for doc in data)
        entry.update({"Retrieved Context" : content})

    elif response:
        entry.update({"FoodChat Response" : data})

    elif expected:
        entry.update({"Expected Response" : data})

    if entry not in sessions:
        sessions.append(entry)
    with open(file_path, "w", encoding="utf-8") as trace_file: 
        json.dump(sessions, trace_file, ensure_ascii=False, indent=4)