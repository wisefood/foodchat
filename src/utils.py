import json
import os
import re
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings

PATH = Path(__file__).parent.resolve()

# Lazy imports for HuggingFace embeddings (requires torch + transformers)
torch = None
AutoModel = None
AutoTokenizer = None
EmbeddingFunction = None


def _load_hf_deps():
    global torch, AutoModel, AutoTokenizer, EmbeddingFunction
    if torch is None:
        import torch as _torch
        from transformers import AutoModel as _AutoModel, AutoTokenizer as _AutoTokenizer
        from chromadb.utils.embedding_functions import EmbeddingFunction as _EmbeddingFunction
        torch = _torch
        AutoModel = _AutoModel
        AutoTokenizer = _AutoTokenizer
        EmbeddingFunction = _EmbeddingFunction

EMBEDDING_MODEL = "davanstrien/autotrain-recipes-2451975973"
TRACE_FILE_PATH = PATH / 'trace.json'


class HuggingFaceEmbeddings:
    """HuggingFace embeddings - requires torch and transformers to be installed."""

    def __init__(self, embedding_model=EMBEDDING_MODEL):
        _load_hf_deps()
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


def pretty_format_recipes(recipe_data: tuple) -> str:
    """Format a tuple of raw recipe strings into a clean, human-readable format.

    Args:
        recipe_data: A tuple of strings, where each string contains a recipe
                     with 'Title:', 'Ingredients:', and 'Directions:' sections.

    Returns:
        Formatted string with meal plan details.
    """
    result = "Here is your meal plan:"

    meal_types = {0: "Breakfast", 1: "Lunch", 2: "Dinner"}

    for i, raw_recipe in enumerate(recipe_data):
        try:
            # Clean up known data issues
            clean_recipe = raw_recipe.strip().replace(",t.',", "").replace("',", "")

            # Use regex to find sections
            title_match = re.search(
                r"Title:(.*?)\s*Ingredients:", clean_recipe, re.IGNORECASE | re.DOTALL
            )
            ingredients_match = re.search(
                r"Ingredients:(.*?)\s*Directions:",
                clean_recipe,
                re.IGNORECASE | re.DOTALL,
            )
            directions_match = re.search(
                r"Directions:(.*)", clean_recipe, re.IGNORECASE | re.DOTALL
            )

            if not all([title_match, ingredients_match, directions_match]):
                result += f"\n\n--- Could not parse Recipe {i + 1} ---"
                continue

            title = title_match.group(1).strip().title()
            ingredients_raw = ingredients_match.group(1).strip()
            directions_raw = directions_match.group(1).strip()

            # Format header
            meal_name = meal_types.get(i, f"Recipe {i + 1}")
            result += f"\n\n--- {meal_name}: {title} ---\n\n"

            # Format ingredients
            result += "Ingredients:"
            ingredient_list = [
                ing.strip().capitalize()
                for ing in ingredients_raw.split(",")
                if ing.strip()
            ]
            for item in ingredient_list:
                result += f"\n  - {item}"

            # Format directions
            result += "\n\nDirections:"
            steps = [
                step.strip().capitalize()
                for step in directions_raw.split(".")
                if step.strip()
            ]
            for j, step in enumerate(steps, 1):
                result += f"\n  {j}. {step}."

        except Exception:
            result += f"\n\n--- Error parsing Recipe {i + 1} ---"

    return result