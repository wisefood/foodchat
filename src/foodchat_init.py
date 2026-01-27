import argparse
import os

from foodchat import initialize_system
from utils import get_embeddings

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FoodChat: A meal-planning assistant.")
    
    parser.add_argument("--data_type", type = str, choices=['pdf', 'csv'], default = 'csv')
    parser.add_argument("--dataset", type = str, choices=['hummus', 'culinary'], default = 'hummus')
    parser.add_argument("--model", type=str, choices=["Llama_FoodChat", "mistral_FoodChat", "DeepSeek_FoodChat"], default="Llama_FoodChat")
    parser.add_argument("--retrieval", type=str, choices=["similarity_score_threshold", "mmr", "similarity"], default="similarity")
    parser.add_argument("--lambda_param", type=float, default=0.5)
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--ragas", action="store_true", help="Run in test mode without groound truth (RAGAs)")
    parser.add_argument("--llm_as_judges", action="store_true", help="Run in test mode with ground truth. Each category is evaluated by different LLM")
    parser.add_argument("--split", type=str, choices=["page_split", "recipe_split", "recursive_character_split"], default="recipe_split")
    parser.add_argument("--query_method", type=str, choices=["standard", "multiquery"], default="standard")
    parser.add_argument("--embedding_model", type=str, choices=["nomic-embed-text:latest", "mxbai-embed-large:latest", "fine_tuned-recipes", "BAAI/bge-small-en-v1.5"], default = 'nomic-embed-text:latest')
    parser.add_argument("--max_retrieval", type=int, default = 3)
    parser.add_argument("--vectorstore", type=str, choices=["chroma", "faiss", "milvus"], default = "chroma")
    parser.add_argument("--hybrid_search", action="store_true")
    parser.add_argument("--ks_weight", type=float, default = 0.5)
    parser.add_argument("--vs_weight", type=float, default = 0.5)
    parser.add_argument("--adaptive_rag", action="store_true")
    
    return parser.parse_args()

class Config:
    dataset = os.getenv("DATASET", "hummus") # Default to 'hummus' if not set
    model = os.getenv("MODEL", "Llama_FoodChat")
    data_type = os.getenv("DATA_TYPE", "csv")
    retrieval = os.getenv("RETRIEVAL", "similarity")
    lambda_param = os.getenv("LAMBDA", 0.5)
    test = os.getenv("TEST_MODE", False)
    ragas = os.getenv("RAGAS", False)
    llm_as_judges = os.getenv("JUDGE", False)
    split = os.getenv("SPLIT", "recipe_split")
    query_method = os.getenv("QUERY_METHOD", "standard")
    embedding_model = os.getenv("EMBEDDINGS", "nomic-embed-text:latest")
    max_retrieval = os.getenv("MAX_RETRIEVAL", 3)
    vectorstore = os.getenv("VECTORSTORE", "chroma")
    hybrid_search = os.getenv("HYBRID_SEARCH", False)
    ks_weight = os.getenv("KS_WEIGHT", 0.5)
    vs_weight = os.getenv("VS_WEIGHT", 0.5)
    adaptive_rag = os.getenv("ADAPTIVE_RAG", False)

config = Config()

def initialize_foodchat():
    #args = parse_arguments()
    embeddings = get_embeddings(config.embedding_model)
    foodchat = initialize_system(config, embeddings)
    return foodchat, config