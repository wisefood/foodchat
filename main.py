import argparse

from foodchat import *
from multiple_evaluation import *
from ragas_eval import *
from user_interface import UI, launch_interface
from utils import *

from fastapi import FastAPI
from src.routers import foodchat_router  # Import the new router

app = FastAPI()

# Include the routers
app.include_router(foodchat_router.router)

@app.get("/")
def root():
    return {"message": "Welcome to WiseFood API"}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FoodChat: A meal-planning assistant.")
    
    parser.add_argument("--data_type", type = str, choices=['pdf', 'csv'], default = 'csv')
    parser.add_argument("--dataset", type = str, choices=['hummus', 'culinary'], default = 'hummus')
    parser.add_argument("--model", type=str, choices=["Llama_FoodChat", "mistral_FoodChat", "DeepSeek_FoodChat"], default="Llama_FoodChat")
    parser.add_argument("--retrieval", type=str, choices=["similarity_score_threshold", "mmr", "similarity"], default="similarity")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Lambda parameter for MMR (0-1). If 1: equivalent to standard similarity search, nodiversity.\
                        If 0: no relevance considered only diversity. Default: 0.5")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--ragas", action="store_true", help="Run in test mode without groound truth (RAGAs)")
    parser.add_argument("--llm_as_judges", action="store_true", help="Run in test mode with ground truth. Each category is evaluated by different LLM")
    parser.add_argument("--split", type=str, choices=["page_split", "recipe_split", "recursive_character_split"], default="recipe_split", help="PDF splitting method")
    parser.add_argument("--query_method", type=str, choices=["standard", "multiquery"], default="standard", help="How is the user query improved")
    parser.add_argument("--embedding_model", type=str, choices=["nomic-embed-text:latest", "mxbai-embed-large:latest", "fine_tuned-recipes"], default = 'nomic-embed-text:latest')
    parser.add_argument("--max_retrieval", type=int, default = 3, help = "Number of maximum retrieved documents per query" )
    parser.add_argument("--vectorstore", type=str, choices=["chroma", "faiss", "milvus"], default = "chroma", help = "Type of vector database" )
    parser.add_argument("--hybrid_search", action="store_true", help = "Enable hybrid search, otherwise classic semantic search." )
    
    parser.add_argument("--ks_weight", type=float, default = 0.5, help="Weight given to the score for documents retrieved with BM25 [0-1]")
    parser.add_argument("--vs_weight", type=float, default = 0.5, help="Weight given to the score for documents retrieved with Vector Search [0-1]")

    parser.add_argument("--adaptive_rag", action="store_true", help="Run with Adaptive RAG system")
    
    args = parser.parse_args()
    if args.retrieval == "similarity" and args.lambda_param != 0.5:
        parser.error(
            "The --lambda parameter is only valid for --retrieval mmr.\
            Do not specify --lambda_param when using similarity-based retrieval"
        )
    if args.data_type == "csv" and args.dataset not in ["hummus", "culinary"] : 
        parser.error(
            "The --dataset parameter can be either 'culinary', or 'hummus' and it is only valid with --data_type csv"
        )
    if args.vs_weight + args.ks_weight != 1.0 : 
        parser.error(
            "The weights (score vector search and keyword search) for hybrid search must sum up to one."
        )
    return args

def main_run() :
    args = parse_arguments()
    embeddings = get_embeddings(args.embedding_model)
    foodchat = initialize_system(args, embeddings)
    ui = UI(foodchat, args)
    # rag_chain = foodchat.create_chain(args, ask_user_fn = None)
    # run_interactive_chat(foodchat, rag_chain, args)
    launch_interface(ui, args)
    return foodchat

if __name__ == "__main__":
    main_run()
    
