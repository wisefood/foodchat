"""FoodChat initialization and configuration from environment variables."""
import os

from foodchat import initialize_system
from utils import get_embeddings


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    return float(os.getenv(key, str(default)))


def _get_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    return int(os.getenv(key, str(default)))


class Config:
    """
    FoodChat configuration loaded from environment variables.

    All settings can be overridden via environment variables.
    See .env.example for full list of available options.
    """

    # Data settings
    dataset: str = os.getenv("DATASET", "hummus")
    data_type: str = os.getenv("DATA_TYPE", "csv")
    split: str = os.getenv("SPLIT", "recipe_split")

    # Model settings
    model: str = os.getenv("MODEL", "Llama_FoodChat")
    embedding_model: str = os.getenv("EMBEDDINGS", "nomic-embed-text:latest")

    # Retrieval settings
    retrieval: str = os.getenv("RETRIEVAL", "similarity")
    vectorstore: str = os.getenv("VECTORSTORE", "chroma")
    max_retrieval: int = _get_int("MAX_RETRIEVAL", 3)
    query_method: str = os.getenv("QUERY_METHOD", "standard")

    # Hybrid search settings
    hybrid_search: bool = _get_bool("HYBRID_SEARCH", False)
    ks_weight: float = _get_float("KS_WEIGHT", 0.5)
    vs_weight: float = _get_float("VS_WEIGHT", 0.5)
    lambda_param: float = _get_float("LAMBDA_PARAM", 0.5)

    # Advanced settings
    adaptive_rag: bool = _get_bool("ADAPTIVE_RAG", False)

    # Test/evaluation settings
    test: bool = _get_bool("TEST_MODE", False)
    ragas: bool = _get_bool("RAGAS", False)
    llm_as_judges: bool = _get_bool("JUDGE", False)


config = Config()


def initialize_foodchat():
    """
    Initialize the FoodChat system with configuration from environment.

    Returns:
        tuple: (FoodChat instance, Config instance)
    """
    embeddings = get_embeddings(config.embedding_model)
    foodchat = initialize_system(config, embeddings)
    return foodchat, config