import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from colorama import Fore, Style
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi

from agents import DocumentGrader, QueryReconciler
from csv_processor import CSV_CULINARY_PATH, CSV_HUMMUS_PATH, CSVProcessor
from pdf_processor import PDFProcessor, SOURCE_NAME
from prompts import template_foodchat
from utils import PATH

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from KG_neo4j.query_kg import get_filtered_recipe_ids

PDF_PATH = f"./data/{SOURCE_NAME}.pdf"
TEST_TRACE_FILE_PATH = PATH / "test_trace.json"
VECTORSTORE_PATH = PATH / "VECTORSTORE"

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database. Provide these alternative questions separated by newlines.
            Original question: {question}""",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_user_profile(user_profile: dict) -> str:
    """Format a user profile dict for display/logging."""
    lines = []
    if user_profile.get("diet"):
        lines.append(f"Diet: {', '.join(user_profile['diet'])}")
    if user_profile.get("allergies"):
        lines.append(f"Allergies: {', '.join(user_profile['allergies'])}")
    if user_profile.get("preferences"):
        lines.append(f"Preferences: {', '.join(user_profile['preferences'])}")
    if user_profile.get("history"):
        lines.append(f"History: {user_profile['history']}")
    return "\n".join(lines) if lines else "No profile information"


class Retriever:
    def __init__(self, persist_dir, chroma_mode="local", chroma_host="localhost", chroma_port=8000):
        self.persist_dir = persist_dir
        self.chroma_mode = chroma_mode
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port

    def _get_chroma_client(self):
        """Get ChromaDB client based on mode (local or remote)."""
        if self.chroma_mode == "remote":
            print(f"Connecting to remote ChromaDB at {self.chroma_host}:{self.chroma_port}")
            return chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        return None  # Local mode uses persist_directory directly

    def get_vector_db(
        self,
        data_type,
        dataset_name,
        data,
        processor,
        vectorstore_name,
        embedding_model,
    ):
        persist_dir = str(
            VECTORSTORE_PATH
            / f"{vectorstore_name}"
            / f"{embedding_model.split('-')[0]}"
            / f"{data_type.upper()}"
            / f"{dataset_name.upper()}_SOURCE"
        )

        # Remote ChromaDB mode
        if self.chroma_mode == "remote":
            print(f"Using remote ChromaDB at {self.chroma_host}:{self.chroma_port}")
            client = self._get_chroma_client()
            vector_db = Chroma(
                client=client,
                collection_name="vector-db",
                embedding_function=processor.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            return vector_db

        # Local ChromaDB mode (default)
        if data_type == "pdf":
            if os.path.exists(persist_dir):
                print("Loading Vector Database (local)")
                vector_db = Chroma(
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    embedding_function=processor.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                print("Generating Vector Database (local)")
                vector_db = Chroma.from_documents(
                    data,
                    embedding=processor.embeddings,
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )

        else:
            selected_cols = (
                [
                    "title",
                    "ingredients",
                    "directions",
                    "recipe_id",
                    "allergens",
                    "meal_course",
                    "diet",
                    "dish_type",
                ]
                if dataset_name == "hummus"
                else ["title", "ingredients"]
            )
            if os.path.exists(persist_dir):
                print("Loading Vector Database (local)")
                vector_db = Chroma(
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    embedding_function=processor.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                print("Generating Vector Database (local)")
                vector_db = Chroma.from_texts(
                    texts=data["combined_text"].to_list(),
                    embedding=processor.embeddings,
                    collection_name="vector-db",
                    metadatas=(
                        data[selected_cols].to_dict("records")
                        if data_type == "csv"
                        else data[selected_cols].to_dict("records")
                    ),
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )

        return vector_db

    def build_vector_retriever(
        self, retrieval_method, vector_db, query_method, mmr_lambda, max_retrieval
    ):
        """Defines the retriever."""
        if retrieval_method == "mmr":
            base_retriever = vector_db.as_retriever(
                search_type=retrieval_method,
                search_kwargs={"k": max_retrieval, "lambda_mult": mmr_lambda},
            )
        elif retrieval_method == "similarity":
            base_retriever = vector_db.as_retriever(
                search_type=retrieval_method, search_kwargs={"k": max_retrieval}
            )
        else:
            base_retriever = vector_db.as_retriever(
                search_type=retrieval_method,
                search_kwargs={"k": max_retrieval, "score_threshold": 0.0},
            )

        if query_method == "standard":
            return base_retriever
        else:
            return MultiQueryRetriever.from_llm(
                llm=OllamaLLM(model="llama3.2:latest"),
                retriever=base_retriever,
                prompt=QUERY_PROMPT,
            )

    def build_bm25_retriever(self, persist_dir, data_type, data, max_retrieval):
        texts_path = str(Path(persist_dir) / "bm_25_texts.json")

        if os.path.exists(texts_path):
            print("Loading BM25 data")
            with open(texts_path, "r") as f:
                texts = json.load(f)
        else:
            print("Generating BM25 data")

            if data_type == "pdf":
                texts = [doc.page_content for doc in data]
            else:
                texts = data["combined_text"].to_list()

            with open(texts_path, "w") as f:
                json.dump(texts, f)

        corpus = [text.lower().split(" ") for text in texts]
        bm25_retriever = BM25Okapi(corpus)

        return bm25_retriever

    def get_vector_search_results(
        self, query, max_retrieval, retrieval_method, retriever, vectorstore, recipe_ids
    ):
        print(
            "FILTERING WITH RECIPE_IDS: ",
            len(recipe_ids),
            " ",
            retrieval_method,
            " ",
            query,
        )
        if retrieval_method == "mmr":
            docs = retriever.invoke(query)
            return docs
        else:
            if retrieval_method == "similarity":
                results = vectorstore.similarity_search_with_score(
                    query, k=max_retrieval, filter={"recipe_id": {"$in": recipe_ids}}
                )
                print("RESULTS: ", results)
            elif retrieval_method == "similarity_score_threshold":
                results = vectorstore.similarity_search_with_score(
                    query,
                    k=max_retrieval,
                    score_threshold=0.0,
                    filter={"recipe_id": {"$in": recipe_ids}},
                )
            else:
                raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

            docs = []
            for doc, score in results:
                doc.metadata["score"] = score
                docs.append(doc)
            for doc in docs:
                doc.metadata["score"] = 1 - doc.metadata["score"]

            return sorted(
                [doc for doc in docs],
                key=lambda x: x.metadata.get("score"),
                reverse=True,
            )

    def get_keyword_search_results(
        self, query, bm25_retriever, data_type, data, max_retrieval
    ):
        tokenized_query = query.lower().split(" ")
        bm25_results = bm25_retriever.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_results)[::-1][:max_retrieval]
        top_scores = bm25_results[top_indices]

        max_score = np.max(top_scores) if len(top_scores) > 0 else 1.0
        if max_score <= 0:
            max_score = 1.0
        normalized_scores = np.maximum(0, top_scores / max_score)

        with open(Path(self.persist_dir) / "bm_25_texts.json", "r") as f:
            docs = json.load(f)

        bm25_docs = []
        for idx, score in zip(top_indices, normalized_scores):
            doc = Document(page_content=docs[idx], metadata={"score": score})
            bm25_docs.append(doc)

        return bm25_docs

    def get_relevant_docs_with_scores(
        self,
        data_type,
        data,
        query,
        retriever,
        max_retrieval,
        retrieval_method,
        vectorstore,
        ks_weight,
        vs_weight,
        recipe_ids,
    ):
        vector_docs = self.get_vector_search_results(
            query, max_retrieval, retrieval_method, retriever, vectorstore, recipe_ids
        )
        if isinstance(retriever, BM25Okapi):
            bm25_docs = self.get_keyword_search_results(
                query, retriever, data_type, data, max_retrieval
            )
            bm25_docs = {
                elem.page_content: elem.metadata["score"] for elem in bm25_docs
            }
            vector_docs = {
                elem.page_content: elem.metadata["score"] for elem in vector_docs
            }

            vector_docs_content = list(vector_docs.keys())
            bm25_docs_content = list(bm25_docs.keys())
            all_page_contents = set(vector_docs_content).union(bm25_docs_content)
            merged_docs = []

            for doc in all_page_contents:
                if doc in bm25_docs_content and doc in vector_docs_content:
                    print(
                        Fore.GREEN
                        + "DOC FOUND IN BOTH SEARCH METHOD RESULTS"
                        + Style.RESET_ALL
                    )
                    bm25_score = bm25_docs.get(doc)
                    vector_score = vector_docs.get(doc)
                elif doc in bm25_docs_content:
                    bm25_score = bm25_docs.get(doc)
                    vector_score = 0.0
                elif doc in vector_docs_content:
                    bm25_score = 0.0
                    vector_score = vector_docs.get(doc)

                hybrid_score = bm25_score * ks_weight + vector_score * vs_weight
                doc = Document(page_content=doc, metadata={"score": hybrid_score})
                merged_docs.append(doc)

            docs_with_scores = sorted(
                merged_docs, key=lambda x: x.metadata["score"], reverse=True
            )

            return docs_with_scores

        else:
            return vector_docs

    def get_retriever(
        self,
        data_type,
        dataset_name,
        data,
        processor,
        vectorstore_name,
        embedding_model,
        retrieval_method,
        max_retrieval,
        query_method,
        mmr_lambda,
        hybrid_search,
    ):
        vector_db = self.get_vector_db(
            data_type, dataset_name, data, processor, vectorstore_name, embedding_model
        )
        vector_retriever = self.build_vector_retriever(
            retrieval_method, vector_db, query_method, mmr_lambda, max_retrieval
        )
        if hybrid_search:
            print("HYBRID SEARCH INITIALISATION")
            bm25_retriever = self.build_bm25_retriever(
                persist_dir=self.persist_dir,
                data_type=data_type,
                data=data,
                max_retrieval=max_retrieval,
            )
            return bm25_retriever, vector_db
        return vector_retriever, vector_db


class FoodChat:
    def __init__(
        self,
        retriever,
        vectorstore,
        retrieval_method,
        query_method,
        mmr_lambda,
        model,
        max_retrieval,
        data,
        dataset,
        data_type,
        persist_dir,
        ks_weight,
        vs_weight,
    ):
        self.llm = OllamaLLM(model=model)
        self.retrieval_method = retrieval_method
        self.query_method = query_method
        self.max_retrieval = max_retrieval
        self.retriever = retriever
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.data = data
        self.dataset = dataset
        self.data_type = data_type
        self.persist_dir = persist_dir
        self.vectorstore = vectorstore
        self.ks_weight = ks_weight
        self.vs_weight = vs_weight

    def get_test_response(self, message):
        return "This was your message: " + message + "\n\n"

    def format_docs(self, docs):
        return "\n\n".join(
            f"Page {doc.metadata['page']}:\n{doc.page_content}" for doc in docs
        )

    def retrieve_with_score(self, query, recipe_ids):
        daily_plan = {}
        retriever_instance = Retriever(self.persist_dir)
        for course in recipe_ids.keys():
            docs = retriever_instance.get_relevant_docs_with_scores(
                data_type=self.data_type,
                data=self.data,
                query=query,
                retriever=self.retriever,
                max_retrieval=self.max_retrieval,
                retrieval_method=self.retrieval_method,
                vectorstore=self.vectorstore,
                ks_weight=self.ks_weight,
                vs_weight=self.vs_weight,
                recipe_ids=recipe_ids[course],
            )
            daily_plan[course] = docs
        return daily_plan

    def log_retrieval(self, docs):
        """Print retrieved context"""
        print("\n=== RETRIEVED CONTEXT ===")
        print(f"NUMBER OF RETRIEVED DOCS: {len(docs)}")
        for doc in docs:
            score = doc[1].get("score", "N/A")
            reason = doc[1].get("reasoning")
            print(f"Daily Plan Score: {score}")
            print(f"Explanation: {reason}")
            print("Content Preview:")
            print(doc[0])
            print("-----------------------")
        return docs

    def create_pre_clarification_chain(self, args):
        """Create the pre-clarification RAG chain.

        The chain expects input with:
        - question: The user's query
        - user_id: The member ID (for reference)
        - user_profile_data: The user profile dict (passed directly, not fetched)
        """
        prompt = ChatPromptTemplate.from_template(template_foodchat)
        query_reconciler = QueryReconciler()

        if args.data_type == "pdf":
            return (
                # Profile is passed directly in input, just format it
                RunnablePassthrough.assign(
                    formatted_profile=RunnableLambda(
                        lambda x: format_user_profile(x.get("user_profile_data", {}))
                    )
                )
                | RunnablePassthrough.assign(
                    context=lambda x: self.format_docs(
                        self.log_retrieval(
                            self.retrieve_with_score(x["question"], x["recipe_ids"])
                        )
                    )
                )
                | RunnablePassthrough.assign(
                    final_input=lambda x: {
                        "context": x["context"],
                        "question": x["question"],
                        "user_profile_data": x["formatted_profile"],
                    }
                )
                | (lambda x: x["final_input"])
                | prompt
                | self.llm
            )
        elif args.data_type == "csv":
            return (
                # Profile is passed directly in input
                RunnablePassthrough.assign(
                    formatted_profile=RunnableLambda(
                        lambda x: format_user_profile(x.get("user_profile_data", {}))
                    )
                )
                | RunnablePassthrough.assign(
                    # TEMPORARILY modify the user data if there is conflict
                    modified_user_data=RunnableLambda(
                        lambda x: query_reconciler.reconcile(
                            x["question"], x.get("user_profile_data", {})
                        )
                    )
                )
            )

    def create_post_clarification_chain(self):
        doc_grader = DocumentGrader()
        prompt = ChatPromptTemplate.from_template(template_foodchat)

        return (
            RunnablePassthrough.assign(
                recipe_list=RunnableLambda(
                    lambda x: get_filtered_recipe_ids(
                        x["modified_user_data"]["allergies"],
                        x["modified_user_data"]["diet"],
                    )
                )
            )
            | RunnablePassthrough.assign(
                context=lambda x: self.log_retrieval(
                    doc_grader.grading_daily_plans(
                        x["question"],
                        x["recipe_list"],
                        x["modified_user_data"],
                    )
                )
            )
            | RunnablePassthrough.assign(
                final_input=lambda x: {
                    "docs": x["context"],
                    "question": x["question"],
                    "user_profile_context": x["modified_user_data"],
                }
            )
            | RunnableLambda(lambda x: x["final_input"])
        )


def initialize_system(args, embeddings):
    """Initialize the FoodChat system with retriever and vectorstore."""
    persist_dir = str(
        VECTORSTORE_PATH
        / f"{args.vectorstore}"
        / f"{args.embedding_model.split('-')[0]}"
        / f"{args.data_type.upper()}"
        / f"{args.dataset.upper()}_SOURCE"
    )

    # Get chroma settings with defaults for backward compatibility
    chroma_mode = getattr(args, "chroma_mode", "local")
    chroma_host = getattr(args, "chroma_host", "chromadb")
    chroma_port = getattr(args, "chroma_port", 8000)

    retriever_instance = Retriever(
        persist_dir,
        chroma_mode=chroma_mode,
        chroma_host=chroma_host,
        chroma_port=chroma_port
    )

    if args.data_type == "pdf":
        processor = PDFProcessor(PDF_PATH, args.split, embeddings)
        print("Splitting PDF Document")
        docs = processor.load_and_chunk_pdf()
        retriever, vector_db = retriever_instance.get_retriever(
            args.data_type,
            args.dataset,
            docs,
            processor,
            args.vectorstore,
            args.embedding_model,
            args.retrieval,
            args.max_retrieval,
            args.query_method,
            args.lambda_param,
            args.hybrid_search,
        )
        return FoodChat(
            retriever=retriever,
            vectorstore=vector_db,
            retrieval_method=args.retrieval,
            mmr_lambda=args.lambda_param,
            query_method=args.query_method,
            model=args.model,
            max_retrieval=args.max_retrieval,
            data=docs,
            dataset=args.dataset,
            data_type=args.data_type,
            persist_dir=persist_dir,
            ks_weight=args.ks_weight,
            vs_weight=args.vs_weight,
        )

    elif args.data_type == "csv":
        data_path = CSV_HUMMUS_PATH if args.dataset == "hummus" else CSV_CULINARY_PATH
        if not os.path.exists(data_path):
            logging.warning(f"CSV file not found at {data_path}, skipping CSV initialization")
            return None
        processor = CSVProcessor(data_path, embeddings)
        df = processor.load_csv_data(args.dataset)
        retriever, vector_db = retriever_instance.get_retriever(
            args.data_type,
            args.dataset,
            df,
            processor,
            args.vectorstore,
            args.embedding_model,
            args.retrieval,
            args.max_retrieval,
            args.query_method,
            args.lambda_param,
            args.hybrid_search,
        )

        return FoodChat(
            retriever=retriever,
            vectorstore=vector_db,
            retrieval_method=args.retrieval,
            mmr_lambda=args.lambda_param,
            query_method=args.query_method,
            model=args.model,
            max_retrieval=args.max_retrieval,
            data=df,
            dataset=args.dataset,
            data_type=args.data_type,
            persist_dir=persist_dir,
            ks_weight=args.ks_weight,
            vs_weight=args.vs_weight,
        )
