import json
import logging
import os
import sys
from pathlib import Path

from colorama import Fore, Style
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaLLM
# from langchain_milvus import Milvus
from rank_bm25 import BM25Okapi

from agents import *
from csv_processor import *
from pdf_processor import *
from prompts import *
from schemas import *
# from user_interface import *
from user_profiles import *
from utils import *

project_root = Path(__file__).parent.parent  # Goes up to WiseFood/
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, persist_dir):
        self.persist_dir = persist_dir

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

        if data_type == "pdf":
            if os.path.exists(persist_dir):
                print("Loading Vector Database")
                vector_db = Chroma(
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    embedding_function=processor.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                print("Generating Vector Database")
                vector_db = Chroma.from_documents(
                    data,
                    embedding=processor.embeddings,
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )

        else:
            selected_cols = (
                ["title", "ingredients", "directions",
                'recipe_id', 'allergens', 'meal_course',
                'diet', 'dish_type']
                if dataset_name == "hummus"
                else ["title", "ingredients"]
            )
            if os.path.exists(persist_dir):
                print("Loading Vector Database")
                vector_db = Chroma(
                    collection_name="vector-db",
                    persist_directory=persist_dir,
                    embedding_function=processor.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                print("Generating Vector Database")
                vector_db = Chroma.from_texts(
                    texts=data["combined_text"].to_list(),
                    embedding=processor.embeddings,
                    collection_name="vector-db",
                    metadatas=data[selected_cols].to_dict("records")
                    if data_type == "csv"
                    else data[selected_cols].to_dict("records"),
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )  # Force use of Cosine Distance (score between 0 and 2 with lower better => need to convert)

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
        # bm25_retriever = BM25Retriever.from_texts(
        #     texts=texts
        # )
        # bm25_retriever.k = max_retrieval

        return bm25_retriever

    def get_vector_search_results(
        self, query, max_retrieval, retrieval_method, retriever, vectorstore,
        recipe_ids
    ):
        print("FILTERING WITH RECIPE_IDS: ", len(recipe_ids), ' ', type(recipe_ids), ' ', type(recipe_ids[0]))
        # Loop through the different meal courses
        # sample = vectorstore._collection.get(limit=10)
        if retrieval_method == "mmr":
            # MMR doesn't provide scores directly; return as-is
            docs = retriever.invoke(query)
            return docs
        else:
            # Fetch documents and scores using the original query
            if retrieval_method == "similarity":
                results = vectorstore.similarity_search_with_score(
                    query, k=max_retrieval,
                    filter={"recipe_id" : {"$in" : recipe_ids}}
                )
            elif retrieval_method == "similarity_score_threshold":
                results = vectorstore.similarity_search_with_score(
                    query, k=max_retrieval, score_threshold=0.0, 
                    filter={"recipe_id" : {"$in" : recipe_ids}}
                )
            else:
                raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

            # Attach scores to document metadata
            docs = []
            for doc, score in results:
                doc.metadata["score"] = score
                docs.append(doc)
            # Sort documents by score in descending order
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
        # bm25_retriever = self.build_bm25_retriever(self.persist_dir, data_type, data, max_retrieval)
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
            query, max_retrieval, retrieval_method, retriever, vectorstore, 
            recipe_ids
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

            # print(Fore.GREEN + "Here are the vector search results : " + str(vector_docs) + Style.RESET_ALL)
            # print(Fore.MAGENTA + "Here are the KEYWORD search results : " + str(bm25_docs) + Style.RESET_ALL)
            # vector_docs_content = [doc.page_content for doc in vector_docs]
            vector_docs_content = list(vector_docs.keys())
            # bm25_docs_content = [doc.page_content for doc in bm25_docs]
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
            # hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights = [0.4, 0.6]) #random weights need tests
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

    def format_docs(self, docs):
        return "\n\n".join(
            f"Page {doc.metadata['page']}:\n{doc.page_content}" for doc in docs
        )

    def retrieve_with_score(self, query, recipe_ids):
        daily_plan = {}
        retriever_instance = Retriever(self.persist_dir)
        for course in recipe_ids.keys() : 
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
                recipe_ids = recipe_ids[course]
            )
            daily_plan[course] = docs
        return daily_plan

    def log_retrieval(self, docs):
        """Print retrieved context"""
        print("\n=== RETRIEVED CONTEXT ===")
        print(f"NUMBER OF RETRIEVED DOCS: {len(docs)}")
        # write_to_file(docs, context=True)
        # write_to_file(docs, context=True, test=True)
        for doc in docs:
            # score = doc.metadata.get("score", "N/A")
            score = doc[1].get("score", "N/A")
            reason = doc[1].get("reasoning")
            # print(f"Page {doc.metadata['page']} - Similarity score: {score}") #instead of page put in the recipe id ?
            print(f"Daily Plan Score: {score}")
            print(f"Explanation: {reason}")
            print("Content Preview:")
            # print(doc.page_content[:1000] + "...")
            print(doc[0])
            print("-----------------------")
        return docs

    def create_pre_clarification_chain(self, args):
        prompt = ChatPromptTemplate.from_template(template_foodchat)
        query_reconciler = QueryReconciler()

        if args.data_type == "pdf":
            return (
                RunnablePassthrough.assign(
                    user_profile_data=RunnableLambda(
                        lambda x: get_user_profile(x["user_id"])
                    )
                )
                | RunnablePassthrough.assign(
                    formatted_profile=RunnableLambda(
                        lambda x: format_user_profile(x["user_profile_data"])
                    )
                )
                | RunnablePassthrough.assign(
                    context=lambda x: self.format_docs(
                        self.log_retrieval(self.retrieve_with_score(x["question"], x['recipe_ids']))
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
                RunnablePassthrough.assign(
                    # We fetch the user profile data
                    user_profile_data=RunnableLambda(
                        lambda x: get_user_profile(x["user_id"])
                    )
                )
                | RunnablePassthrough.assign(
                    # format the fetched user profile to make it easier in next steps.
                    formatted_profile=RunnableLambda(
                        lambda x: format_user_profile(x["user_profile_data"])
                    )
                )
                | RunnablePassthrough.assign(
                    # TEMPORARILY modify the user data (if there is conflict between user data and user query)
                    modified_user_data=RunnableLambda(
                        lambda x: query_reconciler.reconcile(
                            x["question"], x["user_profile_data"]
                        )
                    )
                )
                
            )

    def create_post_clarification_chain(self):
        doc_grader = DocumentGrader()
        prompt = ChatPromptTemplate.from_template(template_foodchat) #todo: refine the prompt

        return(
            # RunnablePassthrough.assign(
            #     check = lambda x: rag_preparator.query_check(x['question'], x['modified_user_data'])
            # )
            # |
            RunnablePassthrough.assign(
                    # Based on the diet and allergens list, provide the recipe ids for each course of the day (breakfast,
                    # lunch, dinner)
                    recipe_ids = RunnableLambda(
                        lambda x: get_filtered_recipe_ids(x['modified_user_data']['allergies'], 
                                                                   x['modified_user_data']['diet']
                        )
                    )
                ) 
                | RunnablePassthrough.assign(
                # Retrieval: build the context with a set of graded and retrieved document. 
                # Make the retriever a "naive" knowledge recommender based on the modified user data
                context=lambda x: self.log_retrieval(
                    doc_grader.grading_daily_plans(
                        x["question"],
                        self.retrieve_with_score(x["question"], x['recipe_ids']),
                        x["modified_user_data"],
                    )
                )
            )
            | RunnablePassthrough.assign(
                # Combine the different parts to ge the final prompt to provide to the LLM
                final_input=lambda x: {
                    "context": x["context"],
                    "question": x["question"],
                    "user_profile_context": x[
                        "modified_user_data"
                    ],  # OLD: x["formatted_profile"]
                }
            )
            | RunnableLambda(lambda x: x["final_input"])
         #   | (lambda x: x["final_input"])
         #   | prompt
         #   | self.llm
        )

def initialize_system(args, embeddings):
    # Add check to see if vector db exists already then no need to split doc, ect...

    persist_dir = str(
        VECTORSTORE_PATH
        / f"{args.vectorstore}"
        / f"{args.embedding_model.split('-')[0]}"
        / f"{args.data_type.upper()}"
        / f"{args.dataset.upper()}_SOURCE"
    )
    retriever_instance = Retriever(persist_dir)

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


def run_interactive_chat(foodchat, rag_chain, args):

    current_profile = setup_user_session()

    if current_profile["user_id"] == "default_user":
        print("Session with default user")
    else:
        print(Fore.GREEN + f"Chatting as {current_profile['user_id']}")
        print(format_user_profile(current_profile) + Style.RESET_ALL)

    chatbot = SimpleChatBot()
    check_answer = FoodChatResponseEvaluator()
    query_rewriter = QueryRewriter()

    MAX_RAG_ITERATIONS = 3

    while True:
        query = input("\nUser: ")
        write_to_file(query, user_query=True)

        routing = QueryClassifier().router(query)

        if query.lower() == "exit":
            break

        if routing["source"] == "vectorstore":
            print("VECTORSTORE")
            iterations = 0

            while iterations < MAX_RAG_ITERATIONS:
                response = rag_chain.invoke(
                    {"question": query, "user_id": current_profile["user_id"]}
                )

                if args.adaptive_rag == True:
                    question_list = check_answer.evaluate(query, response)

                    if len(question_list) != 0:
                        query = query_rewriter.rewrite(query, question_list)
                        iterations += 1
                        print(
                            Fore.GREEN
                            + "RAG ITERATIONS: "
                            + str(f"{iterations}/{MAX_RAG_ITERATIONS}")
                            + Style.RESET_ALL
                        )
                    else:
                        break
                else:
                    iterations = MAX_RAG_ITERATIONS
        else:
            print("CHATBOT")
            response = chatbot.chat(query)

        print(f"\nFoodChat: {response}")

        foodchat.memory.save_context({"question": query}, {"answer": response})

        update_user_profile_history(current_profile["user_id"], response)

        write_to_file(response, response=True)
