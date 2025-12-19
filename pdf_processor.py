from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re

SOURCE_NAME = "20_International_Recipes"


class PDFProcessor:
    def __init__(self, data_path, split_method, embedding_model):
        self.data_path = data_path
        # self.embeddings = OllamaEmbeddings(model=embedding_model) #https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html
        self.embeddings = embedding_model
        self.split_method = split_method

    def load_and_chunk_pdf(self):
        """Load and process PDF based on splitting method."""
        loader = UnstructuredPDFLoader(self.data_path, mode="elements")
        docs = loader.load()

        if self.split_method == "page_split" : 
            processed_docs = self.split_by_page(docs)
        elif self.split_method == "recipe_split": 
            processed_docs = self.split_by_recipe(docs)
        elif self.split_method == "recursive_character_split" : 
            processed_docs = self.split_by_recursive_character(docs)
        
        return processed_docs
    
    def clean_content(self, text) : 
        # Remove special characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f\uf000-\uf0ff-\uf0b7]', ' ', text)
        # Fix hyphenated words
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        # Remove page headers/footers
        text = re.sub(r'Page \d+ of \d+|Images from allrecipes\.com.*', '', text)

        text = re.sub('20 Easy International Recipes', '', text)
        # Collapse whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def split_by_page(self, docs):
        """Split PDF content by page."""
        pages = {}
        for doc in docs:
            page_num = doc.metadata.get("page_number", 1)
            pages.setdefault(page_num, []).append(doc.page_content)
        for k in pages.keys(): 
            pages[k] = self.clean_content(','.join(pages[k]))
        return [Document(page_content=content, metadata={"source" : SOURCE_NAME, "page": page}) 
       for page, content in pages.items()]

    def split_by_recipe(self, docs):
        """Split content by recipe"""
        recipes = {}
        processed_docs = []
        recipe_pattern =  r"(?=\d+\.\s[A-Z][a-zA-Z\s]+(?:\s\([^)]*\))?\sIngredients)"
        pages = {}
        for doc in docs:
            page_num = doc.metadata.get("page_number", 1)
            pages.setdefault(page_num, []).append(doc.page_content)
        
        for k in pages.keys(): 
            pages[k] = self.clean_content('\n\n'.join(pages[k]))

        for k, content in pages.items() : 
            splitted_content = re.split(recipe_pattern, content)
            splitted_content = [elem for elem in splitted_content if elem !='' and len(elem)>1]
            recipes[k+(k-1)] = splitted_content[0]
            recipes[k+k] = splitted_content[1]

        for page, content in recipes.items():
            processed_docs.append(
                Document(page_content = content,metadata={"source" : SOURCE_NAME, "page":page, "content_type" : "recipe"})
                )
        return processed_docs

    def split_by_recursive_character(self, docs):
        new_splits = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        splits = text_splitter.split_documents(docs)

        for i in range(len(splits)) : 
            if len(self.clean_content(splits[i].page_content)) > 1: 
                new_splits.append(splits[i])
        
        return new_splits