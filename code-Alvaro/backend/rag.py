# backend/rag.py

import os
from typing import List
import chromadb

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class BasicRAG:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="llama-3.2-3b-instruct",
            base_url="http://127.0.0.1:1234/v1",
            api_key="not_required",
            temperature=0.1
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        

        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name="basic_rag",
            embedding_function=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_documents_from_files(self, file_paths: List[str]):

        new_docs = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue

            raw_docs = loader.load()
            splits = self.text_splitter.split_documents(raw_docs)

            for i, doc in enumerate(splits):
                doc.metadata["source"] = filename
                doc.metadata["chunk_index"] = i

            new_docs.extend(splits)

        if new_docs:
            self.vectorstore.add_documents(new_docs)
            return f"Added {len(new_docs)} chunks."

        return "No valid documents added."

    def query(self, question: str, selected_files: List[str]):

        #lo blindeamos por seguridad
        selected_files = [os.path.basename(f) for f in selected_files]
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"source": {"$in": selected_files}}
            }
        )

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant context found."

        template = """
        Use ONLY the provided context to answer.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        context_text = "\n\n".join(d.page_content for d in docs)

        return chain.invoke({
            "context": context_text,
            "question": question
        })
        
    
    def list_documents(self):

        # Obtener colección directamente de chromaDB para acceder a los metadatos sin necesidad de recuperar embeddings
        collection = self.vectorstore._collection

        # Traer solo metadatos (sin embeddings)
        results = collection.get(include=["metadatas"])

        if not results or "metadatas" not in results:
            return []

        sources = [
            metadata.get("source")
            for metadata in results["metadatas"]
            if metadata and "source" in metadata
        ]

        # Quitar duplicados y ordenar
        unique_sources = sorted(list(set(sources)))

        return unique_sources