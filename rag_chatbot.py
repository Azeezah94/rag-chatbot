"""
RAG Chatbot — Document Q&A Agent
Author: Azeezat Akinola

A retrieval-augmented generation chatbot that ingests documents (PDF, TXT, CSV),
builds a vector store, and answers questions grounded in that content with
source citations. Built with LangChain + Chroma + an LLM of your choice.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate


PERSIST_DIR = "chroma_db"

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant answering questions based on the provided documents.
Use ONLY the context below to answer. If the answer isn't in the context, say so honestly.
Always cite which part of the context supports your answer.

Context:
{context}

Question: {question}

Answer (with source reference):"""
)


class DocumentLoader:
    """Loads and chunks documents from a folder into LangChain Document objects."""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_folder(self, folder_path: str) -> list:
        """Load all supported files from a folder and split into chunks."""
        documents = []
        for file_path in Path(folder_path).glob("**/*"):
            ext = file_path.suffix.lower()
            if ext in self.LOADER_MAP:
                loader = self.LOADER_MAP[ext](str(file_path))
                documents.extend(loader.load())
        return self.splitter.split_documents(documents)

    def load_single_file(self, file_path: str) -> list:
        ext = Path(file_path).suffix.lower()
        if ext not in self.LOADER_MAP:
            raise ValueError(f"Unsupported file type: {ext}")
        loader = self.LOADER_MAP[ext](file_path)
        return self.splitter.split_documents(loader.load())


class RAGChatbot:
    """End-to-end RAG pipeline: ingest -> embed -> retrieve -> generate."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", k: int = 4):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatAnthropic(model=model, temperature=0)
        self.k = k
        self.vectorstore = None
        self.qa_chain = None

    def ingest(self, documents: list):
        """Build (or rebuild) the vector store from a list of document chunks."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k}),
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
        )
        print(f"Ingested {len(documents)} chunks into vector store.")

    def load_existing(self):
        """Reload a previously persisted vector store without re-ingesting."""
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=self.embeddings
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k}),
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
        )

    def ask(self, question: str) -> dict:
        """Ask a question and get an answer grounded in the ingested documents."""
        if self.qa_chain is None:
            raise RuntimeError("No documents ingested yet. Call .ingest() first.")
        result = self.qa_chain.invoke({"query": question})
        sources = [
            {"source": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:150]}
            for doc in result.get("source_documents", [])
        ]
        return {"answer": result["result"], "sources": sources}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = RAGChatbot()
    loader = DocumentLoader()

    # Point this at a folder of your own PDFs/TXT/CSV files
    docs = loader.load_folder("./sample_docs")
    bot.ingest(docs)

    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        response = bot.ask(question)
        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources: {[s['source'] for s in response['sources']]}")
