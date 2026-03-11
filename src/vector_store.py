# ============================================================
# src/vector_store.py
# ChromaDB vector store — uses HuggingFace embeddings on cloud
# (no Ollama needed for embeddings)
# ============================================================

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_K,
    USE_GROQ,
)

logger = logging.getLogger(__name__)


def _build_embeddings():
    """
    Use HuggingFace embeddings (free, no API key, works everywhere).
    Falls back to Ollama embeddings only when running locally.
    HuggingFace model runs in-process — no external API call.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        logger.info("HuggingFace not available, falling back to Ollama embeddings")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")


class DMEVectorStore:
    """
    ChromaDB vector store with auto-selected embedding model.
    Works on Streamlit Cloud, local machine, Docker — anywhere.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = _build_embeddings()

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        count = self._safe_count()
        if count > 0:
            logger.info(f"Loaded existing vector store: {count} chunks")
        else:
            logger.info("Created new empty vector store")

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        if not documents:
            return
        logger.info(f"Indexing {len(documents)} chunks...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
        logger.info(f"Vector store total: {self._safe_count()} chunks")

    def delete_collection(self) -> None:
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("Vector store cleared")

    def similarity_search(
        self, query: str, k: int = RETRIEVAL_K, filter_dict: Optional[dict] = None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def get_retriever(self, k: int = RETRIEVAL_K, search_type: str = "similarity"):
        return self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )

    def get_document_count(self) -> int:
        return self._safe_count()

    def _safe_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
