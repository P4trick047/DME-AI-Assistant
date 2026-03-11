# ============================================================
# src/vector_store.py
# ChromaDB vector store — stores and retrieves document embeddings
# Think of this as a semantic search engine for your documents
# ============================================================

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
)

logger = logging.getLogger(__name__)


class DMEVectorStore:
    """
    Manages the ChromaDB vector store for semantic document retrieval.

    How it works:
      1. Text chunks are converted to numerical vectors (embeddings)
         by the embedding model (nomic-embed-text)
      2. These vectors are stored in ChromaDB on disk
      3. At query time, the question is also converted to a vector
      4. ChromaDB finds the most similar stored vectors (nearest neighbors)
      5. The matching text chunks are returned as context for the LLM

    This means searching "wheelchair approval criteria" will also find
    chunks about "power mobility device authorization requirements" because
    they are semantically similar, even with different words.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing embedding model: {embedding_model}")
        self.embeddings = OllamaEmbeddings(model=embedding_model)

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

    # ── Write ────────────────────────────────────────────────

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Add document chunks to the vector store.
        Batched to avoid out-of-memory errors on large document sets.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} chunks to vector store...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(
                f"  Batch {i // batch_size + 1} / {(len(documents) - 1) // batch_size + 1} done"
            )

        logger.info(f"Vector store total: {self._safe_count()} chunks")

    def delete_collection(self) -> None:
        """Wipe the entire vector store (used when re-indexing all documents)."""
        self.vectorstore.delete_collection()
        # Recreate empty collection
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("Vector store cleared and recreated")

    # ── Read ─────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = RETRIEVAL_K,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        """
        Find the most semantically relevant chunks for a query.

        Args:
            query:       The user's question or search text
            k:           Number of chunks to retrieve
            filter_dict: Optional metadata filter e.g. {"file_type": "pdf"}
        """
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = RETRIEVAL_K,
    ) -> List[Tuple[Document, float]]:
        """Same as similarity_search but also returns relevance scores (0–1)."""
        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

    def get_retriever(self, k: int = RETRIEVAL_K, search_type: str = "similarity"):
        """
        Return a LangChain-compatible Retriever object.
        Plug this directly into any LangChain chain or agent.
        """
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    # ── Info ─────────────────────────────────────────────────

    def get_document_count(self) -> int:
        """Return number of stored chunks."""
        return self._safe_count()

    def _safe_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
