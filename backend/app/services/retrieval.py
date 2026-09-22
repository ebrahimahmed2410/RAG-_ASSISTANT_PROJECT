"""
Vector Store and Retrieval Service using ChromaDB and Sentence-Transformers
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from app.utils.logging_config import logger


class VectorStoreService:
    """
    Manages persistent ChromaDB vector store and semantic search retrieval.
    Loaded once at application startup during the FastAPI lifespan.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "cs_documents",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_fn = None
        self._is_ready = False

    def initialize(self) -> bool:
        """
        Initializes the ChromaDB persistent client, embedding function, and collection.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info(f"Initializing VectorStoreService at '{self.persist_dir}'...")
            os.makedirs(self.persist_dir, exist_ok=True)

            # Initialize sentence-transformers embedding function
            logger.info(f"Loading embedding model: '{self.embedding_model_name}'...")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )

            # Initialize ChromaDB persistent client
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

            count = self.collection.count()
            logger.info(
                f"Vector store initialized successfully. Collection '{self.collection_name}' contains {count} chunks."
            )
            self._is_ready = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {str(e)}", exc_info=True)
            self._is_ready = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.collection is not None

    def get_document_count(self) -> int:
        """Returns the total number of indexed chunks in the collection."""
        if not self.is_ready:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the indexed document chunks.
        
        Returns a list of dictionaries with keys:
            - text: str
            - document: str
            - page: int
            - chunk_id: str
            - distance: float
            - relevance_score: float
        """
        if not self.is_ready:
            logger.warning("Vector store is not ready. Returning empty search results.")
            return []

        try:
            count = self.get_document_count()
            if count == 0:
                logger.warning("Vector store collection is empty. Please run indexing.")
                return []

            actual_k = min(top_k, count)
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_chunks = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc_text, meta, dist in zip(docs, metas, distances):
                    # For cosine distance in [0, 2], relevance = 1.0 - (distance / 2)
                    # or 1.0 / (1.0 + distance)
                    score = round(max(0.0, 1.0 - (dist / 2.0)), 4) if dist is not None else 1.0
                    retrieved_chunks.append({
                        "text": doc_text,
                        "document": meta.get("document", "Unknown Document"),
                        "page": int(meta.get("page", 1)),
                        "chunk_id": meta.get("chunk_id", "unknown"),
                        "distance": round(dist, 4) if dist is not None else None,
                        "relevance_score": score,
                    })

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:40]}...'")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error executing similarity search: {str(e)}", exc_info=True)
            return []
