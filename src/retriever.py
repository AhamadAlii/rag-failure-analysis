"""Retrieve relevant chunks for a query using embeddings + FAISS."""

from __future__ import annotations

from src.chunker import Chunk
from src.embeddings import Embedder
from src.vector_store import VectorStore


class Retriever:
    """Combines an embedder and a vector store to fetch relevant chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        """Initialize the retriever.

        Args:
            embedder: The embedder used to encode queries.
            vector_store: The populated vector store to search.
        """
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int) -> list[Chunk]:
        """Return the ``top_k`` chunks most relevant to ``question``.

        Args:
            question: The user question.
            top_k: Number of chunks to retrieve.

        Returns:
            A list of retrieved :class:`Chunk` objects ordered by relevance.
        """
        query_embedding = self._embedder.encode([question])
        results = self._vector_store.search(query_embedding, top_k)
        return [chunk for chunk, _score in results]
