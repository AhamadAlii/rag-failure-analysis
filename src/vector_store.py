"""A minimal FAISS-backed vector store for chunk retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from src.chunker import Chunk
from src.utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Stores chunk embeddings in a FAISS index alongside the chunk texts.

    Attributes:
        dimension: Dimensionality of the stored embedding vectors.
    """

    def __init__(self, dimension: int) -> None:
        """Initialize an empty inner-product FAISS index.

        Args:
            dimension: Embedding vector dimension.
        """
        self.dimension = dimension
        self._index: faiss.Index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def build_index(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Populate the index with embeddings and their matching chunks.

        Args:
            embeddings: A ``(n, dimension)`` float32 array of vectors.
            chunks: The chunks corresponding row-for-row to ``embeddings``.

        Raises:
            ValueError: If the number of embeddings and chunks differ.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info("Built FAISS index with %d vectors.", self._index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks for a single query.

        Args:
            query_embedding: A ``(1, dimension)`` or ``(dimension,)`` float32 array.
            top_k: Number of results to return.

        Returns:
            A list of ``(chunk, score)`` tuples ordered by descending similarity.
        """
        query = np.atleast_2d(query_embedding).astype(np.float32)
        scores, indices = self._index.search(query, top_k)

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save_index(self, path: Path) -> None:
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            path: Destination path for the FAISS index. Chunk metadata is saved
                next to it with a ``.meta`` suffix.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        with open(path.with_suffix(".meta"), "wb") as meta_file:
            pickle.dump(self._chunks, meta_file)
        logger.info("Saved FAISS index to %s", path)

    @classmethod
    def load_index(cls, path: Path) -> "VectorStore":
        """Load a previously saved index and its chunk metadata.

        Args:
            path: Path to the saved FAISS index.

        Returns:
            A ready-to-search :class:`VectorStore` instance.
        """
        index = faiss.read_index(str(path))
        with open(path.with_suffix(".meta"), "rb") as meta_file:
            chunks = pickle.load(meta_file)

        store = cls(dimension=index.d)
        store._index = index
        store._chunks = chunks
        logger.info("Loaded FAISS index from %s (%d vectors).", path, index.ntotal)
        return store
