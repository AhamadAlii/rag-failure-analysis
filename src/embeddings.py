"""Generate embeddings using a configurable sentence-transformers model."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import get_logger

logger = get_logger(__name__)


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    Attributes:
        model_name: Name of the underlying sentence-transformers model.
    """

    def __init__(self, model_name: str) -> None:
        """Load the embedding model.

        Args:
            model_name: A sentence-transformers model name.
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into normalized float32 vectors.

        Vectors are L2-normalized so that inner-product search on a FAISS
        ``IndexFlatIP`` is equivalent to cosine similarity.

        Args:
            texts: Texts to embed.

        Returns:
            A ``(len(texts), dimension)`` float32 numpy array.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
