"""Split passages into overlapping word-based chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A single text chunk derived from a source example.

    Attributes:
        text: The chunk text.
        source_id: Identifier of the example the chunk came from.
    """

    text: str
    source_id: str


def chunk_text(
    text: str, source_id: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks measured in words.

    Args:
        text: The passage to split.
        source_id: Identifier attached to every chunk produced from ``text``.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of :class:`Chunk` objects. Returns an empty list for empty text.

    Raises:
        ValueError: If ``chunk_size`` is not positive or ``chunk_overlap`` is
            negative or greater than or equal to ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in the range [0, chunk_size).")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), source_id=source_id))
        if start + chunk_size >= len(words):
            break
    return chunks
