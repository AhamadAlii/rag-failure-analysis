"""Shared helper utilities for the RAG failure-analysis project."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# Project directory layout, resolved relative to this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


@dataclass
class PipelineConfig:
    """Configuration for a single RAG pipeline run.

    Attributes:
        num_examples: Number of SQuAD v2 examples to load.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
        top_k: Number of chunks to retrieve for each question.
        embedding_model: Sentence-transformers model name used for embeddings.
        llm_model: Groq chat model used to generate answers.
        index_path: File path where the FAISS index is saved/loaded.
        results_path: CSV file path where per-question results are stored.
    """

    num_examples: int = 20
    chunk_size: int = 100
    chunk_overlap: int = 20
    top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    index_path: Path = field(default_factory=lambda: DATA_DIR / "faiss.index")
    results_path: Path = field(default_factory=lambda: RESULTS_DIR / "results.csv")


def ensure_directories() -> None:
    """Create the ``data/`` and ``results/`` directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger.

    Args:
        name: Logger name, usually ``__name__`` of the calling module.

    Returns:
        A logger that writes INFO-level messages to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
