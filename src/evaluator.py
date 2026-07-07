"""Evaluation metrics (Exact Match, token F1) and CSV result storage."""

from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd


def normalize_answer(text: str) -> str:
    """Normalize an answer for fair comparison.

    Lowercases, removes punctuation and articles, and collapses whitespace,
    following the standard SQuAD normalization scheme.

    Args:
        text: The raw answer text.

    Returns:
        The normalized answer string.
    """
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if the normalized answers match exactly, else 0.0.

    Args:
        prediction: The predicted answer.
        ground_truth: The reference answer.

    Returns:
        ``1.0`` for an exact match, otherwise ``0.0``.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute the token-level F1 score between two answers.

    Args:
        prediction: The predicted answer.
        ground_truth: The reference answer.

    Returns:
        The F1 score in the range ``[0.0, 1.0]``.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        # If either is empty, F1 is 1.0 only when both are empty.
        return float(pred_tokens == truth_tokens)

    common = 0
    truth_pool = list(truth_tokens)
    for token in pred_tokens:
        if token in truth_pool:
            common += 1
            truth_pool.remove(token)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def save_results(rows: list[dict[str, object]], path: Path) -> pd.DataFrame:
    """Save per-question results to a CSV file.

    Args:
        rows: A list of dictionaries, one per evaluated question.
        path: Destination CSV path.

    Returns:
        The results as a pandas DataFrame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame
