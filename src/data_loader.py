"""Load a small, configurable slice of the SQuAD v2 dataset."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class QAExample:
    """A single question-answering example.

    Attributes:
        example_id: Unique identifier from the source dataset.
        question: The natural-language question.
        context: The passage that may contain the answer.
        answer: The ground-truth answer, or an empty string if unanswerable.
    """

    example_id: str
    question: str
    context: str
    answer: str


def load_squad_examples(num_examples: int) -> list[QAExample]:
    """Load ``num_examples`` answerable examples from SQuAD v2.

    Only answerable questions are kept so the baseline metrics stay meaningful.

    Args:
        num_examples: Maximum number of examples to return.

    Returns:
        A list of :class:`QAExample` objects.
    """
    logger.info("Loading SQuAD v2 validation split from HuggingFace...")
    # Use the canonical namespaced repo id. Newer huggingface_hub rejects the
    # bare name "squad_v2" with an "Invalid HF URI" error.
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    examples: list[QAExample] = []
    for row in dataset:
        answer_texts = row["answers"]["text"]
        if not answer_texts:
            # Skip unanswerable questions for the baseline.
            continue
        examples.append(
            QAExample(
                example_id=str(row["id"]),
                question=row["question"].strip(),
                context=row["context"].strip(),
                answer=answer_texts[0].strip(),
            )
        )
        if len(examples) >= num_examples:
            break

    logger.info("Loaded %d examples.", len(examples))
    return examples
