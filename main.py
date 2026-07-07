"""Run the baseline RAG pipeline end-to-end over a SQuAD v2 sample.

Steps:
    1. Load data.
    2. Build embeddings and a FAISS index.
    3. Retrieve context and generate answers with the LLM.
    4. Compute Exact Match and token F1 metrics.
    5. Save per-question results to a CSV file.
"""

from __future__ import annotations

from tqdm import tqdm

from src.data_loader import load_squad_examples
from src.embeddings import Embedder
from src.evaluator import exact_match, save_results, token_f1
from src.llm import GroqLLM
from src.rag_pipeline import RAGPipeline
from src.utils import PipelineConfig, ensure_directories, get_logger

logger = get_logger(__name__)


def run(config: PipelineConfig) -> None:
    """Execute the full pipeline for a given configuration.

    Args:
        config: The pipeline configuration to run.
    """
    ensure_directories()

    examples = load_squad_examples(config.num_examples)

    embedder = Embedder(config.embedding_model)
    llm = GroqLLM(config.llm_model)
    pipeline = RAGPipeline(config, embedder, llm)
    pipeline.index_examples(examples)

    rows: list[dict[str, object]] = []
    total_em = 0.0
    total_f1 = 0.0

    for example in tqdm(examples, desc="Answering questions"):
        prediction = pipeline.answer(example.question)
        em = exact_match(prediction, example.answer)
        f1 = token_f1(prediction, example.answer)
        total_em += em
        total_f1 += f1

        print(f"\nQ: {example.question}")
        print(f"Predicted: {prediction}")
        print(f"Expected:  {example.answer}")
        print(f"EM={em:.0f}  F1={f1:.2f}")

        rows.append(
            {
                "id": example.example_id,
                "question": example.question,
                "expected": example.answer,
                "predicted": prediction,
                "exact_match": em,
                "token_f1": f1,
            }
        )

    num = max(len(examples), 1)
    print("\n=== Summary ===")
    print(f"Examples:        {len(examples)}")
    print(f"Avg Exact Match: {total_em / num:.3f}")
    print(f"Avg Token F1:    {total_f1 / num:.3f}")

    save_results(rows, config.results_path)
    logger.info("Saved results to %s", config.results_path)


def main() -> None:
    """Entry point using the default configuration."""
    run(PipelineConfig())


if __name__ == "__main__":
    main()
