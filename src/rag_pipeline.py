"""End-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from src.chunker import Chunk, chunk_text
from src.data_loader import QAExample
from src.embeddings import Embedder
from src.llm import GroqLLM
from src.retriever import Retriever
from src.utils import PipelineConfig, get_logger
from src.vector_store import VectorStore

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are an expert question-answering system.

Rules:
1. Use ONLY the provided context.
2. Return ONLY the final answer.
3. Keep the answer as short as possible.
4. Do NOT explain your reasoning.
5. Do NOT add extra words, sentences, or punctuation.
6. If the answer cannot be found in the context, reply exactly:
I don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Build a grounded prompt from retrieved chunks.

    Args:
        question: The user question.
        chunks: The retrieved context chunks.

    Returns:
        A prompt string instructing the model to answer only from the context.
    """
    context = "\n\n".join(f"- {chunk.text}" for chunk in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class RAGPipeline:
    """Orchestrates chunking, indexing, retrieval, and answer generation."""

    def __init__(self, config: PipelineConfig, embedder: Embedder, llm: GroqLLM) -> None:
        """Initialize the pipeline.

        Args:
            config: The pipeline configuration.
            embedder: The embedder used for chunks and queries.
            llm: The LLM used to generate answers.
        """
        self._config = config
        self._embedder = embedder
        self._llm = llm
        self._retriever: Retriever | None = None

    def index_examples(self, examples: list[QAExample]) -> None:
        """Chunk, embed, and index the contexts of all examples.

        Args:
            examples: The examples whose contexts form the knowledge base.
        """
        chunks: list[Chunk] = []
        for example in examples:
            chunks.extend(
                chunk_text(
                    text=example.context,
                    source_id=example.example_id,
                    chunk_size=self._config.chunk_size,
                    chunk_overlap=self._config.chunk_overlap,
                )
            )
        logger.info("Created %d chunks from %d examples.", len(chunks), len(examples))

        embeddings = self._embedder.encode([chunk.text for chunk in chunks])
        vector_store = VectorStore(dimension=self._embedder.dimension)
        vector_store.build_index(embeddings, chunks)
        vector_store.save_index(self._config.index_path)

        self._retriever = Retriever(self._embedder, vector_store)

    def answer(self, question: str) -> str:
        """Retrieve context and generate an answer for a question.

        Args:
            question: The question to answer.

        Returns:
            The generated answer string.

        Raises:
            RuntimeError: If called before :meth:`index_examples`.
        """
        if self._retriever is None:
            raise RuntimeError("Call index_examples() before answer().")

        chunks = self._retriever.retrieve(question, self._config.top_k)
        prompt = build_prompt(question, chunks)
        return self._llm.generate(prompt)
