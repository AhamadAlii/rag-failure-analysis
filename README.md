# Failure Analysis of Retrieval-Augmented Generation (RAG)

A simple, reproducible RAG baseline built for a Research & Solution Development
take-home assignment.

**Research question:** How do retrieval parameters such as chunk size, top-k
retrieval, and embedding model affect question-answering performance and
retrieval failures?

This repository intentionally implements only a clean **baseline pipeline**. The
research experiments are left as placeholders to be implemented later.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for question answering on the SQuAD v2 dataset.

The baseline system uses:
- Sentence-Transformers for embeddings
- FAISS for vector search
- Groq Llama 3.3 70B for answer generation
- Exact Match (EM) and Token F1 for evaluation

As an initial research extension, a prompt engineering experiment was conducted to evaluate the impact of concise prompting on QA performance.

## Installation

Requires **Python 3.12+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env           # Windows: copy .env.example .env
# then edit .env and set GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com/keys.

## Dataset

[SQuAD v2](https://huggingface.co/datasets/squad_v2) is loaded from HuggingFace.
Only a configurable number of **answerable** examples from the validation split
are used. Each example provides a `context`, a `question`, and an `answer`.

## Architecture

```
Load Dataset
   ↓
Split Context into Chunks
   ↓
Generate Embeddings
   ↓
Store in FAISS
   ↓
Retrieve Top-K
   ↓
Build Prompt
   ↓
Send Prompt to LLM (Groq)
   ↓
Generate Answer
   ↓
Compare Against Ground Truth (Exact Match + Token F1)
```

## How to Run

```bash
python main.py
```

This will load the data, build embeddings and a FAISS index, run RAG over every
question, print each answer, compute the average metrics, and save per-question
results to `results/results.csv`.

To change parameters (number of examples, chunk size, top-k, embedding model),
edit the defaults in `src/utils.py` (`PipelineConfig`).

## Folder Structure

```
rag-failure-analysis/
├── data/                    # Cached dataset artifacts and saved FAISS index
├── experiments/             # Research experiment placeholders (TODO only)
│   ├── chunk_size.py
│   ├── topk.py
│   └── embedding_models.py
├── results/                 # Saved metric CSVs
├── src/
│   ├── data_loader.py       # Load SQuAD v2 examples
│   ├── chunker.py           # Split context into overlapping chunks
│   ├── embeddings.py        # sentence-transformers wrapper
│   ├── vector_store.py      # FAISS build/save/load/search
│   ├── retriever.py         # Embed query + search index
│   ├── llm.py               # Groq chat client
│   ├── rag_pipeline.py      # Orchestrates the full RAG flow
│   ├── evaluator.py         # Exact Match + token F1 + CSV output
│   └── utils.py             # Config, paths, logging
├── main.py                  # End-to-end entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Experimental Results

| Experiment | Exact Match | Token F1 |
|------------|------------:|---------:|
| Baseline Prompt | 0.37 | 0.53 |
| Improved Prompt | **0.62** | **0.682** |

### Observation

A more constrained prompt significantly reduced verbose answers and improved both Exact Match and Token F1 without modifying the retrieval pipeline.

## Notes

This project uses the Groq API for answer generation.

If a `429 RateLimitError` occurs, the free-tier token quota has been exhausted. The project can be rerun after the quota resets or by using another valid Groq API key.

## Future Work

- Evaluate different chunk sizes (100, 200, 400, 800)
- Compare top-k retrieval settings (1, 3, 5)
- Evaluate alternative embedding models
- Perform detailed retrieval failure analysis
- Visualize performance across configurations