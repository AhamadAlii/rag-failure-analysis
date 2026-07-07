# Failure Analysis of Retrieval-Augmented Generation (RAG)

A simple, reproducible RAG baseline built for a Research & Solution Development
take-home assignment.

**Research question:** How do retrieval parameters such as chunk size, top-k
retrieval, and embedding model affect question-answering performance and
retrieval failures?

This repository intentionally implements only a clean **baseline pipeline**. The
research experiments are left as placeholders to be implemented later.

## Project Overview

The pipeline loads a small slice of SQuAD v2, chunks each context, embeds the
chunks with a sentence-transformers model, stores them in a FAISS index,
retrieves the top-k chunks for each question, prompts a Groq-hosted LLM to answer
using only that context, and finally scores the answers with Exact Match and
token F1.

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

## Future Work

- Implement the three experiments in `experiments/` (chunk size, top-k,
  embedding model sweeps).
- Add retrieval-failure diagnostics (e.g. was the gold answer present in any
  retrieved chunk?).
- Include unanswerable SQuAD v2 questions to measure "I don't know" behavior.
- Add plots comparing metrics across configurations.
