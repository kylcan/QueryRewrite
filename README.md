# QueryRewrite — LLM-based Query Intent Alignment for Search Optimization

## Overview

This repo is a research workspace for improving retrieval through **query rewriting** and **embedding-based search**.

Current focus (working end-to-end):

- Build an MS MARCO derived dataset with **tiered negatives** and (optional) **LLM rewrites**
- Evaluate retrieval quality, rewrite impact, and **rewrite gating** (“when to rewrite”)

Longer-term (scaffolded / partially implemented):

- Trainable models (rewriter/embedder/alignment)
- General-purpose data preprocessing and training CLI

## Project Structure

```
QueryRewrite/
├── configs/                # Configuration
│   ├── config.py           # Typed dataclass configs + YAML loader
│   └── default.yaml        # Default hyper-parameters
├── data/                   # Dataset construction
│   ├── dataset.py          # PyTorch Dataset (query, pos_doc, neg_doc)
│   ├── collator.py         # Batch tokenisation & padding
│   ├── preprocessing.py    # Raw data cleaning & splitting
│   └── MS_MARCO/           # Working MS MARCO pipeline
│       ├── dataset_build.py # Build intent-aware dataset + mine negatives + (optional) LLM rewrite
│       ├── intent_dataset.jsonl
│       └── (outputs only)   # Eval outputs are NOT stored here
├── models/                 # Model definitions
│   ├── rewriter.py         # LLM + LoRA query rewriter
│   ├── embedder.py         # Bi-encoder embedding model
│   ├── alignment.py        # Joint alignment model
│   └── losses.py           # InfoNCE, Triplet, factory
├── training/               # Training pipeline
│   ├── trainer.py          # Training loop, validation, checkpointing
│   └── scheduler.py        # LR scheduler factory
├── retrieval/              # Retrieval system
│   ├── indexer.py          # FAISS index builder
│   └── searcher.py         # End-to-end search interface
├── evaluation/             # Evaluation
│   ├── shared.py           # Shared load/encode/index context
│   ├── metrics.py          # Metrics (array + list APIs)
│   ├── eval_dataset.py      # RQ1+RQ2: dataset quality
│   ├── eval_rewrite.py      # RQ3: rewrite effect
│   ├── eval_gating.py       # RQ4: rewrite gating threshold sweep
│   ├── run_eval.py          # Unified runner (RQ1–RQ4)
│   ├── eval_retrieval_legacy.py # Legacy single-file evaluator (kept for reference)
│   └── result/              # ALL eval outputs are stored here
├── scripts/                # CLI entry-points
│   ├── train.py            # Train the model
│   ├── evaluate.py         # Evaluate a checkpoint
│   ├── build_index.py      # Build FAISS index
│   └── preprocess.py       # Preprocess raw data
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start (MS MARCO pipeline)

### 1) Install

```bash
pip install -r requirements.txt
```

If you use a virtualenv, activate it first (example):

```bash
source .venv/bin/activate
```

### 2) (Optional) Configure LLM rewriting

LLM rewrite is optional. If environment variables are present, the dataset builder will call the LLM; otherwise it will fall back to rule-based rewriting.

Supported env vars:

- `GPT5_KEY` (or `OPENAI_API_KEY`)
- `CHATGPT_MODEL`
- `CHATGPT_BASE_URL`

### 3) Build the dataset

This produces `data/MS_MARCO/intent_dataset.jsonl`.

```bash
python data/MS_MARCO/dataset_build.py
```

### 4) Run evaluation (RQ1–RQ4)

All evaluation outputs are saved under `evaluation/result/`.

```bash
python -m evaluation.run_eval
```

Run a subset:

```bash
python -m evaluation.run_eval --only dataset
python -m evaluation.run_eval --only rewrite
python -m evaluation.run_eval --only gating
```

Individual modules also work:

```bash
python -m evaluation.eval_dataset
python -m evaluation.eval_rewrite
python -m evaluation.eval_gating
```

## Notes

- The current evaluation uses `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` with L2-normalised embeddings (inner product == cosine).
- If you see HuggingFace rate-limit warnings, set `HF_TOKEN` to improve download speed.

## What’s implemented vs WIP

- ✅ Working: MS MARCO dataset build + retrieval evaluation + rewrite evaluation + rewrite gating
- ⚠️ WIP: `data/collator.py`, `data/preprocessing.py`, and parts of the training/evaluation scaffolding under `scripts/`

## Design Principles

| Principle | How |
|---|---|
| **Decoupled modules** | Data, models, training, retrieval, and evaluation are independent packages |
| **Typed configuration** | Dataclass configs parsed from YAML; no magic strings |
| **Easy to extend** | Swap embedder, loss, or index by changing one config field |
| **Research-friendly** | Skeleton-first design; fill in logic incrementally |
