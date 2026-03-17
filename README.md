# QueryRewrite — LLM-based Query Intent Alignment for Search Optimization

## Overview

This project improves retrieval performance through:

- **Query rewriting** — an LLM (LoRA-finetuned) rewrites user queries for better intent coverage.
- **Embedding-based retrieval** — a bi-encoder maps queries and documents into a shared vector space.
- **Alignment training** — contrastive learning aligns query and document representations.

## Project Structure

```
QueryRewrite/
├── configs/                # Configuration
│   ├── config.py           # Typed dataclass configs + YAML loader
│   └── default.yaml        # Default hyper-parameters
├── data/                   # Dataset construction
│   ├── dataset.py          # PyTorch Dataset (query, pos_doc, neg_doc)
│   ├── collator.py         # Batch tokenisation & padding
│   └── preprocessing.py    # Raw data cleaning & splitting
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
│   ├── metrics.py          # Recall@K, NDCG@K, MRR
│   └── evaluator.py        # High-level evaluation runner
├── scripts/                # CLI entry-points
│   ├── train.py            # Train the model
│   ├── evaluate.py         # Evaluate a checkpoint
│   ├── build_index.py      # Build FAISS index
│   └── preprocess.py       # Preprocess raw data
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python scripts/preprocess.py --raw_dir data/raw --output_dir data/processed

# 3. Train
python scripts/train.py --config configs/default.yaml

# 4. Build index
python scripts/build_index.py --config configs/default.yaml \
    --checkpoint outputs/best/ --corpus data/raw/corpus.jsonl

# 5. Evaluate
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best/
```

## Design Principles

| Principle | How |
|---|---|
| **Decoupled modules** | Data, models, training, retrieval, and evaluation are independent packages |
| **Typed configuration** | Dataclass configs parsed from YAML; no magic strings |
| **Easy to extend** | Swap embedder, loss, or index by changing one config field |
| **Research-friendly** | Skeleton-first design; fill in logic incrementally |
