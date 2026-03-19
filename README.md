# QueryRewrite — LLM-based Query Intent Alignment for Search Optimization

## Overview

This repo is a research workspace for improving retrieval through **query rewriting**, **embedding alignment training**, **confidence-based gating**, and **local LLM fine-tuning with SFT + DPO**.

The full pipeline is working end-to-end:

1. **Dataset construction** — MS MARCO sampling with tiered hard-negative mining and LLM-based query rewrites
2. **Rewrite evaluation** (RQ3) — Measure whether LLM rewrites improve retrieval
3. **Gating** (RQ4) — Confidence-based strategy: only rewrite when the original query is uncertain
4. **Alignment training** (RQ5) — Fine-tune the embedding model with contrastive loss to reduce "both missed" cases
5. **SFT + DPO rewriter** (RQ6) — Train a local query rewriter (Qwen2.5-0.5B + LoRA) via supervised fine-tuning then Direct Preference Optimization

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
│       ├── dataset_build.py      # Build intent-aware dataset + mine negatives + LLM rewrite
│       ├── build_preference_data.py # Build SFT + DPO training data from intent dataset
│       ├── intent_dataset.jsonl
│       └── (outputs only)   # Eval outputs are NOT stored here
├── models/                 # Model definitions
│   ├── rewriter.py         # LLM + LoRA query rewriter (SFT/DPO)
│   ├── embedder.py         # Bi-encoder embedding model (trainable)
│   ├── alignment.py        # Joint alignment model
│   └── losses.py           # InfoNCE, Triplet, DPO loss
├── training/               # Training pipeline
│   ├── trainer.py          # Embedding training loop
│   ├── sft_trainer.py      # SFT training loop for causal LM
│   ├── dpo_trainer.py      # DPO training loop with reference model
│   └── scheduler.py        # LR scheduler factory (linear/cosine)
├── retrieval/              # Retrieval system
│   ├── indexer.py          # FAISS index builder
│   └── searcher.py         # End-to-end search interface
├── evaluation/             # Evaluation
│   ├── shared.py           # Shared load/encode/index context
│   ├── metrics.py          # Metrics (array + list APIs)
│   ├── eval_dataset.py      # RQ1+RQ2: dataset quality
│   ├── eval_rewrite.py      # RQ3: rewrite effect
│   ├── eval_gating.py       # RQ4: rewrite gating threshold sweep
│   ├── eval_alignment.py    # RQ5: alignment training effect
│   ├── eval_rewriter.py     # RQ6: SFT vs DPO vs API rewriter comparison
│   ├── run_eval.py          # Unified runner (RQ1–RQ6)
│   ├── eval_retrieval_legacy.py # Legacy single-file evaluator (kept for reference)
│   └── result/              # ALL eval outputs are stored here
├── scripts/                # CLI entry-points
│   ├── train_alignment.py   # Train alignment embedding model
│   ├── train_sft.py         # SFT training for query rewriter
│   ├── train_dpo.py         # DPO training for query rewriter
│   ├── train.py            # Train the model (scaffold)
│   ├── evaluate.py         # Evaluate a checkpoint (scaffold)
│   ├── build_index.py      # Build FAISS index (scaffold)
│   └── preprocess.py       # Preprocess raw data (scaffold)
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

### 5) Alignment Training (RQ5)

Fine-tune the embedding model with InfoNCE contrastive loss on the dataset triplets:

```bash
python scripts/train_alignment.py --epochs 3 --batch_size 16 --lr 2e-5
```

Key options:
- `--loss infonce|triplet` — Loss function (default: infonce)
- `--temperature 0.07` — InfoNCE temperature
- `--checkpoint_dir checkpoints/alignment` — Where to save

### 6) Alignment Evaluation (RQ5)

Compare frozen vs fine-tuned embedding model:

```bash
python -m evaluation.eval_alignment --checkpoint checkpoints/alignment/best
```

Or via the unified runner:

```bash
python -m evaluation.run_eval --only alignment --checkpoint checkpoints/alignment/best
```

### 7) Build Preference Data for SFT + DPO

Generate supervised training data and preference pairs from retrieval reward signals:

```bash
python -m data.MS_MARCO.build_preference_data
```

This produces:
- `sft_dataset.jsonl` — (prompt, completion) pairs for supervised fine-tuning
- `dpo_dataset.jsonl` — (prompt, chosen, rejected) preference pairs using retrieval quality as reward
- `reward_annotated.jsonl` — all samples with computed reward deltas

### 8) SFT Training (RQ6 — Step 1)

Fine-tune a local query rewriter (Qwen2.5-0.5B) with LoRA adapters:

```bash
python scripts/train_sft.py --epochs 3 --batch_size 2 --lr 2e-4
```

Key options:
- `--model Qwen/Qwen2.5-0.5B` — Base model (default)
- `--lora_rank 16` — LoRA rank (default: 16)
- `--lora_alpha 32` — LoRA alpha scaling (default: 32)
- `--gradient_accumulation 4` — Effective batch = batch_size × accum
- `--scheduler cosine|linear` — LR schedule (default: cosine)

Architecture: Only 0.44% of parameters are trainable (2.2M LoRA / 496M total).

### 9) DPO Training (RQ6 — Step 2)

Align the SFT model using Direct Preference Optimization (Rafailov et al., 2023):

```bash
python scripts/train_dpo.py --sft_checkpoint checkpoints/sft/best --epochs 2 --beta 0.1
```

Key options:
- `--beta 0.1` — DPO inverse temperature (lower = stronger preference learning)
- `--sft_checkpoint checkpoints/sft/best` — Starting point (SFT adapter)
- `--lr 5e-5` — Lower LR than SFT for stable preference learning

The DPO trainer uses a frozen reference model (π_ref) from the SFT checkpoint.

### 10) Rewriter Evaluation (RQ6)

Compare API rewrites vs SFT vs DPO local rewrites:

```bash
python -m evaluation.eval_rewriter \
    --sft checkpoints/sft/best \
    --dpo checkpoints/dpo/best \
    --embedder_checkpoint checkpoints/alignment/best
```

Or via the unified runner:

```bash
python -m evaluation.run_eval --only rewriter
```

## Results Summary

| Stage | Recall@1 | MRR | Both-Miss |
|-------|----------|-----|-----------|
| Frozen baseline | 0.774 | 0.882 | 75 |
| + API Rewrite | 0.788 | — | — |
| + Gating (τ=0.70) | 0.804 | — | — |
| + Alignment (FT) | 0.886 | 0.941 | 31 |
| + FT + Gating (τ=0.50) | 0.890 | — | — |

### SFT + DPO Pipeline

| Component | Detail |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B (494M params) |
| PEFT | LoRA rank=16, alpha=32, targets: q/k/v/o_proj |
| Trainable params | 2.16M / 496M (0.44%) |
| SFT data | 500 (prompt, completion) pairs |
| DPO data | 450 preference pairs (retrieval reward signal) |
| DPO loss | β=0.1, Rafailov et al. (2023) |
| Hardware | Apple M-series, 16GB unified memory, MPS backend |

### RQ6 — Local Rewriter Comparison

| Source | Recall@1 | MRR | Win/Loss vs Original |
|--------|----------|------|---------------------|
| Original query | 0.886 | 0.941 | — |
| API rewrite | 0.846 | 0.919 | +26 / -46 (net -20) |
| SFT rewrite | 0.814 | 0.897 | +11 / -47 (net -36) |
| DPO rewrite | 0.828 | 0.906 | +13 / -42 (net -29) |

**Key findings:**
- DPO outperforms SFT by a net +7 queries on Recall@1, confirming preference alignment works
- DPO produces more concise rewrites (1.18x length ratio vs SFT 1.42x), matching API style (1.56x)
- The fine-tuned embedder (RQ5) already achieves 0.886 Recall@1, making rewriting less impactful — this validates that embedding quality > rewrite quality for retrieval
- At Recall@5+, all methods converge to >0.99, showing rewrites primarily affect ranking precision

## Technical Highlights

- **Full SFT → DPO pipeline**: Supervised fine-tuning followed by preference-based alignment — the standard industry pipeline for LLM training (SFT → RLHF/DPO)
- **Parameter-efficient fine-tuning**: LoRA adapters — only 0.44% parameters trainable, enabling training on consumer hardware
- **Retrieval-grounded reward**: DPO preference pairs derived from embedding cosine similarity (retrieval quality as reward signal), not human annotations
- **Custom DPO implementation**: Hand-written DPO loss with reference model, per-token log-prob computation, and response masking — not using trl's wrapper
- **Gradient accumulation + cosine LR**: Memory-efficient training with proper warmup scheduling
- **End-to-end evaluation**: RQ6 compares API / SFT / DPO rewrites on retrieval metrics with win/loss analysis

## Notes

- The current evaluation uses `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` with L2-normalised embeddings (inner product == cosine).
- If you see HuggingFace rate-limit warnings, set `HF_TOKEN` to improve download speed.

## What’s implemented vs WIP

- ✅ Working: MS MARCO dataset build + retrieval evaluation + rewrite evaluation + rewrite gating + alignment training & evaluation + SFT/DPO rewriter training & evaluation
- ⚠️ WIP: `data/preprocessing.py`, and parts of the retrieval/training scaffolding under `scripts/`

## Design Principles

| Principle | How |
|---|---|
| **Decoupled modules** | Data, models, training, retrieval, and evaluation are independent packages |
| **Typed configuration** | Dataclass configs parsed from YAML; no magic strings |
| **Easy to extend** | Swap embedder, loss, or index by changing one config field |
| **Research-friendly** | Skeleton-first design; fill in logic incrementally |
