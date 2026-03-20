"""Train the query rewriter with SFT (Supervised Fine-Tuning).

Uses (prompt, completion) pairs from sft_dataset.jsonl to fine-tune
a small causal LM (Qwen2.5-0.5B) with LoRA adapters.

Usage::

    python scripts/train_sft.py
    python scripts/train_sft.py --model Qwen/Qwen2.5-0.5B --epochs 3
    python scripts/train_sft.py --lora_rank 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.rewriter import QueryRewriter
from training.sft_trainer import SFTDataset, SFTTrainer


def _load_jsonl(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for query rewriter")
    parser.add_argument("--dataset", default="data/MS_MARCO/sft_dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints/sft")
    parser.add_argument("--scheduler", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deepspeed", default=None, metavar="DS_CONFIG",
        help="Path to DeepSpeed JSON config (e.g. configs/deepspeed_zero2.json). "
             "Launch with: deepspeed --num_gpus N scripts/train_sft.py --deepspeed ...",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    print(f"\n  Loading SFT data from {args.dataset} …")
    records = _load_jsonl(args.dataset)
    print(f"  Total samples: {len(records)}")

    # Build model
    print(f"\n  Loading model: {args.model}")
    model = QueryRewriter(
        base_model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Train/val split
    n_val = int(len(records) * args.val_ratio)
    val_records = records[:n_val] if n_val > 0 else None
    train_records = records[n_val:]
    print(f"  Train: {len(train_records)}, Val: {n_val}")

    train_dataset = SFTDataset(train_records, model.tokenizer, max_length=args.max_length)
    val_dataset = SFTDataset(val_records, model.tokenizer, max_length=args.max_length) if val_records else None

    # Train
    print(f"\n  Config:")
    print(f"    LR: {args.lr}")
    print(f"    Batch: {args.batch_size} × {args.gradient_accumulation} accum")
    print(f"    LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"    Scheduler: {args.scheduler}")

    if deepspeed_config := args.deepspeed:
        print(f"  DeepSpeed config: {deepspeed_config}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        scheduler_type=args.scheduler,
        deepspeed_config=args.deepspeed,
    )

    history = trainer.train()
    print(f"\n  Training complete. Adapter saved to {args.checkpoint_dir}/best")

    # Quick inference test
    print("\n  ── Quick inference test ──")
    test_queries = ["what is rba", "how to install python"]
    rewrites = model.rewrite_queries(test_queries, device=device)
    for q, r in zip(test_queries, rewrites):
        print(f"  Q: {q}")
        print(f"  R: {r}\n")


if __name__ == "__main__":
    main()
