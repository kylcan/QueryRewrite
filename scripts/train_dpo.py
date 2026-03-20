"""Train the query rewriter with DPO (Direct Preference Optimization).

Starts from a SFT checkpoint and further aligns the rewriter using
(prompt, chosen, rejected) preference pairs from dpo_dataset.jsonl.

Usage::

    python scripts/train_dpo.py
    python scripts/train_dpo.py --sft_checkpoint checkpoints/sft/best
    python scripts/train_dpo.py --beta 0.05 --epochs 2
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
from training.dpo_trainer import DPODataset, DPOTrainer


def _load_jsonl(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training for query rewriter")
    parser.add_argument("--dataset", default="data/MS_MARCO/dpo_dataset.jsonl")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft/best",
                        help="Path to the SFT adapter to start from")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO inverse temperature (lower = stronger preference)")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints/dpo")
    parser.add_argument("--scheduler", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deepspeed", default=None, metavar="DS_CONFIG",
        help="Path to DeepSpeed JSON config (e.g. configs/deepspeed_zero3.json). "
             "Launch with: deepspeed --num_gpus N scripts/train_dpo.py --deepspeed ...",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    print(f"\n  Loading DPO data from {args.dataset} …")
    records = _load_jsonl(args.dataset)
    print(f"  Total preference pairs: {len(records)}")

    # Load SFT-trained model
    print(f"\n  Loading base model: {args.model}")
    model = QueryRewriter(
        base_model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    sft_path = Path(args.sft_checkpoint)
    if sft_path.exists():
        print(f"  Loading SFT adapter from {sft_path}")
        model.load_adapter(str(sft_path))
    else:
        print(f"  ⚠ SFT checkpoint not found at {sft_path}, training from base model")

    # Create reference model (frozen copy)
    print("  Creating frozen reference model …")
    ref_model = model.get_ref_model()

    # Train/val split
    n_val = int(len(records) * args.val_ratio)
    val_records = records[:n_val] if n_val > 0 else None
    train_records = records[n_val:]
    print(f"  Train: {len(train_records)}, Val: {n_val}")

    train_dataset = DPODataset(train_records, model.tokenizer, max_length=args.max_length)
    val_dataset = DPODataset(val_records, model.tokenizer, max_length=args.max_length) if val_records else None

    # Train
    print(f"\n  Config:")
    print(f"    LR: {args.lr}")
    print(f"    Beta: {args.beta}")
    print(f"    Batch: {args.batch_size} × {args.gradient_accumulation} accum")
    print(f"    LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"    Scheduler: {args.scheduler}")

    if args.deepspeed:
        print(f"  DeepSpeed config: {args.deepspeed}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        deepspeed_config=args.deepspeed,
    )

    history = trainer.train()
    print(f"\n  DPO training complete. Adapter saved to {args.checkpoint_dir}/best")

    # Quick inference test
    print("\n  ── Quick inference test ──")
    test_queries = ["what is rba", "how to install python"]
    rewrites = model.rewrite_queries(test_queries, device=device)
    for q, r in zip(test_queries, rewrites):
        print(f"  Q: {q}")
        print(f"  R: {r}\n")


if __name__ == "__main__":
    main()
