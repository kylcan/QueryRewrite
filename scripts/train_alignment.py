"""Train the embedding model with contrastive alignment loss.

Uses the existing intent_dataset.jsonl triplets to fine-tune
sentence-transformers/all-MiniLM-L6-v2 so that query and positive-doc
embeddings are pulled closer while hard-negative embeddings are pushed away.

Usage::

    python scripts/train_alignment.py
    python scripts/train_alignment.py --epochs 5 --batch_size 16 --lr 2e-5
    python scripts/train_alignment.py --loss triplet --margin 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.collator import QueryDocCollator
from data.dataset import QueryDocDataset
from models.embedder import EmbeddingModel
from models.losses import AlignmentLossFactory
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train alignment embedding model")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss", default="infonce", choices=["infonce", "triplet"])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints/alignment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset}")
    full_ds = QueryDocDataset(args.dataset)
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    # ── Model + tokenizer ─────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = EmbeddingModel(model_name=args.model)
    collator = QueryDocCollator(
        tokenizer=model.tokenizer,
        max_query_length=64,
        max_doc_length=256,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
    )

    # ── Loss ──────────────────────────────────────────────────
    loss_kwargs = {}
    if args.loss == "infonce":
        loss_kwargs["temperature"] = args.temperature
    else:
        loss_kwargs["margin"] = args.margin
    loss_fn = AlignmentLossFactory.create(args.loss, **loss_kwargs)
    print(f"Loss: {args.loss}  {loss_kwargs}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Train ─────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs …\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    history = trainer.train()

    print(f"\nTraining complete. Best checkpoint → {args.checkpoint_dir}/best/")
    print(f"History: {history}")


if __name__ == "__main__":
    main()
