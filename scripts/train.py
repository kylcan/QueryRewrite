"""Train the alignment model (rewriter + embedder + contrastive loss).

Usage::

    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train alignment model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    """Entry-point: load config → build data → build model → train.

    Steps
    -----
    1. Parse CLI args and load ``ProjectConfig``.
    2. Set random seeds.
    3. Build train / val ``DataLoader``s.
    4. Instantiate ``QueryRewriter``, ``EmbeddingModel``, ``AlignmentModel``.
    5. Build loss, optimizer, scheduler.
    6. Create ``Trainer`` and call ``trainer.train()``.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
