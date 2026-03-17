"""Evaluate a trained model on the test set.

Usage::

    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best/
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint dir.")
    return parser.parse_args()


def main() -> None:
    """Entry-point: load model → build index → evaluate.

    Steps
    -----
    1. Parse CLI args and load config.
    2. Load model from checkpoint.
    3. Build FAISS index over corpus.
    4. Run ``RetrievalEvaluator`` on the test set.
    5. Print results.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
