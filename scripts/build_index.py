"""Build a FAISS index from a trained embedder and a document corpus.

Usage::

    python scripts/build_index.py --config configs/default.yaml \
        --checkpoint outputs/best/ --corpus data/raw/corpus.jsonl
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL.")
    parser.add_argument("--output", type=str, default="outputs/index.faiss")
    return parser.parse_args()


def main() -> None:
    """Entry-point: load embedder → encode corpus → save index.

    Steps
    -----
    1. Load config and embedder from checkpoint.
    2. Read corpus JSONL.
    3. Encode all documents.
    4. Build and save FAISS index.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
