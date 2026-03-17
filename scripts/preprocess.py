"""Preprocess raw data into train/val/test JSONL splits.

Usage::

    python scripts/preprocess.py --raw_dir data/raw --output_dir data/processed
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """Entry-point: clean → split → save.

    Steps
    -----
    1. Instantiate ``DataPreprocessor``.
    2. Call ``preprocessor.run()``.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
