"""Preprocessing utilities for raw datasets.

Handles cleaning, deduplication, and train/val/test splitting before
records are consumed by ``QueryDocDataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataPreprocessor:
    """Cleans and splits raw data into train / val / test JSONL files.

    Parameters
    ----------
    raw_dir : str | Path
        Directory containing raw source files.
    output_dir : str | Path
        Directory where processed JSONL files will be written.
    """

    def __init__(self, raw_dir: str | Path, output_dir: str | Path) -> None:
        raise NotImplementedError

    def clean(self, records: List[Dict]) -> List[Dict]:
        """Remove duplicates, empty fields, and invalid records.

        Parameters
        ----------
        records : list[dict]
            Raw records.

        Returns
        -------
        list[dict]
            Cleaned records.
        """
        raise NotImplementedError

    def split(
        self,
        records: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split records into train / val / test sets.

        Parameters
        ----------
        records : list[dict]
            Cleaned records.
        train_ratio : float
            Fraction for training.
        val_ratio : float
            Fraction for validation.  Test = 1 - train - val.

        Returns
        -------
        tuple[list, list, list]
            (train, val, test) record lists.
        """
        raise NotImplementedError

    def save_jsonl(self, records: List[Dict], path: str | Path) -> None:
        """Write records to a JSONL file.

        Parameters
        ----------
        records : list[dict]
            Records to persist.
        path : str | Path
            Destination file path.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Execute the full preprocessing pipeline: load → clean → split → save."""
        raise NotImplementedError
