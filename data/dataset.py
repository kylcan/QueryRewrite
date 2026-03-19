"""Dataset for query–document triplets.

Expected JSONL schema per line (compatible with intent_dataset.jsonl)::

    {
        "query_original": "...",
        "query_rewrite": "...",
        "pos_doc": "...",
        "hard_neg_doc": "...",
        ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset


class QueryDocDataset(Dataset):
    """PyTorch Dataset that loads (query, positive_doc, negative_doc) triplets.

    Parameters
    ----------
    data_path : str | Path
        Path to a JSONL file containing the triplets.
    query_key : str
        JSON key for the query text.
    pos_key : str
        JSON key for the positive document text.
    neg_key : str
        JSON key for the negative document text.
    """

    def __init__(
        self,
        data_path: str | Path,
        query_key: str = "query_original",
        pos_key: str = "pos_doc",
        neg_key: str = "hard_neg_doc",
    ) -> None:
        self.records = self.load_jsonl(data_path)
        self.query_key = query_key
        self.pos_key = pos_key
        self.neg_key = neg_key

    def __len__(self) -> int:
        """Return the number of triplets in the dataset."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Return a single triplet as a dict.

        Returns
        -------
        dict
            Keys: ``query``, ``positive_doc``, ``negative_doc``.
        """
        r = self.records[index]
        return {
            "query": r[self.query_key],
            "positive_doc": r[self.pos_key],
            "negative_doc": r[self.neg_key],
        }

    @staticmethod
    def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
        """Read a JSONL file into a list of dicts."""
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
