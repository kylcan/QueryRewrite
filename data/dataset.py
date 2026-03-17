"""Dataset for query–document triplets.

Expected JSONL schema per line::

    {
        "query": "original user query",
        "positive_doc": "relevant document text",
        "negative_doc": "non-relevant document text"
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


class QueryDocDataset(Dataset):
    """PyTorch Dataset that loads (query, positive_doc, negative_doc) triplets.

    Parameters
    ----------
    data_path : str | Path
        Path to a JSONL file containing the triplets.
    max_query_length : int
        Maximum token length for queries.
    max_doc_length : int
        Maximum token length for documents.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_query_length: int = 64,
        max_doc_length: int = 256,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of triplets in the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Return a single triplet as a dict.

        Returns
        -------
        dict
            Keys: ``query``, ``positive_doc``, ``negative_doc``.
        """
        raise NotImplementedError

    @staticmethod
    def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
        """Read a JSONL file into a list of dicts.

        Parameters
        ----------
        path : str | Path
            Path to the JSONL file.

        Returns
        -------
        list[dict]
            Parsed records.
        """
        raise NotImplementedError
