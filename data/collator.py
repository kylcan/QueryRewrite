"""Collator that tokenises and pads a batch of query–document triplets."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class QueryDocCollator:
    """Tokenise and pad a batch of (query, positive_doc, negative_doc) dicts.

    Parameters
    ----------
    tokenizer : Any
        A HuggingFace-compatible tokenizer.
    max_query_length : int
        Max token length for queries.
    max_doc_length : int
        Max token length for documents.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_query_length: int = 64,
        max_doc_length: int = 256,
    ) -> None:
        raise NotImplementedError

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of raw string triplets into padded tensors.

        Parameters
        ----------
        batch : list[dict]
            Each dict has keys ``query``, ``positive_doc``, ``negative_doc``.

        Returns
        -------
        dict[str, torch.Tensor]
            Tokenised and padded tensors ready for model consumption.
        """
        raise NotImplementedError
