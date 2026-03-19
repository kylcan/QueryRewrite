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
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def _tokenize(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

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
        queries = [item["query"] for item in batch]
        pos_docs = [item["positive_doc"] for item in batch]
        neg_docs = [item["negative_doc"] for item in batch]

        q_enc = self._tokenize(queries, self.max_query_length)
        p_enc = self._tokenize(pos_docs, self.max_doc_length)
        n_enc = self._tokenize(neg_docs, self.max_doc_length)

        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
            "neg_input_ids": n_enc["input_ids"],
            "neg_attention_mask": n_enc["attention_mask"],
        }
