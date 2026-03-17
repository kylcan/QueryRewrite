"""High-level search interface combining embedding + index lookup."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.embedder import EmbeddingModel
from retrieval.indexer import FaissIndexer


class Searcher:
    """End-to-end retrieval: encode queries → search FAISS → return results.

    Parameters
    ----------
    embedder : EmbeddingModel
        Model used to encode queries.
    indexer : FaissIndexer
        Pre-built FAISS index over the document corpus.
    tokenizer : Any
        Tokenizer compatible with the embedder.
    corpus : list[dict], optional
        Original documents for result display.  Each dict should contain
        at least a ``"text"`` key.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        indexer: FaissIndexer,
        tokenizer: Any,
        corpus: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        queries: List[str],
        top_k: int = 10,
        device: str = "cuda",
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve top-K documents for a batch of queries.

        Parameters
        ----------
        queries : list[str]
            Raw query strings.
        top_k : int
            Number of documents to retrieve per query.
        device : str
            Device for encoding.

        Returns
        -------
        list[list[dict]]
            For each query, a ranked list of dicts with keys
            ``"doc_id"``, ``"score"``, and optionally ``"text"``.
        """
        raise NotImplementedError

    def build_index(
        self,
        documents: List[str],
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        """Encode all documents and build the FAISS index.

        Parameters
        ----------
        documents : list[str]
            Raw document texts.
        batch_size : int
            Encoding batch size.
        device : str
            Device for encoding.
        """
        raise NotImplementedError
