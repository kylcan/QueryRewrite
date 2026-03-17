"""FAISS index builder and manager."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FaissIndexer:
    """Builds and manages a FAISS index over document embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of vectors.
    index_type : str
        FAISS index factory string: ``"flat_ip"`` (inner product),
        ``"ivf"``, or ``"hnsw"``.
    nprobe : int
        Number of probes for IVF indices.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "flat_ip",
        nprobe: int = 10,
    ) -> None:
        raise NotImplementedError

    def build(self, embeddings: np.ndarray) -> None:
        """Train (if needed) and add vectors to the index.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings of shape ``(N, embedding_dim)``.
        """
        raise NotImplementedError

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for nearest neighbours.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query vectors ``(Q, embedding_dim)``.
        top_k : int
            Number of neighbours to return per query.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(scores, indices)`` each of shape ``(Q, top_k)``.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index to disk.

        Parameters
        ----------
        path : str | Path
            File path for the index.
        """
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load a FAISS index from disk.

        Parameters
        ----------
        path : str | Path
            File path of the saved index.
        """
        raise NotImplementedError
