"""Retrieval evaluation metrics."""

from __future__ import annotations

from typing import List

import numpy as np


def recall_at_k(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
    k: int,
) -> float:
    """Compute Recall@K averaged over all queries.

    Parameters
    ----------
    retrieved_ids : list[list[int]]
        For each query, ranked list of retrieved document IDs.
    relevant_ids : list[list[int]]
        For each query, set of ground-truth relevant document IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Mean Recall@K.
    """
    raise NotImplementedError


def ndcg_at_k(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
    k: int,
) -> float:
    """Compute NDCG@K averaged over all queries.

    Parameters
    ----------
    retrieved_ids : list[list[int]]
        Ranked retrieved document IDs per query.
    relevant_ids : list[list[int]]
        Ground-truth relevant document IDs per query.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Mean NDCG@K.
    """
    raise NotImplementedError


def mrr(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
) -> float:
    """Compute Mean Reciprocal Rank over all queries.

    Parameters
    ----------
    retrieved_ids : list[list[int]]
        Ranked retrieved document IDs per query.
    relevant_ids : list[list[int]]
        Ground-truth relevant document IDs per query.

    Returns
    -------
    float
        MRR score.
    """
    raise NotImplementedError
