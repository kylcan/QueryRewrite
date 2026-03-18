"""Retrieval evaluation metrics.

Two API styles:
  - **Array API** (used by eval scripts): operates on numpy arrays from FAISS.
  - **List API** (general purpose): operates on list-of-lists, compatible with
    ``RetrievalEvaluator``.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ================================================================
#  Array API — for FAISS-based evaluation (numpy arrays)
# ================================================================

def recall_at_k_arr(
    retrieved: np.ndarray,
    gold: List[int],
    k: int,
) -> float:
    """Recall@K — fraction of queries where gold doc appears in top-K.

    Parameters
    ----------
    retrieved : ndarray of shape (n_queries, top_k)
        FAISS result indices.
    gold : list[int]
        One gold document index per query.
    """
    hits = sum(1 for i, g in enumerate(gold) if g in retrieved[i, :k])
    return hits / len(gold)


def mrr_arr(
    retrieved: np.ndarray,
    gold: List[int],
    max_k: int,
) -> float:
    """Mean Reciprocal Rank (up to max_k)."""
    rr_sum = 0.0
    for i, g in enumerate(gold):
        for rank in range(min(max_k, retrieved.shape[1])):
            if retrieved[i, rank] == g:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / len(gold)


def hard_neg_in_topk(
    retrieved: np.ndarray,
    neg_ids: List[int],
    k: int,
) -> float:
    """Fraction of queries where the hard-negative appears in top-K."""
    hits = sum(1 for i, n in enumerate(neg_ids) if n in retrieved[i, :k])
    return hits / len(neg_ids)


# ================================================================
#  List API — general purpose (list-of-lists)
# ================================================================

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
    """
    total = 0.0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        rel_set = set(rel)
        hits = sum(1 for doc_id in ret[:k] if doc_id in rel_set)
        total += hits / max(len(rel_set), 1)
    return total / max(len(retrieved_ids), 1)


def ndcg_at_k(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
    k: int,
) -> float:
    """Compute NDCG@K averaged over all queries."""
    total = 0.0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        rel_set = set(rel)
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, doc_id in enumerate(ret[:k])
            if doc_id in rel_set
        )
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        total += dcg / max(idcg, 1e-10)
    return total / max(len(retrieved_ids), 1)


def mrr(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
) -> float:
    """Compute Mean Reciprocal Rank over all queries."""
    rr_sum = 0.0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        rel_set = set(rel)
        for rank, doc_id in enumerate(ret):
            if doc_id in rel_set:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / max(len(retrieved_ids), 1)
