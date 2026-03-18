"""Minimal runnable retrieval pipeline with semantic-gap analysis.

Usage::

    python -m pipeline.run_retrieval          # from project root
    python pipeline/run_retrieval.py          # direct

The pipeline:
1. Loads the curated semantic-gap dataset.
2. Encodes all texts with sentence-transformers.
3. Builds a FAISS inner-product index over documents.
4. Runs top-K retrieval for every query.
5. Reports per-query hits, cosine similarities, and Recall@K.
"""

# pyright: reportCallIssue=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Allow running as ``python pipeline/run_retrieval.py`` from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.semantic_gap_data import Triplet, load_semantic_gap_dataset


# ── Embedding helpers ────────────────────────────────────────────


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of strings into dense vectors.

    Parameters
    ----------
    model : SentenceTransformer
        Pre-trained sentence encoder.
    texts : list[str]
        Raw strings.
    batch_size : int
        Inference batch size.
    normalize : bool
        L2-normalise embeddings (so inner product == cosine similarity).

    Returns
    -------
    np.ndarray
        Embeddings of shape ``(len(texts), dim)`` in float32.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)


# ── FAISS index ──────────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build a flat inner-product FAISS index.

    With L2-normalised vectors, inner product equals cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings ``(N, dim)``.

    Returns
    -------
    Any
        Populated FAISS index ready for search.
    """
    dim = embeddings.shape[1]
    index: Any = faiss.IndexFlatIP(dim)
    getattr(index, "add")(embeddings)
    return index


def search_index(
    index: Any,
    query_embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search the index and return (scores, doc_indices).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(scores, indices)`` each of shape ``(Q, top_k)``.
    """
    scores, indices = getattr(index, "search")(query_embeddings, top_k)
    return scores, indices


# ── Metrics ──────────────────────────────────────────────────────


def recall_at_k(
    retrieved_ids: List[List[int]],
    relevant_ids: List[int],
    k: int,
) -> float:
    """Compute Recall@K (macro-averaged).

    Parameters
    ----------
    retrieved_ids : list[list[int]]
        For each query, ranked list of retrieved doc indices.
    relevant_ids : list[int]
        For each query, the single ground-truth doc index.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Recall@K in [0, 1].
    """
    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        if relevant in retrieved[:k]:
            hits += 1
    return hits / len(relevant_ids)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (assumes L2-normalised)."""
    return float(np.dot(a, b))


# ── Display helpers ──────────────────────────────────────────────


_SEPARATOR = "─" * 80


def print_query_result(
    idx: int,
    triplet: Triplet,
    query_emb: np.ndarray,
    pos_emb: np.ndarray,
    neg_emb: np.ndarray,
    retrieved_indices: np.ndarray,
    retrieved_scores: np.ndarray,
    pos_doc_idx: int,
    all_docs: List[str],
    top_k: int,
) -> bool:
    """Print detailed retrieval result for one query.  Returns True if hit."""
    cos_pos = cosine_similarity(query_emb, pos_emb)
    cos_neg = cosine_similarity(query_emb, neg_emb)

    hit = pos_doc_idx in retrieved_indices[:top_k]
    rank = (
        int(np.where(retrieved_indices == pos_doc_idx)[0][0]) + 1
        if pos_doc_idx in retrieved_indices
        else -1
    )

    status = "✅ HIT" if hit else "❌ MISS"

    print(f"\n{_SEPARATOR}")
    print(f"[Query {idx + 1}]  ({triplet.gap_type})")
    print(f"  Q : {triplet.query}")
    print(f"  P : {triplet.positive_doc[:100]}...")
    print(f"  cos(Q, P+) = {cos_pos:.4f}    cos(Q, P−) = {cos_neg:.4f}    gap = {cos_pos - cos_neg:+.4f}")
    print(f"  Pos doc global index = {pos_doc_idx}  | Rank in retrieval = {rank}  | {status}")
    print(f"  Top-{top_k} retrieved doc indices: {retrieved_indices[:top_k].tolist()}")

    return hit


# ── Main pipeline ────────────────────────────────────────────────


def main(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> Dict[str, float]:
    """Run the full retrieval pipeline and print analysis.

    Parameters
    ----------
    model_name : str
        sentence-transformers model identifier.
    top_k : int
        Number of docs to retrieve per query.

    Returns
    -------
    dict[str, float]
        Metric results keyed like ``"recall@5"``.
    """
    # ── 1. Load data ──────────────────────────────────────────
    triplets = load_semantic_gap_dataset()
    print(f"Loaded {len(triplets)} triplets")

    queries = [t.query for t in triplets]
    pos_docs = [t.positive_doc for t in triplets]
    neg_docs = [t.negative_doc for t in triplets]

    # Build a global document corpus: [pos_0, neg_0, pos_1, neg_1, ...]
    # so pos_doc for query i lives at index 2*i  and neg at 2*i+1.
    all_docs: List[str] = []
    for p, n in zip(pos_docs, neg_docs):
        all_docs.append(p)
        all_docs.append(n)

    pos_doc_indices = [2 * i for i in range(len(triplets))]

    print(f"Corpus size: {len(all_docs)} documents")

    # ── 2. Encode ─────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding queries ...")
    query_embs = encode_texts(model, queries)

    print("Encoding documents ...")
    doc_embs = encode_texts(model, all_docs)

    # Also encode pos/neg separately for per-pair cosine analysis
    pos_embs = doc_embs[::2]   # even indices
    neg_embs = doc_embs[1::2]  # odd indices

    print(f"Embedding dim: {query_embs.shape[1]}")

    # ── 3. Build index ────────────────────────────────────────
    print("Building FAISS index ...")
    index = build_faiss_index(doc_embs)

    # ── 4. Retrieve ───────────────────────────────────────────
    scores, indices = search_index(index, query_embs, top_k=top_k)

    # ── 5. Analyse and print ──────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  RETRIEVAL RESULTS  (top_k={top_k}, model={model_name})")
    print(f"{'═' * 80}")

    hits = 0
    cos_pos_scores: List[float] = []
    cos_neg_scores: List[float] = []

    for i, triplet in enumerate(triplets):
        hit = print_query_result(
            idx=i,
            triplet=triplet,
            query_emb=query_embs[i],
            pos_emb=pos_embs[i],
            neg_emb=neg_embs[i],
            retrieved_indices=indices[i],
            retrieved_scores=scores[i],
            pos_doc_idx=pos_doc_indices[i],
            all_docs=all_docs,
            top_k=top_k,
        )
        if hit:
            hits += 1
        cos_pos_scores.append(cosine_similarity(query_embs[i], pos_embs[i]))
        cos_neg_scores.append(cosine_similarity(query_embs[i], neg_embs[i]))

    # ── 6. Aggregate metrics ──────────────────────────────────
    retrieved_id_lists = [row.tolist() for row in indices]
    k_values = [1, 3, 5, 10]
    results: Dict[str, float] = {}
    for k in k_values:
        if k > top_k:
            continue
        r = recall_at_k(retrieved_id_lists, pos_doc_indices, k)
        results[f"recall@{k}"] = r

    avg_cos_pos = float(np.mean(cos_pos_scores))
    avg_cos_neg = float(np.mean(cos_neg_scores))

    print(f"\n{'═' * 80}")
    print("  SUMMARY")
    print(f"{'═' * 80}")
    print(f"  Model           : {model_name}")
    print(f"  Queries         : {len(triplets)}")
    print(f"  Corpus size     : {len(all_docs)}")
    print(f"  Top-K           : {top_k}")
    print()
    for metric, value in results.items():
        print(f"  {metric:15s} : {value:.4f}")
    print()
    print(f"  Avg cos(Q, P+)  : {avg_cos_pos:.4f}")
    print(f"  Avg cos(Q, P−)  : {avg_cos_neg:.4f}")
    print(f"  Avg gap (P+ − P−): {avg_cos_pos - avg_cos_neg:+.4f}")
    print()

    # ── Per gap-type breakdown ────────────────────────────────
    gap_types = sorted(set(t.gap_type for t in triplets))
    print("  Per gap-type cos(Q, P+):")
    for gt in gap_types:
        gt_scores = [
            cos_pos_scores[i]
            for i, t in enumerate(triplets)
            if t.gap_type == gt
        ]
        print(f"    {gt:30s}  avg={np.mean(gt_scores):.4f}  "
              f"min={np.min(gt_scores):.4f}  max={np.max(gt_scores):.4f}")
    print(f"{'═' * 80}\n")

    return results


if __name__ == "__main__":
    main(top_k=5)
