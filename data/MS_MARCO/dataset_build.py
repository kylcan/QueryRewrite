

"""
MS MARCO → Intent-Aware Retrieval Dataset Builder
==================================================

Improvements over v1:
- **Large distractor corpus**: all candidate passages (not just pos_docs)
  are indexed, simulating realistic retrieval difficulty.
- **Hard-negative mining with score-band filtering**: negatives are
  selected from a configurable similarity window so they are *confusing
  but incorrect*, not trivially unrelated.
- **LLM query rewrite** with real API path (OpenAI-compatible) +
  robust rule-based fallback.
- **Analytic fields**: per-sample cosine similarities are persisted for
  downstream gap analysis.

Output JSONL schema (one object per line)::

    {
        "query_id":          int,
        "query_original":    str,
        "query_rewrite":     str,
        "pos_doc":           str,
        "hard_neg_doc":      str,
        "cos_q_pos":         float,   # cos(query, pos_doc)
        "cos_q_neg":         float,   # cos(query, hard_neg)
        "cos_q_rewrite":     float,   # cos(query, rewrite)  — gap proxy
        "neg_rank":          int,     # rank of hard_neg among all corpus docs
        "gap_type":          str      # "hard" | "medium" | "easy"
    }
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ================================================================
# 1. Load MS MARCO  — query + ALL candidate passages
# ================================================================

def load_msmarco_subset(
    n_samples: int = 500,
    n_distractor_pool: int = 5000,
    dataset_version: str = "v1.1",
) -> Tuple[List[Dict], List[str]]:
    """Load queries with positive docs AND build a large distractor corpus.

    Parameters
    ----------
    n_samples : int
        Number of (query, pos_doc) pairs to collect.
    n_distractor_pool : int
        How many extra MS MARCO rows to scan for distractor passages.
        All passages (positive or not) from these rows enter the corpus.
    dataset_version : str
        MS MARCO HuggingFace config name.

    Returns
    -------
    data : list[dict]
        Each dict has keys ``query``, ``pos_doc``, ``all_passages``.
    corpus : list[str]
        De-duplicated global passage pool (for FAISS indexing).
    """
    dataset = load_dataset("ms_marco", dataset_version, split="train")

    data: List[Dict] = []
    corpus_set: set[str] = set()
    corpus_list: List[str] = []

    scan_limit = max(n_samples * 3, n_distractor_pool)
    scanned = 0

    for item in dataset:  # type: ignore[union-attr]
        if scanned >= scan_limit:
            break
        scanned += 1

        passages: List[str] = item["passages"]["passage_text"]  # type: ignore[index]
        labels: List[int] = item["passages"]["is_selected"]  # type: ignore[index]

        # Add ALL passages to the global corpus
        for p in passages:
            if p not in corpus_set:
                corpus_set.add(p)
                corpus_list.append(p)

        # Collect labelled (query, pos_doc) pairs up to n_samples
        if len(data) < n_samples:
            pos_docs = [p for p, l in zip(passages, labels) if l == 1]
            if pos_docs:
                data.append({
                    "query": item["query"],  # type: ignore[index]
                    "pos_doc": pos_docs[0],
                    "all_passages": passages,
                })

    print(f"  Loaded {len(data)} query–pos pairs, corpus size = {len(corpus_list)}")
    return data, corpus_list


# ================================================================
# 2. Query Rewrite — rule-based fallback + LLM path
# ================================================================

_REWRITE_RULES: List[Callable[[str], str]] = [
    lambda q: q.replace("what is", "define"),
    lambda q: q.replace("how to", "steps for"),
    lambda q: q.replace("best", "top-rated"),
    lambda q: q.replace("cheap", "budget-friendly"),
    lambda q: "explain " + q if not q.startswith("explain") else q,
    lambda q: q.rstrip("?").rstrip(".") + " overview",
    lambda q: "guide: " + q,
    lambda q: q + " tips and advice",
]


def rule_based_rewrite(query: str) -> str:
    """Apply 1–2 random transformation rules.

    Always produces a string that differs from the original so the
    semantic gap between original query and rewrite is non-trivial.
    """
    q = query
    n_rules = random.randint(1, 2)
    chosen = random.sample(_REWRITE_RULES, min(n_rules, len(_REWRITE_RULES)))
    for fn in chosen:
        q = fn(q)
    # Guarantee the rewrite differs
    if q == query:
        q = "detailed guide: " + query
    return q


def llm_rewrite(
    query: str,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> str:
    """Rewrite via LLM if available, else fall back to rules.

    Parameters
    ----------
    query : str
        Original user query.
    llm_callable : callable, optional
        A function that takes a prompt string and returns the LLM's text
        response.  Compatible with ``openai.ChatCompletion`` wrappers or
        any local model serving the same interface.

    Returns
    -------
    str
        Rewritten query.
    """
    if llm_callable is None:
        return rule_based_rewrite(query)

    prompt = (
        "Rewrite the following search query so that it better captures "
        "the user's intent while using different vocabulary.  Return ONLY "
        "the rewritten query, nothing else.\n\n"
        f"Query: {query}\nRewrite:"
    )
    try:
        result = llm_callable(prompt).strip()
        return result if result else rule_based_rewrite(query)
    except Exception:
        return rule_based_rewrite(query)


# ================================================================
# 3. Hard Negative Mining (score-band filtering)
# ================================================================

class HardNegativeMiner:
    """Mine negatives at three difficulty tiers using rank-based selection.

    Instead of a single score band (which tends to collapse to one tier),
    we use the **rank** of non-positive passages as the difficulty proxy:

    - **hard**   : rank 1–5 among non-positive passages (very confusing)
    - **medium** : rank 6–30 (topically related but distinguishable)
    - **easy**   : rank 100+ or random (clearly irrelevant)

    All similarity computations are batched for performance.

    Parameters
    ----------
    corpus : list[str]
        All candidate passages.
    model_name : str
        sentence-transformers model for encoding.
    batch_size : int
        Encoding batch size.
    """

    # Rank ranges (0-indexed, among non-positive passages)
    TIER_RANGES: Dict[str, Tuple[int, int]] = {
        "hard":   (0, 5),     # top-5 non-positive
        "medium": (5, 30),    # rank 6–30
        "easy":   (80, 200),  # rank 80+
    }

    def __init__(
        self,
        corpus: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
    ) -> None:
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)

        print(f"  Encoding corpus ({len(corpus)} passages) …")
        self.corpus_emb_np: np.ndarray = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def batch_mine(
        self,
        queries: List[str],
        pos_docs: List[str],
        target_tiers: List[str],
        batch_size: int = 128,
    ) -> List[Tuple[str, float, int, str]]:
        """Mine negatives for all queries in one vectorised pass.

        Parameters
        ----------
        queries : list[str]
            Original query strings.
        pos_docs : list[str]
            Corresponding positive documents.
        target_tiers : list[str]
            Desired difficulty tier per query.
        batch_size : int
            Encoding batch size.

        Returns
        -------
        list[tuple]
            For each query: ``(neg_doc, cos_score, rank, tier_label)``.
        """
        # Build a fast pos_doc → corpus-index lookup
        doc_to_idx: Dict[str, int] = {}
        for ci, doc in enumerate(self.corpus):
            if doc not in doc_to_idx:
                doc_to_idx[doc] = ci

        print("  Encoding queries for mining …")
        query_emb_np: np.ndarray = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # (n_queries, corpus_size) similarity matrix via matmul
        print("  Computing similarity matrix …")
        sim_matrix = query_emb_np @ self.corpus_emb_np.T  # cosine (both normalised)

        results: List[Tuple[str, float, int, str]] = []

        for i in tqdm(range(len(queries)), desc="Mining negatives"):
            scores = sim_matrix[i]  # (corpus_size,)
            sorted_indices = np.argsort(-scores)  # descending

            # Mask out the positive doc
            pos_idx = doc_to_idx.get(pos_docs[i], -1)

            # Build ranked non-positive list (index, score, global_rank)
            # Only need up to rank ~200 for any tier
            non_pos: List[Tuple[int, float, int]] = []
            for global_rank_pos, ci in enumerate(sorted_indices[:300]):
                ci_int = int(ci)
                if ci_int == pos_idx:
                    continue
                non_pos.append((ci_int, float(scores[ci_int]), global_rank_pos))
                if len(non_pos) >= 200:
                    break

            target_tier = target_tiers[i]
            chosen = self._pick_from_tier(non_pos, target_tier)
            results.append(chosen)

        return results

    def _pick_from_tier(
        self,
        non_pos: List[Tuple[int, float, int]],
        target_tier: str,
    ) -> Tuple[str, float, int, str]:
        """Select a negative from the target tier, with fallback cascade."""
        lo, hi = self.TIER_RANGES.get(target_tier, (0, 5))
        hi = min(hi, len(non_pos))
        candidates = non_pos[lo:hi]

        if candidates:
            chosen_idx, chosen_score, chosen_rank = random.choice(candidates)
            return self.corpus[chosen_idx], chosen_score, chosen_rank, target_tier

        # Fallback cascade: hard → medium → easy → random
        for fallback_tier in ("hard", "medium", "easy"):
            flo, fhi = self.TIER_RANGES[fallback_tier]
            fhi = min(fhi, len(non_pos))
            fc = non_pos[flo:fhi]
            if fc:
                chosen_idx, chosen_score, chosen_rank = random.choice(fc)
                return self.corpus[chosen_idx], chosen_score, chosen_rank, fallback_tier

        chosen = random.choice(self.corpus)
        return chosen, 0.0, -1, "easy"


# ================================================================
# 4. Build Dataset
# ================================================================

def build_dataset(
    n_samples: int = 500,
    n_distractor_pool: int = 5000,
    output_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    llm_callable: Optional[Callable[[str], str]] = None,
    neg_ratio: Optional[Dict[str, float]] = None,
) -> None:
    """End-to-end dataset construction pipeline.

    Parameters
    ----------
    n_samples : int
        Number of query–pos_doc pairs to collect.
    n_distractor_pool : int
        Rows scanned for distractor passages (controls corpus size).
    output_path : str
        Where to write the JSONL output.
    llm_callable : callable, optional
        LLM function for query rewriting.
    neg_ratio : dict, optional
        Target distribution of negative difficulty tiers.
        Default: ``{"hard": 0.4, "medium": 0.4, "easy": 0.2}``.
    """
    if neg_ratio is None:
        neg_ratio = {"hard": 0.4, "medium": 0.4, "easy": 0.2}

    print("[1/5] Loading MS MARCO …")
    data, corpus = load_msmarco_subset(n_samples, n_distractor_pool)

    print("[2/5] Building negative miner …")
    miner = HardNegativeMiner(corpus)

    # Pre-assign a difficulty tier to each sample based on target ratio
    tier_sequence: List[str] = []
    for tier, ratio in neg_ratio.items():
        tier_sequence.extend([tier] * int(round(ratio * len(data))))
    while len(tier_sequence) < len(data):
        tier_sequence.append(random.choice(list(neg_ratio.keys())))
    tier_sequence = tier_sequence[: len(data)]
    random.shuffle(tier_sequence)

    # Generate rewrites
    print("[3/5] Generating query rewrites …")
    queries = [d["query"] for d in data]
    pos_docs = [d["pos_doc"] for d in data]
    rewrites = [llm_rewrite(q, llm_callable) for q in tqdm(queries, desc="Rewriting")]

    # Batch mine negatives
    print("[4/5] Mining negatives (batched) …")
    print(f"       Target ratio: {neg_ratio}")
    neg_results = miner.batch_mine(queries, pos_docs, tier_sequence)

    # Batch encode for analytic similarities
    print("  Computing analytic similarities …")
    all_texts = queries + pos_docs + rewrites
    all_embs = miner.model.encode(
        all_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    n = len(queries)
    q_embs = all_embs[:n]
    p_embs = all_embs[n : 2 * n]
    r_embs = all_embs[2 * n : 3 * n]

    # Per-sample cosines via element-wise dot (both normalised)
    cos_q_pos_arr = np.sum(q_embs * p_embs, axis=1)
    cos_q_rew_arr = np.sum(q_embs * r_embs, axis=1)

    # Assemble results
    results: List[Dict[str, Any]] = []
    gap_counts: Dict[str, int] = {"hard": 0, "medium": 0, "easy": 0}

    for i in range(n):
        hard_neg, cos_q_neg, neg_rank, gap_type = neg_results[i]
        gap_counts[gap_type] += 1

        results.append({
            "query_id": i,
            "query_original": queries[i],
            "query_rewrite": rewrites[i],
            "pos_doc": pos_docs[i],
            "hard_neg_doc": hard_neg,
            "cos_q_pos": round(float(cos_q_pos_arr[i]), 4),
            "cos_q_neg": round(cos_q_neg, 4),
            "cos_q_rewrite": round(float(cos_q_rew_arr[i]), 4),
            "neg_rank": neg_rank,
            "gap_type": gap_type,
        })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[5/5] Saved {len(results)} samples → {output_path}")
    print(f"  Hard: {gap_counts['hard']}  Medium: {gap_counts['medium']}  Easy: {gap_counts['easy']}")

    cos_neg_arr = [r["cos_q_neg"] for r in results]
    print(f"  Avg cos(Q, P+) = {np.mean(cos_q_pos_arr):.4f}")
    print(f"  Avg cos(Q, N−) = {np.mean(cos_neg_arr):.4f}")
    print(f"  Avg gap        = {np.mean(cos_q_pos_arr) - np.mean(cos_neg_arr):+.4f}")

    # Per-tier stats
    for tier in ("hard", "medium", "easy"):
        tier_negs = [r["cos_q_neg"] for r in results if r["gap_type"] == tier]
        if tier_negs:
            print(f"  {tier:8s} neg: avg cos={np.mean(tier_negs):.4f}  "
                  f"min={np.min(tier_negs):.4f}  max={np.max(tier_negs):.4f}  n={len(tier_negs)}")


# ================================================================
# 5. Main
# ================================================================

def main() -> None:
    build_dataset(
        n_samples=500,
        n_distractor_pool=5000,
        output_path="data/MS_MARCO/intent_dataset.jsonl",
        neg_ratio={"hard": 0.4, "medium": 0.4, "easy": 0.2},
    )


if __name__ == "__main__":
    main()