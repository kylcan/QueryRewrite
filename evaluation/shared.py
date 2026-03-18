# pyright: reportCallIssue=false
"""Shared infrastructure for evaluation scripts.

Provides dataset loading, embedding encoding, and FAISS index construction
so that individual evaluation modules (dataset / rewrite / gating) stay DRY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------ #
#  Dataset loading
# ------------------------------------------------------------------ #

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_corpus(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
    """De-duplicate pos + hard_neg docs → (texts, text→index)."""
    seen: Dict[str, int] = {}
    corpus: List[str] = []
    for r in records:
        for key in ("pos_doc", "hard_neg_doc"):
            doc = r[key]
            if doc not in seen:
                seen[doc] = len(corpus)
                corpus.append(doc)
    return corpus, seen


# ------------------------------------------------------------------ #
#  Encoding + FAISS
# ------------------------------------------------------------------ #

def encode(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """L2-normalised encoding (inner-product == cosine)."""
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def build_index(embeddings: np.ndarray) -> Any:
    """Flat inner-product FAISS index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore[attr-defined]
    return index


def search(
    index: Any, queries: np.ndarray, top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) each of shape (Q, top_k)."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    scores, indices = index.search(q, top_k)  # type: ignore[attr-defined]
    return scores, indices


# ------------------------------------------------------------------ #
#  Retrieval context — shared state for all eval modules
# ------------------------------------------------------------------ #

@dataclass
class RetrievalContext:
    """Pre-computed retrieval results shared across evaluation modules.

    Construct via ``RetrievalContext.build()`` to avoid re-encoding.
    """

    records: List[Dict[str, Any]]
    corpus: List[str]
    doc2idx: Dict[str, int]

    gold_pos_ids: List[int]
    gold_neg_ids: List[int]

    orig_emb: np.ndarray
    rewrite_emb: np.ndarray
    corpus_emb: np.ndarray

    orig_scores: np.ndarray
    orig_indices: np.ndarray
    rew_scores: np.ndarray
    rew_indices: np.ndarray

    model_name: str = ""
    top_k: int = 20
    index: Any = field(default=None, repr=False)

    @classmethod
    def build(
        cls,
        dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 20,
    ) -> "RetrievalContext":
        """Load data, encode, index, retrieve — return ready context."""
        print(f"[1/5] Loading dataset from {dataset_path}")
        records = load_dataset(dataset_path)
        print(f"       {len(records)} samples loaded")

        print("[2/5] Building corpus …")
        corpus, doc2idx = build_corpus(records)
        print(f"       Corpus size: {len(corpus)} unique documents")

        gold_pos_ids = [doc2idx[r["pos_doc"]] for r in records]
        gold_neg_ids = [doc2idx[r["hard_neg_doc"]] for r in records]

        print(f"[3/5] Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        print("       Encoding corpus …")
        corpus_emb = encode(model, corpus)

        print("       Encoding original queries …")
        orig_emb = encode(model, [r["query_original"] for r in records])

        print("       Encoding rewritten queries …")
        rewrite_emb = encode(model, [r["query_rewrite"] for r in records])

        print(f"[4/5] Building FAISS index (dim={corpus_emb.shape[1]}, n={len(corpus)}) …")
        index = build_index(corpus_emb)

        print(f"       Searching top-{top_k} for original queries …")
        orig_scores, orig_indices = search(index, orig_emb, top_k)

        print(f"       Searching top-{top_k} for rewritten queries …")
        rew_scores, rew_indices = search(index, rewrite_emb, top_k)

        print("[5/5] Context ready.\n")

        return cls(
            records=records,
            corpus=corpus,
            doc2idx=doc2idx,
            gold_pos_ids=gold_pos_ids,
            gold_neg_ids=gold_neg_ids,
            orig_emb=orig_emb,
            rewrite_emb=rewrite_emb,
            corpus_emb=corpus_emb,
            orig_scores=orig_scores,
            orig_indices=orig_indices,
            rew_scores=rew_scores,
            rew_indices=rew_indices,
            model_name=model_name,
            top_k=top_k,
            index=index,
        )


# ------------------------------------------------------------------ #
#  Pretty-print helper
# ------------------------------------------------------------------ #

def print_table(title: str, rows: List[Tuple[str, ...]]) -> None:
    """Print an aligned ASCII table."""
    col_widths = [
        max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(f"\n  {title}")
    print(f"  {sep}")
    for i, row in enumerate(rows):
        cells = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  | {cells} |")
        if i == 0:
            print(f"  {sep}")
    print(f"  {sep}")


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Write metrics dict to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved → {path}\n")
