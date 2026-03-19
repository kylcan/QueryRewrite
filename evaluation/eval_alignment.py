"""
RQ5 — Alignment Training Effect
================================

Compares retrieval quality between the frozen pre-trained embedding model
and the fine-tuned (alignment-trained) embedding model.

Measures:
  - Recall@K / MRR with fine-tuned model
  - Reduction in "both missed" samples
  - Combined effect: fine-tuned model + rewrite + gating

Usage::

    python -m evaluation.eval_alignment
    python -m evaluation.eval_alignment --checkpoint checkpoints/alignment/best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Metrics are pure numpy — safe to import at module level
from evaluation.metrics import mrr_arr, recall_at_k_arr


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL (no faiss dependency)."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _build_corpus(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
    """De-duplicate pos + neg docs → (texts, text→index)."""
    seen: Dict[str, int] = {}
    corpus: List[str] = []
    for r in records:
        for key in ("pos_doc", "hard_neg_doc"):
            doc = r[key]
            if doc not in seen:
                seen[doc] = len(corpus)
                corpus.append(doc)
    return corpus, seen


def _encode(model: Any, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode with EmbeddingModel → numpy float32."""
    embs = model.encode(texts, batch_size=batch_size)
    return embs.numpy().astype(np.float32)


def _print_table(title: str, rows: List[Tuple[str, ...]]) -> None:
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


def _save_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def evaluate_alignment(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    checkpoint_path: str = "checkpoints/alignment/best",
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 20,
    k_values: List[int] | None = None,
) -> Dict[str, Any]:
    """RQ5: frozen vs fine-tuned embedding comparison."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    # ── Load models (BEFORE faiss is imported) ────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.embedder import EmbeddingModel

    ckpt_file = Path(checkpoint_path) / "checkpoint.pt"
    if not ckpt_file.exists():
        print(f"  ✗ Checkpoint not found: {ckpt_file}")
        print("    Run `python scripts/train_alignment.py` first.")
        return {"error": "checkpoint_not_found"}

    print(f"[1/7] Loading fine-tuned model from {checkpoint_path}")
    ft_model = EmbeddingModel(model_name=base_model_name)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    ft_model.load_state_dict(ckpt["model_state_dict"])
    ft_model.eval()

    print("[2/7] Loading frozen model (fresh pre-trained) …")
    frozen_model = EmbeddingModel(model_name=base_model_name)
    frozen_model.eval()

    records = _load_dataset(dataset_path)
    corpus, doc2idx = _build_corpus(records)

    gold_pos_ids = [doc2idx[r["pos_doc"]] for r in records]

    orig_queries = [r["query_original"] for r in records]
    rew_queries = [r["query_rewrite"] for r in records]

    # ── Encode everything (still no faiss) ────────────────────
    print("[3/7] Encoding with frozen model …")
    frozen_corpus_emb = _encode(frozen_model, corpus)
    frozen_orig_emb = _encode(frozen_model, orig_queries)
    frozen_rew_emb = _encode(frozen_model, rew_queries)

    print("[4/7] Encoding with fine-tuned model …")
    ft_corpus_emb = _encode(ft_model, corpus)
    ft_orig_emb = _encode(ft_model, orig_queries)
    ft_rew_emb = _encode(ft_model, rew_queries)

    # Free model memory before retrieval
    del frozen_model, ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Retrieve top-k via numpy (avoids faiss/torch BLAS conflict on macOS) ──
    print("[5/7] Computing similarity & retrieving top-k …")

    def _topk_ip(queries: np.ndarray, corpus: np.ndarray, k: int):
        """Inner-product top-k search using numpy. Returns (scores, indices)."""
        # queries: (n_q, d), corpus: (n_c, d) — both L2-normalised
        sim = queries @ corpus.T  # (n_q, n_c) — cosine similarity
        # argsort descending → take top k
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_scores = np.take_along_axis(sim, topk_idx, axis=1)
        return topk_scores, topk_idx

    frozen_orig_scores, frozen_orig_idx = _topk_ip(frozen_orig_emb, frozen_corpus_emb, top_k)
    _, frozen_rew_idx = _topk_ip(frozen_rew_emb, frozen_corpus_emb, top_k)

    ft_orig_scores, ft_orig_idx = _topk_ip(ft_orig_emb, ft_corpus_emb, top_k)
    _, ft_rew_idx = _topk_ip(ft_rew_emb, ft_corpus_emb, top_k)

    print("[6/7] Computing metrics …\n")
    results: Dict[str, Any] = {
        "base_model": base_model_name,
        "checkpoint": checkpoint_path,
        "n_queries": len(records),
        "corpus_size": len(corpus),
    }

    # ── Recall@K comparison ───────────────────────────────────
    rows: List[Tuple[str, ...]] = [("K", "Frozen(orig)", "FT(orig)", "Delta", "FT(rew)", "FT+rew delta")]
    for k in k_values:
        if k > top_k:
            continue
        r_frozen = recall_at_k_arr(frozen_orig_idx, gold_pos_ids, k)
        r_ft = recall_at_k_arr(ft_orig_idx, gold_pos_ids, k)
        r_ft_rew = recall_at_k_arr(ft_rew_idx, gold_pos_ids, k)
        rows.append((
            str(k),
            f"{r_frozen:.4f}", f"{r_ft:.4f}", f"{r_ft - r_frozen:+.4f}",
            f"{r_ft_rew:.4f}", f"{r_ft_rew - r_frozen:+.4f}",
        ))
        results[f"recall@{k}_frozen_orig"] = round(r_frozen, 4)
        results[f"recall@{k}_ft_orig"] = round(r_ft, 4)
        results[f"recall@{k}_ft_rewrite"] = round(r_ft_rew, 4)
    _print_table("RQ5 — Recall@K: Frozen vs Fine-Tuned", rows)

    # ── MRR ───────────────────────────────────────────────────
    m_frozen = mrr_arr(frozen_orig_idx, gold_pos_ids, top_k)
    m_ft = mrr_arr(ft_orig_idx, gold_pos_ids, top_k)
    m_ft_rew = mrr_arr(ft_rew_idx, gold_pos_ids, top_k)
    results["mrr_frozen_orig"] = round(m_frozen, 4)
    results["mrr_ft_orig"] = round(m_ft, 4)
    results["mrr_ft_rewrite"] = round(m_ft_rew, 4)

    mrr_rows: List[Tuple[str, ...]] = [
        ("Model", "MRR"),
        ("Frozen (orig)", f"{m_frozen:.4f}"),
        ("Fine-tuned (orig)", f"{m_ft:.4f}"),
        ("Fine-tuned (rewrite)", f"{m_ft_rew:.4f}"),
    ]
    _print_table("MRR Comparison", mrr_rows)

    # ── Both-missed analysis ──────────────────────────────────
    frozen_both_miss = 0
    ft_both_miss = 0
    ft_saved_from_miss = 0

    for i in range(len(records)):
        fo = gold_pos_ids[i] in frozen_orig_idx[i, :1]
        fr = gold_pos_ids[i] in frozen_rew_idx[i, :1]
        fto = gold_pos_ids[i] in ft_orig_idx[i, :1]
        ftr = gold_pos_ids[i] in ft_rew_idx[i, :1]

        if not fo and not fr:
            frozen_both_miss += 1
            if fto or ftr:
                ft_saved_from_miss += 1
        if not fto and not ftr:
            ft_both_miss += 1

    results["frozen_both_miss"] = frozen_both_miss
    results["ft_both_miss"] = ft_both_miss
    results["ft_saved_from_miss"] = ft_saved_from_miss

    print(f"\n  ── Both-Missed Analysis (Recall@1) ──")
    print(f"  Frozen model both-miss:     {frozen_both_miss}")
    print(f"  Fine-tuned model both-miss: {ft_both_miss}")
    print(f"  Saved (was miss, now hit):   {ft_saved_from_miss}")

    # ── Fine-tuned + Gating ───────────────────────────────────
    print("[7/7] Sweeping gating thresholds on fine-tuned model …")
    best_tau = 0.0
    best_gated_r1 = 0.0
    for tau_int in range(10, 19):
        tau = tau_int * 0.05
        gated_emb = np.empty_like(ft_orig_emb)
        for i in range(len(records)):
            if ft_orig_scores[i, 0] >= tau:
                gated_emb[i] = ft_orig_emb[i]
            else:
                gated_emb[i] = ft_rew_emb[i]
        _, gated_idx = _topk_ip(gated_emb, ft_corpus_emb, top_k)
        g_r1 = recall_at_k_arr(gated_idx, gold_pos_ids, 1)
        if g_r1 > best_gated_r1:
            best_gated_r1 = g_r1
            best_tau = tau

    results["ft_gated_best_tau"] = best_tau
    results["ft_gated_best_recall1"] = round(best_gated_r1, 4)

    r1_frozen_orig = results.get("recall@1_frozen_orig", 0)
    print(f"\n  Fine-tuned + Gating (best τ={best_tau:.2f}): Recall@1 = {best_gated_r1:.4f}")
    print(f"  Total gain over frozen baseline: {best_gated_r1 - r1_frozen_orig:+.4f}")

    return results


# ── CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RQ5: Alignment evaluation")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/alignment/best")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save", default="evaluation/result/eval_alignment.json")
    args = parser.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    results = evaluate_alignment(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        base_model_name=args.model,
        top_k=args.top_k,
    )
    if "error" not in results:
        _save_metrics(results, args.save)
        print(f"\n  Results saved to {args.save}")


if __name__ == "__main__":
    main()
