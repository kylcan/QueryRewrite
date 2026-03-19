"""
RQ6 — Local Rewriter Evaluation (SFT vs DPO vs API)
=====================================================

Compares query rewriting quality across three sources:

  1. **API (baseline)**: Rewrites from the dataset (LLM API rewrites)
  2. **SFT model**: Locally trained rewriter after supervised fine-tuning
  3. **DPO model**: SFT model further aligned with DPO preference data

Evaluation axes:
  - Retrieval quality: Recall@K, MRR using the fine-tuned embedder
  - Rewrite diversity: length ratio, distinct n-gram overlap with original
  - Win/loss analysis: per-query comparison showing where each method wins

Usage::

    python -m evaluation.eval_rewriter
    python -m evaluation.eval_rewriter --sft checkpoints/sft/best --dpo checkpoints/dpo/best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import mrr_arr, recall_at_k_arr


# ── Helpers ────────────────────────────────────────────────────

def _load_dataset(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _build_corpus(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
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
    embs = model.encode(texts, batch_size=batch_size)
    return embs.numpy().astype(np.float32)


def _topk_ip(queries: np.ndarray, corpus: np.ndarray, k: int):
    """Inner-product top-k via numpy (avoids faiss BLAS conflict on macOS)."""
    sim = queries @ corpus.T
    topk_idx = np.argsort(-sim, axis=1)[:, :k]
    topk_scores = np.take_along_axis(sim, topk_idx, axis=1)
    return topk_scores, topk_idx


def _print_table(title: str, rows: List[Tuple[str, ...]]) -> None:
    col_widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _distinct_ngrams(text: str, n: int) -> set:
    tokens = text.lower().split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ── Main evaluation ──────────────────────────────────────────

def evaluate_rewriter(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    embedder_checkpoint: str = "checkpoints/alignment/best",
    sft_checkpoint: Optional[str] = "checkpoints/sft/best",
    dpo_checkpoint: Optional[str] = "checkpoints/dpo/best",
    base_rewriter_model: str = "Qwen/Qwen2.5-0.5B",
    embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 20,
    k_values: Optional[List[int]] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> Dict[str, Any]:
    """RQ6: Compare API / SFT / DPO rewrites on retrieval quality."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    from models.embedder import EmbeddingModel
    from models.rewriter import QueryRewriter

    records = _load_dataset(dataset_path)
    corpus, doc2idx = _build_corpus(records)
    gold_pos_ids = [doc2idx[r["pos_doc"]] for r in records]

    orig_queries = [r["query_original"] for r in records]
    api_rewrites = [r["query_rewrite"] for r in records]

    # ── Generate rewrites with local models ───────────────────
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sources: Dict[str, List[str]] = {
        "original": orig_queries,
        "api_rewrite": api_rewrites,
    }

    # SFT rewrites
    sft_path = Path(sft_checkpoint) if sft_checkpoint else None
    if sft_path and sft_path.exists():
        print("[1/6] Generating SFT rewrites …")
        sft_model = QueryRewriter(
            base_model_name=base_rewriter_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        sft_model.load_adapter(str(sft_path))
        sft_model.to(device)
        sources["sft_rewrite"] = sft_model.rewrite_queries(
            orig_queries, device=device, batch_size=4,
        )
        del sft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  Generated {len(sources['sft_rewrite'])} SFT rewrites")
    else:
        print("[1/6] SFT checkpoint not found — skipping SFT rewrites")

    # DPO rewrites
    dpo_path = Path(dpo_checkpoint) if dpo_checkpoint else None
    if dpo_path and dpo_path.exists():
        print("[2/6] Generating DPO rewrites …")
        dpo_model = QueryRewriter(
            base_model_name=base_rewriter_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        dpo_model.load_adapter(str(dpo_path))
        dpo_model.to(device)
        sources["dpo_rewrite"] = dpo_model.rewrite_queries(
            orig_queries, device=device, batch_size=4,
        )
        del dpo_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  Generated {len(sources['dpo_rewrite'])} DPO rewrites")
    else:
        print("[2/6] DPO checkpoint not found — skipping DPO rewrites")

    # ── Load embedder ─────────────────────────────────────────
    print("[3/6] Loading embedding model …")
    emb_model = EmbeddingModel(model_name=embedder_model_name)
    ckpt_file = Path(embedder_checkpoint) / "checkpoint.pt"
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        emb_model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded fine-tuned embedder from {embedder_checkpoint}")
    else:
        print(f"  Using frozen embedder (no checkpoint at {embedder_checkpoint})")
    emb_model.eval()

    # ── Encode all queries & corpus ───────────────────────────
    print("[4/6] Encoding corpus + queries …")
    corpus_emb = _encode(emb_model, corpus)

    query_embs: Dict[str, np.ndarray] = {}
    for name, texts in sources.items():
        query_embs[name] = _encode(emb_model, texts)

    del emb_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Retrieve & evaluate ───────────────────────────────────
    print("[5/6] Computing retrieval metrics …\n")
    results: Dict[str, Any] = {
        "n_queries": len(records),
        "corpus_size": len(corpus),
        "sources_evaluated": list(sources.keys()),
    }

    # Recall@K table
    header = ("K",) + tuple(sources.keys())
    rows: List[Tuple[str, ...]] = [header]

    for k in k_values:
        row: List[str] = [str(k)]
        for name in sources:
            _, idx = _topk_ip(query_embs[name], corpus_emb, top_k)
            r = recall_at_k_arr(idx, gold_pos_ids, k)
            results[f"recall@{k}_{name}"] = round(r, 4)
            row.append(f"{r:.4f}")
        rows.append(tuple(row))
    _print_table("RQ6 — Recall@K by Rewrite Source", rows)

    # MRR
    mrr_rows: List[Tuple[str, ...]] = [("Source", "MRR")]
    for name in sources:
        _, idx = _topk_ip(query_embs[name], corpus_emb, top_k)
        m = mrr_arr(idx, gold_pos_ids, top_k)
        results[f"mrr_{name}"] = round(m, 4)
        mrr_rows.append((name, f"{m:.4f}"))
    _print_table("MRR by Rewrite Source", mrr_rows)

    # ── Rewrite quality analysis ──────────────────────────────
    print("\n  ── Rewrite Quality Analysis ──")
    rewrite_sources = {k: v for k, v in sources.items() if k != "original"}
    for name, texts in rewrite_sources.items():
        avg_len_ratio = np.mean([
            len(t.split()) / max(len(o.split()), 1)
            for t, o in zip(texts, orig_queries)
        ])
        exact_copy = sum(1 for t, o in zip(texts, orig_queries) if t.strip().lower() == o.strip().lower())
        # Distinct unigram overlap with original
        overlap_ratios = []
        for t, o in zip(texts, orig_queries):
            orig_unigrams = _distinct_ngrams(o, 1)
            rew_unigrams = _distinct_ngrams(t, 1)
            if rew_unigrams:
                overlap_ratios.append(len(orig_unigrams & rew_unigrams) / len(rew_unigrams))
            else:
                overlap_ratios.append(1.0)
        avg_overlap = np.mean(overlap_ratios)

        print(f"  {name}:")
        print(f"    Avg length ratio (rewrite/original): {avg_len_ratio:.2f}")
        print(f"    Exact copies of original: {exact_copy}/{len(texts)}")
        print(f"    Avg unigram overlap with original: {avg_overlap:.3f}")
        results[f"{name}_avg_len_ratio"] = round(float(avg_len_ratio), 3)
        results[f"{name}_exact_copy"] = exact_copy
        results[f"{name}_avg_unigram_overlap"] = round(float(avg_overlap), 3)

    # ── Per-query win/loss analysis (Recall@1) ────────────────
    print("\n[6/6] Per-query win/loss analysis (Recall@1) …")
    retrieved_r1: Dict[str, np.ndarray] = {}
    for name in sources:
        _, idx = _topk_ip(query_embs[name], corpus_emb, top_k)
        retrieved_r1[name] = idx[:, :1]

    # Compare each rewrite source vs original
    for name in rewrite_sources:
        wins = losses = ties = 0
        for i in range(len(records)):
            orig_hit = gold_pos_ids[i] in retrieved_r1["original"][i]
            rew_hit = gold_pos_ids[i] in retrieved_r1[name][i]
            if rew_hit and not orig_hit:
                wins += 1
            elif orig_hit and not rew_hit:
                losses += 1
            else:
                ties += 1
        net = wins - losses
        results[f"{name}_vs_original_wins"] = wins
        results[f"{name}_vs_original_losses"] = losses
        results[f"{name}_vs_original_net"] = net
        print(f"  {name} vs original: +{wins} / -{losses} / ={ties} (net {net:+d})")

    # Compare DPO vs SFT if both exist
    if "sft_rewrite" in sources and "dpo_rewrite" in sources:
        wins = losses = ties = 0
        for i in range(len(records)):
            sft_hit = gold_pos_ids[i] in retrieved_r1["sft_rewrite"][i]
            dpo_hit = gold_pos_ids[i] in retrieved_r1["dpo_rewrite"][i]
            if dpo_hit and not sft_hit:
                wins += 1
            elif sft_hit and not dpo_hit:
                losses += 1
            else:
                ties += 1
        results["dpo_vs_sft_wins"] = wins
        results["dpo_vs_sft_losses"] = losses
        results["dpo_vs_sft_net"] = wins - losses
        print(f"  DPO vs SFT: +{wins} / -{losses} / ={ties} (net {wins - losses:+d})")

    return results


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ6: Rewriter evaluation")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--embedder_checkpoint", default="checkpoints/alignment/best")
    parser.add_argument("--sft", default="checkpoints/sft/best")
    parser.add_argument("--dpo", default="checkpoints/dpo/best")
    parser.add_argument("--rewriter_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--embedder_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save", default="evaluation/result/eval_rewriter.json")
    args = parser.parse_args()

    results = evaluate_rewriter(
        dataset_path=args.dataset,
        embedder_checkpoint=args.embedder_checkpoint,
        sft_checkpoint=args.sft,
        dpo_checkpoint=args.dpo,
        base_rewriter_model=args.rewriter_model,
        embedder_model_name=args.embedder_model,
        top_k=args.top_k,
    )
    if "error" not in results:
        _save_metrics(results, args.save)
        print(f"\n  Results saved to {args.save}")


if __name__ == "__main__":
    main()
