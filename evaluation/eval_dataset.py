"""
RQ1 & RQ2 — Dataset Quality Evaluation
=======================================

Validates the constructed dataset:
  RQ1  Is baseline retrieval hard enough?  → Recall@K with original queries
  RQ2  Are hard negatives effective?       → Hard-neg confusion rate + per-tier

Usage::

    python -m evaluation.eval_dataset
    python -m evaluation.eval_dataset --dataset data/MS_MARCO/intent_dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation.metrics import hard_neg_in_topk, mrr_arr, recall_at_k_arr
from evaluation.shared import (
    RetrievalContext,
    print_table,
    save_metrics,
)


def evaluate_dataset(
    ctx: RetrievalContext,
    k_values: List[int] | None = None,
) -> Dict[str, Any]:
    """RQ1 + RQ2: dataset difficulty and hard-negative quality."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    results: Dict[str, Any] = {
        "model": ctx.model_name,
        "n_queries": len(ctx.records),
        "corpus_size": len(ctx.corpus),
    }

    # ── RQ1: Recall@K with original queries ───────────────────
    recall_rows: List[Tuple[str, ...]] = [("K", "Recall(orig)", "MRR")]
    for k in k_values:
        if k > ctx.top_k:
            continue
        r = recall_at_k_arr(ctx.orig_indices, ctx.gold_pos_ids, k)
        recall_rows.append((str(k), f"{r:.4f}", ""))
        results[f"recall@{k}_orig"] = round(r, 4)

    m = mrr_arr(ctx.orig_indices, ctx.gold_pos_ids, ctx.top_k)
    recall_rows[1] = (*recall_rows[1][:2], f"{m:.4f}")
    results["mrr_orig"] = round(m, 4)

    print_table("RQ1 — Baseline Retrieval Difficulty", recall_rows)

    # ── RQ2: Hard-negative confusion ──────────────────────────
    neg_rows: List[Tuple[str, ...]] = [("K", "HardNeg-in-TopK")]
    for k in k_values:
        if k > ctx.top_k:
            continue
        hn = hard_neg_in_topk(ctx.orig_indices, ctx.gold_neg_ids, k)
        neg_rows.append((str(k), f"{hn:.4f}"))
        results[f"hardneg_in_top{k}"] = round(hn, 4)

    print_table("RQ2 — Hard Negative Confusion Rate", neg_rows)

    # ── RQ2b: Per-tier breakdown ──────────────────────────────
    tiers = sorted({r.get("gap_type", "unknown") for r in ctx.records})
    if len(tiers) > 1:
        tier_rows: List[Tuple[str, ...]] = [
            ("Tier", "N", "ConfTop1", "ConfTop5", "AvgCos(Q,N-)", "AvgRank"),
        ]
        for tier in tiers:
            idx = [i for i, r in enumerate(ctx.records) if r.get("gap_type") == tier]
            if not idx:
                continue
            tier_neg = [ctx.gold_neg_ids[i] for i in idx]
            tier_ret = ctx.orig_indices[idx]
            c1 = hard_neg_in_topk(tier_ret, tier_neg, 1)
            c5 = hard_neg_in_topk(tier_ret, tier_neg, 5)
            avg_cos = np.mean([ctx.records[i]["cos_q_neg"] for i in idx])
            avg_rank = np.mean([ctx.records[i].get("neg_rank", -1) for i in idx])
            tier_rows.append((
                tier, str(len(idx)),
                f"{c1:.4f}", f"{c5:.4f}",
                f"{avg_cos:.4f}", f"{avg_rank:.1f}",
            ))
            results[f"tier_{tier}_n"] = len(idx)
            results[f"tier_{tier}_conf_top1"] = round(c1, 4)
            results[f"tier_{tier}_conf_top5"] = round(c5, 4)
        print_table("RQ2b — Per-Tier Negative Confusion", tier_rows)

    # ── Embedding similarity stats ────────────────────────────
    cos_qp = [r["cos_q_pos"] for r in ctx.records]
    cos_qn = [r["cos_q_neg"] for r in ctx.records]
    sim_rows: List[Tuple[str, ...]] = [
        ("Pair", "Mean", "Std", "Min", "Max"),
        ("cos(Q, P+)",
         f"{np.mean(cos_qp):.4f}", f"{np.std(cos_qp):.4f}",
         f"{np.min(cos_qp):.4f}", f"{np.max(cos_qp):.4f}"),
        ("cos(Q, N-)",
         f"{np.mean(cos_qn):.4f}", f"{np.std(cos_qn):.4f}",
         f"{np.min(cos_qn):.4f}", f"{np.max(cos_qn):.4f}"),
        ("Gap (P+ − N-)",
         f"{np.mean(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.std(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.min(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.max(np.array(cos_qp) - np.array(cos_qn)):.4f}"),
    ]
    print_table("Embedding Similarity Analysis", sim_rows)

    # ── Verdict ───────────────────────────────────────────────
    r1 = results.get("recall@1_orig", 0)
    hn5 = results.get("hardneg_in_top5", 0)

    print("\n  ── Verdict ──")
    print(f"  RQ1  Baseline Recall@1 = {r1:.4f}", end="")
    if r1 < 0.60:
        print("  → Challenging ✅")
    elif r1 < 0.80:
        print("  → Moderate difficulty ⚠️")
    else:
        print("  → Too easy — consider larger corpus ❌")

    print(f"  RQ2  HardNeg-in-Top5 = {hn5:.4f}", end="")
    if hn5 > 0.30:
        print("  → Negatives are confusing ✅")
    else:
        print("  → Negatives may be too easy ⚠️")

    if len(tiers) > 1:
        for tier in tiers:
            c1 = results.get(f"tier_{tier}_conf_top1", 0)
            print(f"       {tier:8s}  ConfTop1={c1:.4f}", end="")
            if tier == "hard" and c1 > 0.3:
                print("  → Truly confusing ✅")
            elif tier == "easy" and c1 < 0.1:
                print("  → Easily filtered ✅")
            else:
                print()

    print()
    return results


# ── CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1+RQ2: Dataset quality")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save", default="evaluation/result/eval_dataset.json")
    args = parser.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    ctx = RetrievalContext.build(args.dataset, args.model, args.top_k)
    results = evaluate_dataset(ctx)
    save_metrics(results, args.save)


if __name__ == "__main__":
    main()
