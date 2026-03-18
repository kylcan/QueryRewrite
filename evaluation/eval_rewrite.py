"""
RQ3 — Query Rewrite Effect Evaluation
======================================

Measures how query rewriting affects retrieval quality:
  - Recall@K / MRR comparison (original vs rewrite)
  - Per-sample breakdown: helped / hurt / both_hit / both_miss

Usage::

    python -m evaluation.eval_rewrite
    python -m evaluation.eval_rewrite --dataset data/MS_MARCO/intent_dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.metrics import hard_neg_in_topk, mrr_arr, recall_at_k_arr
from evaluation.shared import (
    RetrievalContext,
    print_table,
    save_metrics,
)


def evaluate_rewrite(
    ctx: RetrievalContext,
    k_values: List[int] | None = None,
    max_examples: int = 10,
) -> Dict[str, Any]:
    """RQ3: original vs rewrite comparison + per-sample analysis."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    results: Dict[str, Any] = {
        "model": ctx.model_name,
        "n_queries": len(ctx.records),
    }

    # ── Recall@K comparison ───────────────────────────────────
    rows: List[Tuple[str, ...]] = [("K", "Recall(orig)", "Recall(rewrite)", "Delta")]
    for k in k_values:
        if k > ctx.top_k:
            continue
        r_o = recall_at_k_arr(ctx.orig_indices, ctx.gold_pos_ids, k)
        r_r = recall_at_k_arr(ctx.rew_indices, ctx.gold_pos_ids, k)
        rows.append((str(k), f"{r_o:.4f}", f"{r_r:.4f}", f"{r_r - r_o:+.4f}"))
        results[f"recall@{k}_orig"] = round(r_o, 4)
        results[f"recall@{k}_rewrite"] = round(r_r, 4)
    print_table("RQ3 — Recall@K: Original vs Rewrite", rows)

    # ── MRR ───────────────────────────────────────────────────
    m_o = mrr_arr(ctx.orig_indices, ctx.gold_pos_ids, ctx.top_k)
    m_r = mrr_arr(ctx.rew_indices, ctx.gold_pos_ids, ctx.top_k)
    results["mrr_orig"] = round(m_o, 4)
    results["mrr_rewrite"] = round(m_r, 4)
    mrr_rows: List[Tuple[str, ...]] = [
        ("Metric", "Original", "Rewrite", "Delta"),
        ("MRR", f"{m_o:.4f}", f"{m_r:.4f}", f"{m_r - m_o:+.4f}"),
    ]
    print_table("MRR Comparison", mrr_rows)

    # ── Hard-neg confusion: orig vs rewrite ───────────────────
    hn_rows: List[Tuple[str, ...]] = [("K", "HN-TopK(orig)", "HN-TopK(rew)")]
    for k in k_values:
        if k > ctx.top_k:
            continue
        h_o = hard_neg_in_topk(ctx.orig_indices, ctx.gold_neg_ids, k)
        h_r = hard_neg_in_topk(ctx.rew_indices, ctx.gold_neg_ids, k)
        hn_rows.append((str(k), f"{h_o:.4f}", f"{h_r:.4f}"))
    print_table("Hard-Negative Confusion: Orig vs Rewrite", hn_rows)

    # ── Per-sample analysis (Recall@1) ────────────────────────
    helped: List[int] = []
    hurt: List[int] = []
    both_hit: List[int] = []
    both_miss: List[int] = []

    for i in range(len(ctx.records)):
        o = ctx.gold_pos_ids[i] in ctx.orig_indices[i, :1]
        r = ctx.gold_pos_ids[i] in ctx.rew_indices[i, :1]
        if o and r:
            both_hit.append(i)
        elif not o and r:
            helped.append(i)
        elif o and not r:
            hurt.append(i)
        else:
            both_miss.append(i)

    n = len(ctx.records)
    print("\n\n  ── Per-Sample Rewrite Analysis (Recall@1) ──")
    print(f"  Both hit:      {len(both_hit):4d}  ({len(both_hit)/n*100:.1f}%)")
    print(f"  Rewrite helped:{len(helped):4d}  ({len(helped)/n*100:.1f}%)  — orig missed, rewrite hit")
    print(f"  Rewrite hurt:  {len(hurt):4d}  ({len(hurt)/n*100:.1f}%)  — orig hit, rewrite missed")
    print(f"  Both missed:   {len(both_miss):4d}  ({len(both_miss)/n*100:.1f}%)")

    results["rewrite_helped"] = len(helped)
    results["rewrite_hurt"] = len(hurt)
    results["rewrite_both_hit"] = len(both_hit)
    results["rewrite_both_miss"] = len(both_miss)

    def _show(label: str, indices: List[int]) -> None:
        if not indices:
            return
        print(f"\n  ▸ {label} (showing up to {min(max_examples, len(indices))}):")
        for idx in indices[:max_examples]:
            rec = ctx.records[idx]
            print(f"    [{idx:3d}] orig: \"{rec['query_original'][:80]}\"")
            print(f"          rew:  \"{rec['query_rewrite'][:80]}\"")
            print(f"          cos(Q,P+)={rec['cos_q_pos']:.3f}"
                  f"  cos(Rew,P+)~={rec['cos_q_rewrite']:.3f}"
                  f"  tier={rec.get('gap_type', '?')}")

    _show("Rewrite HELPED", helped)
    _show("Rewrite HURT", hurt)
    _show("BOTH MISSED", both_miss)

    # ── Verdict ───────────────────────────────────────────────
    r1_o = results.get("recall@1_orig", 0)
    r1_r = results.get("recall@1_rewrite", 0)
    delta = r1_r - r1_o
    print(f"\n  ── Verdict ──")
    print(f"  Recall@1 delta = {delta:+.4f}", end="")
    if delta > 0.01:
        print("  → Rewrite helps ✅")
    elif delta > -0.01:
        print("  → Neutral ⚠️")
    else:
        print("  → Rewrite hurts ❌")
    print(f"  Net (helped − hurt) = {len(helped) - len(hurt):+d}")
    print()

    return results


# ── CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3: Rewrite evaluation")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save", default="evaluation/result/eval_rewrite.json")
    args = parser.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    ctx = RetrievalContext.build(args.dataset, args.model, args.top_k)
    results = evaluate_rewrite(ctx)
    save_metrics(results, args.save)


if __name__ == "__main__":
    main()
