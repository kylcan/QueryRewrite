"""
RQ4 — Retrieval Confidence Gating
==================================

Implements Plan A: selective rewriting based on retrieval confidence.

Gate logic:
  cos(orig_query, top1_doc) ≥ τ  →  keep original query
  cos(orig_query, top1_doc) <  τ  →  use LLM rewrite

Sweeps thresholds to find optimal τ and provides detailed analysis.

Usage::

    python -m evaluation.eval_gating
    python -m evaluation.eval_gating --dataset data/MS_MARCO/intent_dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation.metrics import mrr_arr, recall_at_k_arr
from evaluation.shared import (
    RetrievalContext,
    print_table,
    save_metrics,
    search,
)


def evaluate_gating(
    ctx: RetrievalContext,
    thresholds: List[float] | None = None,
) -> Dict[str, Any]:
    """RQ4: threshold sweep for retrieval confidence gating."""

    if thresholds is None:
        thresholds = [round(t * 0.05 + 0.50, 2) for t in range(10)]

    r1_orig = recall_at_k_arr(ctx.orig_indices, ctx.gold_pos_ids, 1)
    r1_rew = recall_at_k_arr(ctx.rew_indices, ctx.gold_pos_ids, 1)

    results: Dict[str, Any] = {
        "recall@1_orig": round(r1_orig, 4),
        "recall@1_ungated_rewrite": round(r1_rew, 4),
    }

    print("\n  ══ RQ4 — Retrieval Confidence Gating ══")
    print("  Gate: cos(orig_query, top1_doc) ≥ τ → keep original; < τ → use rewrite\n")

    gate_rows: List[Tuple[str, ...]] = [
        ("τ", "N_kept", "N_rewrite", "Recall@1", "MRR",
         "Helped", "Hurt", "Net", "vs_ungated"),
    ]

    best_tau = 0.0
    best_recall = 0.0

    for tau in thresholds:
        gated_emb = np.empty_like(ctx.orig_emb)
        n_kept = 0
        for i in range(len(ctx.records)):
            if ctx.orig_scores[i, 0] >= tau:
                gated_emb[i] = ctx.orig_emb[i]
                n_kept += 1
            else:
                gated_emb[i] = ctx.rewrite_emb[i]

        n_rewrite = len(ctx.records) - n_kept
        _, gated_idx = search(ctx.index, gated_emb, ctx.top_k)

        g_r1 = recall_at_k_arr(gated_idx, ctx.gold_pos_ids, 1)
        g_mrr = mrr_arr(gated_idx, ctx.gold_pos_ids, ctx.top_k)

        g_helped = g_hurt = 0
        for i in range(len(ctx.records)):
            o = ctx.gold_pos_ids[i] in ctx.orig_indices[i, :1]
            g = ctx.gold_pos_ids[i] in gated_idx[i, :1]
            if not o and g:
                g_helped += 1
            elif o and not g:
                g_hurt += 1

        net = g_helped - g_hurt
        vs = g_r1 - r1_rew

        gate_rows.append((
            f"{tau:.2f}", str(n_kept), str(n_rewrite),
            f"{g_r1:.4f}", f"{g_mrr:.4f}",
            str(g_helped), str(g_hurt), f"{net:+d}", f"{vs:+.4f}",
        ))

        if g_r1 > best_recall:
            best_recall = g_r1
            best_tau = tau

    print_table("Threshold Sweep — Gated Rewrite", gate_rows)

    gain_orig = best_recall - r1_orig
    gain_rew = best_recall - r1_rew

    print(f"\n  Best τ = {best_tau:.2f}  →  Recall@1 = {best_recall:.4f}"
          f"  (orig={r1_orig:.4f}, ungated_rew={r1_rew:.4f})")
    print(f"  Gain over original:        {gain_orig:+.4f}")
    print(f"  Gain over ungated rewrite: {gain_rew:+.4f}")

    results["gate_best_tau"] = best_tau
    results["gate_best_recall1"] = round(best_recall, 4)
    results["gate_gain_over_orig"] = round(gain_orig, 4)
    results["gate_gain_over_ungated"] = round(gain_rew, 4)

    # ── Detail analysis at best τ ─────────────────────────────
    gated_emb = np.empty_like(ctx.orig_emb)
    for i in range(len(ctx.records)):
        if ctx.orig_scores[i, 0] >= best_tau:
            gated_emb[i] = ctx.orig_emb[i]
        else:
            gated_emb[i] = ctx.rewrite_emb[i]

    _, gated_best = search(ctx.index, gated_emb, ctx.top_k)

    saved: List[int] = []
    lost: List[int] = []
    for i in range(len(ctx.records)):
        o = ctx.gold_pos_ids[i] in ctx.orig_indices[i, :1]
        r = ctx.gold_pos_ids[i] in ctx.rew_indices[i, :1]
        g = ctx.gold_pos_ids[i] in gated_best[i, :1]
        if o and not r and g:
            saved.append(i)
        if not o and r and not g:
            lost.append(i)

    print(f"\n  At τ={best_tau:.2f}:")
    print(f"    Gating SAVED (hurt→hit):    {len(saved)}")
    print(f"    Gating LOST  (helped→miss): {len(lost)}")

    if saved:
        print(f"\n  ▸ Gating SAVED (up to 5):")
        for idx in saved[:5]:
            rec = ctx.records[idx]
            print(f"    [{idx:3d}] \"{rec['query_original'][:70]}\"")
            print(f"          top1_score={ctx.orig_scores[idx,0]:.3f}"
                  f"  cos(Q,P+)={rec['cos_q_pos']:.3f}")

    if lost:
        print(f"\n  ▸ Gating LOST (up to 5):")
        for idx in lost[:5]:
            rec = ctx.records[idx]
            print(f"    [{idx:3d}] \"{rec['query_original'][:70]}\"")
            print(f"          top1_score={ctx.orig_scores[idx,0]:.3f}"
                  f"  cos(Q,P+)={rec['cos_q_pos']:.3f}")

    print()
    return results


# ── CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RQ4: Gating strategy")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save", default="evaluation/result/eval_gating.json")
    args = parser.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    ctx = RetrievalContext.build(args.dataset, args.model, args.top_k)
    results = evaluate_gating(ctx)
    save_metrics(results, args.save)


if __name__ == "__main__":
    main()
