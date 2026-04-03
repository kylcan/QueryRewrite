# pyright: reportCallIssue=false

"""
MS MARCO Intent-Aware Dataset — Retrieval Evaluation
=====================================================

Validates the constructed dataset by answering three research questions:

  RQ1  Does the larger corpus make baseline retrieval harder?
       → Recall@K with original queries should be significantly < 1.0

  RQ2  Are hard negatives effective?
       → Hard negatives should appear in top-K more often than random docs

  RQ3  Does query rewrite improve retrieval?
       → Recall@K(rewrite) > Recall@K(original)

Pipeline:
  1. Load intent_dataset.jsonl
  2. Build a FAISS corpus from ALL unique docs (pos + hard_neg + extras)
  3. Encode original queries + rewritten queries
  4. Retrieve top-K for both
  5. Compute Recall@K, MRR, and hard-neg confusion rate
  6. Print comparison tables

Usage::

    python data/MS_MARCO/eval_retrieval.py
    python data/MS_MARCO/eval_retrieval.py --top_k 10 --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ================================================================
# Data loading
# ================================================================

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the JSONL dataset into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_corpus(records: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
    """Extract a de-duplicated corpus and return (texts, text→index map).

    The corpus includes both positive docs and hard negatives so the
    retrieval task is non-trivial (the model must distinguish them).
    """
    seen: Dict[str, int] = {}
    corpus: List[str] = []
    for r in records:
        for key in ("pos_doc", "hard_neg_doc"):
            doc = r[key]
            if doc not in seen:
                seen[doc] = len(corpus)
                corpus.append(doc)
    return corpus, seen


# ================================================================
# Encoding + FAISS
# ================================================================

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
    index: Any = faiss.IndexFlatIP(dim)
    getattr(index, "add")(embeddings)
    return index


def search(index: Any, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) each of shape (Q, top_k)."""
    scores, indices = getattr(index, "search")(queries, top_k)
    return scores, indices


# ================================================================
# Metrics
# ================================================================

def recall_at_k(retrieved: np.ndarray, gold: List[int], k: int) -> float:
    """Recall@K — fraction of queries where gold doc appears in top-K."""
    hits = sum(1 for i, g in enumerate(gold) if g in retrieved[i, :k])
    return hits / len(gold)


def mrr(retrieved: np.ndarray, gold: List[int], max_k: int) -> float:
    """Mean Reciprocal Rank (up to max_k)."""
    rr_sum = 0.0
    for i, g in enumerate(gold):
        for rank in range(min(max_k, retrieved.shape[1])):
            if retrieved[i, rank] == g:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / len(gold)


def hard_neg_in_topk(retrieved: np.ndarray, neg_ids: List[int], k: int) -> float:
    """Fraction of queries where the hard-negative appears in top-K.

    High values = negatives are confusing the model (good dataset property).
    """
    hits = sum(1 for i, n in enumerate(neg_ids) if n in retrieved[i, :k])
    return hits / len(neg_ids)


# ================================================================
# Reporting
# ================================================================

def print_table(title: str, rows: List[Tuple[str, ...]]) -> None:
    """Print an aligned ASCII table."""
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


# ================================================================
# Main evaluation
# ================================================================

def evaluate(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    model_name: str = "all-MiniLM-L6-v2",
    k_values: List[int] | None = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Run the full evaluation and return metrics dict."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    # ── 1. Load data ──────────────────────────────────────────
    print(f"[1/5] Loading dataset from {dataset_path}")
    records = load_dataset(dataset_path)
    print(f"       {len(records)} samples loaded")

    # ── 2. Build corpus ───────────────────────────────────────
    print("[2/5] Building corpus …")
    corpus, doc2idx = build_corpus(records)
    print(f"       Corpus size: {len(corpus)} unique documents")

    gold_pos_ids = [doc2idx[r["pos_doc"]] for r in records]
    gold_neg_ids = [doc2idx[r["hard_neg_doc"]] for r in records]

    original_queries = [r["query_original"] for r in records]
    rewrite_queries = [r["query_rewrite"] for r in records]

    # ── 3. Encode ─────────────────────────────────────────────
    print(f"[3/5] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("       Encoding corpus …")
    corpus_emb = encode(model, corpus)

    print("       Encoding original queries …")
    orig_emb = encode(model, original_queries)

    print("       Encoding rewritten queries …")
    rewrite_emb = encode(model, rewrite_queries)

    # ── 4. Retrieve ───────────────────────────────────────────
    print(f"[4/5] Building FAISS index (dim={corpus_emb.shape[1]}, n={len(corpus)}) …")
    index = build_index(corpus_emb)

    print(f"       Searching top-{top_k} for original queries …")
    orig_scores, orig_indices = search(index, orig_emb, top_k)

    print(f"       Searching top-{top_k} for rewritten queries …")
    rew_scores, rew_indices = search(index, rewrite_emb, top_k)

    # ── 5. Metrics ────────────────────────────────────────────
    print("[5/5] Computing metrics …\n")

    results: Dict[str, Any] = {
        "model": model_name,
        "n_queries": len(records),
        "corpus_size": len(corpus),
    }

    # --- Table 1: Recall@K comparison ---
    recall_rows: List[Tuple[str, ...]] = [("K", "Recall(orig)", "Recall(rewrite)", "Delta")]
    for k in k_values:
        if k > top_k:
            continue
        r_orig = recall_at_k(orig_indices, gold_pos_ids, k)
        r_rew = recall_at_k(rew_indices, gold_pos_ids, k)
        delta = r_rew - r_orig
        recall_rows.append((str(k), f"{r_orig:.4f}", f"{r_rew:.4f}", f"{delta:+.4f}"))
        results[f"recall@{k}_orig"] = round(r_orig, 4)
        results[f"recall@{k}_rewrite"] = round(r_rew, 4)

    print_table("RQ1 & RQ3 — Recall@K: Original vs Rewrite", recall_rows)

    # --- Table 2: MRR comparison ---
    mrr_orig = mrr(orig_indices, gold_pos_ids, top_k)
    mrr_rew = mrr(rew_indices, gold_pos_ids, top_k)
    results["mrr_orig"] = round(mrr_orig, 4)
    results["mrr_rewrite"] = round(mrr_rew, 4)

    mrr_rows: List[Tuple[str, ...]] = [
        ("Metric", "Original", "Rewrite", "Delta"),
        ("MRR", f"{mrr_orig:.4f}", f"{mrr_rew:.4f}", f"{mrr_rew - mrr_orig:+.4f}"),
    ]
    print_table("MRR Comparison", mrr_rows)

    # --- Table 3: Hard-negative confusion ---
    neg_rows: List[Tuple[str, ...]] = [("K", "HardNeg-in-TopK(orig)", "HardNeg-in-TopK(rew)")]
    for k in k_values:
        if k > top_k:
            continue
        hn_orig = hard_neg_in_topk(orig_indices, gold_neg_ids, k)
        hn_rew = hard_neg_in_topk(rew_indices, gold_neg_ids, k)
        neg_rows.append((str(k), f"{hn_orig:.4f}", f"{hn_rew:.4f}"))
        results[f"hardneg_in_top{k}_orig"] = round(hn_orig, 4)
        results[f"hardneg_in_top{k}_rewrite"] = round(hn_rew, 4)

    print_table("RQ2 — Hard Negative Confusion Rate", neg_rows)

    # --- Table 3b: Per-tier hard-neg confusion ---
    tiers = sorted({r.get("gap_type", "unknown") for r in records})
    if len(tiers) > 1:
        tier_rows: List[Tuple[str, ...]] = [
            ("Tier", "N", "ConfTop1(orig)", "ConfTop5(orig)", "AvgCos(Q,N-)", "AvgRank"),
        ]
        for tier in tiers:
            tier_idx = [i for i, r in enumerate(records) if r.get("gap_type") == tier]
            if not tier_idx:
                continue
            tier_neg_ids = [gold_neg_ids[i] for i in tier_idx]
            tier_orig_ret = orig_indices[tier_idx]
            t_conf1 = hard_neg_in_topk(tier_orig_ret, tier_neg_ids, 1)
            t_conf5 = hard_neg_in_topk(tier_orig_ret, tier_neg_ids, 5)
            t_cos = np.mean([records[i]["cos_q_neg"] for i in tier_idx])
            t_rank = np.mean([records[i].get("neg_rank", -1) for i in tier_idx])
            tier_rows.append((
                tier, str(len(tier_idx)),
                f"{t_conf1:.4f}", f"{t_conf5:.4f}",
                f"{t_cos:.4f}", f"{t_rank:.1f}",
            ))
            results[f"tier_{tier}_n"] = len(tier_idx)
            results[f"tier_{tier}_conf_top1"] = round(t_conf1, 4)
            results[f"tier_{tier}_conf_top5"] = round(t_conf5, 4)
            results[f"tier_{tier}_avg_cos_neg"] = round(float(t_cos), 4)
        print_table("RQ2b — Per-Tier Negative Confusion Breakdown", tier_rows)

    # --- Table 4: Cosine similarity analysis (from dataset fields) ---
    cos_qp = [r["cos_q_pos"] for r in records]
    cos_qn = [r["cos_q_neg"] for r in records]
    cos_qr = [r["cos_q_rewrite"] for r in records]

    sim_rows: List[Tuple[str, ...]] = [
        ("Pair", "Mean", "Std", "Min", "Max"),
        ("cos(Q, P+)",
         f"{np.mean(cos_qp):.4f}", f"{np.std(cos_qp):.4f}",
         f"{np.min(cos_qp):.4f}", f"{np.max(cos_qp):.4f}"),
        ("cos(Q, N-)",
         f"{np.mean(cos_qn):.4f}", f"{np.std(cos_qn):.4f}",
         f"{np.min(cos_qn):.4f}", f"{np.max(cos_qn):.4f}"),
        ("cos(Q, Rewrite)",
         f"{np.mean(cos_qr):.4f}", f"{np.std(cos_qr):.4f}",
         f"{np.min(cos_qr):.4f}", f"{np.max(cos_qr):.4f}"),
        ("Gap (P+ − N-)",
         f"{np.mean(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.std(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.min(np.array(cos_qp) - np.array(cos_qn)):.4f}",
         f"{np.max(np.array(cos_qp) - np.array(cos_qn)):.4f}"),
    ]
    print_table("Embedding Similarity Analysis (from dataset)", sim_rows)

    # --- Summary verdict ---
    r1_orig = results.get("recall@1_orig", 0)
    r1_rew = results.get("recall@1_rewrite", 0)
    hn5_orig = results.get("hardneg_in_top5_orig", 0)

    print("\n  ── Verdict ──")
    print(f"  RQ1  Baseline Recall@1 = {r1_orig:.4f}", end="")
    if r1_orig < 0.60:
        print("  → Task is challenging ✅")
    elif r1_orig < 0.80:
        print("  → Moderate difficulty ⚠️")
    else:
        print("  → Still too easy — consider larger corpus ❌")

    print(f"  RQ2  HardNeg-in-Top5 = {hn5_orig:.4f}", end="")
    if hn5_orig > 0.30:
        print("  → Negatives are confusing the model ✅")
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

    delta_r1 = r1_rew - r1_orig
    print(f"  RQ3  Recall@1 delta (rewrite − orig) = {delta_r1:+.4f}", end="")
    if delta_r1 > 0.01:
        print("  → Rewrite helps ✅")
    elif delta_r1 > -0.01:
        print("  → Rewrite is neutral ⚠️")
    else:
        print("  → Rewrite hurts — rules may be too noisy ❌")

    # --- Per-sample rewrite effect analysis ---
    print("\n\n  ── Per-Sample Rewrite Analysis (Recall@1) ──")

    helped: List[int] = []    # orig miss, rewrite hit
    hurt: List[int] = []      # orig hit, rewrite miss
    both_hit: List[int] = []
    both_miss: List[int] = []

    for i in range(len(records)):
        o_hit = gold_pos_ids[i] in orig_indices[i, :1]
        r_hit = gold_pos_ids[i] in rew_indices[i, :1]
        if o_hit and r_hit:
            both_hit.append(i)
        elif not o_hit and r_hit:
            helped.append(i)
        elif o_hit and not r_hit:
            hurt.append(i)
        else:
            both_miss.append(i)

    print(f"  Both hit:      {len(both_hit):4d}  ({len(both_hit)/len(records)*100:.1f}%)")
    print(f"  Rewrite helped:{len(helped):4d}  ({len(helped)/len(records)*100:.1f}%)  — orig missed, rewrite hit")
    print(f"  Rewrite hurt:  {len(hurt):4d}  ({len(hurt)/len(records)*100:.1f}%)  — orig hit, rewrite missed")
    print(f"  Both missed:   {len(both_miss):4d}  ({len(both_miss)/len(records)*100:.1f}%)")

    results["rewrite_helped"] = len(helped)
    results["rewrite_hurt"] = len(hurt)
    results["rewrite_both_hit"] = len(both_hit)
    results["rewrite_both_miss"] = len(both_miss)

    max_examples = 10

    if helped:
        print(f"\n  ▸ Rewrite HELPED (showing up to {min(max_examples, len(helped))}):")
        for idx in helped[:max_examples]:
            r = records[idx]
            print(f"    [{idx:3d}] orig: \"{r['query_original'][:80]}\"")
            print(f"          rew:  \"{r['query_rewrite'][:80]}\"")
            print(f"          cos(Q,P+)={r['cos_q_pos']:.3f}  cos(Rew,P+)~={r['cos_q_rewrite']:.3f}  tier={r.get('gap_type','?')}")

    if hurt:
        print(f"\n  ▸ Rewrite HURT (showing up to {min(max_examples, len(hurt))}):")
        for idx in hurt[:max_examples]:
            r = records[idx]
            print(f"    [{idx:3d}] orig: \"{r['query_original'][:80]}\"")
            print(f"          rew:  \"{r['query_rewrite'][:80]}\"")
            print(f"          cos(Q,P+)={r['cos_q_pos']:.3f}  cos(Rew,P+)~={r['cos_q_rewrite']:.3f}  tier={r.get('gap_type','?')}")

    if both_miss:
        print(f"\n  ▸ BOTH MISSED (showing up to {min(max_examples, len(both_miss))}):")
        for idx in both_miss[:max_examples]:
            r = records[idx]
            print(f"    [{idx:3d}] orig: \"{r['query_original'][:80]}\"")
            print(f"          rew:  \"{r['query_rewrite'][:80]}\"")
            print(f"          cos(Q,P+)={r['cos_q_pos']:.3f}  cos(Rew,P+)~={r['cos_q_rewrite']:.3f}  tier={r.get('gap_type','?')}")

    # ── RQ4: Retrieval Confidence Gating ──────────────────────
    # Gate logic: if top-1 retrieval score for original query >= τ,
    # the original is already good — skip rewrite; otherwise use rewrite.
    print("\n\n  ══ RQ4 — Retrieval Confidence Gating (Plan A) ══")
    print("  Gate: cos(orig_query, top1_doc) ≥ τ → keep original; < τ → use rewrite\n")

    thresholds = [round(t * 0.05 + 0.50, 2) for t in range(10)]  # 0.50 .. 0.95

    gate_rows: List[Tuple[str, ...]] = [
        ("τ", "N_kept", "N_rewrite", "Recall@1", "MRR", "Helped", "Hurt", "Net", "vs_ungated"),
    ]

    best_tau = 0.0
    best_recall = 0.0

    for tau in thresholds:
        # Build gated embedding: original if confident, rewrite otherwise
        gated_emb = np.empty_like(orig_emb)
        n_kept = 0
        for i in range(len(records)):
            if orig_scores[i, 0] >= tau:
                gated_emb[i] = orig_emb[i]
                n_kept += 1
            else:
                gated_emb[i] = rewrite_emb[i]

        n_rewrite = len(records) - n_kept

        # Search with gated queries
        _, gated_indices = search(index, gated_emb, top_k)

        # Metrics
        g_recall1 = recall_at_k(gated_indices, gold_pos_ids, 1)
        g_mrr = mrr(gated_indices, gold_pos_ids, top_k)

        # Per-sample analysis vs original
        g_helped = 0
        g_hurt = 0
        for i in range(len(records)):
            o_hit = gold_pos_ids[i] in orig_indices[i, :1]
            g_hit = gold_pos_ids[i] in gated_indices[i, :1]
            if not o_hit and g_hit:
                g_helped += 1
            elif o_hit and not g_hit:
                g_hurt += 1

        net = g_helped - g_hurt
        vs_ungated = g_recall1 - r1_rew  # vs ungated rewrite

        gate_rows.append((
            f"{tau:.2f}",
            str(n_kept), str(n_rewrite),
            f"{g_recall1:.4f}", f"{g_mrr:.4f}",
            str(g_helped), str(g_hurt),
            f"{net:+d}", f"{vs_ungated:+.4f}",
        ))

        if g_recall1 > best_recall:
            best_recall = g_recall1
            best_tau = tau

    print_table("Threshold Sweep — Gated Rewrite", gate_rows)

    print(f"\n  Best τ = {best_tau:.2f}  →  Recall@1 = {best_recall:.4f}"
          f"  (orig={r1_orig:.4f}, ungated_rew={r1_rew:.4f})")
    gain_over_orig = best_recall - r1_orig
    gain_over_rew = best_recall - r1_rew
    print(f"  Gain over original:        {gain_over_orig:+.4f}")
    print(f"  Gain over ungated rewrite: {gain_over_rew:+.4f}")

    results["gate_best_tau"] = best_tau
    results["gate_best_recall1"] = round(best_recall, 4)
    results["gate_gain_over_orig"] = round(gain_over_orig, 4)
    results["gate_gain_over_ungated"] = round(gain_over_rew, 4)

    # ── Detail for best τ ─────────────────────────────────────
    gated_emb_best = np.empty_like(orig_emb)
    gate_decisions: List[str] = []  # "keep" or "rewrite"
    for i in range(len(records)):
        if orig_scores[i, 0] >= best_tau:
            gated_emb_best[i] = orig_emb[i]
            gate_decisions.append("keep")
        else:
            gated_emb_best[i] = rewrite_emb[i]
            gate_decisions.append("rewrite")

    _, gated_best_indices = search(index, gated_emb_best, top_k)

    # Show examples where gating fixed a "hurt" case
    gating_saved: List[int] = []   # was hurt by rewrite, gating kept original → still hit
    gating_missed: List[int] = []  # was helped by rewrite, but gating refused → still miss

    for i in range(len(records)):
        o_hit = gold_pos_ids[i] in orig_indices[i, :1]
        r_hit = gold_pos_ids[i] in rew_indices[i, :1]
        g_hit = gold_pos_ids[i] in gated_best_indices[i, :1]
        if o_hit and not r_hit and g_hit:
            gating_saved.append(i)
        if not o_hit and r_hit and not g_hit:
            gating_missed.append(i)

    print(f"\n  At τ={best_tau:.2f}:")
    print(f"    Gating SAVED (hurt→hit):  {len(gating_saved)}")
    print(f"    Gating LOST  (helped→miss): {len(gating_missed)}")

    if gating_saved:
        print(f"\n  ▸ Gating SAVED — would have been hurt, gate kept original (up to 5):")
        for idx in gating_saved[:5]:
            r = records[idx]
            print(f"    [{idx:3d}] \"{r['query_original'][:70]}\"")
            print(f"          top1_score={orig_scores[idx,0]:.3f}  cos(Q,P+)={r['cos_q_pos']:.3f}")

    if gating_missed:
        print(f"\n  ▸ Gating LOST — would have been helped, gate blocked rewrite (up to 5):")
        for idx in gating_missed[:5]:
            r = records[idx]
            print(f"    [{idx:3d}] \"{r['query_original'][:70]}\"")
            print(f"          top1_score={orig_scores[idx,0]:.3f}  cos(Q,P+)={r['cos_q_pos']:.3f}")

    print()

    # Save metrics to JSON
    metrics_path = str(Path("evaluation/result") / "eval_metrics.json")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved → {metrics_path}\n")

    return results


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MS MARCO intent dataset")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    evaluate(
        dataset_path=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
