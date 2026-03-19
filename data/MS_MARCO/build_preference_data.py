"""
Preference Dataset Builder for SFT + DPO Training
===================================================

Constructs training data for the query rewriter model from the existing
intent_dataset.jsonl and eval_rewrite results.

Three output formats:
  1. **SFT data** — (prompt, completion) pairs using the LLM rewrites as targets
  2. **DPO preference pairs** — (prompt, chosen, rejected) based on retrieval
     signal: rewrites that improve retrieval are "chosen", those that hurt are
     "rejected"
  3. **Reward-labeled data** — each rewrite annotated with a retrieval reward
     score for analysis

The retrieval signal comes from comparing cos(rewrite, pos_doc) vs
cos(original, pos_doc) and whether the rewrite improved Recall@1.

Usage::

    python data/MS_MARCO/build_preference_data.py
    python data/MS_MARCO/build_preference_data.py --dataset data/MS_MARCO/intent_dataset.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ── Prompt template ────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a search query optimizer. Given a user's search query, "
    "rewrite it to better match relevant encyclopedia-style passages. "
    "Output ONLY the rewritten query, nothing else."
)

def format_sft_prompt(query: str) -> str:
    """Build the instruction prompt for SFT/DPO training."""
    return f"Rewrite the following search query to improve retrieval.\nQuery: {query}\nRewrite:"


# ── Retrieval reward computation ───────────────────────────────
def _compute_retrieval_rewards(
    records: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Compute retrieval-based reward signals for each sample.

    Returns enriched records with:
      - reward_cos_delta: cos(rewrite, pos) - cos(orig, pos)
      - reward_recall: 1 if rewrite hits top-1, -1 if miss, 0 if same as orig
      - reward_combined: weighted combination
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from models.embedder import EmbeddingModel
    import torch

    model = EmbeddingModel(model_name=model_name)
    model.eval()

    # Collect all texts
    all_orig = [r["query_original"] for r in records]
    all_rew = [r["query_rewrite"] for r in records]
    all_pos = [r["pos_doc"] for r in records]

    # Encode
    print("  Encoding queries and documents …")
    with torch.no_grad():
        orig_emb = model.encode(all_orig, batch_size=batch_size).numpy()
        rew_emb = model.encode(all_rew, batch_size=batch_size).numpy()
        pos_emb = model.encode(all_pos, batch_size=batch_size).numpy()

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute per-sample rewards
    enriched = []
    for i, r in enumerate(records):
        cos_orig_pos = float(np.dot(orig_emb[i], pos_emb[i]))
        cos_rew_pos = float(np.dot(rew_emb[i], pos_emb[i]))

        delta = cos_rew_pos - cos_orig_pos
        # Reward: positive if rewrite is closer to pos_doc
        reward_cos = round(delta, 6)
        # Binary signal
        reward_binary = 1 if delta > 0.01 else (-1 if delta < -0.01 else 0)

        enriched.append({
            **r,
            "cos_orig_pos": round(cos_orig_pos, 4),
            "cos_rew_pos": round(cos_rew_pos, 4),
            "reward_cos_delta": reward_cos,
            "reward_binary": reward_binary,
        })

    return enriched


# ── Dataset builders ───────────────────────────────────────────
def build_sft_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build SFT pairs: every LLM rewrite is treated as a valid target.

    Returns list of {prompt, completion, query_id}.
    """
    sft_data = []
    for r in records:
        sft_data.append({
            "query_id": r["query_id"],
            "prompt": format_sft_prompt(r["query_original"]),
            "completion": r["query_rewrite"],
        })
    return sft_data


def build_dpo_dataset(
    enriched: List[Dict[str, Any]],
    min_reward_gap: float = 0.01,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Build DPO preference pairs from retrieval reward signals.

    Strategy:
      - If rewrite improved retrieval (reward > threshold):
          chosen = rewrite, rejected = original
      - If rewrite hurt retrieval (reward < -threshold):
          chosen = original, rejected = rewrite
      - Neutral samples are discarded (no clear preference signal)

    Parameters
    ----------
    enriched : list
        Records with reward_cos_delta field.
    min_reward_gap : float
        Minimum |reward| to include a sample.

    Returns
    -------
    dpo_data : list
        Each item has {prompt, chosen, rejected, query_id, reward_delta}.
    stats : dict
        Counts of chosen_rewrite, chosen_original, skipped.
    """
    dpo_data = []
    stats = {"chosen_rewrite": 0, "chosen_original": 0, "skipped": 0}

    for r in enriched:
        delta = r["reward_cos_delta"]
        prompt = format_sft_prompt(r["query_original"])

        if delta > min_reward_gap:
            # Rewrite is better → chosen=rewrite, rejected=original
            dpo_data.append({
                "query_id": r["query_id"],
                "prompt": prompt,
                "chosen": r["query_rewrite"],
                "rejected": r["query_original"],
                "reward_delta": round(delta, 4),
            })
            stats["chosen_rewrite"] += 1
        elif delta < -min_reward_gap:
            # Original is better → chosen=original, rejected=rewrite
            dpo_data.append({
                "query_id": r["query_id"],
                "prompt": prompt,
                "chosen": r["query_original"],
                "rejected": r["query_rewrite"],
                "reward_delta": round(abs(delta), 4),
            })
            stats["chosen_original"] += 1
        else:
            stats["skipped"] += 1

    return dpo_data, stats


# ── I/O ────────────────────────────────────────────────────────
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _save_jsonl(data: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → Saved {len(data)} samples to {path}")


# ── Main ───────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build preference dataset for SFT/DPO")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--output_dir", default="data/MS_MARCO")
    parser.add_argument("--min_gap", type=float, default=0.01,
                        help="Min cosine delta to include in DPO pairs")
    args = parser.parse_args()

    print("=" * 60)
    print("  Building Preference Dataset for SFT + DPO")
    print("=" * 60)

    # Load
    records = _load_jsonl(args.dataset)
    print(f"\n  Loaded {len(records)} records from {args.dataset}")

    # Compute retrieval rewards
    print("\n  Computing retrieval reward signals …")
    enriched = _compute_retrieval_rewards(records)

    # Build SFT dataset
    print("\n  Building SFT dataset …")
    sft_data = build_sft_dataset(records)
    _save_jsonl(sft_data, f"{args.output_dir}/sft_dataset.jsonl")

    # Build DPO dataset
    print("\n  Building DPO preference pairs …")
    dpo_data, dpo_stats = build_dpo_dataset(enriched, min_reward_gap=args.min_gap)
    _save_jsonl(dpo_data, f"{args.output_dir}/dpo_dataset.jsonl")

    # Save reward-annotated data for analysis
    _save_jsonl(enriched, f"{args.output_dir}/reward_annotated.jsonl")

    # Summary
    pos_count = sum(1 for r in enriched if r["reward_cos_delta"] > 0)
    neg_count = sum(1 for r in enriched if r["reward_cos_delta"] < 0)
    avg_delta = np.mean([r["reward_cos_delta"] for r in enriched])

    print(f"\n  ── Summary ──")
    print(f"  Total samples:        {len(records)}")
    print(f"  Rewrite improved:     {pos_count} ({pos_count/len(records)*100:.1f}%)")
    print(f"  Rewrite hurt:         {neg_count} ({neg_count/len(records)*100:.1f}%)")
    print(f"  Avg cosine delta:     {avg_delta:+.4f}")
    print(f"  DPO pairs:            {len(dpo_data)}")
    print(f"    chosen=rewrite:     {dpo_stats['chosen_rewrite']}")
    print(f"    chosen=original:    {dpo_stats['chosen_original']}")
    print(f"    skipped (neutral):  {dpo_stats['skipped']}")


if __name__ == "__main__":
    main()
