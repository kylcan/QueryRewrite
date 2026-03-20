"""
Rejection Sampling Fine-Tuning (RFT) Data Builder
==================================================

Implements the Rejection Sampling (RS) paradigm from:
  "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (Yuan et al., 2023)

Applied to query rewriting:
  1. Use the current SFT model to generate N candidate rewrites per query
  2. Score each candidate with a retrieval reward (cosine similarity to pos_doc)
  3. Keep only the top-K candidates as SFT training targets
  4. Discard candidates below a quality threshold

This creates a **data flywheel**: the trained model generates better candidates,
which become better training data for the next iteration.

Why this matters for 7B training:
  - Rejection sampling ensures the model trains on *achievable* high-quality outputs
  - Eliminates the "train on mediocre API outputs" noise in vanilla SFT
  - Used in LLaMA-2's chat training and Qwen's instruction tuning

Usage::

    python data/MS_MARCO/rejection_sampling.py --sft_checkpoint checkpoints/sft/best
    python data/MS_MARCO/rejection_sampling.py --n_candidates 8 --top_k 2
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data.MS_MARCO.build_preference_data import format_sft_prompt


def _compute_cosine_reward(
    candidate: str,
    pos_doc: str,
    embedder: Any,
) -> float:
    """Compute cos(candidate_emb, pos_doc_emb) as reward."""
    import torch
    with torch.no_grad():
        embs = embedder.encode([candidate, pos_doc], batch_size=2)
    e = embs.numpy()
    return float(np.dot(e[0], e[1]))


def build_rejection_sampled_dataset(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    sft_checkpoint: str = "checkpoints/sft/best",
    output_path: str = "data/MS_MARCO/sft_dataset_rs.jsonl",
    embedder_checkpoint: Optional[str] = "checkpoints/alignment/best",
    base_model: str = "Qwen/Qwen2.5-0.5B",
    n_candidates: int = 8,
    top_k: int = 2,
    min_reward: float = 0.0,
    temperature: float = 0.8,
    max_new_tokens: int = 64,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Generate N candidate rewrites per query, keep top-K by retrieval reward.

    Parameters
    ----------
    n_candidates : int
        Number of stochastic samples per query.
    top_k : int
        Number of best candidates to keep as SFT targets.
    min_reward : float
        Minimum cosine reward to include a candidate. Filters low-quality outputs.
    temperature : float
        Sampling temperature (>0 for diversity).

    Returns
    -------
    list of {prompt, completion, query_id, reward, rank}
    """
    import torch
    from models.rewriter import QueryRewriter
    from models.embedder import EmbeddingModel

    # Select device
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load data
    records: List[Dict[str, Any]] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} queries")

    # Load SFT model
    print(f"  Loading rewriter from {sft_checkpoint} …")
    rewriter = QueryRewriter(
        base_model_name=base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    if Path(sft_checkpoint).exists():
        rewriter.load_adapter(sft_checkpoint)
    else:
        print(f"  ⚠ No checkpoint found, using base model")
    rewriter.model.to(device)
    rewriter.model.eval()

    # Load embedder for reward
    print(f"  Loading embedder …")
    embedder = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if embedder_checkpoint and Path(embedder_checkpoint + "/checkpoint.pt").exists():
        ckpt = torch.load(
            embedder_checkpoint + "/checkpoint.pt",
            map_location="cpu", weights_only=True,
        )
        embedder.load_state_dict(ckpt["model_state_dict"])
    embedder.eval()

    # Pre-encode all pos_docs (batch)
    print(f"  Encoding {len(records)} positive documents …")
    pos_docs = [r["pos_doc"] for r in records]
    with torch.no_grad():
        pos_embs = embedder.encode(pos_docs, batch_size=64).numpy()  # (N, d)

    results: List[Dict[str, Any]] = []
    reward_stats: List[float] = []
    kept = 0
    discarded = 0

    print(f"\n  Generating {n_candidates} candidates per query, keeping top {top_k} …")

    for i, r in enumerate(records):
        query = r["query_original"]
        prompt = format_sft_prompt(query)
        pos_emb = pos_embs[i]

        # Generate N candidates with sampling
        candidates: List[str] = []
        # Generate in one batch (n_candidates copies of same prompt)
        prompts_batch = [prompt] * n_candidates
        encoded = rewriter.tokenizer(
            prompts_batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = rewriter.model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=rewriter.tokenizer.pad_token_id,
            )

        for j in range(output_ids.shape[0]):
            gen_ids = output_ids[j, prompt_len:]
            raw = rewriter.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = (raw if isinstance(raw, str) else raw[0]).strip()
            if text and text != query:  # skip empty or exact copies
                candidates.append(text)

        if not candidates:
            discarded += 1
            continue

        # Score all candidates
        with torch.no_grad():
            cand_embs = embedder.encode(candidates, batch_size=len(candidates)).numpy()

        rewards = [float(np.dot(cand_embs[ci], pos_emb)) for ci in range(len(candidates))]

        # Sort by reward descending
        scored = sorted(
            zip(candidates, rewards), key=lambda x: x[1], reverse=True
        )

        # Keep top-k above min_reward
        added = 0
        for rank, (cand, reward) in enumerate(scored[:top_k]):
            if reward < min_reward:
                break
            results.append({
                "query_id": r.get("query_id", i),
                "prompt": prompt,
                "completion": cand,
                "reward": round(reward, 4),
                "rank": rank + 1,
                "source": "rejection_sampling",
            })
            reward_stats.append(reward)
            added += 1
            kept += 1
        if added == 0:
            discarded += 1

        if (i + 1) % 100 == 0:
            avg_r = np.mean(reward_stats[-100:]) if reward_stats else 0
            print(f"  [{i+1}/{len(records)}] kept={kept}  avg_reward={avg_r:.4f}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  ── Rejection Sampling Summary ──")
    print(f"  Queries processed:  {len(records)}")
    print(f"  Pairs kept:         {kept}")
    print(f"  Queries discarded:  {discarded}")
    if reward_stats:
        print(f"  Reward mean:        {np.mean(reward_stats):.4f}")
        print(f"  Reward std:         {np.std(reward_stats):.4f}")
        print(f"  Reward > 0.8:       {sum(1 for r in reward_stats if r > 0.8)}")
    print(f"\n  Saved {len(results)} RS samples to {output_path}")
    return results


def merge_with_original_sft(
    original_sft_path: str = "data/MS_MARCO/sft_dataset.jsonl",
    rs_sft_path: str = "data/MS_MARCO/sft_dataset_rs.jsonl",
    output_path: str = "data/MS_MARCO/sft_dataset_merged.jsonl",
) -> int:
    """Merge original SFT data with rejection-sampled data."""
    all_records: List[Dict[str, str]] = []
    for path in [original_sft_path, rs_sft_path]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                all_records.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Merged {len(all_records)} samples → {output_path}")
    return len(all_records)


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Rejection Sampling SFT data builder")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft/best")
    parser.add_argument("--embedder_checkpoint", default="checkpoints/alignment/best")
    parser.add_argument("--output", default="data/MS_MARCO/sft_dataset_rs.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Candidates to generate per query")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Best candidates to keep per query")
    parser.add_argument("--min_reward", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging with original SFT data")
    args = parser.parse_args()

    build_rejection_sampled_dataset(
        dataset_path=args.dataset,
        sft_checkpoint=args.sft_checkpoint,
        output_path=args.output,
        embedder_checkpoint=args.embedder_checkpoint,
        base_model=args.model,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        min_reward=args.min_reward,
        temperature=args.temperature,
    )

    if not args.no_merge:
        merge_with_original_sft(output_path=args.output)


if __name__ == "__main__":
    main()
