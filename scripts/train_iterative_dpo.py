"""Train the query rewriter with Iterative (Self-Play) DPO.

Each round:
  1. The current policy generates N stochastic rewrites per query.
  2. Rewrites are scored by a retrieval-based cosine reward (or LLM judge).
  3. Best (chosen) vs worst (rejected) rewrites form new preference pairs.
  4. DPO runs for `--steps_per_round` gradient steps on the accumulated buffer.
  5. Round checkpoint saved; repeat for `--n_rounds` rounds.

The reference model is the SFT checkpoint, kept frozen throughout all rounds
(static-reference iterative DPO).

Usage::

    # Quick local run (CPU / MPS)
    python scripts/train_iterative_dpo.py \\
        --sft_checkpoint checkpoints/sft/best \\
        --embedder_checkpoint checkpoints/alignment/best \\
        --n_rounds 3 --steps_per_round 100

    # 4-GPU A100 run with DeepSpeed ZeRO-3
    deepspeed --num_gpus 4 scripts/train_iterative_dpo.py \\
        --sft_checkpoint checkpoints/sft/best \\
        --embedder_checkpoint checkpoints/alignment/best \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --n_rounds 3 --steps_per_round 500 \\
        --deepspeed configs/deepspeed_zero3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.rewriter import QueryRewriter
from models.embedder import EmbeddingModel
from training.iterative_dpo_trainer import IterativeDPOTrainer, cosine_reward_fn


def _load_jsonl(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _build_raw_queries(intent_records, rewriter: QueryRewriter) -> list:
    """Build list of {query, prompt} dicts from the intent dataset."""
    from data.MS_MARCO.build_preference_data import format_sft_prompt  # type: ignore
    raw = []
    for r in intent_records:
        query = (
            r.get("query")
            or r.get("query_original")
            or r.get("original_query")
            or ""
        )
        prompt = format_sft_prompt(query)
        raw.append({"query": query, "prompt": prompt})
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative (self-play) DPO training for query rewriter"
    )
    # Data
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl",
                        help="Intent dataset (queries + positive doc ids)")
    parser.add_argument("--seed_dpo_dataset", default="data/MS_MARCO/dpo_dataset.jsonl",
                        help="Seed DPO pairs to pre-populate the buffer (can be empty)")
    parser.add_argument("--corpus_file", default="data/MS_MARCO/corpus.jsonl",
                        help="Passage corpus (id, text)")
    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft/best")
    parser.add_argument("--embedder_checkpoint", default="checkpoints/alignment/best")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    # Iterative DPO hyper-params
    parser.add_argument("--n_rounds", type=int, default=3,
                        help="Number of self-play rounds")
    parser.add_argument("--steps_per_round", type=int, default=200,
                        help="DPO gradient steps per round")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Stochastic candidates to generate per query per round")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_reward_gap", type=float, default=0.05,
                        help="Minimum reward delta to keep a preference pair")
    parser.add_argument("--buffer_size", type=int, default=None,
                        help="Max pairs in rolling buffer (None = unlimited)")
    # DPO optimiser
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    # Infrastructure
    parser.add_argument("--checkpoint_dir", default="checkpoints/iterative_dpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deepspeed", default=None, metavar="DS_CONFIG",
        help="Path to DeepSpeed JSON config.  Launch with: "
             "deepspeed --num_gpus N scripts/train_iterative_dpo.py --deepspeed ...",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if args.deepspeed:
        print(f"DeepSpeed config: {args.deepspeed}")

    # ── Load intent dataset ────────────────────────────────────────────────
    print(f"\n  Loading intent dataset from {args.dataset} …")
    intent_records = _load_jsonl(args.dataset)
    print(f"  {len(intent_records)} queries")

    # ── Load seed DPO data ─────────────────────────────────────────────────
    seed_dpo: list = []
    if Path(args.seed_dpo_dataset).exists():
        seed_dpo = _load_jsonl(args.seed_dpo_dataset)
        print(f"  Seed DPO pairs loaded: {len(seed_dpo)}")
    else:
        print(f"  ⚠ No seed DPO dataset found at {args.seed_dpo_dataset}, starting from scratch")

    # ── Load corpus & pre-compute embeddings ──────────────────────────────
    print(f"\n  Loading corpus from {args.corpus_file} …")
    corpus_records = _load_jsonl(args.corpus_file)
    corpus_texts = [r["text"] for r in corpus_records]
    print(f"  Corpus size: {len(corpus_texts)}")

    print(f"  Loading embedder from {args.embedder_checkpoint} …")
    embedder = EmbeddingModel()
    embedder_ckpt = Path(args.embedder_checkpoint)
    ckpt_file = embedder_ckpt / "checkpoint.pt"
    if ckpt_file.exists():
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        embedder.load_state_dict(state["model_state_dict"])
        print("  Fine-tuned embedder weights loaded.")
    else:
        print("  ⚠ Embedder checkpoint not found, using pre-trained weights.")
    embedder.to(device)
    embedder.eval()

    print("  Encoding corpus (this may take a few minutes) …")
    corpus_embeddings = embedder.encode(corpus_texts, device=device)
    if isinstance(corpus_embeddings, torch.Tensor):
        corpus_embeddings = corpus_embeddings.cpu().numpy()
    print(f"  Corpus embeddings: {corpus_embeddings.shape}")

    # Build pos_doc_indices: for each query record, find the index of its
    # positive passage in the corpus.
    corpus_id_to_idx = {str(r["id"]): i for i, r in enumerate(corpus_records)}
    corpus_text_to_idx = {r["text"]: i for i, r in enumerate(corpus_records)}
    pos_doc_indices: list[int] = []
    valid_records: list = []
    for r in intent_records:
        pos_id = r.get("positive_passage_id") or r.get("pos_id")
        if pos_id is not None and str(pos_id) in corpus_id_to_idx:
            pos_doc_indices.append(corpus_id_to_idx[str(pos_id)])
            valid_records.append(r)
            continue

        pos_doc = r.get("pos_doc")
        if isinstance(pos_doc, str) and pos_doc in corpus_text_to_idx:
            pos_doc_indices.append(corpus_text_to_idx[pos_doc])
            valid_records.append(r)
    print(f"  Queries with valid positive passages: {len(valid_records)}")

    if not valid_records:
        raise ValueError(
            "No valid positive passages found. Ensure the dataset contains "
            "positive_passage_id/pos_id or pos_doc text that matches corpus.jsonl."
        )

    # ── Load policy model ──────────────────────────────────────────────────
    print(f"\n  Loading model: {args.model}")
    model = QueryRewriter(
        base_model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    sft_path = Path(args.sft_checkpoint)
    if sft_path.exists():
        print(f"  Loading SFT adapter from {sft_path}")
        model.load_adapter(str(sft_path))
    else:
        print(f"  ⚠ SFT checkpoint not found at {sft_path}, using base model")

    # ── Create frozen reference model ─────────────────────────────────────
    print("  Creating frozen reference model …")
    ref_model = model.get_ref_model()

    # ── Build raw query list for generation ───────────────────────────────
    raw_queries = _build_raw_queries(valid_records, model)

    # ── Build reward function ──────────────────────────────────────────────
    reward_fn = cosine_reward_fn(embedder, corpus_embeddings, pos_doc_indices, device=device)

    # ── Launch iterative DPO ───────────────────────────────────────────────
    print(f"\n  Config:")
    print(f"    Rounds: {args.n_rounds}  steps/round: {args.steps_per_round}")
    print(f"    Candidates/query: {args.n_candidates}  temp: {args.temperature}")
    print(f"    β={args.beta}  lr={args.lr}  batch={args.batch_size}×{args.gradient_accumulation}")
    print(f"    Min reward gap: {args.min_reward_gap}")
    if args.buffer_size:
        print(f"    Buffer size (FIFO): {args.buffer_size}")

    trainer = IterativeDPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        raw_queries=raw_queries,
        seed_dpo_records=seed_dpo,
        n_rounds=args.n_rounds,
        steps_per_round=args.steps_per_round,
        n_candidates=args.n_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        min_reward_gap=args.min_reward_gap,
        buffer_size=args.buffer_size,
        beta=args.beta,
        batch_size=args.batch_size,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        deepspeed_config=args.deepspeed,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    all_history = trainer.train()

    # Save training history
    history_path = Path(args.checkpoint_dir) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n  History saved → {history_path}")

    # ── Quick inference test ───────────────────────────────────────────────
    print("\n  ── Quick inference test ──")
    test_queries = ["what is rba", "how to install python"]
    rewrites = model.rewrite_queries(test_queries, device=device)
    for q, r in zip(test_queries, rewrites):
        print(f"  Q: {q}")
        print(f"  R: {r}\n")


if __name__ == "__main__":
    main()
