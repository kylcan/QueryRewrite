"""Iterative DPO (Direct Preference Optimization) trainer.

Implements self-play iterative DPO: the policy is used to generate fresh
on-policy candidates every round, which are scored by a reward function,
turned into new preference pairs, and used for another round of DPO.

Algorithm (Xu et al., 2023; Yuan et al., 2024 — "RAFT / Self-Play Fine-Tuning"):

    for round r = 1 … R:
        1. Generate N candidate rewrites per query using the current policy
           (temperature > 0 for diversity).
        2. Score each candidate with reward_fn(query, rewrite).
        3. For each query, pair the highest-scored candidate (chosen) against
           the lowest-scored candidate (rejected), filtering pairs whose
           reward gap is below a minimum threshold.
        4. Combine new pairs with a rolling data buffer (optional).
        5. Run DPO for `steps_per_round` gradient steps on the new dataset.
        6. Save the round checkpoint; continue to round r+1.

The reference model is the SFT checkpoint and stays frozen for all rounds
(static-reference variant).  This keeps DPO training stable and means a
single reference model is shared across rounds.

Usage::

    from training.iterative_dpo_trainer import IterativeDPOTrainer, cosine_reward_fn

    trainer = IterativeDPOTrainer(
        model=rewriter,
        ref_model=frozen_sft_model,
        reward_fn=cosine_reward_fn(embedder, corpus_embs, pos_doc_ids),
        raw_queries=[{"query": "...", "prompt": "..."}, ...],
        seed_dpo_records=[{"prompt": ..., "chosen": ..., "rejected": ...}, ...],
        n_rounds=3,
        steps_per_round=200,
    )
    trainer.train()
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from training.dpo_trainer import DPODataset, DPOTrainer


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PreferencePair = Dict[str, Any]  # {prompt, chosen, rejected, [chosen_reward, ...]}  


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def cosine_reward_fn(
    embedder: Any,
    corpus_embeddings: np.ndarray,
    pos_doc_indices: List[int],
    device: str = "cpu",
) -> Callable[[List[str], List[str], List[int]], List[float]]:
    """Factory that returns a reward function using cosine similarity.

    The reward for a rewritten query is the cosine similarity between its
    embedding and the embedding of the ground-truth positive document.

    Parameters
    ----------
    embedder : Embedder
        A sentence-level embedder (``models/embedder.py``).
    corpus_embeddings : np.ndarray
        Pre-computed corpus embeddings, shape ``[N_docs, D]``.
    pos_doc_indices : List[int]
        Ground-truth document index for each training query.
    device : str
        Device to run the embedder on.

    Returns
    -------
    Callable[[queries, rewrites, sample_indices], rewards]
        ``queries``       – original (pre-rewrite) queries (not used here, kept
                            for API symmetry with LLM-judge variant).
        ``rewrites``      – rewritten queries to score.
        ``sample_indices``– index into ``pos_doc_indices`` for each sample.
    """
    # Normalise corpus once
    norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_norm = corpus_embeddings / np.maximum(norms, 1e-8)

    def _reward(
        queries: List[str],
        rewrites: List[str],
        sample_indices: List[int],
    ) -> List[float]:
        with torch.no_grad():
            rewrite_embs = embedder.encode(rewrites, device=device)  # [B, D]
        if isinstance(rewrite_embs, torch.Tensor):
            rewrite_embs = rewrite_embs.cpu().numpy()
        rewrite_norms = np.linalg.norm(rewrite_embs, axis=1, keepdims=True)
        rewrite_norm = rewrite_embs / np.maximum(rewrite_norms, 1e-8)

        rewards = []
        for i, idx in enumerate(sample_indices):
            pos_emb = corpus_norm[pos_doc_indices[idx]]      # [D]
            sim = float(rewrite_norm[i] @ pos_emb)
            rewards.append(sim)
        return rewards

    return _reward


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class IterativeDPOTrainer:
    """Iterative / on-policy DPO trainer.

    Parameters
    ----------
    model : Any (QueryRewriter)
        The policy model.  Must expose ``rewrite_queries_stochastic()`` or
        ``rewrite_queries()`` for candidate generation.
    ref_model : nn.Module
        Frozen reference model (static across all rounds).
    reward_fn : Callable
        Signature: ``(queries, rewrites, sample_indices) -> List[float]``.
        Higher is better.  Use ``cosine_reward_fn(...)`` for retrieval reward
        or wrap an LLM judge.
    raw_queries : List[Dict]
        List of dicts with at least ``"prompt"`` (SFT-style prompt) and
        ``"query"`` (original user query text) keys.
    seed_dpo_records : List[Dict]
        Initial preference pairs (``prompt``, ``chosen``, ``rejected``) used
        for round 0 warm-start.  May be empty.
    n_rounds : int
        Number of self-play rounds.
    steps_per_round : int
        Gradient steps per round (DPO).
    n_candidates : int
        Candidates to generate per query per round.
    temperature : float
        Sampling temperature for candidate generation.
    top_p : float
        Top-p for nucleus sampling.
    min_reward_gap : float
        Minimum ``reward(chosen) - reward(rejected)`` to include a pair.
    buffer_size : int | None
        Maximum number of preference pairs to keep in the rolling buffer
        (FIFO).  ``None`` = unlimited (keep all rounds' data).
    beta : float
        DPO inverse temperature.
    batch_size : int
        DPO batch size per round.
    lr : float
        Learning rate.
    gradient_accumulation_steps : int
        Gradient accumulation.
    max_grad_norm : float
        Gradient clipping norm.
    warmup_ratio : float
        Warmup fraction of steps_per_round.
    device : str
        Target device.
    checkpoint_dir : str
        Root directory for round checkpoints.
    deepspeed_config : str | None
        Optional path to a DeepSpeed config.
    val_ratio : float
        Fraction of preference pairs to withhold for validation each round.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Any,
        ref_model: torch.nn.Module,
        reward_fn: Callable[[List[str], List[str], List[int]], List[float]],
        raw_queries: List[Dict[str, str]],
        seed_dpo_records: List[Dict[str, str]],
        n_rounds: int = 3,
        steps_per_round: int = 200,
        n_candidates: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_reward_gap: float = 0.05,
        buffer_size: Optional[int] = None,
        beta: float = 0.1,
        batch_size: int = 2,
        lr: float = 5e-5,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.05,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints/iterative_dpo",
        deepspeed_config: Optional[str] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.raw_queries = raw_queries
        self.n_rounds = n_rounds
        self.steps_per_round = steps_per_round
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.top_p = top_p
        self.min_reward_gap = min_reward_gap
        self.buffer_size = buffer_size
        self.beta = beta
        self.batch_size = batch_size
        self.lr = lr
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.deepspeed_config = deepspeed_config
        self.val_ratio = val_ratio
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        # Rolling preference buffer — seeded with provided DPO data
        self._buffer: List[Dict[str, Any]] = list(seed_dpo_records)

        # Tokenizer lives on the model
        self.tokenizer = getattr(model, "tokenizer", None)

        # Query indices: maps position in raw_queries to an integer id used
        # by the reward function's pos_doc_indices lookup.
        self._query_indices = list(range(len(raw_queries)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """Run all self-play rounds and return per-round history."""
        all_history: List[Dict[str, float]] = []

        print(f"\n{'='*60}")
        print(f"  Iterative DPO  |  {self.n_rounds} rounds × {self.steps_per_round} steps")
        print(f"{'='*60}")

        for round_num in range(self.n_rounds):
            print(f"\n── Round {round_num + 1}/{self.n_rounds} ──────────────────────────────")

            # 1. Generate new on-policy preference pairs
            new_pairs = self._generate_preference_pairs(round_num)
            print(f"  New preference pairs generated: {len(new_pairs)}")

            # 2. Add to rolling buffer
            self._buffer.extend(new_pairs)
            if self.buffer_size is not None and len(self._buffer) > self.buffer_size:
                # Keep most recent pairs (FIFO)
                self._buffer = self._buffer[-self.buffer_size:]
            print(f"  Buffer size: {len(self._buffer)}")

            if len(self._buffer) < self.batch_size:
                print(f"  ⚠ Not enough data for a batch (need {self.batch_size}), skipping DPO.")
                continue

            # 3. DPO training on the current buffer
            round_history = self._run_dpo_round(round_num)
            all_history.append(round_history)

            # 4. Save round checkpoint
            ckpt_dir = self.checkpoint_dir / f"round_{round_num + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_fn = getattr(self.model, "save_adapter", None)
            if save_fn is not None:
                save_fn(str(ckpt_dir))
                print(f"  Checkpoint saved → {ckpt_dir}")

        print(f"\n{'='*60}")
        print(f"  Iterative DPO complete.")
        return all_history

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _generate_preference_pairs(
        self, round_num: int
    ) -> List[Dict[str, Any]]:
        """Generate N stochastic rewrites per query and create preference pairs.

        For each query:
          - Generate ``n_candidates`` rewrites with temperature sampling.
          - Score all candidates with ``reward_fn``.
          - Pair ``argmax`` (chosen) vs ``argmin`` (rejected) if reward gap
            exceeds ``min_reward_gap``.
        """
        self.model.eval()
        pairs: List[Dict[str, Any]] = []

        # Shuffle query order each round for variety
        indices = list(range(len(self.raw_queries)))
        random.shuffle(indices)

        queries = [self.raw_queries[i]["query"] for i in indices]
        prompts = [self.raw_queries[i]["prompt"] for i in indices]

        print(f"  Generating {self.n_candidates} candidates for {len(queries)} queries…", flush=True)

        # Generate candidates in a loop (n_candidates passes)
        all_candidates: List[List[str]] = [[] for _ in queries]

        for cand_idx in range(self.n_candidates):
            generated = self._generate_batch(queries, prompts)
            for qi, rewrite in enumerate(generated):
                all_candidates[qi].append(rewrite)

        # Score all candidates and create pairs
        for qi, (query, prompt) in enumerate(zip(queries, prompts)):
            candidates = all_candidates[qi]
            sample_idx = indices[qi]

            # Deduplicate (case-insensitive)
            seen: set = set()
            unique_candidates = []
            for c in candidates:
                key = c.strip().lower()
                if key not in seen and key:
                    seen.add(key)
                    unique_candidates.append(c)

            if len(unique_candidates) < 2:
                continue

            # Score unique candidates
            rewards = self.reward_fn(
                [query] * len(unique_candidates),
                unique_candidates,
                [sample_idx] * len(unique_candidates),
            )

            best_idx = int(np.argmax(rewards))
            worst_idx = int(np.argmin(rewards))
            if best_idx == worst_idx:
                continue

            reward_gap = rewards[best_idx] - rewards[worst_idx]
            if reward_gap < self.min_reward_gap:
                continue

            pairs.append({
                "prompt": prompt,
                "chosen": unique_candidates[best_idx],
                "rejected": unique_candidates[worst_idx],
                "chosen_reward": float(rewards[best_idx]),
                "rejected_reward": float(rewards[worst_idx]),
                "round": round_num + 1,
            })

        return pairs

    def _generate_batch(
        self,
        queries: List[str],
        prompts: List[str],
    ) -> List[str]:
        """Generate one rewrite per query using temperature sampling."""
        if self.tokenizer is None:
            raise RuntimeError("model must have a tokenizer attribute")

        # Use the model's built-in generation helper if available
        gen_fn = getattr(self.model, "rewrite_queries", None)
        if gen_fn is not None:
            # Pass generation kwargs as supported by the model API
            try:
                return gen_fn(
                    queries,
                    device=self.device,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )
            except TypeError:
                # Fallback: model.rewrite_queries() without extra kwargs
                return gen_fn(queries, device=self.device)

        # Manual generation (tokenize prompt → generate → decode response)
        return self._manual_generate(prompts)

    def _manual_generate(self, prompts: List[str]) -> List[str]:
        """Tokenize prompts and call model.generate() directly."""
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("model must have a tokenizer attribute")
        model = self.model
        results: List[str] = []

        for prompt in prompts:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            new_ids = out_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append(text)

        return results

    def _run_dpo_round(self, round_num: int) -> Dict[str, float]:
        """Train the policy for ``steps_per_round`` steps on the current buffer."""
        # Train / val split
        data = list(self._buffer)
        random.shuffle(data)
        n_val = max(1, int(len(data) * self.val_ratio))
        val_data = data[:n_val]
        train_data = data[n_val:]

        tokenizer = self.tokenizer

        # Determine max_length from buffer (short since queries are short)
        max_length = 256

        train_ds = DPODataset(train_data, tokenizer, max_length=max_length)
        val_ds = DPODataset(val_data, tokenizer, max_length=max_length) if val_data else None

        # Use a per-round subdirectory so each round's state is independent
        round_ckpt = str(self.checkpoint_dir / "dpo_working")

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            beta=self.beta,
            epochs=1,             # epochs=1 with max_steps limits to steps_per_round
            batch_size=self.batch_size,
            lr=self.lr,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.grad_accum,
            max_grad_norm=self.max_grad_norm,
            device=self.device,
            checkpoint_dir=round_ckpt,
            deepspeed_config=self.deepspeed_config,
            max_steps=self.steps_per_round,
        )

        print(f"  DPO: {len(train_data)} train pairs | {len(val_data)} val pairs")
        history = trainer.train()
        return {f"round_{round_num + 1}_{k}": v for k, v in history.items()}
