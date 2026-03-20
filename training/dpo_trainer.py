"""DPO (Direct Preference Optimization) trainer for causal language models.

Trains a QueryRewriter using preference pairs (prompt, chosen, rejected)
with a frozen reference model, implementing Rafailov et al. (2023).

Architecture::

    ┌──────────────────────────────────────────────┐
    │ Policy model (trainable) → log π(y|x)       │
    │ Reference model (frozen)  → log π_ref(y|x)  │
    │                                              │
    │ L = -E[log σ(β · (Δ_chosen - Δ_rejected))]  │
    │     where Δ = log π(y|x) - log π_ref(y|x)   │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from models.losses import DPOLoss


class DPODataset(Dataset):
    """Dataset for DPO: (prompt, chosen, rejected) preference triplets.

    Each sample produces two tokenised sequences (chosen and rejected),
    both sharing the same prompt prefix. A ``response_mask`` indicates
    which tokens belong to the response (for log-prob computation).
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(
            response, add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]

        total = len(prompt_ids) + len(resp_ids)
        if total > self.max_length:
            resp_ids = resp_ids[: self.max_length - len(prompt_ids)]

        input_ids = prompt_ids + resp_ids
        attention_mask = [1] * len(input_ids)
        response_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        r = self.records[idx]
        chosen = self._tokenize_pair(r["prompt"], r["chosen"])
        rejected = self._tokenize_pair(r["prompt"], r["rejected"])
        return {"chosen": chosen, "rejected": rejected}


def _pad_sequence(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Left-pad a list of 1-D tensors to the same length."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        padded.append(
            torch.cat([torch.full((pad_len,), pad_value, dtype=t.dtype), t])
        )
    return torch.stack(padded)


def dpo_collate_fn(batch: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collate DPO samples, padding chosen and rejected separately."""
    result: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in ("chosen", "rejected"):
        items = [b[key] for b in batch]
        result[key] = {
            "input_ids": _pad_sequence([x["input_ids"] for x in items], 0),
            "attention_mask": _pad_sequence([x["attention_mask"] for x in items], 0),
            "response_mask": _pad_sequence([x["response_mask"] for x in items], 0),
        }
    return result


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Parameters
    ----------
    model : nn.Module
        The trainable policy model (QueryRewriter with LoRA).
    ref_model : nn.Module
        Frozen reference model (deepcopy of policy before DPO).
    train_dataset : DPODataset
        Training preference pairs.
    val_dataset : DPODataset | None
        Validation data.
    beta : float
        DPO inverse temperature.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    warmup_ratio : float
        Fraction of steps for LR warmup.
    weight_decay : float
        AdamW weight decay.
    max_grad_norm : float
        Gradient clipping norm.
    gradient_accumulation_steps : int
        Accumulate gradients over this many steps (ignored when using DeepSpeed).
    device : str
        Target device.
    checkpoint_dir : str
        Where to save checkpoints.
    label_smoothing : float
        DPO label smoothing (0 = standard DPO).
    deepspeed_config : str | None
        Path to a DeepSpeed JSON config.  Only the policy model is wrapped;
        the reference model stays as a plain frozen module.
    max_steps : int | None
        Stop training after this many gradient steps (across all epochs).
        Useful for iterative DPO where each round trains for a fixed budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        train_dataset: DPODataset,
        val_dataset: Optional[DPODataset] = None,
        beta: float = 0.1,
        epochs: int = 1,
        batch_size: int = 2,
        lr: float = 5e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints/dpo",
        label_smoothing: float = 0.0,
        deepspeed_config: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.grad_accum = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_steps = max_steps
        self.use_deepspeed = False
        self.engine = None

        # Models
        self.model = model
        self.ref_model = ref_model

        # Device placement (DeepSpeed handles it for the policy)
        if deepspeed_config is None:
            if hasattr(model, "model"):
                model.model.to(device)
            else:
                model.to(device)
        ref_model.to(device)

        # Loss
        self.dpo_loss = DPOLoss(beta=beta, label_smoothing=label_smoothing)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dpo_collate_fn,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dpo_collate_fn,
            )

        if deepspeed_config is not None:
            # ── DeepSpeed path ───────────────────────────────────────────────
            # Only the trainable policy is wrapped; the frozen ref model is
            # kept as a plain module on the same device.
            import json
            import deepspeed as _ds  # type: ignore[import]

            with open(deepspeed_config) as _f:
                ds_cfg = json.load(_f)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.engine, self.optimizer, _, self.scheduler = _ds.initialize(
                model=model,
                model_parameters=trainable_params,
                config=ds_cfg,
            )
            self.use_deepspeed = True
            self.device = self.engine.device
            # Also move ref model to the same device
            ref_model.to(self.device)
        else:
            # ── Standard PyTorch path ─────────────────────────────────────────
            self.optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            )
            from training.scheduler import build_scheduler
            num_steps = len(self.train_loader) * epochs // self.grad_accum
            self.scheduler = build_scheduler(
                self.optimizer, num_steps,
                warmup_ratio=warmup_ratio,
                scheduler_type="cosine",
            )

        self.best_val_loss = float("inf")
        self._global_step: int = 0

    def _get_logps(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sequence log-probs for a batch.

        ``model`` may be the raw policy model, the DeepSpeed engine that wraps
        it, or the frozen reference model — all support the same forward call.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        response_mask = batch["response_mask"].to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return DPOLoss.compute_logps(logits, input_ids, response_mask)

    def train(self) -> Dict[str, float]:
        """Run full DPO training loop."""
        print(f"\n  Starting DPO for {self.epochs} epochs (β={self.dpo_loss.beta}) …")
        history: Dict[str, float] = {}

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch()
            print(
                f"  Epoch {epoch+1}/{self.epochs}"
                f"  dpo_loss={train_loss:.4f}  acc={train_acc:.3f}",
                end="",
            )

            if self.val_loader is not None:
                val_loss, val_acc = self._validate()
                print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}", end="")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save(epoch, "best")
                    print("  ★ saved best", end="")
            else:
                self._save(epoch, "best")

            print()
            history[f"epoch_{epoch+1}_loss"] = train_loss
            history[f"epoch_{epoch+1}_acc"] = train_acc

        return history

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0
        global_step = getattr(self, "_global_step", 0)

        # Determine which model handle to use for policy forwards
        policy: Any = self.engine if self.use_deepspeed else self.model

        if self.use_deepspeed:
            # ── DeepSpeed training loop ───────────────────────────────────────
            for batch in self.train_loader:
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                policy_chosen_logps = self._get_logps(policy, batch["chosen"])
                policy_rejected_logps = self._get_logps(policy, batch["rejected"])

                with torch.no_grad():
                    ref_chosen_logps = self._get_logps(self.ref_model, batch["chosen"])
                    ref_rejected_logps = self._get_logps(self.ref_model, batch["rejected"])

                loss = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                )
                self.engine.backward(loss)
                self.engine.step()

                with torch.no_grad():
                    chosen_rewards = policy_chosen_logps - ref_chosen_logps
                    rejected_rewards = policy_rejected_logps - ref_rejected_logps
                    acc = (chosen_rewards > rejected_rewards).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                n_steps += 1
                global_step += 1
        else:
            # ── Standard PyTorch training loop ───────────────────────────────
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader):
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                policy_chosen_logps = self._get_logps(policy, batch["chosen"])
                policy_rejected_logps = self._get_logps(policy, batch["rejected"])

                with torch.no_grad():
                    ref_chosen_logps = self._get_logps(self.ref_model, batch["chosen"])
                    ref_rejected_logps = self._get_logps(self.ref_model, batch["rejected"])

                loss = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                ) / self.grad_accum
                loss.backward()

                with torch.no_grad():
                    chosen_rewards = policy_chosen_logps - ref_chosen_logps
                    rejected_rewards = policy_rejected_logps - ref_rejected_logps
                    acc = (chosen_rewards > rejected_rewards).float().mean().item()

                total_loss += loss.item() * self.grad_accum
                total_acc += acc
                n_steps += 1
                global_step += 1

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            if n_steps % self.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        self._global_step = global_step
        return total_loss / max(n_steps, 1), total_acc / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        # Always use the underlying model (not the engine) for validation;
        # safe for ZeRO-2 where weights are replicated.
        policy = self.model

        for batch in self.val_loader:  # type: ignore
            policy_chosen_logps = self._get_logps(policy, batch["chosen"])
            policy_rejected_logps = self._get_logps(policy, batch["rejected"])
            ref_chosen_logps = self._get_logps(self.ref_model, batch["chosen"])
            ref_rejected_logps = self._get_logps(self.ref_model, batch["rejected"])

            loss = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            chosen_rewards = policy_chosen_logps - ref_chosen_logps
            rejected_rewards = policy_rejected_logps - ref_rejected_logps
            acc = (chosen_rewards > rejected_rewards).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n += 1

        return total_loss / max(n, 1), total_acc / max(n, 1)

    def _save(self, epoch: int, tag: str) -> None:
        path = self.checkpoint_dir / tag
        if self.use_deepspeed:
            # For ZeRO-2: weights replicated, adapter save works directly.
            # For ZeRO-3: use engine.save_checkpoint() and consolidate offline.
            save_fn = getattr(self.model, "save_adapter", None)
            if save_fn is not None:
                save_fn(str(path))
            else:
                self.engine.save_checkpoint(str(self.checkpoint_dir), tag=tag)
            return
        save_fn = getattr(self.model, "save_adapter", None)
        if save_fn is not None:
            save_fn(str(path))
        else:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                path / "checkpoint.pt",
            )
