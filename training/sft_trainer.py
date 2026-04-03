"""SFT (Supervised Fine-Tuning) trainer for causal language models.

Trains a QueryRewriter on (prompt, completion) pairs using standard
language modelling loss. Only the completion tokens contribute to the loss.

Architecture::

    ┌──────────────────┐
    │ prompt tokens    │  labels = -100 (ignored)
    ├──────────────────┤
    │ completion tokens│  labels = token_ids (LM loss)
    └──────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from training.deepspeed_utils import prepare_deepspeed_config


class SFTDataset(Dataset):
    """Dataset for SFT: (prompt, completion) pairs as causal LM sequences.

    Each sample is tokenised as ``[prompt] [completion] [eos]`` with labels
    set to ``-100`` for prompt tokens (so they don't contribute to loss).
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        prompt = r["prompt"]
        completion = r["completion"]

        # Tokenize prompt and completion separately to know boundary
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        comp_ids = self.tokenizer.encode(
            completion, add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]

        # Truncate if needed (prefer keeping prompt intact)
        total = len(prompt_ids) + len(comp_ids)
        if total > self.max_length:
            comp_ids = comp_ids[: self.max_length - len(prompt_ids)]

        input_ids = prompt_ids + comp_ids
        attention_mask = [1] * len(input_ids)

        # Labels: -100 for prompt, actual ids for completion
        labels = [-100] * len(prompt_ids) + comp_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of SFT samples to the same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), b["input_ids"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), b["attention_mask"]])
        )
        labels.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), b["labels"]])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


class SFTTrainer:
    """Supervised Fine-Tuning trainer for causal LM query rewriter.

    Parameters
    ----------
    model : nn.Module
        The QueryRewriter model (with LoRA adapters).
    train_dataset : SFTDataset
        Training data.
    val_dataset : SFTDataset | None
        Validation data.
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
        Accumulate gradients over this many steps (ignored when using DeepSpeed,
        which reads ``gradient_accumulation_steps`` from its own config).
    device : str
        Target device.
    checkpoint_dir : str
        Where to save checkpoints.
    scheduler_type : str
        LR scheduler: "linear" or "cosine".
    deepspeed_config : str | None
        Path to a DeepSpeed JSON config.  When set, DeepSpeed is used for
        gradient accumulation, mixed-precision, gradient clipping, and the
        optimizer/LR schedule (all taken from the config file).  The standard
        PyTorch optimizer/scheduler below is skipped.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: SFTDataset,
        val_dataset: Optional[SFTDataset] = None,
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 2e-4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints/sft",
        scheduler_type: str = "cosine",
        deepspeed_config: Optional[str] = None,
        log_every_steps: int = 100,
    ) -> None:
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.grad_accum = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every_steps = max(1, log_every_steps)
        self.use_deepspeed = False
        self.engine = None

        # Move model (DeepSpeed handles device placement when active)
        if deepspeed_config is None:
            if hasattr(model, "model"):
                model.model.to(device)
            else:
                model.to(device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=sft_collate_fn,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=sft_collate_fn,
            )

        if deepspeed_config is not None:
            # ── DeepSpeed path ───────────────────────────────────────────────
            # Optimizer, LR schedule, gradient clipping and mixed-precision are
            # all specified inside the DeepSpeed JSON config file.
            import json
            import deepspeed as _ds  # type: ignore[import]

            with open(deepspeed_config) as _f:
                ds_cfg = json.load(_f)
            total_steps = max(1, len(self.train_loader) * epochs // max(1, self.grad_accum))
            ds_cfg = prepare_deepspeed_config(
                ds_cfg,
                batch_size=batch_size,
                gradient_accumulation_steps=self.grad_accum,
                lr=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                total_steps=total_steps,
                max_grad_norm=max_grad_norm,
            )

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            # deepspeed.initialize returns (engine, optimizer, dataloader, lr_sched)
            self.engine, self.optimizer, _, self.scheduler = _ds.initialize(
                model=model,
                model_parameters=trainable_params,
                config=ds_cfg,
            )
            self.use_deepspeed = True
            # Update device to what DeepSpeed chose
            self.device = self.engine.device
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
                scheduler_type=scheduler_type,
            )

        self.best_val_loss = float("inf")

    def _is_main_process(self) -> bool:
        if self.engine is not None and hasattr(self.engine, "global_rank"):
            return self.engine.global_rank == 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _log_step(self, prefix: str, step: int, total_steps: int, loss: float) -> None:
        if not self._is_main_process():
            return
        if step % self.log_every_steps != 0:
            return
        print(f"    {prefix} step {step}/{total_steps}  loss={loss:.4f}")

    def train(self) -> Dict[str, float]:
        """Run full SFT training loop."""
        print(f"\n  Starting SFT for {self.epochs} epochs …")
        history: Dict[str, float] = {}

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            print(f"  Epoch {epoch+1}/{self.epochs}  train_loss={train_loss:.4f}", end="")

            if self.val_loader is not None:
                val_loss = self._validate()
                print(f"  val_loss={val_loss:.4f}", end="")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save(epoch, "best")
                    print("  ★ saved best", end="")
            else:
                self._save(epoch, "best")

            print()
            history[f"epoch_{epoch+1}_loss"] = train_loss

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_steps = 0
        total_steps = len(self.train_loader)

        if self.use_deepspeed:
            # ── DeepSpeed training loop ───────────────────────────────────────
            # engine.backward() + engine.step() handle gradient accumulation,
            # mixed-precision scaling, gradient clipping, optimizer.step(),
            # scheduler.step(), and zero_grad() — all per the JSON config.
            for step, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
                self.engine.backward(loss)
                self.engine.step()
                total_loss += loss.item()
                n_steps += 1
                self._log_step(
                    f"SFT epoch {epoch+1}/{self.epochs}",
                    step,
                    total_steps,
                    total_loss / n_steps,
                )
        else:
            # ── Standard PyTorch training loop ───────────────────────────────
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.grad_accum
                loss.backward()
                total_loss += outputs["loss"].item()
                n_steps += 1

                if step % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self._log_step(
                    f"SFT epoch {epoch+1}/{self.epochs}",
                    step,
                    total_steps,
                    total_loss / n_steps,
                )

            # Handle remaining gradients
            if n_steps % self.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        total_steps = len(self.val_loader) if self.val_loader is not None else 0
        for batch in self.val_loader:  # type: ignore
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs["loss"].item()
            n += 1
            self._log_step("SFT validation", n, total_steps, total_loss / n)
        return total_loss / max(n, 1)

    def _save(self, epoch: int, tag: str) -> None:
        path = self.checkpoint_dir / tag
        if self.use_deepspeed:
            # For ZeRO-2 the weights are replicated on every GPU, so saving
            # the LoRA adapter via self.model works.  For ZeRO-3 (partitioned
            # weights) use engine.save_checkpoint() and consolidate offline
            # with `deepspeed.utils.zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict`.
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
