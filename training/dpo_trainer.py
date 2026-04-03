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

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from models.losses import DPOLoss
from training.deepspeed_utils import prepare_deepspeed_config


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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        chosen = self._tokenize_pair(r["prompt"], r["chosen"])
        rejected = self._tokenize_pair(r["prompt"], r["rejected"])
        out: Dict[str, Any] = {"chosen": chosen, "rejected": rejected}
        if "_ref_chosen_logp" in r:
            out["ref_chosen_logp"] = torch.tensor(r["_ref_chosen_logp"])
            out["ref_rejected_logp"] = torch.tensor(r["_ref_rejected_logp"])
        return out


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


def dpo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate DPO samples, padding chosen and rejected separately."""
    result: Dict[str, Any] = {}
    for key in ("chosen", "rejected"):
        items = [b[key] for b in batch]
        result[key] = {
            "input_ids": _pad_sequence([x["input_ids"] for x in items], 0),
            "attention_mask": _pad_sequence([x["attention_mask"] for x in items], 0),
            "response_mask": _pad_sequence([x["response_mask"] for x in items], 0),
        }
    if "ref_chosen_logp" in batch[0]:
        result["ref_chosen_logp"] = torch.stack([b["ref_chosen_logp"] for b in batch])
        result["ref_rejected_logp"] = torch.stack([b["ref_rejected_logp"] for b in batch])
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
    precompute_ref_log_probs : bool
        If True, compute all reference model log-probs before training starts,
        cache them in the dataset records, then free the reference model from
        GPU.  Essential for multi-GPU setups where the ref model does not fit
        alongside the policy model.
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
        precompute_ref_log_probs: bool = False,
        log_every_steps: int = 100,
        ref_model_device: str = "auto",
        save_every_steps: int = 500,
        seed: int = 42,
        resume: bool = False,
    ) -> None:
        self.device = device
        self.policy_device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.grad_accum = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_steps = max_steps
        self.log_every_steps = max(1, log_every_steps)
        self.save_every_steps = max(0, save_every_steps)
        self.use_deepspeed = False
        self.engine = None
        self._ref_precomputed = precompute_ref_log_probs
        self.seed = seed
        self.resume = resume
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.ref_cache_path = self.checkpoint_dir / "ref_logps_cache.pt"
        self.resume_dir = self.checkpoint_dir / "resume"
        self.resume_state_path = self.resume_dir / "training_state.pt"
        self.ref_device = self._resolve_ref_device(
            requested=ref_model_device,
            default_device=device,
            precompute_ref_log_probs=precompute_ref_log_probs,
        )

        # Models
        self.model = model
        self.ref_model = ref_model

        # Device placement (DeepSpeed handles it for the policy)
        if deepspeed_config is None:
            if hasattr(model, "model"):
                model.model.to(device)
            else:
                model.to(device)
        self.policy_device = device
        if not precompute_ref_log_probs:
            ref_model.to(self.ref_device)

        # Loss
        self.dpo_loss = DPOLoss(beta=beta, label_smoothing=label_smoothing)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dpo_collate_fn,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dpo_collate_fn,
            )

        # ── Pre-compute ref logprobs (before DeepSpeed init) ────────────
        if precompute_ref_log_probs:
            self._precompute_ref_logps(ref_model, train_dataset, val_dataset)

        if deepspeed_config is not None:
            # ── DeepSpeed path ───────────────────────────────────────────────
            # Only the trainable policy is wrapped; the frozen ref model is
            # kept as a plain module on the same device.
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
            self.engine, self.optimizer, _, self.scheduler = _ds.initialize(
                model=model,
                model_parameters=trainable_params,
                config=ds_cfg,
            )
            self.use_deepspeed = True
            self.device = str(self.engine.device)
            self.policy_device = str(self.engine.device)
            if not precompute_ref_log_probs:
                ref_model.to(self.ref_device)
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
        self._resume_epoch: int = 0
        self._resume_step_in_epoch: int = 0

        if self.resume:
            self._load_training_state()

    @staticmethod
    def _resolve_ref_device(
        requested: str,
        default_device: str,
        precompute_ref_log_probs: bool,
    ) -> str:
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            return default_device if not precompute_ref_log_probs else "cuda:0"
        if precompute_ref_log_probs and torch.cuda.is_available():
            return "cuda:0"
        return default_device

    def _is_cache_leader(self) -> bool:
        return self.rank == 0

    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _build_train_loader(self, epoch: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dpo_collate_fn,
            generator=generator,
        )

    def _wait_for_ref_cache(self, timeout_seconds: int = 7200) -> None:
        start = time.time()
        while not self.ref_cache_path.exists():
            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for ref log-prob cache at {self.ref_cache_path}"
                )
            time.sleep(2)

    @staticmethod
    def _assign_cached_logps(dataset: Optional[DPODataset], cached: Optional[Dict[str, torch.Tensor]]) -> None:
        if dataset is None or cached is None:
            return
        chosen = cached["chosen"]
        rejected = cached["rejected"]
        if len(dataset.records) != chosen.shape[0] or len(dataset.records) != rejected.shape[0]:
            raise ValueError("Cached ref log-probs do not match dataset size.")
        for index, record in enumerate(dataset.records):
            record["_ref_chosen_logp"] = float(chosen[index].item())
            record["_ref_rejected_logp"] = float(rejected[index].item())

    def _log_step(
        self,
        prefix: str,
        step: int,
        total_steps: int,
        loss: float,
        acc: float,
    ) -> None:
        if not self._is_main_process():
            return
        if step % self.log_every_steps != 0:
            return
        print(f"    {prefix} step {step}/{total_steps}  loss={loss:.4f}  acc={acc:.3f}")

    def _save_training_state(self, epoch: int, step_in_epoch: int) -> None:
        if self.use_deepspeed:
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            self.engine.save_checkpoint(
                str(self.resume_dir),
                tag="latest",
                client_state={
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": self._global_step,
                    "best_val_loss": self.best_val_loss,
                },
            )
        elif self._is_main_process():
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                },
                self.resume_dir / "model_state.pt",
            )

        if self._is_main_process():
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": self._global_step,
                    "best_val_loss": self.best_val_loss,
                },
                self.resume_state_path,
            )

    def _load_training_state(self) -> None:
        if not self.resume_state_path.exists():
            if self._is_main_process():
                print(f"  Resume requested but no state found at {self.resume_state_path}; starting fresh.")
            return

        state = torch.load(self.resume_state_path, map_location="cpu")
        self._resume_epoch = int(state.get("epoch", 0))
        self._resume_step_in_epoch = int(state.get("step_in_epoch", 0))
        self._global_step = int(state.get("global_step", 0))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))

        if self.use_deepspeed:
            _, client_state = self.engine.load_checkpoint(str(self.resume_dir), tag="latest")
            if client_state:
                self._resume_epoch = int(client_state.get("epoch", self._resume_epoch))
                self._resume_step_in_epoch = int(client_state.get("step_in_epoch", self._resume_step_in_epoch))
                self._global_step = int(client_state.get("global_step", self._global_step))
                self.best_val_loss = float(client_state.get("best_val_loss", self.best_val_loss))
        elif self._is_main_process():
            model_state_path = self.resume_dir / "model_state.pt"
            if model_state_path.exists():
                model_state = torch.load(model_state_path, map_location="cpu")
                self.model.load_state_dict(model_state["model_state_dict"])
                self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
                self.scheduler.load_state_dict(model_state["scheduler_state_dict"])

        if self._is_main_process():
            print(
                "  Resuming training from "
                f"epoch {self._resume_epoch + 1}, step {self._resume_step_in_epoch + 1}"
            )

    # ── ref logprob pre-computation ───────────────────────────────────────

    @torch.no_grad()
    def _precompute_ref_logps(
        self,
        ref_model: torch.nn.Module,
        train_dataset: DPODataset,
        val_dataset: Optional[DPODataset],
    ) -> None:
        """Compute ref log-probs once, save to disk, then load into all ranks."""
        import gc

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self._is_cache_leader() and not self.ref_cache_path.exists():
            print(
                f"\n  [rank{self.rank}] Pre-computing reference log-probs on {self.ref_device} …"
            )
            ref_model.eval()
            ref_model.to(self.ref_device)

            cache_payload: Dict[str, Dict[str, torch.Tensor]] = {}
            for tag, dataset in [("train", train_dataset), ("val", val_dataset)]:
                if dataset is None:
                    continue
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=dpo_collate_fn,
                )
                chosen_batches: List[torch.Tensor] = []
                rejected_batches: List[torch.Tensor] = []
                for batch in loader:
                    c_logp = self._get_logps(ref_model, batch["chosen"], device=self.ref_device)
                    r_logp = self._get_logps(ref_model, batch["rejected"], device=self.ref_device)
                    chosen_batches.append(c_logp.cpu())
                    rejected_batches.append(r_logp.cpu())

                cache_payload[tag] = {
                    "chosen": torch.cat(chosen_batches),
                    "rejected": torch.cat(rejected_batches),
                }
                print(f"    {tag}: {cache_payload[tag]['chosen'].shape[0]} samples cached")

            tmp_path = self.ref_cache_path.with_suffix(".tmp")
            torch.save(cache_payload, tmp_path)
            os.replace(tmp_path, self.ref_cache_path)
            print(f"  Saved ref log-prob cache to {self.ref_cache_path}")

        if not self._is_cache_leader():
            print(f"\n  [rank{self.rank}] Waiting for ref log-prob cache …")
        self._wait_for_ref_cache()

        cached_payload = torch.load(self.ref_cache_path, map_location="cpu")
        self._assign_cached_logps(train_dataset, cached_payload.get("train"))
        self._assign_cached_logps(val_dataset, cached_payload.get("val"))

        ref_model.cpu()
        del ref_model
        self.ref_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [rank{self.rank}] Ref log-probs loaded; ref model freed from GPU.\n")

    def _get_logps(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute per-sequence log-probs for a batch.

        ``model`` may be the raw policy model, the DeepSpeed engine that wraps
        it, or the frozen reference model — all support the same forward call.
        """
        target_device = self.device if device is None else device
        input_ids = batch["input_ids"].to(target_device)
        attention_mask = batch["attention_mask"].to(target_device)
        response_mask = batch["response_mask"].to(target_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return DPOLoss.compute_logps(logits, input_ids, response_mask)

    def train(self) -> Dict[str, float]:
        """Run full DPO training loop."""
        print(f"\n  Starting DPO for {self.epochs} epochs (β={self.dpo_loss.beta}) …")
        history: Dict[str, float] = {}

        for epoch in range(self._resume_epoch, self.epochs):
            start_step = self._resume_step_in_epoch if epoch == self._resume_epoch else 0
            train_loss, train_acc = self._train_epoch(epoch, start_step=start_step)
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
            self._resume_step_in_epoch = 0
            if epoch + 1 < self.epochs:
                self._save_training_state(epoch + 1, 0)

        return history

    def _train_epoch(self, epoch: int, start_step: int = 0):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0
        global_step = getattr(self, "_global_step", 0)

        # Determine which model handle to use for policy forwards
        policy: Any = self.engine if self.use_deepspeed else self.model
        train_loader = self._build_train_loader(epoch)
        total_steps = len(train_loader)

        def _get_ref_logps(batch):
            """Return ref logps from cache or live computation."""
            if "ref_chosen_logp" in batch:
                return (batch["ref_chosen_logp"].to(self.device),
                        batch["ref_rejected_logp"].to(self.device))
            with torch.no_grad():
                return (self._get_logps(self.ref_model, batch["chosen"], device=self.ref_device).to(self.device),
                        self._get_logps(self.ref_model, batch["rejected"], device=self.ref_device).to(self.device))

        if self.use_deepspeed:
            # ── DeepSpeed training loop ───────────────────────────────────────
            for step, batch in enumerate(train_loader, start=1):
                if step <= start_step:
                    continue
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                policy_chosen_logps = self._get_logps(policy, batch["chosen"])
                policy_rejected_logps = self._get_logps(policy, batch["rejected"])
                ref_chosen_logps, ref_rejected_logps = _get_ref_logps(batch)

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
                self._global_step = global_step
                self._log_step(
                    "DPO train",
                    step,
                    total_steps,
                    total_loss / n_steps,
                    total_acc / n_steps,
                )
                if self.save_every_steps and step % self.save_every_steps == 0:
                    self._save_training_state(epoch, step)
        else:
            # ── Standard PyTorch training loop ───────────────────────────────
            self.optimizer.zero_grad()
            for step, batch in enumerate(train_loader, start=1):
                if step <= start_step:
                    continue
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                policy_chosen_logps = self._get_logps(policy, batch["chosen"])
                policy_rejected_logps = self._get_logps(policy, batch["rejected"])
                ref_chosen_logps, ref_rejected_logps = _get_ref_logps(batch)

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
                self._global_step = global_step

                if step % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self._log_step(
                    "DPO train",
                    step,
                    total_steps,
                    total_loss / n_steps,
                    total_acc / n_steps,
                )
                if self.save_every_steps and step % self.save_every_steps == 0:
                    self._save_training_state(epoch, step)

            if n_steps % self.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        self._global_step = global_step
        self._save_training_state(epoch + 1, 0)
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

        def _get_ref_logps_val(batch):
            if "ref_chosen_logp" in batch:
                return (batch["ref_chosen_logp"].to(self.device),
                        batch["ref_rejected_logp"].to(self.device))
            return (self._get_logps(self.ref_model, batch["chosen"]),
                    self._get_logps(self.ref_model, batch["rejected"]))

        for batch in self.val_loader:  # type: ignore
            policy_chosen_logps = self._get_logps(policy, batch["chosen"])
            policy_rejected_logps = self._get_logps(policy, batch["rejected"])
            ref_chosen_logps, ref_rejected_logps = _get_ref_logps_val(batch)

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