"""Main training loop orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader


class Trainer:
    """Handles the full training lifecycle: train, validate, checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The embedding model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader | None
        Validation data loader (optional).
    loss_fn : torch.nn.Module
        Loss function (e.g. InfoNCE).
    optimizer : torch.optim.Optimizer
        Optimizer.
    epochs : int
        Number of training epochs.
    scheduler : optional
        Learning-rate scheduler (step per epoch).
    device : str
        Target device (``"cuda"`` or ``"cpu"``).
    checkpoint_dir : str | Path
        Directory to save model checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 3,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        checkpoint_dir: str | Path = "checkpoints/alignment",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_val_loss = float("inf")

    def train(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns
        -------
        dict[str, float]
            Final training metrics.
        """
        history: Dict[str, float] = {}
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            print(f"  Epoch {epoch+1}/{self.epochs}  train_loss={train_loss:.4f}", end="")

            if self.val_loader is not None:
                val_metrics = self._validate()
                val_loss = val_metrics["val_loss"]
                print(f"  val_loss={val_loss:.4f}", end="")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(self.checkpoint_dir / "best", epoch)
                    print("  ★ saved best", end="")
            else:
                self.save_checkpoint(self.checkpoint_dir / "best", epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            print()
            history[f"epoch_{epoch+1}_train_loss"] = train_loss

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for a single epoch.

        Returns
        -------
        float
            Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            q_emb = self.model(
                batch["query_input_ids"].to(self.device),
                batch["query_attention_mask"].to(self.device),
            )
            p_emb = self.model(
                batch["pos_input_ids"].to(self.device),
                batch["pos_attention_mask"].to(self.device),
            )
            n_emb = self.model(
                batch["neg_input_ids"].to(self.device),
                batch["neg_attention_mask"].to(self.device),
            )

            loss = self.loss_fn(q_emb, p_emb, n_emb)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:  # type: ignore[union-attr]
            q_emb = self.model(
                batch["query_input_ids"].to(self.device),
                batch["query_attention_mask"].to(self.device),
            )
            p_emb = self.model(
                batch["pos_input_ids"].to(self.device),
                batch["pos_attention_mask"].to(self.device),
            )
            n_emb = self.model(
                batch["neg_input_ids"].to(self.device),
                batch["neg_attention_mask"].to(self.device),
            )
            loss = self.loss_fn(q_emb, p_emb, n_emb)
            total_loss += loss.item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Persist model state.

        Parameters
        ----------
        path : str | Path
            Checkpoint directory.
        epoch : int
            Current epoch.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path / "checkpoint.pt",
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore training state from a checkpoint.

        Parameters
        ----------
        path : str | Path
            Checkpoint directory.

        Returns
        -------
        int
            The epoch to resume from.
        """
        ckpt = torch.load(Path(path) / "checkpoint.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]
