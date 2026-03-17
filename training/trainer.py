"""Main training loop orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from configs.config import ProjectConfig


class Trainer:
    """Handles the full training lifecycle: train, validate, checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train (can be ``AlignmentModel`` or any ``nn.Module``).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    loss_fn : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    config : ProjectConfig
        Project-level configuration.
    scheduler : optional
        Learning-rate scheduler.
    device : str
        Target device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ProjectConfig,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError

    def train(self) -> Dict[str, float]:
        """Run the full training loop for ``config.training.epochs`` epochs.

        Returns
        -------
        dict[str, float]
            Final training metrics.
        """
        raise NotImplementedError

    def _train_epoch(self, epoch: int) -> float:
        """Train for a single epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index.

        Returns
        -------
        float
            Average training loss for this epoch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation and return metrics.

        Returns
        -------
        dict[str, float]
            Validation metrics.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Persist model, optimizer, and scheduler state.

        Parameters
        ----------
        path : str | Path
            Checkpoint directory.
        epoch : int
            Current epoch.
        """
        raise NotImplementedError

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
        raise NotImplementedError
