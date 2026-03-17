"""Learning-rate scheduler factory."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear",
) -> _LRScheduler:
    """Build a learning-rate scheduler with warmup.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer to schedule.
    num_training_steps : int
        Total number of optimisation steps.
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    scheduler_type : str
        ``"linear"`` or ``"cosine"``.

    Returns
    -------
    _LRScheduler
        Configured scheduler instance.
    """
    raise NotImplementedError
