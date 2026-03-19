"""Learning-rate scheduler factory with warmup support."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear",
) -> LambdaLR:
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
    LambdaLR
        Configured scheduler instance.
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "linear":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

    elif scheduler_type == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type!r}")

    return LambdaLR(optimizer, lr_lambda)
