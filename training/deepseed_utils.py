"""Utilities for preparing DeepSpeed configs at runtime."""

from __future__ import annotations

import copy
from typing import Any, Dict


def prepare_deepspeed_config(
    config: Dict[str, Any],
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
    max_grad_norm: float,
) -> Dict[str, Any]:
    """Resolve ``\"auto\"`` placeholders used by the checked-in configs."""
    cfg = copy.deepcopy(config)

    _replace_auto(cfg, "train_micro_batch_size_per_gpu", batch_size)
    _replace_auto(cfg, "gradient_accumulation_steps", gradient_accumulation_steps)
    _replace_auto(cfg, "gradient_clipping", max_grad_norm)

    optimizer_params = cfg.get("optimizer", {}).get("params", {})
    _replace_auto(optimizer_params, "lr", lr)
    _replace_auto(optimizer_params, "weight_decay", weight_decay)

    scheduler = cfg.get("scheduler")
    if isinstance(scheduler, dict):
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler_params = scheduler.get("params", {})
        _replace_auto(scheduler_params, "warmup_num_steps", warmup_steps)
        _replace_auto(scheduler_params, "total_num_steps", total_steps)

        # DeepSpeed 0.18 on this machine does not recognize the checked-in
        # WarmupCosineAnnealing scheduler name. Fall back to a fixed LR instead
        # of failing engine initialization.
        if scheduler.get("type") == "WarmupCosineAnnealing":
            cfg.pop("scheduler", None)

    zero_optimization = cfg.get("zero_optimization", {})
    _replace_auto(zero_optimization, "reduce_bucket_size", 50_000_000)
    _replace_auto(zero_optimization, "stage3_prefetch_bucket_size", 50_000_000)
    _replace_auto(zero_optimization, "stage3_param_persistence_threshold", 1_000_000)

    return cfg


def _replace_auto(config: Dict[str, Any], key: str, value: Any) -> None:
    if config.get(key) == "auto":
        config[key] = value