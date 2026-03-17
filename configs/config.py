"""Configuration loader.

Reads YAML configs and exposes them as nested, dot-accessible objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Typed sub-configs
# ------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and length limits for dataset construction."""
    train_path: str = "data/raw/train.jsonl"
    val_path: str = "data/raw/val.jsonl"
    test_path: str = "data/raw/test.jsonl"
    max_query_length: int = 64
    max_doc_length: int = 256


@dataclass
class LoRAConfig:
    """LoRA-specific hyper-parameters."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class RewriteModelConfig:
    """Config for the LLM-based query rewriter."""
    base_model: str = "meta-llama/Llama-2-7b-hf"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class EmbeddingModelConfig:
    """Config for the embedding / bi-encoder model."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    pooling: str = "mean"


@dataclass
class AlignmentConfig:
    """Config for the alignment / contrastive loss."""
    loss_type: str = "infonce"
    temperature: float = 0.07
    margin: float = 0.2


@dataclass
class RetrievalConfig:
    """Config for FAISS index and retrieval."""
    index_type: str = "flat_ip"
    top_k: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    nprobe: int = 10


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "outputs/"
    logging_dir: str = "outputs/logs/"


@dataclass
class EvaluationConfig:
    """Metrics configuration."""
    metrics: List[str] = field(default_factory=lambda: ["recall", "ndcg", "mrr"])
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Root configuration aggregating all sub-configs."""
    name: str = "query_rewrite"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    rewrite_model: RewriteModelConfig = field(default_factory=RewriteModelConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config file and return a fully typed ``ProjectConfig``.

    Parameters
    ----------
    path : str | Path
        Path to a YAML configuration file.

    Returns
    -------
    ProjectConfig
        Parsed and validated project configuration.
    """
    raise NotImplementedError
