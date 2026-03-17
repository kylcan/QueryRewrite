"""LLM-based query rewriter with optional LoRA fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class QueryRewriter(nn.Module):
    """Wraps a causal LM and applies LoRA adapters for query rewriting.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    lora_rank : int
        Rank for LoRA decomposition.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability inside LoRA layers.
    target_modules : list[str]
        Names of attention projection layers to adapt.
    """

    def __init__(
        self,
        base_model_name: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the causal LM forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.
        labels : torch.Tensor, optional
            Target token IDs for language-modelling loss.

        Returns
        -------
        dict[str, torch.Tensor]
            ``logits`` and optionally ``loss``.
        """
        raise NotImplementedError

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Auto-regressively generate rewritten queries.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs.
        attention_mask : torch.Tensor
            Attention mask.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        torch.Tensor
            Generated token IDs.
        """
        raise NotImplementedError

    def save_adapter(self, path: str) -> None:
        """Persist only the LoRA adapter weights.

        Parameters
        ----------
        path : str
            Directory to save adapter weights.
        """
        raise NotImplementedError

    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from disk.

        Parameters
        ----------
        path : str
            Directory containing saved adapter weights.
        """
        raise NotImplementedError
