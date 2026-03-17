"""Bi-encoder embedding model for query and document encoding."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    """Encodes text into dense vector representations.

    Wraps a pre-trained transformer encoder and applies a pooling strategy
    (mean-pooling or CLS) to produce fixed-size embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier
        (e.g. ``sentence-transformers/all-MiniLM-L6-v2``).
    embedding_dim : int
        Expected dimensionality of the output embeddings.
    pooling : str
        Pooling strategy: ``"mean"`` or ``"cls"``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input tokens into dense embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Normalised embeddings of shape ``(batch, embedding_dim)``.
        """
        raise NotImplementedError

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean-pooling over non-padding tokens.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Encoder outputs ``(batch, seq_len, hidden_dim)``.
        attention_mask : torch.Tensor
            Mask ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Pooled embeddings ``(batch, hidden_dim)``.
        """
        raise NotImplementedError

    def _cls_pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the [CLS] token representation.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Encoder outputs ``(batch, seq_len, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            CLS embeddings ``(batch, hidden_dim)``.
        """
        raise NotImplementedError

    def encode(
        self,
        texts: list[str],
        tokenizer: object,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """High-level helper: tokenise raw strings and return embeddings.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.
        tokenizer : object
            HuggingFace tokenizer.
        batch_size : int
            Inference batch size.
        device : str, optional
            Target device override.

        Returns
        -------
        torch.Tensor
            Stacked embeddings ``(N, embedding_dim)``.
        """
        raise NotImplementedError
