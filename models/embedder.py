"""Bi-encoder embedding model for query and document encoding."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = self._cls_pool(hidden)
        else:
            pooled = self._mean_pool(hidden, attention_mask)

        return F.normalize(pooled, p=2, dim=-1)

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean-pooling over non-padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _cls_pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the [CLS] token representation."""
        return hidden_states[:, 0, :]

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """High-level helper: tokenise raw strings and return embeddings.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.
        batch_size : int
            Inference batch size.
        device : str, optional
            Target device override.

        Returns
        -------
        torch.Tensor
            Stacked embeddings ``(N, embedding_dim)``.
        """
        if device is None:
            dev = next(self.parameters()).device
        else:
            dev = torch.device(device)

        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(dev) for k, v in encoded.items()}
            emb = self.forward(encoded["input_ids"], encoded["attention_mask"])
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)
