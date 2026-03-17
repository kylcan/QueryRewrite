"""End-to-end alignment model combining rewriter + embedder + loss."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.rewriter import QueryRewriter
from models.embedder import EmbeddingModel


class AlignmentModel(nn.Module):
    """Orchestrates rewriting, encoding, and contrastive alignment.

    Composes ``QueryRewriter`` and ``EmbeddingModel`` so they can be
    trained jointly or separately via a shared forward interface.

    Parameters
    ----------
    rewriter : QueryRewriter
        The LLM-based query rewriter.
    embedder : EmbeddingModel
        The bi-encoder embedding model.
    freeze_rewriter : bool
        If ``True``, freeze the rewriter during alignment training.
    freeze_embedder : bool
        If ``True``, freeze the embedder during alignment training.
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        embedder: EmbeddingModel,
        freeze_rewriter: bool = False,
        freeze_embedder: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings for query, positive doc, and optional negative doc.

        Returns
        -------
        dict[str, torch.Tensor]
            ``query_emb``, ``pos_emb``, and optionally ``neg_emb``.
        """
        raise NotImplementedError

    def _set_frozen(self, module: nn.Module, frozen: bool) -> None:
        """Freeze or unfreeze all parameters in a module.

        Parameters
        ----------
        module : nn.Module
            The module whose parameters should be frozen/unfrozen.
        frozen : bool
            ``True`` to freeze, ``False`` to unfreeze.
        """
        raise NotImplementedError
