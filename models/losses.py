"""Contrastive and alignment loss functions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss.

    Parameters
    ----------
    temperature : float
        Scaling temperature for the softmax.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Parameters
        ----------
        query_emb : torch.Tensor
            Query embeddings ``(batch, dim)``.
        pos_emb : torch.Tensor
            Positive document embeddings ``(batch, dim)``.
        neg_emb : torch.Tensor, optional
            Explicit negatives ``(batch, dim)`` or ``(batch, num_neg, dim)``.
            If ``None``, in-batch negatives are used.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        raise NotImplementedError


class TripletLoss(nn.Module):
    """Triplet margin loss for (query, pos, neg) triplets.

    Parameters
    ----------
    margin : float
        Minimum margin between positive and negative distances.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet margin loss.

        Parameters
        ----------
        query_emb : torch.Tensor
            ``(batch, dim)``
        pos_emb : torch.Tensor
            ``(batch, dim)``
        neg_emb : torch.Tensor
            ``(batch, dim)``

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        raise NotImplementedError


class AlignmentLossFactory:
    """Factory that instantiates the correct loss based on a config string.

    Supported values: ``"infonce"``, ``"triplet"``.
    """

    @staticmethod
    def create(loss_type: str, **kwargs) -> nn.Module:
        """Return a loss module.

        Parameters
        ----------
        loss_type : str
            One of ``"infonce"`` or ``"triplet"``.
        **kwargs
            Keyword arguments forwarded to the loss constructor.

        Returns
        -------
        nn.Module
            The instantiated loss.

        Raises
        ------
        ValueError
            If ``loss_type`` is not recognised.
        """
        raise NotImplementedError
