"""Contrastive and alignment loss functions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss.

    Parameters
    ----------
    temperature : float
        Scaling temperature for the softmax.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

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
        # Positive scores: (batch,)
        pos_scores = torch.sum(query_emb * pos_emb, dim=-1) / self.temperature

        if neg_emb is None:
            # In-batch negatives: (batch, batch)
            all_scores = torch.mm(query_emb, pos_emb.t()) / self.temperature
            labels = torch.arange(query_emb.size(0), device=query_emb.device)
            return F.cross_entropy(all_scores, labels)

        # Explicit negatives
        if neg_emb.dim() == 2:
            neg_emb = neg_emb.unsqueeze(1)  # (batch, 1, dim)
        # neg_scores: (batch, num_neg)
        neg_scores = torch.bmm(
            neg_emb, query_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # logits: (batch, 1 + num_neg), label = 0 (positive is first)
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """Triplet margin loss for (query, pos, neg) triplets.

    Parameters
    ----------
    margin : float
        Minimum margin between positive and negative distances.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

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
        return self.loss_fn(query_emb, pos_emb, neg_emb)


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss.

    Implements the DPO objective from Rafailov et al. (2023):

        L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x)
                              - log π(y_l|x)/π_ref(y_l|x)))]

    where y_w is the chosen (preferred) response and y_l is the rejected one.

    Parameters
    ----------
    beta : float
        KL penalty coefficient. Higher β → policy stays closer to reference.
        Typical values: 0.1–0.5.
    label_smoothing : float
        Optional label smoothing for robustness (0 = no smoothing).
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss.

        Parameters
        ----------
        policy_chosen_logps : (batch,)
            Log-probabilities of chosen responses under the policy.
        policy_rejected_logps : (batch,)
            Log-probabilities of rejected responses under the policy.
        ref_chosen_logps : (batch,)
            Log-probabilities of chosen responses under reference model.
        ref_rejected_logps : (batch,)
            Log-probabilities of rejected responses under reference model.

        Returns
        -------
        torch.Tensor — Scalar DPO loss.
        """
        # Log-ratio differences
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)

        if self.label_smoothing > 0:
            loss = (
                -self.label_smoothing * F.logsigmoid(-logits)
                - (1 - self.label_smoothing) * F.logsigmoid(logits)
            )
        else:
            loss = -F.logsigmoid(logits)

        return loss.mean()

    @staticmethod
    def compute_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence log-probabilities from logits.

        Parameters
        ----------
        logits : (batch, seq_len, vocab_size)
        labels : (batch, seq_len) — Target token IDs.
        mask : (batch, seq_len) — 1 for response tokens, 0 for prompt/padding.

        Returns
        -------
        (batch,) — Sum of per-token log-probs over response tokens.
        """
        # Shift: predict next token
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = mask[:, 1:]

        # Per-token log-probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask and sum
        return (token_logps * shift_mask).sum(dim=-1)


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
        if loss_type == "infonce":
            return InfoNCELoss(**kwargs)
        elif loss_type == "triplet":
            return TripletLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type!r}. Use 'infonce' or 'triplet'.")
