"""Multi-task loss for evidence selection.

Losses:
  - Focal Loss per node type (replaces BCE to handle class imbalance)
  - Margin ranking loss (positive vs. negative nodes)
  - Weighted sum: total = sum_t lambda_t * L_t + lambda_rank * L_rank
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.utils.logging import get_logger

logger = get_logger(__name__)

SCORED_TYPES = ["textblock", "cell", "image", "caption", "table"]


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this reduces to weighted BCE.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma == 0:
            loss = bce
        else:
            p_t = torch.exp(-bce)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class EvidenceSelectionLoss(nn.Module):
    def __init__(
        self,
        lambda_text: float = 1.0,
        lambda_cell: float = 1.0,
        lambda_image: float = 1.0,
        lambda_caption: float = 0.5,
        lambda_table: float = 0.5,
        lambda_rank: float = 0.5,
        margin: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[Union[float, Dict[str, float]]] = None,
    ):
        super().__init__()
        self.lambdas = {
            "textblock": lambda_text,
            "cell": lambda_cell,
            "image": lambda_image,
            "caption": lambda_caption,
            "table": lambda_table,
        }
        self.lambda_rank = lambda_rank
        self.margin = margin

        # Per-type FocalLoss instances
        self.focal_losses = nn.ModuleDict()
        if isinstance(focal_alpha, dict):
            for ntype in SCORED_TYPES:
                alpha = focal_alpha.get(ntype, None)
                self.focal_losses[ntype] = FocalLoss(gamma=focal_gamma, alpha=alpha)
        else:
            # Single alpha for all types (backward compatible)
            for ntype in SCORED_TYPES:
                self.focal_losses[ntype] = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            logits: dict ntype -> (N,) logit tensor
            data:   HeteroData with .y per node type

        Returns:
            dict with 'total', per-type focal losses, and 'rank'.
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total_tensor = None  # accumulate weighted losses

        # Per-type focal losses
        for ntype in SCORED_TYPES:
            lam = self.lambdas.get(ntype, 0.0)
            if lam == 0.0:
                continue
            if ntype not in logits or logits[ntype].shape[0] == 0:
                continue
            if not hasattr(data[ntype], "y") or data[ntype].y is None:
                continue
            y = data[ntype].y
            if y.shape[0] == 0:
                continue

            logit = logits[ntype]
            focal_loss = self.focal_losses[ntype](logit, y)
            loss_dict[f"focal_{ntype}"] = focal_loss

            contribution = lam * focal_loss
            total_tensor = contribution if total_tensor is None else total_tensor + contribution

        # Ranking loss: encourage positive scores > negative scores + margin
        rank_loss = self._ranking_loss(logits, data)
        if rank_loss is not None:
            loss_dict["rank"] = rank_loss
            rl = self.lambda_rank * rank_loss
            total_tensor = rl if total_tensor is None else total_tensor + rl

        # Handle edge case: no valid losses computed
        if total_tensor is None:
            total_tensor = torch.tensor(0.0, requires_grad=False)
            if next(iter(logits.values()), None) is not None:
                v = next(iter(logits.values()))
                total_tensor = total_tensor.to(v.device)

        loss_dict["total"] = total_tensor
        return loss_dict

    def _ranking_loss(
        self,
        logits: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Optional[torch.Tensor]:
        """Margin ranking loss: score(pos) - score(neg) > margin.

        Applied to primary evidence types (textblock, cell, image, table).
        """
        all_pos = []
        for ntype in ["textblock", "cell", "image", "table"]:
            if ntype not in logits or logits[ntype].shape[0] == 0:
                continue
            if not hasattr(data[ntype], "y") or data[ntype].y is None:
                continue
            y = data[ntype].y
            logit = logits[ntype]
            pos_mask = y > 0.5
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_scores = logit[pos_mask]
            neg_scores = logit[neg_mask]

            # Broadcast: all positive vs all negative pairs
            p = pos_scores.unsqueeze(1)   # (P, 1)
            n = neg_scores.unsqueeze(0)   # (1, N)
            diff = p - n  # (P, N) pairwise differences
            rank_l = F.relu(self.margin - diff).mean()
            all_pos.append(rank_l)

        if not all_pos:
            return None
        return torch.stack(all_pos).mean()
