"""losses.py - Loss functions for classification and reconstruction."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class FocalLoss(nn.Module):
    """
    Focal Loss for rare event detection.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Features:
    - γ (gamma): Reduces weight of easy examples, focuses on hard ones
    - α (alpha): Balances class importance
    - pos_weight: Additional weight for positive class (adaptive scaling)

    Args:
        alpha: Balance factor (dynamic based on prevalence)
        gamma: Focusing parameter (default: 3.0)
        pos_weight: Additional weight for positive class
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 3.0,
        pos_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model logits (shape: [batch, ...])
            targets: Binary labels (0 or 1)

        Returns:
            Tensor with calculated loss
        """
        probs = torch.sigmoid(logits)

        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            modulating_factor = modulating_factor * alpha_t

        if self.pos_weight != 1.0:
            weight_t = self.pos_weight * targets + (1 - targets)
            modulating_factor = modulating_factor * weight_t

        loss = modulating_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss with variable weights per channel/variable.

    Used for autoencoder reconstruction with different importance weights
    for each climate variable.

    Args:
        variable_weights: Tensor of weights per variable/channel
        beta: Threshold for L1/L2 transition (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        variable_weights: torch.Tensor,
        beta: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.register_buffer("weights", variable_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)

        Returns:
            Weighted Smooth L1 Loss
        """
        diff = torch.abs(pred - target)

        # Smooth L1
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        # Apply variable weights
        if self.weights is not None:
            # weights: (C,) -> (1, C, 1, 1)
            weighted_loss = loss * self.weights.view(1, -1, 1, 1)
        else:
            weighted_loss = loss

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with class weights and optional mask.

    Used for classification with imbalanced classes.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model logits (B, 1, H, W)
            targets: Binary targets (B, 1, H, W)
            mask: Optional mask for valid pixels (B, 1, H, W)

        Returns:
            Weighted BCE loss
        """
        # BCE with logits
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply class weights
        weight = self.pos_weight * targets + self.neg_weight * (1 - targets)
        loss = loss * weight

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                loss = loss.sum() / valid_pixels
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()

        return loss


def build_loss(
    loss_name: str,
    pos_weight: float = 1.0,
    gamma: float = 3.0,
    alpha: Optional[float] = None,
    prevalence: Optional[float] = None,
    use_dynamic_alpha: bool = True,
    variable_weights: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    """
    Build loss function with dynamic alpha support.

    Args:
        loss_name: Name of the loss ('focal', 'weighted_bce', 'smooth_l1')
        pos_weight: Weight for positive class
        gamma: Gamma for Focal Loss
        alpha: Fixed alpha (if None and use_dynamic_alpha=True, calculates from prevalence)
        prevalence: Prevalence of positive class (for dynamic alpha)
        use_dynamic_alpha: If True, calculate alpha = 1 - 2*prevalence
        variable_weights: Variable weights for Smooth L1 Loss

    Returns:
        Loss module
    """

    if loss_name == "focal":
        # Dynamic alpha - more conservative
        if alpha is None and use_dynamic_alpha and prevalence is not None:
            # α = 1 - 2*prevalence (more conservative)
            # For extreme (2.9%): α = 1 - 0.058 = 0.942
            # For severe (7.0%): α = 1 - 0.140 = 0.860
            # For moderate (15.3%): α = 1 - 0.306 = 0.694
            alpha = 1.0 - prevalence * 2
            # Clipping to avoid extreme values
            alpha = max(0.25, min(0.75, alpha))

        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            pos_weight=pos_weight,
        )

    elif loss_name == "weighted_bce":
        return WeightedBCEWithLogitsLoss(
            pos_weight=pos_weight,
            neg_weight=1.0,
        )

    elif loss_name == "smooth_l1":
        if variable_weights is None:
            raise ValueError("variable_weights required for smooth_l1 loss")
        return WeightedSmoothL1Loss(
            variable_weights=variable_weights,
            beta=0.1,
        )

    else:
        raise ValueError(f"Unsupported loss: {loss_name}. Use 'focal', 'weighted_bce', or 'smooth_l1'.")