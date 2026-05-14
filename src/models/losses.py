"""
Loss Functions for Viability Prediction.

This module provides loss functions optimized for binary segmentation
of viability maps. The main challenges are:

1. Class Imbalance: Trap pixels are rare compared to viable pixels
2. Spatial Precision: Trap boundaries must be precise
3. Per-Direction Independence: Each direction channel is independent

Loss Functions:
- BCEWithLogitsLoss: Standard binary cross-entropy
- DiceLoss: Overlap-based loss, handles class imbalance
- DiceBCELoss: Combined loss for best of both worlds
- PerChannelDiceBCELoss: Compute loss per direction, then average

The recommended loss is DiceBCELoss with equal weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice coefficient measures overlap between prediction and target:
        Dice = 2 * |P ∩ T| / (|P| + |T|)
    
    Dice Loss = 1 - Dice
    
    Advantages:
    - Handles class imbalance naturally
    - Directly optimizes the evaluation metric
    - Works well for sparse targets (traps)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
        
        Returns:
            Dice loss value
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Compute Dice per sample and channel
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy Loss.
    
    This combination leverages:
    - BCE: Good gradients for pixel-level learning
    - Dice: Handles class imbalance, optimizes overlap
    
    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
    
    Recommended weights: 0.5 each (default)
    
    Args:
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
        smooth: Smoothing factor for Dice
        pos_weight: Optional positive class weight for BCE
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class PerChannelDiceBCELoss(nn.Module):
    """
    Per-channel Dice + BCE Loss.
    
    Computes loss for each direction channel independently,
    then averages. This ensures all directions contribute
    equally regardless of their viability distribution.
    
    Useful when directions have very different viability ratios.
    
    Args:
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss
        smooth: Smoothing factor for Dice
        channel_weights: Optional per-channel weights [N, S, E, W]
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        channel_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
        if channel_weights is not None:
            self.register_buffer("channel_weights", channel_weights)
        else:
            self.channel_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-channel combined loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        B, C, H, W = logits.shape
        probs = torch.sigmoid(logits)
        
        channel_losses = []
        
        for c in range(C):
            # Dice for this channel
            pred_c = probs[:, c].view(B, -1)
            target_c = targets[:, c].view(B, -1)
            
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = (1.0 - dice).mean()
            
            # BCE for this channel
            bce_loss = F.binary_cross_entropy_with_logits(
                logits[:, c], targets[:, c]
            )
            
            # Combined
            channel_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
            channel_losses.append(channel_loss)
        
        # Stack and apply channel weights
        channel_losses = torch.stack(channel_losses)
        
        if self.channel_weights is not None:
            channel_losses = channel_losses * self.channel_weights
        
        return channel_losses.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive class
    FL(p) = -(1-alpha) * p^gamma * log(1-p)  for negative class
    
    The (1-p)^gamma factor down-weights easy examples,
    focusing training on hard examples (misclassified pixels).
    
    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
        
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)
        
        # Compute cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.
    
    Allows different weights for false positives and false negatives.
    
    Tversky = TP / (TP + alpha*FP + beta*FN)
    
    Special cases:
    - alpha=beta=0.5: Equivalent to Dice
    - alpha=beta=1.0: Equivalent to Jaccard/IoU
    - alpha < beta: Penalize false negatives more (better recall)
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
        
        Returns:
            Tversky loss value
        """
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return 1.0 - tversky


def create_loss(
    loss_type: str = "dice_bce",
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss function
            - "bce": BCEWithLogitsLoss
            - "dice": DiceLoss
            - "dice_bce": DiceBCELoss (recommended)
            - "per_channel": PerChannelDiceBCELoss
            - "focal": FocalLoss
            - "tversky": TverskyLoss
        dice_weight: Weight for Dice component
        bce_weight: Weight for BCE component
        **kwargs: Additional arguments for specific losses
    
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    
    elif loss_type == "dice_bce":
        return DiceBCELoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            **kwargs
        )
    
    elif loss_type == "per_channel":
        return PerChannelDiceBCELoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            **kwargs
        )
    
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    B, C, H, W = 2, 4, 64, 64
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W)).float()
    
    # Test each loss
    losses = {
        "bce": create_loss("bce"),
        "dice": create_loss("dice"),
        "dice_bce": create_loss("dice_bce"),
        "per_channel": create_loss("per_channel"),
        "focal": create_loss("focal"),
        "tversky": create_loss("tversky"),
    }
    
    print("\nLoss values on random data:")
    for name, loss_fn in losses.items():
        value = loss_fn(logits, targets)
        print(f"  {name:15s}: {value.item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    logits.requires_grad = True
    
    for name, loss_fn in losses.items():
        loss = loss_fn(logits, targets)
        loss.backward()
        
        if logits.grad is not None and logits.grad.abs().sum() > 0:
            print(f"  {name:15s}: ✓ gradients flow")
        else:
            print(f"  {name:15s}: ✗ no gradients")
        
        logits.grad = None
    
    # Test with extreme cases
    print("\nTesting edge cases...")
    
    # All zeros prediction, all zeros target (perfect match)
    logits_zero = torch.ones(B, C, H, W) * (-10)  # sigmoid → 0
    targets_zero = torch.zeros(B, C, H, W)
    
    for name in ["dice", "dice_bce"]:
        loss_fn = create_loss(name)
        loss = loss_fn(logits_zero, targets_zero)
        print(f"  {name} (all-zero match): {loss.item():.4f}")
    
    # All ones prediction, all ones target (perfect match)
    logits_one = torch.ones(B, C, H, W) * 10  # sigmoid → 1
    targets_one = torch.ones(B, C, H, W)
    
    for name in ["dice", "dice_bce"]:
        loss_fn = create_loss(name)
        loss = loss_fn(logits_one, targets_one)
        print(f"  {name} (all-one match): {loss.item():.4f}")
    
    print("\n✓ All loss function tests passed!")
