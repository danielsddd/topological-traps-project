"""
Models Module - Neural Network Architecture and Training Components.

This module contains:
- U-Net model with ResNet34 encoder for viability prediction
- Loss functions (BCE, Dice, Combined)
- Evaluation metrics (IoU, Dice, Accuracy)
"""

from .unet import (
    MultiRobotViabilityUNet,
    create_model,
)

from .losses import (
    DiceLoss,
    DiceBCELoss,
    PerChannelDiceBCELoss,
    create_loss,
)

from .metrics import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    compute_precision,
    compute_recall,
    compute_per_channel_metrics,
    MetricTracker,
)

__all__ = [
    # Model
    "MultiRobotViabilityUNet",
    "create_model",
    # Losses
    "DiceLoss",
    "DiceBCELoss",
    "PerChannelDiceBCELoss",
    "create_loss",
    # Metrics
    "compute_iou",
    "compute_dice",
    "compute_pixel_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_per_channel_metrics",
    "MetricTracker",
]
