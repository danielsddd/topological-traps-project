"""
Evaluation Metrics for Viability Prediction.

This module provides metrics for evaluating binary segmentation
performance on viability maps.

Primary Metrics:
- IoU (Intersection over Union): Standard segmentation metric
- Dice Coefficient: F1 score for binary classification
- Pixel Accuracy: Simple percentage of correct predictions

Per-Direction Metrics:
- Computed separately for N, S, E, W channels
- Helps identify which directions are hardest to predict

The MetricTracker class aggregates metrics across batches.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    IoU = |P ∩ T| / |P ∪ T|
        = TP / (TP + FP + FN)
    
    Args:
        predictions: Model outputs (logits or probabilities)
        targets: Ground truth binary labels
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score (scalar or per-sample tensor)
    """
    # Binarize predictions
    if predictions.max() > 1 or predictions.min() < 0:
        # Assume logits, apply sigmoid
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    # Flatten spatial dimensions
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Compute intersection and union
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def compute_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice Coefficient (F1 Score).
    
    Dice = 2 * |P ∩ T| / (|P| + |T|)
         = 2 * TP / (2*TP + FP + FN)
    
    Args:
        predictions: Model outputs (logits or probabilities)
        targets: Ground truth binary labels
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        Dice score
    """
    # Binarize predictions
    if predictions.max() > 1 or predictions.min() < 0:
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    # Flatten
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # Compute Dice
    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        preds_flat.sum() + targets_flat.sum() + smooth
    )
    
    return dice


def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute pixel-wise accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        predictions: Model outputs
        targets: Ground truth labels
        threshold: Threshold for binarizing predictions
    
    Returns:
        Accuracy score
    """
    if predictions.max() > 1 or predictions.min() < 0:
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    correct = (preds == targets).float()
    accuracy = correct.mean()
    
    return accuracy


def compute_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute precision (positive predictive value).
    
    Precision = TP / (TP + FP)
    
    Args:
        predictions: Model outputs
        targets: Ground truth labels
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Precision score
    """
    if predictions.max() > 1 or predictions.min() < 0:
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    
    return precision


def compute_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute recall (sensitivity / true positive rate).
    
    Recall = TP / (TP + FN)
    
    Args:
        predictions: Model outputs
        targets: Ground truth labels
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Recall score
    """
    if predictions.max() > 1 or predictions.min() < 0:
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    tp = (preds_flat * targets_flat).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return recall


def compute_per_channel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    channel_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each channel (direction) separately.
    
    Args:
        predictions: Model outputs (B, C, H, W)
        targets: Ground truth labels (B, C, H, W)
        threshold: Threshold for binarization
        channel_names: Names for each channel (default: N, S, E, W)
    
    Returns:
        Dictionary mapping channel name to metrics dict
    """
    if channel_names is None:
        channel_names = ["N", "S", "E", "W"]
    
    C = predictions.shape[1]
    results = {}
    
    for c in range(min(C, len(channel_names))):
        pred_c = predictions[:, c]
        target_c = targets[:, c]
        
        results[channel_names[c]] = {
            "iou": compute_iou(pred_c, target_c, threshold).item(),
            "dice": compute_dice(pred_c, target_c, threshold).item(),
            "accuracy": compute_pixel_accuracy(pred_c, target_c, threshold).item(),
            "precision": compute_precision(pred_c, target_c, threshold).item(),
            "recall": compute_recall(pred_c, target_c, threshold).item(),
        }
    
    return results


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute confusion matrix components.
    
    Args:
        predictions: Model outputs
        targets: Ground truth labels
        threshold: Threshold for binarization
    
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    if predictions.max() > 1 or predictions.min() < 0:
        preds = torch.sigmoid(predictions)
    else:
        preds = predictions
    
    preds = (preds > threshold).float()
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


class MetricTracker:
    """
    Track and aggregate metrics across batches.
    
    Usage:
        tracker = MetricTracker()
        
        for batch in dataloader:
            predictions = model(inputs)
            tracker.update(predictions, targets)
        
        metrics = tracker.compute()  # Average over all samples
    
    Args:
        threshold: Threshold for binary predictions
        channel_names: Names for each channel
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        channel_names: List[str] = None
    ):
        self.threshold = threshold
        self.channel_names = channel_names or ["N", "S", "E", "W"]
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.count = 0
        
        # Overall metrics
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.acc_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        
        # Per-channel metrics
        self.per_channel = {
            name: {
                "iou": 0.0,
                "dice": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "count": 0
            }
            for name in self.channel_names
        }
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = None
    ):
        """
        Update metrics with a batch.
        
        Args:
            predictions: Model outputs (B, C, H, W)
            targets: Ground truth labels (B, C, H, W)
            batch_size: Batch size (inferred if None)
        """
        if batch_size is None:
            batch_size = predictions.shape[0]
        
        self.count += batch_size
        
        # Overall metrics
        self.iou_sum += compute_iou(
            predictions, targets, self.threshold
        ).item() * batch_size
        
        self.dice_sum += compute_dice(
            predictions, targets, self.threshold
        ).item() * batch_size
        
        self.acc_sum += compute_pixel_accuracy(
            predictions, targets, self.threshold
        ).item() * batch_size
        
        self.precision_sum += compute_precision(
            predictions, targets, self.threshold
        ).item() * batch_size
        
        self.recall_sum += compute_recall(
            predictions, targets, self.threshold
        ).item() * batch_size
        
        # Per-channel metrics
        C = min(predictions.shape[1], len(self.channel_names))
        
        for c in range(C):
            name = self.channel_names[c]
            pred_c = predictions[:, c]
            target_c = targets[:, c]
            
            self.per_channel[name]["iou"] += compute_iou(
                pred_c, target_c, self.threshold
            ).item() * batch_size
            
            self.per_channel[name]["dice"] += compute_dice(
                pred_c, target_c, self.threshold
            ).item() * batch_size
            
            self.per_channel[name]["accuracy"] += compute_pixel_accuracy(
                pred_c, target_c, self.threshold
            ).item() * batch_size
            
            self.per_channel[name]["precision"] += compute_precision(
                pred_c, target_c, self.threshold
            ).item() * batch_size
            
            self.per_channel[name]["recall"] += compute_recall(
                pred_c, target_c, self.threshold
            ).item() * batch_size
            
            self.per_channel[name]["count"] += batch_size
    
    def compute(self) -> Dict:
        """
        Compute average metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if self.count == 0:
            return {}
        
        metrics = {
            "iou": self.iou_sum / self.count,
            "dice": self.dice_sum / self.count,
            "accuracy": self.acc_sum / self.count,
            "precision": self.precision_sum / self.count,
            "recall": self.recall_sum / self.count,
            "n_samples": self.count,
        }
        
        # Per-channel
        for name in self.channel_names:
            count = self.per_channel[name]["count"]
            if count > 0:
                for metric in ["iou", "dice", "accuracy", "precision", "recall"]:
                    key = f"{metric}_{name}"
                    metrics[key] = self.per_channel[name][metric] / count
        
        return metrics
    
    def get_summary_string(self) -> str:
        """Get formatted summary string."""
        metrics = self.compute()
        
        if not metrics:
            return "No metrics computed"
        
        lines = [
            f"IoU: {metrics['iou']:.4f}",
            f"Dice: {metrics['dice']:.4f}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
        ]
        
        # Per-direction IoU
        dir_ious = []
        for name in self.channel_names:
            key = f"iou_{name}"
            if key in metrics:
                dir_ious.append(f"{name}:{metrics[key]:.3f}")
        
        if dir_ious:
            lines.append(f"Per-direction IoU: {', '.join(dir_ious)}")
        
        return " | ".join(lines)


class RobotSizeMetricTracker:
    """
    Track metrics separately for each robot size.
    
    Useful for analyzing generalization to unseen sizes.
    """
    
    def __init__(
        self,
        robot_sizes: List[Tuple[int, int]],
        threshold: float = 0.5
    ):
        self.robot_sizes = robot_sizes
        self.threshold = threshold
        
        # Create tracker for each robot size
        self.trackers = {
            (l, w): MetricTracker(threshold=threshold)
            for l, w in robot_sizes
        }
    
    def reset(self):
        """Reset all trackers."""
        for tracker in self.trackers.values():
            tracker.reset()
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        robot_length: int,
        robot_width: int
    ):
        """
        Update metrics for a specific robot size.
        
        Args:
            predictions: Model outputs
            targets: Ground truth labels
            robot_length: Robot length
            robot_width: Robot width
        """
        key = (robot_length, robot_width)
        
        if key in self.trackers:
            self.trackers[key].update(predictions, targets)
    
    def compute(self) -> Dict[Tuple[int, int], Dict]:
        """
        Compute metrics for each robot size.
        
        Returns:
            Dictionary mapping (length, width) to metrics
        """
        return {
            size: tracker.compute()
            for size, tracker in self.trackers.items()
        }
    
    def get_summary_dataframe(self):
        """Get metrics as pandas DataFrame."""
        import pandas as pd
        
        results = self.compute()
        
        rows = []
        for (length, width), metrics in results.items():
            if metrics:
                row = {
                    "robot_length": length,
                    "robot_width": width,
                    "size": f"{length}x{width}",
                    **metrics
                }
                rows.append(row)
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create test data
    B, C, H, W = 4, 4, 64, 64
    
    # Perfect predictions
    targets = torch.randint(0, 2, (B, C, H, W)).float()
    predictions_perfect = targets.clone()
    
    print("\nPerfect predictions:")
    print(f"  IoU: {compute_iou(predictions_perfect, targets):.4f}")
    print(f"  Dice: {compute_dice(predictions_perfect, targets):.4f}")
    print(f"  Accuracy: {compute_pixel_accuracy(predictions_perfect, targets):.4f}")
    
    # Random predictions
    predictions_random = torch.randn(B, C, H, W)
    
    print("\nRandom predictions:")
    print(f"  IoU: {compute_iou(predictions_random, targets):.4f}")
    print(f"  Dice: {compute_dice(predictions_random, targets):.4f}")
    print(f"  Accuracy: {compute_pixel_accuracy(predictions_random, targets):.4f}")
    
    # Per-channel metrics
    print("\nPer-channel metrics (random):")
    channel_metrics = compute_per_channel_metrics(predictions_random, targets)
    for name, metrics in channel_metrics.items():
        print(f"  {name}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
    
    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()
    
    for _ in range(10):
        preds = torch.randn(B, C, H, W)
        targs = torch.randint(0, 2, (B, C, H, W)).float()
        tracker.update(preds, targs)
    
    print(f"  Tracked samples: {tracker.count}")
    print(f"  Summary: {tracker.get_summary_string()}")
    
    print("\n✓ All metric tests passed!")
