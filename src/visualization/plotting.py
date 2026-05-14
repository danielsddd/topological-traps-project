"""
Visualization and Plotting Functions.

This module provides publication-quality visualization tools for:
- Training curves
- Predictions vs ground truth
- Directional viability maps
- Robot size comparisons
- Error analysis

All functions support saving to file and/or displaying interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


# Default colors
COLORS = {
    "viable": [50, 200, 50],      # Green
    "trap": [200, 50, 50],        # Red
    "obstacle": [50, 50, 50],     # Dark gray
    "free": [255, 255, 255],      # White
}

DIRECTION_NAMES = ["North", "South", "East", "West"]
DIRECTION_SHORT = ["N", "S", "E", "W"]


def setup_plotting_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_training_curves(
    history: Dict,
    output_path: Optional[str] = None,
    show: bool = False,
    title: str = "Training History"
) -> None:
    """
    Plot training curves (loss and metrics).
    
    Args:
        history: Dictionary with training history
                 Keys: train_loss, val_loss, val_iou, val_dice, learning_rate
        output_path: Path to save figure
        show: Whether to display figure
        title: Figure title
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], 'b-', label='Train', linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU curve
    ax = axes[0, 1]
    if "val_iou" in history:
        ax.plot(epochs, history["val_iou"], 'g-', label='Val IoU', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Validation IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dice curve
    ax = axes[1, 0]
    if "val_dice" in history:
        ax.plot(epochs, history["val_dice"], 'm-', label='Val Dice', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    if "learning_rate" in history:
        ax.plot(epochs, history["learning_rate"], 'k-', linewidth=2)
        ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_predictions(
    occupancy: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    direction: int = 0,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None
) -> None:
    """
    Plot prediction comparison: Ground Truth vs Prediction vs Error.
    
    Args:
        occupancy: Occupancy grid (H, W)
        ground_truth: Ground truth labels (4, H, W)
        predictions: Model predictions (4, H, W) - probabilities
        direction: Direction to visualize (0=N, 1=S, 2=E, 3=W)
        threshold: Threshold for binary predictions
        output_path: Path to save figure
        show: Whether to display
        title: Figure title
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    dir_name = DIRECTION_NAMES[direction]
    gt = ground_truth[direction]
    pred = predictions[direction]
    pred_binary = (pred > threshold).astype(np.uint8)
    
    # Original map
    axes[0].imshow(occupancy, cmap='gray')
    axes[0].set_title("Occupancy Map")
    axes[0].axis('off')
    
    # Ground truth
    gt_vis = create_viability_visualization(occupancy, gt)
    axes[1].imshow(gt_vis)
    axes[1].set_title(f"Ground Truth (Viable-{dir_name})")
    axes[1].axis('off')
    
    # Prediction
    pred_vis = create_viability_visualization(occupancy, pred_binary)
    axes[2].imshow(pred_vis)
    
    # Compute metrics
    from ..models.metrics import compute_iou, compute_dice
    import torch
    iou = compute_iou(
        torch.from_numpy(pred_binary).float().unsqueeze(0),
        torch.from_numpy(gt).float().unsqueeze(0),
        threshold=0.5
    ).item()
    
    axes[2].set_title(f"Prediction (IoU: {iou:.3f})")
    axes[2].axis('off')
    
    # Error map
    error_vis = create_error_visualization(occupancy, gt, pred_binary)
    axes[3].imshow(error_vis)
    axes[3].set_title("Errors (FP=Blue, FN=Red)")
    axes[3].axis('off')
    
    if title is None:
        title = f"Viability Prediction - {dir_name}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def create_viability_visualization(
    occupancy: np.ndarray,
    viability: np.ndarray
) -> np.ndarray:
    """
    Create RGB visualization of viability map.
    
    Args:
        occupancy: Occupancy grid (H, W)
        viability: Viability mask (H, W) - binary
    
    Returns:
        RGB image (H, W, 3)
    """
    H, W = occupancy.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Obstacles: dark gray
    vis[occupancy == 0] = COLORS["obstacle"]
    
    # Free but not viable: red (trap)
    trap = (occupancy == 1) & (viability == 0)
    vis[trap] = COLORS["trap"]
    
    # Viable: green
    vis[viability == 1] = COLORS["viable"]
    
    return vis


def create_error_visualization(
    occupancy: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray
) -> np.ndarray:
    """
    Create error visualization showing false positives and negatives.
    
    Args:
        occupancy: Occupancy grid (H, W)
        ground_truth: Ground truth (H, W)
        prediction: Prediction (H, W)
    
    Returns:
        RGB image showing errors
    """
    H, W = occupancy.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Background
    vis[occupancy == 0] = COLORS["obstacle"]
    vis[(occupancy == 1) & (ground_truth == prediction)] = [200, 200, 200]  # Correct
    
    # False positives (predicted viable but actually trap): blue
    fp = (prediction == 1) & (ground_truth == 0) & (occupancy == 1)
    vis[fp] = [50, 50, 200]
    
    # False negatives (predicted trap but actually viable): red
    fn = (prediction == 0) & (ground_truth == 1) & (occupancy == 1)
    vis[fn] = [200, 50, 50]
    
    return vis


def plot_per_direction_viability(
    occupancy: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None
) -> None:
    """
    Plot viability for all 4 directions in a 2x2 grid.
    
    Args:
        occupancy: Occupancy grid (H, W)
        labels: Ground truth labels (4, H, W)
        predictions: Model predictions (4, H, W) - optional
        threshold: Threshold for predictions
        output_path: Path to save figure
        show: Whether to display
        title: Figure title
    """
    setup_plotting_style()
    
    if predictions is not None:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
    
    for d in range(4):
        if predictions is not None:
            # Ground truth row
            ax = axes[0, d]
            vis_gt = create_viability_visualization(occupancy, labels[d])
            ax.imshow(vis_gt)
            ax.set_title(f"GT: Viable-{DIRECTION_NAMES[d]}")
            ax.axis('off')
            
            # Prediction row
            ax = axes[1, d]
            pred_binary = (predictions[d] > threshold).astype(np.uint8)
            vis_pred = create_viability_visualization(occupancy, pred_binary)
            ax.imshow(vis_pred)
            
            # Compute IoU
            from ..models.metrics import compute_iou
            import torch
            iou = compute_iou(
                torch.from_numpy(pred_binary).float().unsqueeze(0),
                torch.from_numpy(labels[d]).float().unsqueeze(0)
            ).item()
            
            ax.set_title(f"Pred: {DIRECTION_NAMES[d]} (IoU: {iou:.3f})")
            ax.axis('off')
        else:
            ax = axes[d]
            vis = create_viability_visualization(occupancy, labels[d])
            ax.imshow(vis)
            ax.set_title(f"Viable-{DIRECTION_NAMES[d]}")
            ax.axis('off')
    
    if title is None:
        title = "Directional Viability"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_viability_comparison(
    occupancy: np.ndarray,
    labels_list: List[np.ndarray],
    robot_sizes: List[Tuple[int, int]],
    direction: int = 0,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Compare viability maps across different robot sizes.
    
    Args:
        occupancy: Occupancy grid (H, W)
        labels_list: List of label arrays (4, H, W), one per robot size
        robot_sizes: List of (length, width) tuples
        direction: Direction to visualize
        output_path: Path to save figure
        show: Whether to display
    """
    setup_plotting_style()
    
    n = len(robot_sizes) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    # Original map
    axes[0].imshow(occupancy, cmap='gray')
    axes[0].set_title("Occupancy Map")
    axes[0].axis('off')
    
    # Each robot size
    for i, ((length, width), labels) in enumerate(zip(robot_sizes, labels_list)):
        vis = create_viability_visualization(occupancy, labels[direction])
        axes[i + 1].imshow(vis)
        
        viable_ratio = labels[direction].sum() / occupancy.sum() * 100
        axes[i + 1].set_title(f"{length}×{width}\n{viable_ratio:.1f}% viable")
        axes[i + 1].axis('off')
    
    plt.suptitle(f"Viable-{DIRECTION_NAMES[direction]} by Robot Size", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_robot_size_comparison(
    metrics_by_size: Dict[Tuple[int, int], Dict],
    metric_name: str = "iou",
    train_sizes: List[Tuple[int, int]] = None,
    test_sizes: List[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot metrics comparison across robot sizes.
    
    Args:
        metrics_by_size: Dictionary mapping (length, width) to metrics
        metric_name: Which metric to plot
        train_sizes: Sizes used during training (shown differently)
        test_sizes: Sizes held out for testing
        output_path: Path to save figure
        show: Whether to display
    """
    setup_plotting_style()
    
    # Sort sizes by diagonal
    import math
    sizes = sorted(metrics_by_size.keys(), key=lambda s: math.sqrt(s[0]**2 + s[1]**2))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_labels = [f"{l}×{w}" for l, w in sizes]
    values = [metrics_by_size[s].get(metric_name, 0) for s in sizes]
    x = range(len(sizes))
    
    # Color bars by train/test
    colors = []
    for size in sizes:
        if train_sizes and size in train_sizes:
            colors.append('steelblue')
        elif test_sizes and size in test_sizes:
            colors.append('coral')
        else:
            colors.append('gray')
    
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Robot Size")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{metric_name.upper()} by Robot Size")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Legend
    if train_sizes or test_sizes:
        legend_elements = []
        if train_sizes:
            legend_elements.append(mpatches.Patch(color='steelblue', label='Seen (Train)'))
        if test_sizes:
            legend_elements.append(mpatches.Patch(color='coral', label='Unseen (Test)'))
        ax.legend(handles=legend_elements)
    
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def create_figure_grid(
    images: List[np.ndarray],
    titles: List[str],
    ncols: int = 4,
    figsize_per_image: Tuple[float, float] = (4, 4),
    output_path: Optional[str] = None,
    show: bool = False,
    suptitle: Optional[str] = None
) -> None:
    """
    Create a grid of images.
    
    Args:
        images: List of images (each H, W or H, W, 3)
        titles: List of titles for each image
        ncols: Number of columns
        figsize_per_image: Size per image (width, height)
        output_path: Path to save figure
        show: Whether to display
        suptitle: Overall title
    """
    setup_plotting_style()
    
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    axes = np.array(axes).flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        ax.set_title(title)
        ax.axis('off')
    
    # Hide empty axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")
    
    # Create dummy data
    occupancy = np.ones((100, 100), dtype=np.uint8)
    occupancy[:20, :] = 0
    occupancy[-20:, :] = 0
    occupancy[:, :20] = 0
    occupancy[:, -20:] = 0
    
    labels = np.random.randint(0, 2, (4, 100, 100)).astype(np.uint8)
    predictions = np.random.rand(4, 100, 100).astype(np.float32)
    
    # Test viability visualization
    vis = create_viability_visualization(occupancy, labels[0])
    print(f"Viability visualization shape: {vis.shape}")
    
    # Test error visualization
    error_vis = create_error_visualization(occupancy, labels[0], (predictions[0] > 0.5).astype(np.uint8))
    print(f"Error visualization shape: {error_vis.shape}")
    
    # Test training curves (with dummy history)
    history = {
        "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
        "val_loss": [1.1, 0.9, 0.7, 0.6, 0.5],
        "val_iou": [0.3, 0.5, 0.6, 0.7, 0.75],
        "val_dice": [0.4, 0.55, 0.65, 0.72, 0.78],
        "learning_rate": [1e-4, 1e-4, 5e-5, 5e-5, 1e-5],
    }
    
    print("Plotting training curves...")
    plot_training_curves(history, show=False)
    
    print("\n✓ All visualization tests passed!")
