"""
Step 1 of Oracle: Rotation-Safe Mask.

Erodes free space with a circular kernel of diameter = robot diagonal.
Surviving pixels have enough clearance for the robot to rotate 360°.

Note on conservatism: using the full diagonal is a safe upper bound.
A robot that can translate slightly while turning needs less clearance.
For this project we use the conservative bound — documented as future work.
"""
from __future__ import annotations
import math
import numpy as np
import cv2


def compute_rotation_safe_mask(
    occupancy: np.ndarray,
    robot_length: int,
    robot_width: int,
) -> np.ndarray:
    """
    Compute pixels where the robot can rotate 360°.

    Uses morphological erosion with a circular kernel whose diameter
    equals the robot diagonal (worst-case rotation clearance).

    Args:
        occupancy:    (H, W) uint8, 1=free, 0=obstacle.
        robot_length: Robot longer dimension in pixels.
        robot_width:  Robot shorter dimension in pixels.

    Returns:
        (H, W) uint8 binary mask. 1 = robot can rotate here, 0 = cannot.
    """
    diagonal = math.sqrt(robot_length ** 2 + robot_width ** 2)
    radius   = math.ceil(diagonal / 2)
    diameter = 2 * radius + 1  # always odd

    kernel = _circular_kernel(diameter)
    eroded = cv2.erode(occupancy.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(np.uint8)


def _circular_kernel(diameter: int) -> np.ndarray:
    """Create a filled circular structuring element."""
    r = diameter // 2
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    return (x ** 2 + y ** 2 <= r ** 2).astype(np.uint8)


def visualize_rotation_safe(
    occupancy: np.ndarray,
    robot_sizes: list[tuple[int, int]],
    output_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Visualize rotation-safe regions for multiple robot sizes.

    Green  = rotation-safe (robot can rotate here)
    Red    = free but not rotation-safe (robot fits but cannot turn)
    Dark   = obstacle
    """
    import matplotlib.pyplot as plt

    n = len(robot_sizes)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    axes[0].imshow(occupancy, cmap="gray")
    axes[0].set_title("Occupancy")
    axes[0].axis("off")

    for i, (L, W) in enumerate(robot_sizes):
        mask = compute_rotation_safe_mask(occupancy, L, W)
        vis  = np.zeros((*occupancy.shape, 3), dtype=np.uint8)
        vis[occupancy == 0]                      = [50,  50,  50]   # obstacle
        vis[(occupancy == 1) & (mask == 0)]      = [200, 50,  50]   # trapped
        vis[mask == 1]                           = [50,  200, 50]   # safe

        diag = math.sqrt(L ** 2 + W ** 2)
        pct  = mask.mean() * 100
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(f"{L}×{W} (diag={diag:.0f}px)\n{pct:.1f}% safe")
        axes[i + 1].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()