"""
Step 2 of Oracle: Translation-Safe Mask (per direction).

For each cardinal direction, erodes free space with the robot's oriented
rectangular footprint. Result = pixels where the robot physically fits
while traveling in that heading.

This is the critical fix over the naive approach that only checks rotation:
  - North/South motion: robot is L pixels tall, W pixels wide
  - East/West  motion: robot is W pixels tall, L pixels wide

Without this step, a corridor 1px wide would appear navigable because
every pixel individually is free — but a 30×20 robot cannot fit there.
"""
from __future__ import annotations
import numpy as np
import cv2

# (dy, dx) in image coords — y increases downward
DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "N": (-1,  0),
    "S": ( 1,  0),
    "E": ( 0,  1),
    "W": ( 0, -1),
}
DIRECTIONS = ["N", "S", "E", "W"]


def compute_translation_safe_mask(
    occupancy: np.ndarray,
    robot_length: int,
    robot_width: int,
    direction: str,
) -> np.ndarray:
    """
    Compute pixels where the robot physically fits while moving in `direction`.

    Kernel shapes (rows × cols):
      N/S → (robot_length, robot_width)   — robot is tall
      E/W → (robot_width,  robot_length)  — robot is wide

    Args:
        occupancy:    (H, W) uint8, 1=free, 0=obstacle.
        robot_length: Longer robot dimension (pixels).
        robot_width:  Shorter robot dimension (pixels).
        direction:    One of "N", "S", "E", "W".

    Returns:
        (H, W) uint8 mask. 1 = robot fits here at this heading.
    """
    if direction not in DIRECTION_VECTORS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got '{direction}'")

    if direction in ("N", "S"):
        kernel = np.ones((robot_length, robot_width), dtype=np.uint8)
    else:
        kernel = np.ones((robot_width, robot_length), dtype=np.uint8)

    eroded = cv2.erode(occupancy.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(np.uint8)


def compute_all_translation_masks(
    occupancy: np.ndarray,
    robot_length: int,
    robot_width: int,
) -> dict[str, np.ndarray]:
    """
    Compute translation-safe masks for all 4 directions.

    Returns:
        Dict: direction ("N"/"S"/"E"/"W") → (H, W) uint8 mask.
    """
    return {
        d: compute_translation_safe_mask(occupancy, robot_length, robot_width, d)
        for d in DIRECTIONS
    }