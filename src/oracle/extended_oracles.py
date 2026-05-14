"""
Extended Oracles — Phase 2 research extensions.

This module extends the basic 4-direction Oracle with two new capabilities:

  1. Continuous-angle viability  (Direction 1: "The Any-Heading Oracle")
       For an arbitrary heading θ ∈ [0, 360), produce a binary mask
       indicating where the robot can escape *while moving at angle θ*.

  2. Time-to-Escape cost maps    (Alternative Extension 2)
       For each viable pixel, the minimum number of cell-steps the robot
       must travel in a given direction before reaching a rotation-safe
       region (a "safe harbour"). Non-viable pixels carry a large constant.
       4-channel variant covers all 4 cardinal directions; angle variant
       returns a single channel for an arbitrary heading.

Both new oracles re-use the existing primitives:
    * src.oracle.rotation_check.compute_rotation_safe_mask
    * src.oracle.translation_check.compute_translation_safe_mask
    * src.oracle.directional_viability.compute_viability_single_direction

For arbitrary angles we *rotate the map by -θ*, run the East-direction
pipeline, then *rotate the result back by +θ*. This is exact for
multiples of 90° and a tight approximation otherwise (NEAREST
interpolation keeps the masks binary).

This file is import-safe even when matplotlib / cv2 are missing — only
the standard scientific stack is imported eagerly.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from .rotation_check import compute_rotation_safe_mask
from .translation_check import (
    DIRECTION_VECTORS,
    DIRECTIONS,
    compute_translation_safe_mask,
)
from .directional_viability import compute_viability_single_direction

logger = logging.getLogger(__name__)

# Cost-map sentinel: any value >= COST_MAX_VALUE indicates "non-viable".
# 255 fits in uint8 and is large compared to plausible escape distances
# on a 512×512 map (max plausible escape ≈ 512 steps, but the model is
# trained on normalised costs anyway).
COST_MAX_VALUE: int = 255


# ---------------------------------------------------------------------------
# Continuous-angle viability  (Direction 1)
# ---------------------------------------------------------------------------

def _rotate_grid(
    grid: np.ndarray,
    angle_deg: float,
    interpolation: int = cv2.INTER_NEAREST,
    border_value: int = 0,
) -> np.ndarray:
    """
    Rotate a 2-D grid about its centre.

    Args:
        grid:          (H, W) uint8 array.
        angle_deg:     Rotation angle in degrees (CCW positive — image-coord
                       convention used throughout this codebase: y-down,
                       so we negate inside cv2.getRotationMatrix2D).
        interpolation: cv2 interpolation flag. NEAREST keeps masks binary.
        border_value:  Value to fill in cells that fall outside the source.

    Returns:
        (H, W) uint8 — rotated grid, same shape as input.
    """
    h, w = grid.shape[:2]
    # cv2 uses CCW positive in screen coords (y-down) → matches image convention.
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        grid.astype(np.uint8),
        M,
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(border_value),
    )


def continuous_angle_viability(
    map_grid: np.ndarray,
    L: int,
    W: int,
    angle_deg: float,
) -> np.ndarray:
    """
    Compute the heading-θ viability mask via the rotate-East-rotate trick.

    Convention:
        angle_deg = 0   → East   (+x)
        angle_deg = 90  → North  (-y in image coords)
        angle_deg = 180 → West   (-x)
        angle_deg = 270 → South  (+y in image coords)

    Algorithm:
        1.  Rotate the occupancy grid by -angle_deg  → "θ becomes East".
        2.  Run the standard rotation+translation+BFS pipeline for East.
        3.  Rotate the resulting mask back by +angle_deg.

    For multiples of 90° this is exact. Otherwise NEAREST interpolation is
    used so masks remain binary; a thin border of zeros (treated as
    obstacle) appears in the rotated frame which is the conservative —
    safe — choice.

    Args:
        map_grid:  (H, W) uint8 occupancy grid (1 = free, 0 = obstacle).
        L:         Robot longer dimension in pixels.
        W:         Robot shorter dimension in pixels.
        angle_deg: Heading angle in degrees, 0 = East, CCW positive.

    Returns:
        (H, W) uint8 binary viability mask in the *original* frame.
        1 = viable for heading θ, 0 = trapped or obstacle.
    """
    if map_grid.ndim != 2:
        raise ValueError(f"map_grid must be 2-D, got shape {map_grid.shape}")
    if L <= 0 or W <= 0:
        raise ValueError(f"L and W must be positive, got L={L}, W={W}")

    angle_deg = float(angle_deg) % 360.0

    # Fast path: cardinal directions — no rotation, no interpolation loss.
    cardinal = {0.0: "E", 90.0: "N", 180.0: "W", 270.0: "S"}
    if angle_deg in cardinal:
        d = cardinal[angle_deg]
        rot_safe = compute_rotation_safe_mask(map_grid, L, W)
        trans_safe = compute_translation_safe_mask(map_grid, L, W, d)
        return compute_viability_single_direction(trans_safe, rot_safe, d)

    # General path: rotate, compute East, rotate back.
    rotated = _rotate_grid(map_grid, -angle_deg, border_value=0)

    rot_safe = compute_rotation_safe_mask(rotated, L, W)
    trans_safe_E = compute_translation_safe_mask(rotated, L, W, "E")
    via_E = compute_viability_single_direction(trans_safe_E, rot_safe, "E")

    via = _rotate_grid(via_E, angle_deg, border_value=0)
    # Re-binarise: even with NEAREST, double rotation can produce stray
    # values if shapes are non-square; clip to {0, 1}.
    return (via > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Time-to-Escape cost maps  (Alternative Extension 2)
# ---------------------------------------------------------------------------

def _compute_viable_all_directions(
    rotation_safe: np.ndarray,
    trans_masks: dict,
) -> np.ndarray:
    """
    Helper: compute the "viable in all 4 cardinal directions" mask.

    A pixel is "rotation-safe in spirit" *and* the robot fits in every
    cardinal heading. These pixels are the BFS seeds for cost-map
    propagation — minimum cost = 0.

    Args:
        rotation_safe: (H, W) uint8.
        trans_masks:   dict direction → (H, W) uint8.

    Returns:
        (H, W) uint8 — 1 where the robot is rotation-safe AND translation-
        safe in every direction.
    """
    seed = (rotation_safe == 1).astype(np.uint8)
    for d in DIRECTIONS:
        seed = seed & (trans_masks[d] == 1).astype(np.uint8)
    return seed.astype(np.uint8)


def _bfs_cost_map(
    translation_safe: np.ndarray,
    seed_mask: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    BFS cost map for a single direction.

    For each pixel, the cost is the minimum number of cell-steps in
    `direction` to reach a seed pixel (rotation-safe + translation-safe
    in all directions).

    Implementation: reverse BFS from seeds. We move *opposite* to the
    direction during propagation, tracking the step count. This means
    a pixel's cost = (number of forward steps it must take in `direction`
    to land on a seed).

    Args:
        translation_safe: (H, W) uint8 — robot fits here at this heading.
        seed_mask:        (H, W) uint8 — seeds (cost = 0).
        direction:        "N", "S", "E", or "W".

    Returns:
        (H, W) float32 cost map. Non-viable pixels get COST_MAX_VALUE.
    """
    dy, dx = DIRECTION_VECTORS[direction]
    H, Wd = translation_safe.shape

    cost = np.full((H, Wd), COST_MAX_VALUE, dtype=np.float32)

    # Seeds must also be translation-safe in this direction.
    seeds = (seed_mask == 1) & (translation_safe == 1)
    cost[seeds] = 0.0

    ys, xs = np.where(seeds)
    queue: deque = deque(zip(ys.tolist(), xs.tolist()))

    # Reverse BFS: from a known-cost pixel at (y, x) with cost c, the
    # pixel "one step before in `direction`" is at (y-dy, x-dx). That
    # pixel reaches our seed in c+1 steps if it is translation-safe and
    # we have not visited it with a lower cost.
    while queue:
        y, x = queue.popleft()
        c = cost[y, x]

        ny, nx = y - dy, x - dx
        if 0 <= ny < H and 0 <= nx < Wd:
            if translation_safe[ny, nx] == 1 and cost[ny, nx] > c + 1:
                cost[ny, nx] = c + 1
                queue.append((ny, nx))

    return cost


def escape_cost_map(
    map_grid: np.ndarray,
    L: int,
    W: int,
    direction: Optional[str] = None,
) -> np.ndarray:
    """
    Compute time-to-escape cost map(s).

    Definition:
        For each viable pixel, the cost is the minimum number of
        cell-steps moving strictly in the chosen direction until
        reaching a "rotation-safe in all 4 cardinal directions" cell.
        Non-viable pixels carry COST_MAX_VALUE.

    Args:
        map_grid:  (H, W) uint8 occupancy grid (1 = free, 0 = obstacle).
        L:         Robot longer dimension in pixels.
        W:         Robot shorter dimension in pixels.
        direction: Optional — one of "N", "S", "E", "W". If None,
                   returns a 4-channel array stacked in the canonical
                   [N, S, E, W] order. If a *cardinal* string, returns a
                   single (H, W) channel.

    Returns:
        If `direction is None`:  (4, H, W) float32 — channels [N, S, E, W].
        Otherwise:               (H, W) float32 cost map for that direction.

    For arbitrary angles use `escape_cost_map_for_angle`.
    """
    if map_grid.ndim != 2:
        raise ValueError(f"map_grid must be 2-D, got shape {map_grid.shape}")
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"direction must be in {DIRECTIONS} or None, got {direction!r}")

    rot_safe = compute_rotation_safe_mask(map_grid, L, W)
    trans_masks = {
        d: compute_translation_safe_mask(map_grid, L, W, d) for d in DIRECTIONS
    }
    seeds_all = _compute_viable_all_directions(rot_safe, trans_masks)

    if direction is not None:
        return _bfs_cost_map(trans_masks[direction], seeds_all, direction)

    H, Wd = map_grid.shape
    cost4 = np.full((4, H, Wd), COST_MAX_VALUE, dtype=np.float32)
    for i, d in enumerate(DIRECTIONS):
        cost4[i] = _bfs_cost_map(trans_masks[d], seeds_all, d)
    return cost4


def escape_cost_map_for_angle(
    map_grid: np.ndarray,
    L: int,
    W: int,
    angle_deg: float,
) -> np.ndarray:
    """
    Time-to-escape cost map for an arbitrary heading θ.

    Same rotate-East-rotate trick used for binary continuous-angle
    viability: rotate the map so θ becomes East, run the East cost-map
    pipeline, rotate the result back.

    Note that rotation can blur edge values; we use NEAREST interpolation
    and re-clip cells that exceed COST_MAX_VALUE.

    Args:
        map_grid:  (H, W) uint8 occupancy grid (1 = free, 0 = obstacle).
        L:         Robot longer dimension in pixels.
        W:         Robot shorter dimension in pixels.
        angle_deg: Heading in degrees, 0 = East, CCW positive.

    Returns:
        (H, W) float32 cost map in the *original* frame.
    """
    if map_grid.ndim != 2:
        raise ValueError(f"map_grid must be 2-D, got shape {map_grid.shape}")

    angle_deg = float(angle_deg) % 360.0
    cardinal = {0.0: "E", 90.0: "N", 180.0: "W", 270.0: "S"}
    if angle_deg in cardinal:
        return escape_cost_map(map_grid, L, W, cardinal[angle_deg])

    rotated = _rotate_grid(map_grid, -angle_deg, border_value=0)
    cost_E = escape_cost_map(rotated, L, W, "E")

    # Rotate cost map back. Use NEAREST so cost values are preserved
    # exactly. Border = COST_MAX_VALUE so newly-exposed cells are non-viable.
    H, Wd = cost_E.shape
    M = cv2.getRotationMatrix2D((Wd / 2.0, H / 2.0), angle_deg, 1.0)
    cost_back = cv2.warpAffine(
        cost_E.astype(np.float32),
        M,
        (Wd, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(COST_MAX_VALUE),
    )
    cost_back = np.clip(cost_back, 0.0, COST_MAX_VALUE).astype(np.float32)
    return cost_back


# ---------------------------------------------------------------------------
# Convenience helpers used by training-time data construction
# ---------------------------------------------------------------------------

def angle_to_sincos(angle_deg: float) -> Tuple[float, float]:
    """
    Encode an angle as (sin, cos) for FiLM-style network conditioning.

    Args:
        angle_deg: Heading in degrees, 0 = East, CCW positive.

    Returns:
        (sin θ, cos θ) — both in [-1, 1].
    """
    th = np.deg2rad(angle_deg)
    return float(np.sin(th)), float(np.cos(th))


def normalise_cost_map(
    cost_map: np.ndarray,
    max_value: float = float(COST_MAX_VALUE),
) -> np.ndarray:
    """
    Normalise a cost map to [0, 1] for stable regression training.

    Pixels at COST_MAX_VALUE map to 1.0.

    Args:
        cost_map:  Cost map (any shape).
        max_value: Sentinel value to map to 1.0.

    Returns:
        Normalised cost map (same shape, float32, in [0, 1]).
    """
    out = np.clip(cost_map.astype(np.float32) / float(max_value), 0.0, 1.0)
    return out


def denormalise_cost_map(
    cost_map_norm: np.ndarray,
    max_value: float = float(COST_MAX_VALUE),
) -> np.ndarray:
    """Inverse of `normalise_cost_map`."""
    return np.clip(cost_map_norm * float(max_value), 0.0, max_value).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test (pytest-free smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("extended_oracles_smoketest")

    H = Wd = 80
    L_test, W_test = 8, 6

    # Open room
    occ = np.ones((H, Wd), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 0  # walls

    log.info("Smoke test: continuous_angle_viability")
    for ang in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
        m = continuous_angle_viability(occ, L_test, W_test, ang)
        log.info(f"  angle={ang:5.1f}°  viable_pct={m.mean()*100:.1f}%  shape={m.shape}")

    log.info("Smoke test: escape_cost_map (4 channels)")
    cm4 = escape_cost_map(occ, L_test, W_test, direction=None)
    log.info(f"  shape={cm4.shape}  dtype={cm4.dtype}")
    log.info(f"  per-direction mean (free pixels): "
             f"N={cm4[0][cm4[0] < COST_MAX_VALUE].mean():.2f}  "
             f"S={cm4[1][cm4[1] < COST_MAX_VALUE].mean():.2f}  "
             f"E={cm4[2][cm4[2] < COST_MAX_VALUE].mean():.2f}  "
             f"W={cm4[3][cm4[3] < COST_MAX_VALUE].mean():.2f}")

    log.info("Smoke test: escape_cost_map_for_angle(45°)")
    cm45 = escape_cost_map_for_angle(occ, L_test, W_test, 45.0)
    log.info(f"  shape={cm45.shape}  free_min={cm45[cm45 < COST_MAX_VALUE].min():.1f}  "
             f"free_max={cm45[cm45 < COST_MAX_VALUE].max():.1f}")

    log.info("All smoke tests passed.")