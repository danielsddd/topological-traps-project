"""
Velocity-Dependent Viability Oracle — The Momentum Trap.

Extends the basic 4-direction Oracle with a kinodynamic braking-distance
model.  At speed v with maximum deceleration a_max, the robot needs

    d_brake = v² / (2 · a_max)          [metres]

extra clearance *ahead* of its physical footprint to guarantee a safe
stop.  This distance is converted to pixels and appended to the
translation-erosion kernel along the axis of travel.

The effect:
    • At v ≈ 0 the output matches the basic Oracle exactly.
    • As v grows, viable space shrinks — corridors that are safe at
      walking speed become "momentum traps" at running speed.

Physics mapping (configurable):
    pixels_per_meter   : scale factor from map pixels to physical metres.
                         Default 10 px/m  (1 px = 10 cm on a 512 px map
                         covering ~50 m).
    max_deceleration   : a_max in m/s².  Default 2.0  (gentle AMR braking).

Usage:
    from src.oracle.velocity_oracle import velocity_viability

    # Returns (4, H, W) uint8 — same format as generate_labels_for_map()
    labels = velocity_viability(occupancy, L=30, W=20, velocity=1.5,
                                max_decel=2.0, px_per_m=10.0)
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Core: braking-distance computation
# ---------------------------------------------------------------------------

def braking_distance_px(
    velocity: float,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> int:
    """
    Compute braking distance in pixels.

    d_brake = v² / (2 · a_max)   [metres]  →  round up to integer pixels.

    Args:
        velocity:   Robot speed in m/s  (≥ 0).
        max_decel:  Maximum deceleration in m/s² (> 0).
        px_per_m:   Pixels per metre.

    Returns:
        Braking distance in whole pixels (≥ 0).
    """
    if velocity < 0:
        raise ValueError(f"velocity must be ≥ 0, got {velocity}")
    if max_decel <= 0:
        raise ValueError(f"max_decel must be > 0, got {max_decel}")
    d_m = (velocity ** 2) / (2.0 * max_decel)
    return int(math.ceil(d_m * px_per_m))


# ---------------------------------------------------------------------------
# Velocity-aware translation erosion
# ---------------------------------------------------------------------------

def _velocity_translation_safe_mask(
    occupancy: np.ndarray,
    robot_L: int,
    robot_W: int,
    direction: str,
    d_brake_px: int,
) -> np.ndarray:
    """
    C-space erosion with braking-distance extension.

    The kernel is the robot's oriented bounding box **plus** d_brake_px
    pixels appended along the travel axis (the "braking footprint").

    Direction → kernel mapping (rows × cols):
        N / S :  robot oriented vertically  →  (L + d_brake, W)
        E / W :  robot oriented horizontally →  (W, L + d_brake)

    The braking extension is always added to the dimension aligned with
    the direction of travel.

    Args:
        occupancy:   (H, W) uint8 — 1 = free, 0 = obstacle.
        robot_L:     Robot longer dimension in pixels.
        robot_W:     Robot shorter dimension in pixels.
        direction:   "N", "S", "E", or "W".
        d_brake_px:  Braking distance in pixels.

    Returns:
        (H, W) uint8 — 1 where the velocity-extended robot fits.
    """
    if direction in ("N", "S"):
        # Robot oriented vertically: L is rows, W is cols.
        # Travel is along rows → extend rows by braking distance.
        k_rows = robot_L + d_brake_px
        k_cols = robot_W
    else:
        # Robot oriented horizontally: W is rows, L is cols.
        # Travel is along cols → extend cols by braking distance.
        k_rows = robot_W
        k_cols = robot_L + d_brake_px

    # Ensure kernel is at least 1×1
    k_rows = max(k_rows, 1)
    k_cols = max(k_cols, 1)

    kernel = np.ones((k_rows, k_cols), dtype=np.uint8)
    eroded = cv2.erode(occupancy.astype(np.uint8), kernel)
    return eroded.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full velocity-dependent viability pipeline
# ---------------------------------------------------------------------------

def velocity_viability(
    map_grid: np.ndarray,
    L: int,
    W: int,
    velocity: float,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> np.ndarray:
    """
    Full velocity-aware viability pipeline for one (map, robot, speed).

    Steps:
        1.  Rotation-safe mask          — unchanged (robot can still rotate
            when stationary, so the circular erosion uses physical dims).
        2.  Velocity-extended translation masks  — per direction, the erosion
            kernel grows by d_brake along the travel axis.
        3.  Directional reverse BFS     — seeded from the intersection of
            rotation-safe and velocity-extended translation-safe pixels.

    At v = 0 this returns exactly the same result as the basic Oracle
    (d_brake = 0 → kernel = physical footprint only).

    Args:
        map_grid:   (H, W) uint8 occupancy grid (1 = free, 0 = obstacle).
        L:          Robot longer dimension (pixels).
        W:          Robot shorter dimension (pixels).
        velocity:   Robot speed in m/s.
        max_decel:  Maximum deceleration in m/s².
        px_per_m:   Pixels per metre for the map.

    Returns:
        (4, H, W) uint8 — channels [N, S, E, W].
        1 = viable at this speed, 0 = momentum trap or obstacle.
    """
    if map_grid.ndim != 2:
        raise ValueError(f"map_grid must be 2-D, got shape {map_grid.shape}")

    d_brake = braking_distance_px(velocity, max_decel, px_per_m)
    logger.debug(
        "velocity=%.2f m/s  max_decel=%.2f  d_brake=%d px  (L=%d W=%d)",
        velocity, max_decel, d_brake, L, W,
    )

    # Step 1: rotation-safe (unchanged — robot can rotate when stopped)
    rotation_safe = compute_rotation_safe_mask(map_grid, L, W)

    # Step 2: velocity-extended translation masks per direction
    vel_trans_masks = {
        d: _velocity_translation_safe_mask(map_grid, L, W, d, d_brake)
        for d in DIRECTIONS
    }

    # Step 3: directional reverse BFS
    H, Wd = map_grid.shape
    labels = np.zeros((4, H, Wd), dtype=np.uint8)
    for i, d in enumerate(DIRECTIONS):
        labels[i] = compute_viability_single_direction(
            vel_trans_masks[d], rotation_safe, d,
        )

    return labels


def velocity_viability_all_speeds(
    map_grid: np.ndarray,
    L: int,
    W: int,
    velocities: List[float],
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> Dict[float, np.ndarray]:
    """
    Compute viability for multiple speeds on one (map, robot_size).

    Args:
        map_grid:    (H, W) uint8 occupancy.
        L, W:        Robot dimensions.
        velocities:  List of speeds in m/s.
        max_decel:   Max deceleration in m/s².
        px_per_m:    Scale factor.

    Returns:
        Dict mapping velocity → (4, H, W) uint8 viability.
    """
    return {
        v: velocity_viability(map_grid, L, W, v, max_decel, px_per_m)
        for v in velocities
    }


# ---------------------------------------------------------------------------
# Velocity-aware escape cost map (combines Experiment 1 + 2)
# ---------------------------------------------------------------------------

def velocity_escape_cost_map(
    map_grid: np.ndarray,
    L: int,
    W: int,
    velocity: float,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> np.ndarray:
    """
    Escape cost map with velocity-dependent robot footprint.

    Same BFS cost logic as extended_oracles.escape_cost_map, but the
    translation kernels include braking distance.

    Args:
        map_grid:  (H, W) uint8 occupancy.
        L, W:      Robot dimensions (pixels).
        velocity:  Speed in m/s.
        max_decel: Max deceleration in m/s².
        px_per_m:  Pixels per metre.

    Returns:
        (4, H, W) float32 — escape cost per direction.
        Non-viable pixels get 255.0.
    """
    from .extended_oracles import _bfs_cost_map, COST_MAX_VALUE

    d_brake = braking_distance_px(velocity, max_decel, px_per_m)
    rotation_safe = compute_rotation_safe_mask(map_grid, L, W)

    vel_trans_masks = {
        d: _velocity_translation_safe_mask(map_grid, L, W, d, d_brake)
        for d in DIRECTIONS
    }

    # Seeds: rotation-safe AND translation-safe in ALL directions (at this speed)
    seed = (rotation_safe == 1).astype(np.uint8)
    for d in DIRECTIONS:
        seed = seed & (vel_trans_masks[d] == 1).astype(np.uint8)

    H, Wd = map_grid.shape
    cost4 = np.full((4, H, Wd), COST_MAX_VALUE, dtype=np.float32)
    for i, d in enumerate(DIRECTIONS):
        cost4[i] = _bfs_cost_map(vel_trans_masks[d], seed, d)

    return cost4


# ---------------------------------------------------------------------------
# Normalisation helpers for the NN input velocity channel
# ---------------------------------------------------------------------------

# Default velocity range for training.  max_velocity defines the
# normalisation ceiling: v_norm = v / V_MAX.
# src/oracle/velocity_oracle.py
DEFAULT_VELOCITIES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
V_MAX: float = 6.0


def normalise_velocity(v: float, v_max: float = V_MAX) -> float:
    """Map velocity to [0, 1] for the NN input channel."""
    if v > v_max:
        logger.warning(
            "velocity %.2f exceeds V_MAX=%.2f — input will be clipped to 1.0. "
            "Increase V_MAX in velocity_oracle.py and retrain if you need "
            "higher speeds.",
            v, v_max,
        )
    return min(v / v_max, 1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("velocity_oracle_smoketest")

    H = Wd = 80
    L_test, W_test = 8, 6

    # Open room with walls
    occ = np.ones((H, Wd), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 0

    log.info("=== Velocity Oracle smoke test ===")
    for v in [0.0, 0.5, 1.0, 2.0, 3.0]:
        labels = velocity_viability(occ, L_test, W_test, v, max_decel=2.0, px_per_m=10.0)
        d_brake = braking_distance_px(v, 2.0, 10.0)
        pct = labels.mean() * 100
        log.info(f"  v={v:.1f} m/s  d_brake={d_brake:3d} px  viable_pct={pct:.1f}%")

    # Verify: v=0 matches basic oracle
    from .directional_viability import generate_labels_for_map
    basic = generate_labels_for_map(occ, L_test, W_test)
    vel0  = velocity_viability(occ, L_test, W_test, 0.0)
    assert np.array_equal(basic, vel0), "v=0 must match basic oracle!"
    log.info("  ✓ v=0 matches basic oracle exactly.")

    # Monotonicity: viable space should shrink with speed
    prev_pct = 100.0
    for v in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        labels = velocity_viability(occ, L_test, W_test, v, 2.0, 10.0)
        cur_pct = labels.mean() * 100
        assert cur_pct <= prev_pct + 0.01, (
            f"Monotonicity violated: v={v} has {cur_pct:.1f}% > {prev_pct:.1f}%"
        )
        prev_pct = cur_pct
    log.info("  ✓ Viable space monotonically decreases with speed.")

    log.info("All smoke tests passed.")