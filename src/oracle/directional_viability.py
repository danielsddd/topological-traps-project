"""
Step 3 of Oracle: Reverse BFS Flood-Fill.

A pixel is viable for direction D if, moving only in direction D through
the translation-safe space for D, the robot eventually reaches a pixel
that is ALSO rotation-safe (a "safe harbour").

Algorithm (Reverse BFS):
  Seed:       pixels that are translation-safe-D AND rotation-safe.
  Propagate:  if (y,x) is viable, mark (y-dy, x-dx) as viable,
              provided (y-dy, x-dx) is translation-safe-D.
  Terminate:  when queue is empty.

Correctness:
  - Only translation-safe pixels can be in the path → robot never clips walls.
  - Only seeds from the rotation-safe intersection → guarantees escape.
  - Reverse propagation is O(H×W) — each pixel visited at most once.

Note on rotation-safe islands:
  If a rotation-safe cluster is itself enclosed by corridors the robot
  cannot enter without first turning, the BFS will correctly NOT propagate
  into those corridors (they are not translation-safe). This means some
  rotation-safe pixels may themselves be unreachable — which is correct.
"""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Dict

import numpy as np

from .translation_check import DIRECTION_VECTORS, DIRECTIONS


DIR_INDEX = {d: i for i, d in enumerate(DIRECTIONS)}


def compute_viability_single_direction(
    translation_safe: np.ndarray,
    rotation_safe: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    Compute viability mask for one direction via reverse BFS.

    Args:
        translation_safe: (H,W) uint8 — robot fits here at this heading.
        rotation_safe:    (H,W) uint8 — robot can rotate here.
        direction:        "N", "S", "E", or "W".

    Returns:
        (H,W) uint8. 1=viable (can escape), 0=trapped or obstacle.
    """
    dy, dx = DIRECTION_VECTORS[direction]
    H, W   = translation_safe.shape

    viability = np.zeros((H, W), dtype=np.uint8)

    # Seed: must be both translation-safe and rotation-safe
    seed_mask        = (translation_safe == 1) & (rotation_safe == 1)
    viability[seed_mask] = 1

    # Use deque(zip()) — avoids intermediate Python list (RAM-efficient)
    ys, xs = np.where(seed_mask)
    queue: deque = deque(zip(ys.tolist(), xs.tolist()))

    while queue:
        y, x   = queue.popleft()
        ny, nx = y - dy, x - dx  # reverse step

        if 0 <= ny < H and 0 <= nx < W:
            if translation_safe[ny, nx] == 1 and viability[ny, nx] == 0:
                viability[ny, nx] = 1
                queue.append((ny, nx))

    return viability


def compute_viability_all_directions(
    rotation_safe: np.ndarray,
    trans_masks: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute 4-channel viability label for one (map, robot_size) pair.

    Args:
        rotation_safe: (H,W) rotation-safe mask (Step 1 output).
        trans_masks:   Dict direction→(H,W) translation masks (Step 2 output).

    Returns:
        (4, H, W) uint8 array — channels are [N, S, E, W].
    """
    H, W   = rotation_safe.shape
    labels = np.zeros((4, H, W), dtype=np.uint8)
    for i, d in enumerate(DIRECTIONS):
        labels[i] = compute_viability_single_direction(trans_masks[d], rotation_safe, d)
    return labels


def generate_labels_for_map(
    occupancy: np.ndarray,
    robot_length: int,
    robot_width: int,
) -> np.ndarray:
    """
    Full Oracle pipeline for one (map, robot_size) pair.

    Steps:
        1. Rotation-safe mask  (circular erosion)
        2. Translation masks   (rectangular erosion × 4 directions)
        3. Directional BFS     (reverse BFS × 4 directions)

    Args:
        occupancy:    (H,W) uint8 occupancy grid.
        robot_length: Robot longer dimension (pixels).
        robot_width:  Robot shorter dimension (pixels).

    Returns:
        (4, H, W) uint8 label array — channels [N, S, E, W].
        Values: 1=viable, 0=trapped or obstacle.
    """
    from .rotation_check    import compute_rotation_safe_mask
    from .translation_check import compute_all_translation_masks

    rotation_safe = compute_rotation_safe_mask(occupancy, robot_length, robot_width)
    trans_masks   = compute_all_translation_masks(occupancy, robot_length, robot_width)
    labels        = compute_viability_all_directions(rotation_safe, trans_masks)
    return labels


def generate_labels_all_robots(
    occupancy: np.ndarray,
    robot_sizes: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Generate labels for multiple robot sizes on a single map.

    Args:
        occupancy:   (H,W) occupancy grid.
        robot_sizes: List of (length, width) tuples.

    Returns:
        Dict mapping (length, width) → (4, H, W) label array.
    """
    return {
        (L, W): generate_labels_for_map(occupancy, L, W)
        for L, W in robot_sizes
    }
def compute_viability_naive(
    translation_safe: np.ndarray,
    rotation_safe: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    TRULY NAIVE O((H×W)²) viability — for benchmarking only.

    For each free pixel, runs an independent forward BFS in `direction`.
    No caching, no path reuse — every pixel is treated independently.
    This is the worst-case approach that justifies the reverse BFS.

    WARNING: Do NOT use on full 512×512 maps.
    """
    dy, dx    = DIRECTION_VECTORS[direction]
    H, W      = translation_safe.shape
    viability = np.zeros((H, W), dtype=np.uint8)

    free_ys, free_xs = np.where(translation_safe == 1)

    for start_y, start_x in zip(free_ys.tolist(), free_xs.tolist()):
        # Independent BFS from this pixel — no reuse of previous results
        visited                    = np.zeros((H, W), dtype=bool)
        visited[start_y, start_x] = True
        queue                      = deque([(start_y, start_x)])
        found                      = False

        while queue and not found:
            y, x   = queue.popleft()
            if rotation_safe[y, x] == 1:
                found = True
                break
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if translation_safe[ny, nx] == 1 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        if found:
            viability[start_y, start_x] = 1  # mark ONLY the start pixel

    return viability