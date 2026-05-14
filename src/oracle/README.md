# src/oracle/

Ground-truth viability label generation for non-holonomic robots.

Given an occupancy grid and robot dimensions, the oracle computes a
4-channel binary label: for each pixel and each cardinal direction
(North, South, East, West), is the robot able to escape to a
rotation-safe zone by translating in that direction?

---

## Algorithm

```
Input:  occupancy grid (H×W), robot length L, robot width W

Step 1 — Rotation check
        Erode occupancy with a disc of radius ceil(max(L,W) / 2).
        Result: rotation_safe mask — pixels where the robot can
                rotate freely in place.

Step 2 — Translation check (per direction)
        For each cardinal direction D ∈ {N, S, E, W}:
          Erode occupancy with the robot's axis-aligned bounding box
          (L×W or W×L depending on heading).
          Result: translation_safe[D] — pixels where the robot body
                  fits when moving in direction D.

Step 3 — Directional BFS (per direction)
        For each D:
          Seeds  = pixels that are both rotation_safe AND translation_safe[D]
          BFS    = reverse flood-fill through translation_safe[D] from seeds
          Result: label[D] = 1 if pixel can reach a seed, 0 otherwise

Output: (4, H, W) uint8 array — channels [N, S, E, W]
```

---

## Files

| File | Description |
|---|---|
| `rotation_check.py` | `compute_rotation_safe_mask(occ, L, W)` — morphological erosion with a disc kernel. Returns (H, W) uint8. |
| `translation_check.py` | `compute_translation_safe_mask(occ, L, W, direction)` — axis-aligned bounding-box erosion. Returns (H, W) uint8. |
| `directional_viability.py` | `generate_labels_for_map(occ, L, W)` — runs the full 3-step pipeline. Returns (4, H, W) uint8. Also exposes `compute_viability_single_direction()` for per-direction computation. |
| `extended_oracles.py` | Additional oracle types: `escape_cost_map()` (BFS cost), `continuous_angle_viability()` (arbitrary heading via rotation). |
| `velocity_oracle.py` | `velocity_viability(occ, L, W, velocity, max_decel, px_per_m)` — extends translation kernel by braking distance `v²/(2a)`. At `v=0` produces identical output to the basic oracle. |
| `generator.py` | `generate_labels_batch()` — parallel label generation across maps using `multiprocessing`. Skips maps that already have labels. |

---

## Quick usage

```python
import numpy as np
from src.oracle.directional_viability import generate_labels_for_map
from src.oracle.velocity_oracle import velocity_viability

# Load occupancy grid (1 = free, 0 = obstacle)
occ = np.load("data/processed/some_map.npy").astype(np.uint8)

# Basic oracle (4 cardinal directions)
labels = generate_labels_for_map(occ, robot_length=30, robot_width=20)
# labels.shape == (4, 512, 512), dtype uint8
# labels[0] = North, [1] = South, [2] = East, [3] = West

# Velocity-dependent oracle
labels_fast = velocity_viability(
    occ, L=30, W=20,
    velocity=2.0,      # m/s
    max_decel=2.0,     # m/s²
    px_per_m=10.0,     # map resolution
)
```

---

## Performance

On a 512×512 map with robot 30×20:
- Full oracle (4 directions): ~220 ms
- Single direction: ~55 ms
- Parallelised across 16 CPU cores: ~20 ms effective throughput

The oracle is CPU-only. GPU acceleration is not used for label generation;
the bottleneck is morphological erosion and BFS, which are memory-bound.

---

## Extension: continuous angles

`extended_oracles.py` supports arbitrary heading angles by rotating the
map so the heading aligns with East, running the East-direction oracle,
and rotating back. This is the ground truth used by the
`continuous_angle` training mode.

```python
from src.oracle.extended_oracles import continuous_angle_viability

label = continuous_angle_viability(occ, L=30, W=20, angle_deg=37.0)
# label.shape == (512, 512), dtype uint8
```
