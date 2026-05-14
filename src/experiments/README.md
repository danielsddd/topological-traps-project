# src/experiments/

Extended dataset classes for the two new training experiments.
These datasets compute oracle labels on-the-fly during training,
avoiding the need to pre-generate labels for every velocity or angle.

---

## Experiment 1 — Velocity-Dependent Viability

**File**: `velocity_experiment.py`
**Class**: `VelocityViabilityDataset`

At each `__getitem__`, a velocity is sampled and the oracle is called
with a braking-distance-extended footprint:

```
d_brake = v² / (2 · a_max)   [pixels]
```

The translation kernel for direction D is extended by `d_brake` along
the travel axis. At `v=0` this produces identical output to the basic
oracle (verified by unit test).

The 4th input channel encodes velocity as `v / V_MAX` broadcast to (H, W).

```python
from src.experiments.velocity_experiment import VelocityViabilityDataset

ds = VelocityViabilityDataset(
    map_dir="data/processed",
    manifest_path="data/manifest.csv",
    robot_sizes=[(20,15), (30,20), (40,25)],
    split="train",
    velocities=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # m/s
    max_decel=2.0,   # m/s²
    px_per_m=10.0,
)
x, y, meta = ds[0]
# x.shape == (4, 512, 512)  [occ, L_norm, W_norm, v_norm]
# y.shape == (4, 512, 512)  [N, S, E, W] binary viability
```

---

## Experiment 2 — Escape Distance Cost Maps

**File**: `velocity_experiment.py`
**Class**: `VelocityCostMapDataset`

Same velocity-extended footprint, but the oracle computes a continuous
escape distance (BFS step count to nearest rotation-safe zone) rather
than binary viability. Targets are normalised to [0, 1].

```python
from src.experiments.velocity_experiment import VelocityCostMapDataset

ds = VelocityCostMapDataset(...)
x, y, meta = ds[0]
# x.shape == (4, 512, 512)
# y.shape == (4, 512, 512)  float32 in [0, 1]  (normalised escape cost)
```

---

## Velocity oracle utilities — `src/oracle/velocity_oracle.py`

| Function | Description |
|---|---|
| `braking_distance_px(v, a, px_per_m)` | Compute `v²/(2a)` in pixels. |
| `normalise_velocity(v)` | Map speed to [0, 1]: `v / V_MAX`. |
| `velocity_viability(occ, L, W, v, ...)` | Full velocity-aware oracle. Returns (4,H,W) uint8. |
| `velocity_escape_cost_map(occ, L, W, v, ...)` | Velocity-aware escape cost map. Returns (4,H,W) float32. |

`V_MAX = 3.0 m/s` is the normalisation ceiling. Speeds above this will
clip the normalised channel to 1.0; increase `V_MAX` and retrain if
higher speeds are needed.
