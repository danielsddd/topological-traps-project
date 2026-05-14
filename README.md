# Directional Topological Traps

**TAU Algorithmic Robotics — Final Project**

A deep learning system that predicts heading-dependent viability maps for
non-holonomic robot navigation, with a custom Trap-Aware PRM planner that uses
the trained model to avoid topological traps during path planning.

---

## Problem

Non-holonomic robots (cars, forklifts, AMRs) cannot move in arbitrary directions.
In cluttered environments this creates **topological traps**: regions the robot can
enter but cannot escape due to its turning-radius constraint.

Detecting traps analytically requires morphological erosion followed by directional
BFS — accurate but slow (200–400 ms per map). This project trains a U-Net to predict
the same viability maps in **~14 ms** (up to 20× speedup), enabling real-time
trap-aware planning.

---

## Method

### Oracle (ground truth label generation)
1. **Rotation check** — morphological erosion with a disc of radius `max(L, W) / 2`
   identifies pixels where the robot can rotate freely in place.
2. **Translation check** — oriented bounding-box erosion per heading identifies
   pixels where the robot body fits when translating North / South / East / West.
3. **Directional BFS** — reverse flood-fill from rotation-safe seeds through
   translation-safe pixels labels each free pixel as *viable* or *trapped*
   per cardinal direction.

### Neural Network
- **Architecture**: U-Net with pretrained ResNet34 encoder (~24 M parameters)
- **Input**: 3 channels — occupancy grid, normalised robot length, normalised robot width
- **Output**: 4 channels — binary viability map [North, South, East, West]
- **Loss**: Combined BCE + Dice
- **Training**: Mixed precision (FP16), AdamW, cosine LR schedule, early stopping

### Training Modes (`--oracle_type`)

| Mode | Input channels | Output | Task |
|---|---|---|---|
| `basic` | 3 (occ, L, W) | 4-ch binary | Cardinal-direction viability |
| `continuous_angle` | 5 (+ sin θ, cos θ) | 1-ch binary | Arbitrary heading angle |
| `cost_map` | 3 (occ, L, W) | 4-ch float | Escape distance regression |
| `velocity` | 4 (+ v_norm) | 4-ch binary | Velocity-dependent viability |

---

## Key Results

### Model accuracy vs naive baseline

| Model | Val IoU | Naive baseline | Gap over naive | Speedup vs Oracle |
|---|---|---|---|---|
| Binary viability (4-dir) | 0.9779 | 0.713 | +0.265 | 16× |
| Continuous-angle | 0.9838 | 0.713 | +0.271 | 19× |
| Velocity-aware | 0.9954 | 0.713 | +0.282 | 16× |
| Escape-distance (MAE) | 10.7 px | — | Pearson r = 0.679 | 21× |

*Naive baseline = predict all free space as viable. 28.7% of free pixels are
genuinely trapped, making this a non-trivial task.*

### Velocity experiment — viable area shrinks with speed

| Speed (m/s) | Viable% | IoU |
|---|---|---|
| 0.0 | 38.8% | 0.991 |
| 1.5 | 37.0% | 0.992 |
| 3.0 | 35.2% | 0.991 |

### Warehouse PRM benchmark (67% trap density)

| Planner | Trap rate | Reduction vs Standard |
|---|---|---|
| Standard PRM | 0.673 | — |
| TrapAwarePRM (Oracle) | 0.123 | 81.8% |
| TrapAwarePRM (NN) | 0.125 | **81.4%** |

The NN matches Oracle trap-avoidance quality at 13.6× lower labelling cost.

---

## Stack

| Component | Tool |
|---|---|
| Language | Python 3.10 |
| Deep learning | PyTorch 2.x, segmentation-models-pytorch |
| Data processing | NumPy, OpenCV, SciPy |
| PRM planner | Pure NumPy (StandardPRM + TrapAwarePRM) |
| Experiment tracking | Weights & Biases |
| Compute | GeForce 2080 (8 GB) / Titan XP (12 GB) |
| Conda environment | `traps` |

---

## Setup

```bash
conda activate traps
pip install -r requirements.txt
```

All paths are controlled by `configs/config.yaml` and `configs/.env`.
No hardcoded paths exist anywhere in the codebase.

---

## Directory Structure

```
project2/
├── configs/
│   ├── config.yaml          # Model, training, and data hyperparameters
│   └── .env                 # Environment paths
├── data/
│   ├── processed/           # 512×512 occupancy grids (.npy)
│   ├── labels/              # Oracle viability labels, one dir per robot size
│   │   ├── robot_20x15/
│   │   ├── robot_30x20/
│   │   ├── robot_40x25/
│   │   └── robot_25x18/     # OOD test size — never seen during training
│   └── manifest.csv         # Map-level train / val / test split
├── src/                     # All library code (importable modules)
│   ├── oracle/              # Ground-truth label generation
│   ├── models/              # U-Net, losses, metrics
│   ├── data/                # Dataset classes and augmentations
│   ├── training/            # Trainer, checkpointing, logging
│   ├── evaluation/          # Evaluator, generalization, speed benchmark
│   ├── integration/         # TrapAwarePRM planner (pure NumPy)
│   ├── experiments/         # Extended oracle datasets (velocity, cost map)
│   └── visualization/       # Plotting utilities
├── scripts/                 # All runnable entry points
├── outputs/                 # Checkpoints, figures, JSON results (gitignored)
├── logs/                    # Training logs, TensorBoard events
├── RESULTS.md               # Full experimental results with figures
└── generate_readmes.py      # This script
```

---

## Quick Start

### 1 — Generate oracle labels
```bash
python scripts/local/04_generate_labels.py --config configs/config.yaml
```

### 2 — Train
```bash
# Basic 4-direction binary model
python scripts/train.py --config configs/config.yaml --oracle_type basic

# Velocity-aware model
python scripts/train.py --config configs/config.yaml --oracle_type velocity --epochs 50

# Cost-map regression model
python scripts/train.py --config configs/config.yaml --oracle_type cost_map --epochs 30

# Continuous-angle model
python scripts/train.py --config configs/config.yaml --oracle_type continuous_angle --epochs 30
```

### 3 — Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint outputs/viability_<run>/checkpoints/best_iou.pth \
    --config configs/config.yaml

python scripts/evaluate_velocity.py \
    --checkpoint outputs/viability_velocity_<run>/checkpoints/best_iou.pth \
    --velocities 0.0 0.5 1.0 1.5 2.0 2.5 3.0 \
    --robot-size 30 20

python scripts/evaluate_cost_map.py \
    --checkpoint outputs/viability_cost_map_<run>/checkpoints/last.pth \
    --config configs/config.yaml
```

### 4 — PRM benchmark
```bash
# On test maps from the dataset
python scripts/benchmark_prm.py --num-maps 8 --num-samples 500

# On a synthetic warehouse map
python scripts/benchmark_prm_hard.py \
    --shelves 6 --aisles 10 --dead-ends 8 \
    --num-samples 600 --runs 5
```

### 5 — Demo
```bash
python scripts/demo_angle_sweep.py \
    --checkpoint outputs/viability_continuous_angle_<run>/checkpoints/best_iou.pth \
    --n-angles 24 \
    --out outputs/closed_loop_demo/demo_angle_sweep.gif
```

---

## Checkpointing & Resuming

Every training script saves two checkpoints:
- `best_iou.pth` — best validation metric seen so far
- `last.pth` — end of most recent epoch

To resume training from where it left off:
```bash
python scripts/train.py --config configs/config.yaml \
    --oracle_type basic \
    --resume outputs/viability_<run>/checkpoints/last.pth
```

---

## Results

Full metrics, training curves, and figures: [RESULTS.md](RESULTS.md)
