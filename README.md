# Directional Topological Traps

A deep learning system that predicts heading-dependent viability maps for
non-holonomic robot navigation.  A rectangular robot that enters a narrow
corridor from the wrong heading can become permanently trapped — it cannot
rotate to escape.  This project learns to predict those traps in real time
and uses the predictions to guide a sampling-based motion planner away from
dangerous regions.

**Course:** Algorithmic Robotics and Motion Planning — Fall 2025/2026 — Dan Halperin, TAU  
**Author:** Daniel Simanovsky  
**Submission deadline:** Sunday, March 22, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Data Pipeline](#data-pipeline)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [PRM Benchmark — TrapAwarePRM vs StandardPRM](#prm-benchmark--trapaware-prm-vs-standard-prm)
10. [Generating Report Figures](#generating-report-figures)
11. [SLURM Quick Reference](#slurm-quick-reference)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### Problem

A rectangular non-holonomic robot (e.g. a forklift) moving through an indoor
environment can enter regions where its heading makes escape impossible — it
physically cannot rotate or translate out.  Classical planners discover these
*topological traps* only after expensive collision checking, wasting samples
and computation.

### Approach

1. **Oracle algorithm** — morphological erosion (rotation check) + directional
   BFS flood-fill (escape viability) computes ground-truth binary viability
   maps for 4 cardinal headings (N / S / E / W).  Accurate but slow (~2 s
   per 512×512 map).

2. **U-Net** — learns to predict the Oracle's output from the occupancy grid
   and normalised robot dimensions.  Fast (~5 ms per map on GPU).

3. **TrapAwarePRM** — a pure-NumPy PRM planner that uses the learned viability
   map to reject low-viability samples and penalise trap-crossing edges,
   reducing trap encounters by ~80% compared to uniform sampling.

### Key numbers

| Item | Value |
|---|---|
| Architecture | U-Net, ResNet-34 encoder (ImageNet pretrained) |
| Input | 3 channels — occupancy grid + normalised robot L, W |
| Output | 4 channels — binary viability for N, S, E, W |
| Parameters | ~24 M |
| Training data | 800 HouseExpo maps × 3 robot sizes |
| Generalisation test | 1 held-out robot size |
| Inference | ~5 ms / map on GPU (mixed precision, FP16) |

---

## Repository Structure

```
project2/
├── configs/
│   ├── .env                          # All paths (single source of truth)
│   ├── config.yaml                   # Hyperparameters, model, data settings
│   └── config_schema.py              # Dataclass schema for config.yaml
│
├── data/
│   ├── raw_maps/                     # HouseExpo JSON files
│   ├── processed/                    # 512×512 occupancy grids (.npy)
│   ├── labels/                       # Oracle viability labels
│   │   ├── robot_6x4/               # Small   (TRAIN)
│   │   ├── robot_10x6/              # Medium  (TRAIN)
│   │   ├── robot_14x9/              # Large   (TRAIN)
│   │   └── robot_18x11/             # X-Large (TEST-ONLY — generalisation)
│   └── manifest.csv                  # Map-level train / val / test split
│
├── src/
│   ├── oracle/                       # Erosion + BFS viability oracle
│   ├── models/                       # U-Net architecture, losses, metrics
│   ├── data/                         # Dataset, DataLoader, augmentations
│   ├── training/                     # Trainer, callbacks, checkpointing
│   ├── evaluation/                   # Evaluator, generalisation, speed bench
│   ├── integration/                  # Pure-NumPy PRM planners
│   │   ├── prm.py                   # StandardPRM + TrapAwarePRM
│   │   └── README.md
│   ├── experiments/                  # Extended oracles (angle, cost-map, velocity)
│   ├── visualization/                # Training curves, prediction viewer, figures
│   └── utils/                        # Device helpers, verify_setup, misc
│
├── scripts/
│   ├── local/                        # Data pipeline scripts (run locally / CPU)
│   │   ├── 01_explore_dataset.py
│   │   ├── 02_preprocess_maps.py
│   │   ├── 03_create_manifest.py
│   │   ├── 04_generate_labels.py
│   │   └── 05_verify_labels.py
│   ├── train.py                      # Main training entry point
│   ├── evaluate.py                   # Main evaluation
│   ├── benchmark_prm.py             # TrapAwarePRM vs StandardPRM on test maps
│   ├── benchmark_prm_hard.py        # Warehouse (synthetic) benchmark
│   ├── demo_angle_sweep.py          # Animated heading-sweep demo
│   └── slurm/
│       ├── _header.sh               # Shared SLURM preamble
│       ├── preprocess.sh            # CPU job: map preprocessing
│       ├── oracle.sh                # CPU job: oracle label generation
│       └── train.sh                 # GPU job: model training
│
├── checkpoints/                      # Saved model weights (best + last)
├── logs/                             # SLURM logs, TensorBoard, GPU utilisation
├── outputs/
│   ├── figures/                      # All generated figures
│   └── results/                      # JSON evaluation results
├── requirements.txt
└── README.md                         # ← you are here
```

---

## Prerequisites

- **Python 3.10**
- **CUDA 11.8+** compatible GPU (tested on GeForce 2080 8 GB / Titan XP 12 GB)
- **Conda** (Miniconda or Anaconda)
- **Git**
- (TAU cluster) SSH access to SLURM via university VPN

---

## Installation

### Option A — TAU SLURM cluster

```bash
# 1. Connect via VPN, then SSH to the SLURM client
ssh <TAU_USERNAME>@kilonova.cs.tau.ac.il
ssh -o ServerAliveInterval=60 <TAU_USERNAME>@slurm-client.cs.tau.ac.il
tmux new -s traps
bash

# 2. Set base path (adjust to your allocation)
export BASE_DIR="<YOUR_STORAGE_PATH>"
export PROJECT_DIR="$BASE_DIR/project2"

# 3. Clone repository
mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR"
git clone <REPO_URL> .

# 4. Create conda environment
source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda create -n traps python=3.10 -y
conda activate traps

# 5. Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 6. Install project dependencies
pip install -r requirements.txt

# 7. Create directory structure
mkdir -p data/{raw_maps,processed}
mkdir -p data/labels/{robot_6x4,robot_10x6,robot_14x9,robot_18x11}
mkdir -p logs/{slurm/{train,oracle,preprocess},tensorboard,gpu_usage}
mkdir -p checkpoints
mkdir -p outputs/{figures,results,visualizations}

# 8. Set up persistent cache (keeps ImageNet weights across jobs)
mkdir -p "$BASE_DIR/.cache/torch"
export TORCH_HOME="$BASE_DIR/.cache/torch"
export HF_HOME="$BASE_DIR/.cache"

# 9. Verify installation
python -c "
import torch, cv2, numpy, yaml, segmentation_models_pytorch as smp
print(f'PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')
print(f'OpenCV  {cv2.__version__}    |  SMP: OK')
print('ALL OK')
"
```

### Option B — Local machine / generic Linux

```bash
git clone <REPO_URL>
cd topological-traps

conda create -n traps python=3.10 -y
conda activate traps
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw_maps,processed}
mkdir -p data/labels/{robot_6x4,robot_10x6,robot_14x9,robot_18x11}
mkdir -p logs checkpoints outputs/{figures,results}
```

### requirements.txt

```
segmentation-models-pytorch==0.3.4
opencv-python-headless
numpy
scipy
pandas
matplotlib
seaborn
tqdm
pyyaml
tensorboard
albumentations
```

### Verify setup

A built-in verification utility checks packages, CUDA, directories, and the
config file in one shot:

```bash
python -c "from src.utils import verify_setup; verify_setup()"
```

---

## Configuration

All configuration is centralised in two files.

### `configs/.env` — paths (sourced by every SLURM script)

```bash
export BASE_DIR="<YOUR_STORAGE_PATH>"
export PROJECT_DIR="$BASE_DIR/project2"
export CONDA_SH="$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
export CONDA_ENV="traps"
export TORCH_HOME="$BASE_DIR/.cache/torch"
export HF_HOME="$BASE_DIR/.cache"
export TOKENIZERS_PARALLELISM="false"

export RAW_MAPS_DIR="$PROJECT_DIR/data/raw_maps"
export PROCESSED_DIR="$PROJECT_DIR/data/processed"
export LABELS_DIR="$PROJECT_DIR/data/labels"
export MANIFEST_PATH="$PROJECT_DIR/data/manifest.csv"
export CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
export LOG_DIR="$PROJECT_DIR/logs"
export OUTPUT_DIR="$PROJECT_DIR/outputs"
export FIGURES_DIR="$PROJECT_DIR/outputs/figures"
export RESULTS_DIR="$PROJECT_DIR/outputs/results"
```

If you run on a different machine, edit **only** this file.

### `configs/config.yaml` — hyperparameters

Contains sections for `data`, `robot`, `oracle`, `model`, `training`,
`evaluation`, and `logging`.  Every script loads it via:

```python
from src.config import load_config
cfg = load_config()              # reads configs/config.yaml
cfg = load_config("other.yaml")  # or an explicit path
```

Environment variables in `.env` override the corresponding `paths:` entries
in `config.yaml`, so the YAML can stay generic while `.env` pins the
machine-specific paths.

---

## Data Pipeline

Run these steps **in order**.  Each script is idempotent — re-running skips
files that already exist.

### Step 1 — Explore raw data (optional)

```bash
python scripts/local/01_explore_dataset.py --config configs/config.yaml
```

Prints statistics about the HouseExpo JSON files in `data/raw_maps/`.

### Step 2 — Preprocess maps

Convert HouseExpo JSON floor plans to 512×512 binary occupancy grids (`.npy`).

```bash
# Locally
python scripts/local/02_preprocess_maps.py --config configs/config.yaml

# On SLURM
sbatch scripts/slurm/preprocess.sh
```

Output: `data/processed/<map_id>.npy` — one file per map.

### Step 3 — Create train / val / test manifest

```bash
python scripts/local/03_create_manifest.py --config configs/config.yaml
```

Output: `data/manifest.csv` with columns `map_id, split` (train / val / test).
The split is at map level so no map leaks across splits.

### Step 4 — Generate Oracle labels

The Oracle applies morphological erosion (rotation check) then directional
BFS flood-fill (escape viability) for each of the 4 cardinal headings.
Parallelised across maps and directions via multiprocessing.

```bash
# Locally (slow — one map at a time)
python scripts/local/04_generate_labels.py --config configs/config.yaml

# On SLURM (recommended — 16 CPUs)
sbatch scripts/slurm/oracle.sh

# Quick test with 10 maps
python scripts/local/04_generate_labels.py --config configs/config.yaml --max-maps 10

# Force recompute (ignores existing labels)
python scripts/local/04_generate_labels.py --config configs/config.yaml --force

# Verify existing labels only (no generation)
python scripts/local/04_generate_labels.py --config configs/config.yaml --verify-only
```

Output: `data/labels/robot_<L>x<W>/<map_id>.npy` — shape `(4, 512, 512)`,
channels ordered N, S, E, W.  Values are binary (0 = trapped, 1 = viable).

### Step 5 — Visual sanity check ("eye test")

```bash
python scripts/local/05_verify_labels.py --config configs/config.yaml
```

Renders a grid of (occupancy map, oracle labels) pairs so you can verify the
labels look correct before committing to a training run.

---

## Training

### Basic training (4-direction binary viability)

```bash
# Local / interactive
python scripts/train.py --config configs/config.yaml

# Resume from checkpoint after preemption
python scripts/train.py --config configs/config.yaml \
    --resume checkpoints/last.pth

# On SLURM
sbatch scripts/slurm/train.sh
```

The training script uses mixed precision (FP16), saves `last.pth` every epoch
and `best_iou.pth` on validation improvement, and supports graceful resume
after SLURM preemption.

### Alternative oracle types

```bash
# Continuous-angle viability (5-ch input, 1-ch output per sampled angle)
python scripts/train.py --config configs/config.yaml \
    --oracle_type continuous_angle --epochs 30

# Time-to-escape cost regression (3-ch input, 4-ch float output)
python scripts/train.py --config configs/config.yaml \
    --oracle_type cost_map --epochs 30

# Velocity-aware viability (4-ch input, 4-ch output per speed)
python scripts/train.py --config configs/config.yaml \
    --oracle_type velocity --epochs 30
```

### Key training flags

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to config YAML | `configs/config.yaml` |
| `--resume` | Path to checkpoint to resume from | None |
| `--oracle_type` | `basic`, `continuous_angle`, `cost_map`, `velocity` | `basic` |
| `--epochs` | Override number of epochs | from config |
| `--batch_size` | Override batch size | from config |
| `--lr` | Override learning rate | from config |
| `--experiment_name` | Run name for logging | auto-generated |

---

## Evaluation

### Full evaluation

```bash
python scripts/evaluate.py --config configs/config.yaml \
    --checkpoint checkpoints/best_iou.pth
```

Reports per-direction IoU, Dice score, pixel accuracy, and inference speed
compared to the Oracle.  Results are saved to
`outputs/results/evaluation.json`.

### Generalisation test

The evaluation automatically tests on the held-out robot size
(configured under `robot.test_only_sizes` in `config.yaml`) and compares
metrics against the training sizes.

### Speed benchmark

Included in the evaluation script.  Compares Oracle wall-clock time
(erosion + BFS, single-threaded) vs model inference time (GPU, mixed
precision).  Typical result: **~400× speedup** on a single 512×512 map.

---

## PRM Benchmark — TrapAwarePRM vs StandardPRM

The PRM planners are implemented entirely in **pure NumPy + SciPy**
(`src/integration/prm.py`).  No external motion planning library is required.

### How it works

**StandardPRM** samples uniformly from free space, connects k-nearest
neighbours with straight-line edges (collision checked via pixel
rasterisation), and queries the shortest path with Dijkstra.

**TrapAwarePRM** extends StandardPRM with viability-guided hybrid sampling:
- 85% of nodes are *viability-filtered* — rejected if the NN viability score
  is below a threshold in all 4 directions.
- 15% of nodes are *unconditional uniform* — ensures connectivity even in
  highly trap-dense regions.
- Edges crossing low-viability pixels receive a penalty multiplier (default
  5×) on their Euclidean length.

The local planner is holonomic (straight-line).  Extending to Dubins curves
is documented as future work.

### Running the benchmark

```bash
# On test maps from the dataset
python scripts/benchmark_prm.py \
    --checkpoint checkpoints/best_iou.pth \
    --num-maps 8 \
    --num-samples 500

# On a synthetic warehouse with shelves, aisles, and dead-end alcoves
python scripts/benchmark_prm_hard.py \
    --checkpoint checkpoints/best_iou.pth

# Quick sanity check (3 maps, 300 samples)
python scripts/benchmark_prm.py --num-maps 3 --num-samples 300
```

### Key benchmark flags

| Flag | Description | Default |
|---|---|---|
| `--checkpoint` | Path to trained model | auto-detect latest |
| `--num-maps` | Number of test maps | 8 |
| `--num-samples` | PRM roadmap nodes per planner | 500 |
| `--k-nn` | K nearest neighbours | 10 |
| `--viability-threshold` | Min viability to accept a sample | 0.5 |
| `--trap-penalty` | Edge weight penalty for trap regions | 5.0 |
| `--uniform-ratio` | Fraction of unconditional uniform nodes | 0.15 |

Results are saved to `outputs/results/prm_comparison.json`.  Comparison
figures (roadmap overlays, trap rate bar charts) are saved to
`outputs/figures/`.

### Example results

On a synthetic warehouse map (512×512, 67% trap density, 5 runs):

| Planner | Trap rate | Build time | Path found |
|---|---|---|---|
| StandardPRM | 0.673 | 641 ms | 100% |
| TrapAwarePRM (Oracle labels) | 0.123 | 1137 ms | 100% |
| TrapAwarePRM (NN prediction) | 0.125 | 1132 ms | 100% |

The NN-guided planner matches Oracle-guided quality (~81% trap reduction)
while replacing the ~165 ms Oracle call with a ~12 ms neural network
inference.

---

## Generating Report Figures

```bash
python scripts/08_generate_figures.py --config configs/config.yaml \
    --checkpoint checkpoints/best_iou.pth
```

Generates all figures used in the report:

| Figure | Description |
|---|---|
| Eye-test grid | Sample (map, label) pairs |
| Training curves | Loss + validation IoU over epochs |
| Best predictions | Top-4 highest-IoU examples |
| Failure cases | Bottom-4 with analysis |
| Per-direction IoU | N / S / E / W bar chart |
| Generalisation | IoU vs robot size (train vs test markers) |
| Speed comparison | Oracle vs model (log scale) |
| PRM comparison | Standard PRM vs TrapAwarePRM roadmaps |

All figures are saved at 150 DPI as PNG in `outputs/figures/`.

---

## SLURM Quick Reference

### Submitting jobs

```bash
sbatch scripts/slurm/preprocess.sh           # CPU — map preprocessing
sbatch scripts/slurm/oracle.sh               # CPU — label generation
sbatch scripts/slurm/train.sh                # GPU — training
```

All SLURM scripts source `scripts/slurm/_header.sh`, which activates the
conda environment, prints job metadata, and logs GPU info.

### Monitoring

```bash
squeue -u $USER                               # list your jobs
watch -n 150 squeue -u $USER                  # auto-refresh (polite interval)
tail -f logs/slurm/train/<job_id>_train.out   # live output
scancel <job_id>                              # cancel a job
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
```

### Preemption handling

Jobs on the `studentkillable` partition can be killed at any time.  All
training scripts save `last.pth` every epoch; to resume after preemption,
simply resubmit:

```bash
sbatch scripts/slurm/train.sh
```

The script automatically resumes from the latest checkpoint if one exists.

---

## Troubleshooting

**"Config file not found"** — Make sure you run all commands from the project
root (`project2/`), or pass `--config` with an absolute path.

**"CUDA out of memory"** — Reduce `batch_size` in `config.yaml`.  The default
(8) fits on a 2080 (8 GB); use 16 on a Titan XP (12 GB).

**"No processed maps found"** — Run the preprocessing step first:
`python scripts/local/02_preprocess_maps.py --config configs/config.yaml`

**SLURM job killed with no error** — This is a preemption.  Check
`sacct -j <job_id>` for state `CANCELLED` or `PREEMPTED`.  Resubmit; the
training script resumes from `last.pth`.

**Stale conda environment** — If packages seem missing after a node change:
```bash
source configs/.env
source "$CONDA_SH"
conda activate traps
```

---

## License

Course project — TAU Algorithmic Robotics 2025/2026.
