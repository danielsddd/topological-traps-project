# scripts/

All runnable entry points for the project. Each script reads configuration
from `configs/config.yaml` and writes outputs under `outputs/`.

Scripts are designed to be idempotent: re-running a completed step skips
already-processed files rather than recomputing from scratch.

---

## Training

| Script | Description |
|---|---|
| `train.py` | Main training entry point. Selects oracle type via `--oracle_type`. Supports resume via `--resume <path/to/last.pth>`. |

### Oracle types
```
basic            3-ch input → 4-ch binary viability  (BCE + Dice)
continuous_angle 5-ch input → 1-ch binary viability  (BCE + Dice, per-angle)
cost_map         3-ch input → 4-ch float cost map    (SmoothL1 / Huber)
velocity         4-ch input → 4-ch binary viability  (BCE + Dice, per-speed)
```

---

## Data Pipeline

Run these in order the first time you set up the project.

| Script | Description |
|---|---|
| `local/01_explore_dataset.py` | Inspect raw HouseExpo JSON files and print statistics. |
| `local/02_preprocess_maps.py` | Convert HouseExpo JSON → 512×512 occupancy grids (.npy). |
| `local/03_create_manifest.py` | Create `data/manifest.csv` with train / val / test split. |
| `local/04_generate_labels.py` | Run the oracle pipeline to generate viability labels. Skips maps that already have labels. |
| `local/05_verify_labels.py` | Visual sanity check — renders sample oracle labels as images. |

---

## Evaluation

| Script | Description |
|---|---|
| `evaluate.py` | Main evaluation: per-direction IoU, Dice, pixel accuracy, Oracle vs NN speedup. |
| `evaluate_velocity.py` | Evaluate velocity model at multiple speeds. Produces viable-area shrinkage curve and momentum-trap heatmap. |
| `evaluate_cost_map.py` | Evaluate escape-distance regression model: MAE, RMSE, Pearson r, threshold sensitivity. |

---

## Benchmarks

| Script | Description |
|---|---|
| `benchmark_prm.py` | TrapAwarePRM vs StandardPRM on test maps from the dataset. Reports trap rate, path quality, and build time. |
| `benchmark_prm_hard.py` | Warehouse benchmark on a synthetically generated map with shelves, aisles, and dead-end alcoves. |
| `local/benchmark_oracle.py` | Oracle timing analysis — measures erosion + BFS time across map sizes. |

---

## Demos & Visualisation

| Script | Description |
|---|---|
| `demo_angle_sweep.py` | Animated GIF of the viability map rotating through 0°→360°. Left panel shows robot heading; right panel shows the NN viability map for that heading. |
| `local/plot_training_curves.py` | Plot training/validation loss and IoU from a saved checkpoint or W&B export. |

---

## Utilities

| Script | Description |
|---|---|
| `quick_test.py` | End-to-end smoke test on synthetic data. Verifies the full pipeline (oracle → dataset → model → loss) without requiring real data. |
