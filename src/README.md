# src/

All importable library code. Scripts in `scripts/` import from here.
Nothing in `src/` reads from `configs/` directly — configuration is
passed in as arguments or dictionaries by the calling script.

---

## Modules

| Module | Description |
|---|---|
| `oracle/` | Ground-truth viability label generation pipeline. |
| `models/` | U-Net architecture, loss functions, and evaluation metrics. |
| `data/` | PyTorch Dataset classes, DataLoader factory, and augmentations. |
| `training/` | Trainer class, checkpoint save/load, W&B/TensorBoard logging. |
| `evaluation/` | Evaluator, generalization analysis, speed benchmark. |
| `integration/` | TrapAwarePRM planner and StandardPRM baseline (pure NumPy). |
| `experiments/` | Extended oracle datasets for velocity and cost-map experiments. |
| `visualization/` | Plotting utilities for training curves and predictions. |
| `utils/` | Device helpers, path utilities, and miscellaneous functions. |

Each module has its own `README.md` with API details.
