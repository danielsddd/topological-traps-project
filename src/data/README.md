# src/data/

Data loading, preprocessing, and augmentation.

---

## Files

| File | Description |
|---|---|
| `map_loader.py` | Converts HouseExpo JSON floor plans to 512×512 binary occupancy grids (1 = free, 0 = obstacle). Handles scaling, padding, and border walls. |
| `dataset.py` | `MultiRobotViabilityDataset` — PyTorch Dataset. Loads occupancy grids and pre-computed oracle labels. Supports multi-size training by sampling a random robot size per `__getitem__`. Also exposes `create_dataloaders()` factory. |
| `augmentations.py` | Direction-aware spatial augmentations. Rotations are applied in 90° multiples only, and labels are permuted accordingly (rotating the map 90° clockwise means North → East, etc.). |
| `manifest.py` | Utilities for reading and validating `data/manifest.csv`. |

---

## Dataset

```python
from src.data.dataset import MultiRobotViabilityDataset, create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    config=cfg,
    robot_sizes=[(20, 15), (30, 20), (40, 25)],
)

# Each batch: (inputs, labels, metadata)
# inputs: (B, 3, 512, 512)  float32
# labels: (B, 4, 512, 512)  float32  {0, 1}
# metadata: list of dicts with robot_length, robot_width, map_name
```

---

## Input encoding

```
Channel 0:  occupancy grid             float32, values {0.0, 1.0}
Channel 1:  robot_length / resolution  float32, constant spatial map
Channel 2:  robot_width  / resolution  float32, constant spatial map
```

For the velocity model, a 4th channel is added:
```
Channel 3:  velocity / V_MAX           float32, constant spatial map
```

For the continuous-angle model:
```
Channel 3:  sin(θ)                     float32, constant spatial map
Channel 4:  cos(θ)                     float32, constant spatial map
```

---

## Train / val / test split

Splits are at the **map level** in `data/manifest.csv`:
- 700 maps for training
- 150 maps for validation
- 150 maps for test

The OOD robot size (25×18) is **only** evaluated on the test split and
was never used to generate training labels.
