# src/models/

Neural network architecture, loss functions, and evaluation metrics.

---

## Architecture — `unet.py`

`MultiRobotViabilityUNet` wraps `segmentation_models_pytorch.Unet` with a
ResNet34 encoder pretrained on ImageNet.

```
Input  (B, C_in, 512, 512)   C_in = 3, 4, or 5 depending on oracle type
  └─ ResNet34 encoder         5 stages, pretrained weights
  └─ U-Net decoder            skip connections, channels (256,128,64,32,16)
Output (B, C_out, 512, 512)  C_out = 1 or 4, raw logits
```

Robot size is encoded as **constant spatial channels** (global conditioning):
channel 1 = `robot_L / resolution` broadcast to (H, W), channel 2 = `robot_W / resolution`.
This allows a single model to generalise across robot sizes without
retraining.

```python
from src.models.unet import MultiRobotViabilityUNet

# Basic model
model = MultiRobotViabilityUNet(in_channels=3, classes=4)

# Velocity model
model = MultiRobotViabilityUNet(in_channels=4, classes=4)

# Continuous-angle model
model = MultiRobotViabilityUNet(in_channels=5, classes=1)

# Load from checkpoint
model = MultiRobotViabilityUNet.from_checkpoint("outputs/.../best_iou.pth")
```

---

## Losses — `losses.py`

| Class | Description |
|---|---|
| `DiceLoss` | Soft Dice loss over all channels. |
| `DiceBCELoss` | `0.5 * BCE + 0.5 * Dice` — default for binary viability. |
| `PerChannelDiceBCELoss` | Computes Dice per direction, averages. More stable on imbalanced maps. |

```python
from src.models.losses import create_loss

criterion = create_loss("dice_bce")          # binary classification
criterion = create_loss("smooth_l1")         # regression (cost map)
```

---

## Metrics — `metrics.py`

| Function | Description |
|---|---|
| `compute_iou(logits, target)` | Mean IoU over all output channels. |
| `compute_dice(logits, target)` | Mean Dice score. |
| `compute_per_channel_metrics(logits, target)` | Per-direction IoU and Dice (N/S/E/W). |
| `compute_pixel_accuracy(logits, target)` | Fraction of correctly classified pixels. |

```python
from src.models.metrics import compute_iou, compute_per_channel_metrics

iou = compute_iou(model_output, labels)
per_dir = compute_per_channel_metrics(model_output, labels)
# per_dir = {"iou_N": 0.97, "iou_S": 0.98, "iou_E": 0.97, "iou_W": 0.98, ...}
```

---

## Parameters

~24.4 M total parameters, ~93 MB on disk. All parameters are trainable
(encoder is fine-tuned, not frozen).
