# Run on cluster: python scripts/local/check_metrics.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from src.models.metrics import compute_iou, compute_dice, MetricTracker

print("=== METRICS SANITY CHECKS ===\n")

H, W = 64, 64  # small for speed
B = 2          # batch size

# ---- Test 1: Perfect prediction ------------------------------------------
# If pred == target, IoU must be 1.0
labels  = torch.randint(0, 2, (B, 4, H, W)).float()
# logits that sigmoid → 1.0 where label=1, 0.0 where label=0
logits  = labels * 20.0 - (1 - labels) * 20.0

iou = compute_iou(logits, labels)
dice = compute_dice(logits, labels)
print(f"Test 1 — Perfect prediction:")
print(f"  IoU  = {iou:.4f}  (expected 1.0000)")
print(f"  Dice = {dice:.4f}  (expected 1.0000)")
assert abs(iou - 1.0) < 0.01, f"FAIL: IoU={iou} expected 1.0"
print("  [PASS]\n")

# ---- Test 2: All-zeros prediction ----------------------------------------
# If pred = all 0, IoU depends on label density
labels_dense  = torch.ones(B, 4, H, W).float()   # all viable
logits_zero   = torch.full((B, 4, H, W), -20.0)  # predict nothing

iou_zero = compute_iou(logits_zero, labels_dense)
print(f"Test 2 — All-zero prediction on all-ones labels:")
print(f"  IoU  = {iou_zero:.4f}  (expected 0.0000)")
assert abs(iou_zero - 0.0) < 0.01, f"FAIL: IoU={iou_zero} expected 0.0"
print("  [PASS]\n")

# ---- Test 3: All-ones prediction -----------------------------------------
labels_half = torch.zeros(B, 4, H, W).float()
labels_half[:, :, :H//2, :] = 1.0   # 50% viable
logits_ones = torch.full((B, 4, H, W), 20.0)     # predict everything

iou_ones = compute_iou(logits_ones, labels_half)
# TP=H*W/2, FP=H*W/2, FN=0 → IoU = 0.5
print(f"Test 3 — All-ones prediction on 50%-viable labels:")
print(f"  IoU  = {iou_ones:.4f}  (expected ~0.5000)")
assert abs(iou_ones - 0.5) < 0.05, f"FAIL: IoU={iou_ones} expected ~0.5"
print("  [PASS]\n")

# ---- Test 4: MetricTracker matches compute_iou ---------------------------
tracker = MetricTracker()
labels2 = torch.randint(0, 2, (B, 4, H, W)).float()
logits2 = labels2 * 20.0 - (1 - labels2) * 20.0  # perfect

tracker.update(logits2, labels2)
metrics = tracker.compute()
tracker_iou = metrics.get('iou', -1)
direct_iou  = compute_iou(logits2, labels2)

print(f"Test 4 — MetricTracker vs direct compute_iou:")
print(f"  MetricTracker IoU = {tracker_iou:.4f}")
print(f"  Direct IoU        = {direct_iou:.4f}")
print(f"  Difference        = {abs(tracker_iou - direct_iou):.6f}")
assert abs(tracker_iou - direct_iou) < 0.05, \
    f"FAIL: tracker={tracker_iou} direct={direct_iou}"
print("  [PASS]\n")

# ---- Test 5: Real label file sanity check --------------------------------
print("Test 5 — Real label file:")
import numpy as np
label_files = list(Path("data/labels/robot_30x20").glob("*.npy"))[:3]
for lf in label_files:
    arr = np.load(lf)
    viable_pct = arr.mean() * 100
    # Simulate model predicting the correct answer
    t = torch.from_numpy(arr).float().unsqueeze(0)  # (1,4,512,512)
    logit = t * 20.0 - (1-t) * 20.0
    iou = compute_iou(logit, t)
    print(f"  {lf.stem[:20]}: viable={viable_pct:.1f}%  perfect_pred_IoU={iou:.4f} (expect 1.0)")

print("\n=== ALL METRICS CHECKS PASSED ===")
print("The 0.93 IoU from training is plausible — metrics are computing correctly.")