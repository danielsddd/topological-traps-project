#!/usr/bin/env python3
"""
scripts/local/check_viability_distribution.py

Checks the actual model output values on oracle-confirmed trap pixels.
Run this BEFORE tuning thresholds in demo_dwa_viability.py.

Output: percentile distribution of via predictions for:
  - Trapped pixels   (oracle east_label = 0)
  - Clear pixels     (oracle east_label = 1)
  - Near-exit pixels (east_label=0 but within 60px of east_label=1)

The 25th percentile of trapped pixels → set viability_threshold to that.
The 5th  percentile of trapped pixels → set hard_reject_threshold to that.

Usage:
    python scripts/local/check_viability_distribution.py \\
        --checkpoint outputs/viability_continuous_angle_*/checkpoints/best_iou.pth \\
        --n-maps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import MultiRobotViabilityUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-maps", type=int, default=10)
    p.add_argument("--robot-L", type=int, default=30)
    p.add_argument("--robot-W", type=int, default=20)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {"in_channels": 5, "classes": 1})
    model = MultiRobotViabilityUNet(**cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()
    resolution = int(ckpt.get("resolution", 512))
    print(f"Model loaded: in={cfg['in_channels']} out={cfg['classes']} "
          f"resolution={resolution} device={device}")

    # ---- Paths ------------------------------------------------------------
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        processed_dir = PROJECT_ROOT / yaml_cfg["paths"]["processed_maps"]
        labels_dir    = PROJECT_ROOT / yaml_cfg.get("paths", {}).get(
                            "labels_dir", "data/labels") / f"robot_{args.robot_L}x{args.robot_W}"
    else:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        labels_dir    = PROJECT_ROOT / "data" / "labels" / f"robot_{args.robot_L}x{args.robot_W}"

    map_paths = sorted(processed_dir.glob("*.npy"))[:args.n_maps]
    print(f"Checking {len(map_paths)} maps  |  labels: {labels_dir}\n")

    all_trapped  = []
    all_clear    = []
    all_near_exit = []

    for map_path in map_paths:
        label_path = labels_dir / map_path.name
        if not label_path.exists():
            continue

        occ   = np.load(str(map_path)).astype(np.uint8)
        label = np.load(str(label_path))  # (4, H, W)
        east_label = label[2].astype(np.uint8)  # 0=trapped, 1=clear

        H, W = occ.shape

        # Build model input for east heading (θ=0)
        x = np.zeros((1, cfg["in_channels"], H, W), dtype=np.float32)
        x[0, 0] = occ.astype(np.float32)
        x[0, 1] = float(args.robot_L) / resolution
        x[0, 2] = float(args.robot_W) / resolution
        if cfg["in_channels"] == 5:
            x[0, 3] = float(np.sin(0.0))  # east heading
            x[0, 4] = float(np.cos(0.0))

        with torch.no_grad():
            pred = torch.sigmoid(
                model(torch.from_numpy(x).to(device))
            )[0, 0].cpu().numpy()  # (H, W)

        # Free space only
        free = occ == 1

        trapped_mask  = (east_label == 0) & free
        clear_mask    = (east_label == 1) & free

        # Near-exit: trapped but within 60px of clear
        near_exit_mask = binary_dilation(clear_mask, iterations=60) & trapped_mask

        if trapped_mask.any():
            all_trapped.extend(pred[trapped_mask].tolist())
        if clear_mask.any():
            all_clear.extend(pred[clear_mask].tolist())
        if near_exit_mask.any():
            all_near_exit.extend(pred[near_exit_mask].tolist())

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def fmt(arr, label):
        if not arr:
            print(f"  {label}: NO DATA")
            return
        a = np.array(arr)
        p = np.percentile(a, pcts)
        print(f"  {label} (n={len(a):,}):")
        print(f"    mean={a.mean():.3f}  std={a.std():.3f}  "
              f"min={a.min():.3f}  max={a.max():.3f}")
        pct_str = "  ".join(f"p{q}={v:.3f}" for q, v in zip(pcts, p))
        print(f"    {pct_str}")

    print("=" * 70)
    print("MODEL VIABILITY OUTPUT DISTRIBUTION (east heading, θ=0)")
    print(f"Robot: {args.robot_L}×{args.robot_W}  |  {len(map_paths)} maps")
    print("=" * 70)
    fmt(all_trapped,   "TRAPPED  (oracle=0)")
    print()
    fmt(all_near_exit, "NEAR-EXIT (oracle=0, within 60px of clear)")
    print()
    fmt(all_clear,     "CLEAR    (oracle=1)")
    print("=" * 70)

    if all_trapped:
        p5  = float(np.percentile(all_trapped, 5))
        p25 = float(np.percentile(all_trapped, 25))
        print(f"\nRECOMMENDED THRESHOLDS based on trapped-pixel distribution:")
        print(f"  hard_reject_threshold = {p5:.2f}   (5th  pct of trapped — only confident traps)")
        print(f"  viability_threshold   = {p25:.2f}  (25th pct of trapped — soft penalty zone)")
        print(f"\n  Set these in src/planning/dwa_planner.py DWAConfig defaults")
        print(f"  AND in shared_cfg in demo_dwa_viability.py and demo_dwa_size_sweep.py")


if __name__ == "__main__":
    main()