#!/usr/bin/env python3
"""
scripts/evaluate_velocity.py

Evaluation & visualization for Experiment 1: Velocity-Dependent Viability.

Produces:
  1. Viable-area-vs-speed curve       — proves "space shrinks with speed"
  2. Per-speed IoU table               — NN accuracy at each velocity
  3. Per-direction IoU at each speed   — directional breakdown
  4. Qualitative multi-speed panel     — side-by-side map: v=0..3 m/s
  5. Momentum-trap heatmap             — pixels that flip trap/viable as v grows
  6. Speed comparison (Oracle vs NN)   — timing at each velocity

Usage:
    python scripts/evaluate_velocity.py \
        --checkpoint outputs/viability_velocity_*/checkpoints/best_iou.pth \
        --config configs/config.yaml \
        --output-dir outputs/velocity_evaluation

    # Quick smoke test (5 maps, 3 speeds)
    python scripts/evaluate_velocity.py \
        --checkpoint ... --num-maps 5 --velocities 0.0 1.0 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from src.models.unet import MultiRobotViabilityUNet
from src.oracle.velocity_oracle import (
    DEFAULT_VELOCITIES,
    V_MAX,
    braking_distance_px,
    normalise_velocity,
    velocity_viability,
)
from src.experiments import predict_velocity_viability

logger = logging.getLogger(__name__)


# =========================================================================
# Metrics
# =========================================================================

def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """IoU, Dice, accuracy for binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    iou = float(intersection / max(union, 1))
    dice = float(2 * intersection / max(pred_b.sum() + gt_b.sum(), 1))
    acc = float((pred_b == gt_b).mean())
    return {"iou": iou, "dice": dice, "acc": acc}


def _per_direction_iou(pred4: np.ndarray, gt4: np.ndarray) -> Dict[str, float]:
    """Per-channel IoU for (4, H, W) arrays."""
    names = ["N", "S", "E", "W"]
    result = {}
    for c, n in enumerate(names):
        m = _binary_metrics(pred4[c], gt4[c])
        result[f"iou_{n}"] = m["iou"]
    result["iou_mean"] = float(np.mean([result[f"iou_{n}"] for n in names]))
    return result


# =========================================================================
# Core evaluation loop
# =========================================================================

def evaluate_at_speed(
    model: torch.nn.Module,
    map_files: List[Path],
    L: int,
    W: int,
    velocity: float,
    resolution: int,
    device: str,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate the NN against the Oracle at a single speed."""
    rows = []
    oracle_times = []
    nn_times = []

    for mf in map_files:
        occ = np.load(mf).astype(np.uint8)

        # Oracle
        t0 = time.perf_counter()
        gt = velocity_viability(occ, L, W, velocity, max_decel, px_per_m)
        oracle_times.append(time.perf_counter() - t0)

        # NN
        t0 = time.perf_counter()
        pred = predict_velocity_viability(
            model, occ, L, W, velocity, resolution, device, threshold,
        )
        nn_times.append(time.perf_counter() - t0)

        overall = _binary_metrics(pred, gt)
        per_dir = _per_direction_iou(pred, gt)

        # max(axis=0): pixel is "viable" if escapable in ANY direction
        viable_frac_gt = float(gt.max(axis=0).mean())   # viable in ≥1 direction
        viable_frac_pred = float(pred.max(axis=0).mean())

        rows.append({
            "map": mf.stem,
            **overall,
            **per_dir,
            "viable_frac_gt": viable_frac_gt,
            "viable_frac_pred": viable_frac_pred,
        })

    summary = {
        "velocity": velocity,
        "d_brake_px": braking_distance_px(velocity, max_decel, px_per_m),
        "n_maps": len(map_files),
        "iou_mean": float(np.mean([r["iou"] for r in rows])),
        "iou_std": float(np.std([r["iou"] for r in rows])),
        "dice_mean": float(np.mean([r["dice"] for r in rows])),
        "acc_mean": float(np.mean([r["acc"] for r in rows])),
        "viable_frac_gt_mean": float(np.mean([r["viable_frac_gt"] for r in rows])),
        "viable_frac_pred_mean": float(np.mean([r["viable_frac_pred"] for r in rows])),
        "oracle_ms_mean": float(np.mean(oracle_times) * 1000),
        "nn_ms_mean": float(np.mean(nn_times) * 1000),
    }
    for d in ["N", "S", "E", "W"]:
        summary[f"iou_{d}_mean"] = float(np.mean([r[f"iou_{d}"] for r in rows]))

    return {"per_map": rows, "summary": summary}


# =========================================================================
# Visualization functions
# =========================================================================

def plot_viable_area_vs_speed(
    results: List[Dict],
    out_dir: Path,
) -> None:
    """
    Figure 1: Viable area fraction vs speed.
    Shows that viable space shrinks as velocity increases.
    Plots both Oracle ground truth and NN prediction.
    """
    speeds = [r["summary"]["velocity"] for r in results]
    gt_fracs = [r["summary"]["viable_frac_gt_mean"] for r in results]
    pred_fracs = [r["summary"]["viable_frac_pred_mean"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=130)
    ax.plot(speeds, gt_fracs, "o-", color="#2ecc71", linewidth=2, markersize=8,
            label="Oracle (ground truth)")
    ax.plot(speeds, pred_fracs, "s--", color="#3498db", linewidth=2, markersize=7,
            label="NN prediction")
    ax.set_xlabel("Robot velocity (m/s)", fontsize=12)
    ax.set_ylabel("Mean viable area fraction", fontsize=12)
    ax.set_title("Viable Space Shrinks with Speed\n"
                 "(The Momentum Trap Effect)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add braking distance annotations
    for v, frac in zip(speeds, gt_fracs):
        d = braking_distance_px(v, 2.0, 10.0)
        if d > 0:
            ax.annotate(f"d={d}px", (v, frac), textcoords="offset points",
                        xytext=(5, 10), fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(out_dir / "viable_area_vs_speed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved viable_area_vs_speed.png")


def plot_iou_vs_speed(
    results: List[Dict],
    out_dir: Path,
) -> None:
    """Figure 2: Per-speed IoU bar chart with per-direction breakdown."""
    speeds = [r["summary"]["velocity"] for r in results]
    ious = [r["summary"]["iou_mean"] for r in results]
    iou_stds = [r["summary"]["iou_std"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

    # Left: overall IoU vs speed
    bars = ax1.bar(range(len(speeds)), ious, yerr=iou_stds,
                   color="#3498db", alpha=0.8, capsize=4)
    ax1.set_xticks(range(len(speeds)))
    ax1.set_xticklabels([f"{v:.1f}" for v in speeds])
    ax1.set_xlabel("Velocity (m/s)", fontsize=12)
    ax1.set_ylabel("IoU", fontsize=12)
    ax1.set_title("Overall IoU vs Speed", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    for bar, iou in zip(bars, ious):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{iou:.3f}", ha="center", fontsize=9)

    # Right: per-direction IoU grouped bar chart
    dirs = ["N", "S", "E", "W"]
    x = np.arange(len(speeds))
    width = 0.18
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for j, (d, c) in enumerate(zip(dirs, colors)):
        vals = [r["summary"][f"iou_{d}_mean"] for r in results]
        ax2.bar(x + j * width - 1.5 * width, vals, width,
                label=d, color=c, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:.1f}" for v in speeds])
    ax2.set_xlabel("Velocity (m/s)", fontsize=12)
    ax2.set_ylabel("IoU", fontsize=12)
    ax2.set_title("Per-Direction IoU vs Speed", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "iou_vs_speed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved iou_vs_speed.png")


def plot_multi_speed_panel(
    occ: np.ndarray,
    L: int,
    W: int,
    model: torch.nn.Module,
    velocities: List[float],
    resolution: int,
    device: str,
    out_path: Path,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> None:
    """
    Figure 3: Qualitative panel showing viability at multiple speeds.
    Top row: Oracle,  Bottom row: NN.  Columns = different speeds.
    Uses North direction for visualization (channel 0).
    """
    n_speeds = len(velocities)
    fig, axes = plt.subplots(2, n_speeds, figsize=(4 * n_speeds, 8), dpi=130)
    if n_speeds == 1:
        axes = axes.reshape(2, 1)

    # Custom colormap: red = trapped, green = viable, gray = obstacle
    cmap = LinearSegmentedColormap.from_list(
        "trap_viable", ["#e74c3c", "#f39c12", "#2ecc71"], N=256
    )

    obstacle_mask = (occ == 0)

    for col, v in enumerate(velocities):
        d_brake = braking_distance_px(v, max_decel, px_per_m)

        # Oracle
        gt = velocity_viability(occ, L, W, v, max_decel, px_per_m)
        gt_north = gt[0].astype(np.float32)
        gt_vis = np.where(obstacle_mask, np.nan, gt_north)

        # NN
        pred = predict_velocity_viability(
            model, occ, L, W, v, resolution, device,
        )
        pred_north = pred[0].astype(np.float32)
        pred_vis = np.where(obstacle_mask, np.nan, pred_north)

        # Plot
        for row, data, label in [
            (0, gt_vis, "Oracle"),
            (1, pred_vis, "NN"),
        ]:
            ax = axes[row, col]
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1,
                           origin="upper", interpolation="nearest")
            ax.set_title(f"{label}\nv={v:.1f} m/s  (d={d_brake}px)",
                         fontsize=10)
            ax.axis("off")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
        cax=cbar_ax, label="Viability (North direction)"
    )
    fig.suptitle("Momentum Trap: Viable Space Shrinks with Speed",
                 fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_momentum_trap_heatmap(
    occ: np.ndarray,
    L: int,
    W: int,
    velocities: List[float],
    out_path: Path,
    max_decel: float = 2.0,
    px_per_m: float = 10.0,
) -> None:
    """
    Figure 4: Momentum-trap heatmap.
    For each pixel, compute the maximum speed at which it remains viable.
    Red = only viable at low speed (momentum trap), green = viable at all speeds.
    """
    H, Wd = occ.shape
    max_safe_speed = np.zeros((H, Wd), dtype=np.float32)

    for v in sorted(velocities):
        gt = velocity_viability(occ, L, W, v, max_decel, px_per_m)
        # A pixel is "safe at speed v" if viable in ALL 4 directions
        all_viable = (gt.min(axis=0) == 1)
        max_safe_speed[all_viable] = v

    obstacle_mask = (occ == 0)
    vis = np.where(obstacle_mask, np.nan, max_safe_speed)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=130)
    im = ax.imshow(vis, cmap="RdYlGn", vmin=0, vmax=max(velocities),
                   origin="upper", interpolation="nearest")
    ax.set_title("Maximum Safe Speed per Pixel\n"
                 "(Red = low-speed-only → momentum trap)", fontsize=13)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Max safe speed (m/s)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_timing_vs_speed(
    results: List[Dict],
    out_dir: Path,
) -> None:
    """Figure 5: Oracle vs NN inference time at each speed."""
    speeds = [r["summary"]["velocity"] for r in results]
    oracle_ms = [r["summary"]["oracle_ms_mean"] for r in results]
    nn_ms = [r["summary"]["nn_ms_mean"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=130)
    x = np.arange(len(speeds))
    w = 0.35
    ax.bar(x - w / 2, oracle_ms, w, label="Oracle (CPU)", color="#e74c3c", alpha=0.8)
    ax.bar(x + w / 2, nn_ms, w, label="NN (GPU)", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in speeds])
    ax.set_xlabel("Velocity (m/s)", fontsize=12)
    ax.set_ylabel("Inference time (ms)", fontsize=12)
    ax.set_title("Inference Speed: Oracle vs NN at Each Velocity", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "timing_vs_speed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved timing_vs_speed.png")


# =========================================================================
# Main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Velocity-Dependent Viability (Momentum Trap)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output-dir", type=str, default="outputs/velocity_evaluation")
    p.add_argument("--velocities", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--robot-size", type=int, nargs=2, default=[30, 20],
                   help="Robot (L, W) in pixels")
    p.add_argument("--max-decel", type=float, default=2.0,
                   help="Maximum deceleration (m/s²)")
    p.add_argument("--px-per-m", type=float, default=10.0,
                   help="Pixels per metre")
    p.add_argument("--num-maps", type=int, default=50,
                   help="Number of test maps to evaluate")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_model(ckpt_path: str, device: str) -> Tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        cfg = {"in_channels": 4, "classes": 4}
        logger.warning("No 'config' in checkpoint; using fallback %s", cfg)
    model = MultiRobotViabilityUNet(**cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()
    resolution = ckpt.get("resolution", 512)
    return model, int(resolution)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model, resolution = _load_model(args.checkpoint, device)
    logger.info("Model loaded. Resolution=%d", resolution)

    # Find test maps
    map_dir = Path(config["paths"]["processed_maps"])
    manifest_path = config["paths"]["manifest"]
    import pandas as pd
    df = pd.read_csv(manifest_path)
    test_files = df[df["split"] == "test"]["filename"].tolist()
    test_paths = [map_dir / fn for fn in sorted(test_files) if (map_dir / fn).exists()]
    test_paths = test_paths[:args.num_maps]
    logger.info("Evaluating on %d test maps", len(test_paths))

    L, W = args.robot_size
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Evaluate at each speed ----
    all_results = []
    for v in args.velocities:
        logger.info("Evaluating at v=%.1f m/s  (d_brake=%d px)...",
                     v, braking_distance_px(v, args.max_decel, args.px_per_m))
        result = evaluate_at_speed(
            model, test_paths, L, W, v, resolution, device,
            args.max_decel, args.px_per_m, args.threshold,
        )
        s = result["summary"]
        logger.info("  IoU=%.4f  Dice=%.4f  viable_gt=%.3f  viable_pred=%.3f  "
                     "oracle=%.1fms  nn=%.1fms",
                     s["iou_mean"], s["dice_mean"],
                     s["viable_frac_gt_mean"], s["viable_frac_pred_mean"],
                     s["oracle_ms_mean"], s["nn_ms_mean"])
        all_results.append(result)

    # ---- Save JSON ----
    json_out = out_dir / "velocity_evaluation.json"
    serializable = []
    for r in all_results:
        serializable.append({
            "summary": r["summary"],
            "per_map_count": len(r["per_map"]),
        })
    with open(json_out, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", json_out)

    # ---- Generate figures ----
    logger.info("Generating figures...")
    plot_viable_area_vs_speed(all_results, out_dir)
    plot_iou_vs_speed(all_results, out_dir)
    plot_timing_vs_speed(all_results, out_dir)

    # Qualitative panel on first test map
    first_occ = np.load(test_paths[0]).astype(np.uint8)
    plot_multi_speed_panel(
        first_occ, L, W, model, args.velocities, resolution, device,
        out_dir / "multi_speed_panel.png",
        args.max_decel, args.px_per_m,
    )

    # Momentum-trap heatmap on first test map
    plot_momentum_trap_heatmap(
        first_occ, L, W, args.velocities,
        out_dir / "momentum_trap_heatmap.png",
        args.max_decel, args.px_per_m,
    )

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("VELOCITY-DEPENDENT VIABILITY — SUMMARY")
    print("=" * 80)
    print(f"{'Speed':>8s} {'d_brake':>8s} {'IoU':>8s} {'Dice':>8s} "
          f"{'Viable%':>8s} {'Oracle':>10s} {'NN':>10s} {'Speedup':>8s}")
    print("-" * 80)
    for r in all_results:
        s = r["summary"]
        speedup = s["oracle_ms_mean"] / max(s["nn_ms_mean"], 0.01)
        print(f"{s['velocity']:>7.1f}  {s['d_brake_px']:>7d}  "
              f"{s['iou_mean']:>7.4f}  {s['dice_mean']:>7.4f}  "
              f"{s['viable_frac_gt_mean']*100:>7.1f}  "
              f"{s['oracle_ms_mean']:>9.1f}  {s['nn_ms_mean']:>9.1f}  "
              f"{speedup:>7.1f}×")
    print("=" * 80)

    logger.info("All done. Figures and results in: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())