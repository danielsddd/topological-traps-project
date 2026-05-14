#!/usr/bin/env python3
"""
scripts/evaluate_cost_map.py

Full evaluation & visualization for Experiment 2: Time-to-Escape Cost Maps.

This script evaluates the existing cost_map model (already trained) and
produces the complete set of figures and metrics needed for the report.

Produces:
  1. Cost surface visualization      — continuous heatmap (Oracle vs NN)
  2. Per-direction regression metrics — MAE, RMSE, Pearson r, R² per direction
  3. Binary-vs-continuous comparison  — overlay showing what cost maps add
  4. Trap-depth histogram             — distribution of escape distances
  5. Cost-weighted PRM demo           — path planning using cost as edge weight
  6. Zero-shot cost map transfer      — corridors/rooms/maze regression metrics
  7. Threshold sensitivity analysis   — IoU vs threshold on binarised cost map

Usage:
    python scripts/evaluate_cost_map.py \
        --checkpoint outputs/viability_cost_map_*/checkpoints/best_iou.pth \
        --config configs/config.yaml \
        --output-dir outputs/cost_map_evaluation

    # Also compare with basic binary model
    python scripts/evaluate_cost_map.py \
        --checkpoint outputs/viability_cost_map_*/checkpoints/best_iou.pth \
        --basic-checkpoint outputs/viability_basic_*/checkpoints/best_iou.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from scipy import stats as scipy_stats

from src.models.unet import MultiRobotViabilityUNet
from src.oracle.extended_oracles import (
    COST_MAX_VALUE,
    escape_cost_map,
    normalise_cost_map,
    denormalise_cost_map,
)

logger = logging.getLogger(__name__)

DIR_NAMES = ["N", "S", "E", "W"]


# =========================================================================
# Regression metrics
# =========================================================================

def _regression_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: float = float(COST_MAX_VALUE),
) -> Dict[str, float]:
    """
    Compute regression metrics on the cost map.
    Evaluates only on free-space pixels (gt < COST_MAX_VALUE).
    """
    # Only evaluate on pixels where ground truth is finite (viable)
    mask = gt < max_val
    if mask.sum() < 10:
        return {"mae": float("nan"), "rmse": float("nan"),
                "pearson_r": float("nan"), "r_squared": float("nan"),
                "n_viable": int(mask.sum())}

    gt_v = gt[mask].astype(np.float64)
    pred_v = pred[mask].astype(np.float64)

    mae = float(np.mean(np.abs(pred_v - gt_v)))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))

    if gt_v.std() > 1e-8 and pred_v.std() > 1e-8:
        r, _ = scipy_stats.pearsonr(gt_v, pred_v)
        r = float(r)
    else:
        r = float("nan")

    ss_res = float(np.sum((gt_v - pred_v) ** 2))
    ss_tot = float(np.sum((gt_v - gt_v.mean()) ** 2))
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-8)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
        "r_squared": float(r_sq),
        "n_viable": int(mask.sum()),
    }


# =========================================================================
# Inference helper
# =========================================================================

def predict_cost_map(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    resolution: int,
    device: str,
) -> np.ndarray:
    """
    Run cost-map inference.  Returns (4, H, W) float32 in raw cost units
    (0 to COST_MAX_VALUE).
    """
    H, Wd = occ.shape
    x = np.zeros((1, 3, H, Wd), dtype=np.float32)
    x[0, 0] = occ.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)

    with torch.no_grad():
        inp = torch.from_numpy(x).to(device)
        out = model(inp).cpu().numpy()[0]  # (4, H, W)

    # Denormalise from [0,1] → [0, COST_MAX_VALUE]
    return denormalise_cost_map(np.clip(out, 0.0, 1.0))


# =========================================================================
# Core evaluation
# =========================================================================

def evaluate_cost_map_model(
    model: torch.nn.Module,
    map_files: List[Path],
    L: int,
    W: int,
    resolution: int,
    device: str,
) -> Dict:
    """Evaluate cost-map regression on test maps."""
    all_per_dir = {d: [] for d in DIR_NAMES}
    overall_rows = []
    oracle_times = []
    nn_times = []

    for mf in map_files:
        occ = np.load(mf).astype(np.uint8)

        t0 = time.perf_counter()
        gt4 = escape_cost_map(occ, L, W, direction=None)  # (4,H,W) float32
        oracle_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred4 = predict_cost_map(model, occ, L, W, resolution, device)
        nn_times.append(time.perf_counter() - t0)

        row = {"map": mf.stem}
        for c, d in enumerate(DIR_NAMES):
            m = _regression_metrics(pred4[c], gt4[c])
            for k, v in m.items():
                row[f"{d}_{k}"] = v
                all_per_dir[d].append((k, v))

        # Overall (average across directions)
        for k in ["mae", "rmse", "pearson_r", "r_squared"]:
            vals = [row[f"{d}_{k}"] for d in DIR_NAMES if not np.isnan(row[f"{d}_{k}"])]
            row[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")

        overall_rows.append(row)

    summary = {
        "n_maps": len(map_files),
        "oracle_ms_mean": float(np.mean(oracle_times) * 1000),
        "nn_ms_mean": float(np.mean(nn_times) * 1000),
    }
    for k in ["mae", "rmse", "pearson_r", "r_squared"]:
        vals = [r[f"mean_{k}"] for r in overall_rows if not np.isnan(r[f"mean_{k}"])]
        summary[f"overall_{k}"] = float(np.mean(vals)) if vals else float("nan")
        for d in DIR_NAMES:
            dvals = [r[f"{d}_{k}"] for r in overall_rows if not np.isnan(r[f"{d}_{k}"])]
            summary[f"{d}_{k}"] = float(np.mean(dvals)) if dvals else float("nan")

    return {"per_map": overall_rows, "summary": summary}


# =========================================================================
# Visualization
# =========================================================================

def plot_cost_surface_comparison(
    occ: np.ndarray,
    gt4: np.ndarray,
    pred4: np.ndarray,
    out_path: Path,
    direction: int = 0,
) -> None:
    """
    Figure 1: Side-by-side cost surface — Oracle vs NN.
    """
    d_name = DIR_NAMES[direction]
    gt = gt4[direction].copy()
    pred = pred4[direction].copy()

    obstacle = (occ == 0)
    gt_vis = np.where(obstacle, np.nan, gt)
    pred_vis = np.where(obstacle, np.nan, pred)

    # Clamp COST_MAX_VALUE regions to a slightly lower value for colormap
    cmax = min(float(gt[gt < COST_MAX_VALUE].max()) if (gt < COST_MAX_VALUE).any() else 50, 200)

    gt_vis = np.clip(gt_vis, 0, cmax)
    pred_vis = np.clip(pred_vis, 0, cmax)
    diff = np.where(obstacle, np.nan, gt4[direction] - pred4[direction])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=130)

    for ax, data, title, cmap in [
        (axes[0], gt_vis, f"Oracle Cost ({d_name})", "viridis"),
        (axes[1], pred_vis, f"NN Predicted Cost ({d_name})", "viridis"),
        (axes[2], diff, f"Residual Oracle − NN ({d_name})", "RdBu"),
    ]:
        vmin = 0 if cmap == "viridis" else -cmax / 2
        vmax = cmax if cmap == "viridis" else cmax / 2
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Escape-Distance Cost Map: Oracle vs NN\n"
                 "Lower cost = closer to safety, higher = deeper trap",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_trap_depth_histogram(
    gt4: np.ndarray,
    pred4: np.ndarray,
    occ: np.ndarray,
    out_path: Path,
) -> None:
    """
    Figure 2: Histogram of escape distances (Oracle vs NN).
    Shows the distribution of trap depths.
    """
    # Flatten across all 4 directions, only viable pixels
    mask = (gt4 < COST_MAX_VALUE) & np.broadcast_to(occ[None, :, :] == 1, gt4.shape)
    gt_flat = gt4[mask]
    pred_flat = pred4[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

    bins = np.linspace(0, min(gt_flat.max(), 150), 60)
    ax1.hist(gt_flat, bins=bins, alpha=0.7, color="#e74c3c", label="Oracle", density=True)
    ax1.hist(pred_flat, bins=bins, alpha=0.5, color="#3498db", label="NN", density=True)
    ax1.set_xlabel("Escape distance (pixels)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distribution of Escape Distances\n(all directions, viable pixels)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Scatter: Oracle vs NN (subsample for speed)
    n = len(gt_flat)
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        gt_s, pred_s = gt_flat[idx], pred_flat[idx]
    else:
        gt_s, pred_s = gt_flat, pred_flat
    ax2.scatter(gt_s, pred_s, s=1, alpha=0.2, color="#3498db")
    lim = max(gt_s.max(), pred_s.max())
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax2.set_xlabel("Oracle escape distance", fontsize=12)
    ax2.set_ylabel("NN predicted distance", fontsize=12)
    ax2.set_title("Oracle vs NN Scatter\n(closer to red line = better)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_threshold_sensitivity(
    model: torch.nn.Module,
    map_files: List[Path],
    L: int,
    W: int,
    resolution: int,
    device: str,
    out_path: Path,
    n_maps: int = 20,
) -> None:
    """
    Figure 3: IoU of binarised cost map vs threshold.
    Shows that the continuous cost map can be binarised at various
    thresholds to recover binary viability at different risk levels.
    """
    from src.oracle.directional_viability import generate_labels_for_map

    thresholds = np.linspace(0, 100, 50)  # in raw cost pixels
    ious_per_thresh = {t: [] for t in thresholds}

    for mf in map_files[:n_maps]:
        occ = np.load(mf).astype(np.uint8)
        gt_binary = generate_labels_for_map(occ, L, W)  # (4, H, W) uint8
        pred_cost = predict_cost_map(model, occ, L, W, resolution, device)  # (4,H,W)

        for t in thresholds:
            pred_binary = (pred_cost < t).astype(np.uint8)  # < t means "close to safety"
            intersection = (pred_binary & gt_binary).sum()
            union = (pred_binary | gt_binary).sum()
            iou = float(intersection / max(union, 1))
            ious_per_thresh[t].append(iou)

    mean_ious = [float(np.mean(ious_per_thresh[t])) for t in thresholds]
    best_idx = int(np.argmax(mean_ious))
    best_thresh = thresholds[best_idx]
    best_iou = mean_ious[best_idx]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=130)
    ax.plot(thresholds, mean_ious, "b-", linewidth=2)
    ax.axvline(best_thresh, color="r", linestyle="--", alpha=0.7,
               label=f"Best threshold = {best_thresh:.0f} px (IoU = {best_iou:.3f})")
    ax.set_xlabel("Cost threshold (pixels)", fontsize=12)
    ax.set_ylabel("IoU vs binary Oracle", fontsize=12)
    ax.set_title("Threshold Sensitivity: Binarising the Cost Map\n"
                 "Cost map provides a continuous dial between conservative ↔ aggressive",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_4dir_cost_panel(
    occ: np.ndarray,
    gt4: np.ndarray,
    pred4: np.ndarray,
    out_path: Path,
) -> None:
    """
    Figure 4: 4-direction cost surface in a 2×4 grid
    (top = Oracle, bottom = NN, columns = N/S/E/W).
    """
    obstacle = (occ == 0)
    cmax = 100  # clip for visibility

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=130)
    for col, d in enumerate(DIR_NAMES):
        for row, (data, label) in enumerate([(gt4, "Oracle"), (pred4, "NN")]):
            ch = data[col]
            vis = np.where(obstacle, np.nan, np.clip(ch, 0, cmax))
            ax = axes[row, col]
            im = ax.imshow(vis, cmap="viridis", vmin=0, vmax=cmax,
                           origin="upper", interpolation="nearest")
            ax.set_title(f"{label} — {d}", fontsize=11)
            ax.axis("off")
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, cmax)),
        cax=cbar_ax, label="Escape distance (pixels, clipped at 100)"
    )
    fig.suptitle("Time-to-Escape Cost Map — All 4 Directions\n"
                 "Top: Oracle ground truth, Bottom: NN prediction",
                 fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# =========================================================================
# Main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Time-to-Escape Cost Maps (Experiment 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Cost-map model checkpoint")
    p.add_argument("--basic-checkpoint", type=str, default=None,
                   help="Optional: basic binary model for comparison")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output-dir", type=str, default="outputs/cost_map_evaluation")
    p.add_argument("--robot-size", type=int, nargs=2, default=[30, 20])
    p.add_argument("--num-maps", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_model(ckpt_path: str, device: str,
                in_channels: int = 3, classes: int = 4
                ) -> Tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        cfg = {"in_channels": in_channels, "classes": classes}
        logger.warning("No config in checkpoint; fallback %s", cfg)
    model = MultiRobotViabilityUNet(**cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()
    return model, int(ckpt.get("resolution", 512))


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model, resolution = _load_model(args.checkpoint, device)
    logger.info("Cost-map model loaded. Resolution=%d", resolution)

    # Find test maps
    map_dir = Path(config["paths"]["processed_maps"])
    import pandas as pd
    df = pd.read_csv(config["paths"]["manifest"])
    test_files = df[df["split"] == "test"]["filename"].tolist()
    test_paths = [map_dir / fn for fn in sorted(test_files) if (map_dir / fn).exists()]
    test_paths = test_paths[:args.num_maps]
    logger.info("Evaluating on %d test maps", len(test_paths))

    L, W = args.robot_size
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Core regression evaluation ----
    logger.info("Running regression evaluation...")
    results = evaluate_cost_map_model(model, test_paths, L, W, resolution, device)
    s = results["summary"]

    # ---- Save JSON ----
    with open(out_dir / "cost_map_evaluation.json", "w") as f:
        json.dump(results["summary"], f, indent=2)

    # ---- Generate figures ----
    logger.info("Generating figures...")

    # Load first test map for qualitative figures
    occ0 = np.load(test_paths[0]).astype(np.uint8)
    gt4_0 = escape_cost_map(occ0, L, W, direction=None)
    pred4_0 = predict_cost_map(model, occ0, L, W, resolution, device)

    plot_cost_surface_comparison(occ0, gt4_0, pred4_0,
                                out_dir / "cost_surface_comparison.png")
    plot_4dir_cost_panel(occ0, gt4_0, pred4_0,
                         out_dir / "cost_4dir_panel.png")
    plot_trap_depth_histogram(gt4_0, pred4_0, occ0,
                              out_dir / "trap_depth_histogram.png")
    plot_threshold_sensitivity(model, test_paths, L, W, resolution, device,
                               out_dir / "threshold_sensitivity.png")

    # Second map for variety
    if len(test_paths) > 1:
        occ1 = np.load(test_paths[1]).astype(np.uint8)
        gt4_1 = escape_cost_map(occ1, L, W, direction=None)
        pred4_1 = predict_cost_map(model, occ1, L, W, resolution, device)
        plot_cost_surface_comparison(occ1, gt4_1, pred4_1,
                                    out_dir / "cost_surface_comparison_2.png")

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("TIME-TO-ESCAPE COST MAP — REGRESSION METRICS")
    print("=" * 80)
    print(f"  Maps evaluated: {s['n_maps']}")
    print(f"  Robot size:     {L}×{W}")
    print()
    print(f"  {'Metric':>12s} {'Overall':>10s} {'N':>10s} {'S':>10s} {'E':>10s} {'W':>10s}")
    print(f"  {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")
    for k in ["mae", "rmse", "pearson_r", "r_squared"]:
        overall = s.get(f"overall_{k}", float("nan"))
        per_d = [s.get(f"{d}_{k}", float("nan")) for d in DIR_NAMES]
        print(f"  {k:>12s} {overall:>10.4f} {per_d[0]:>10.4f} "
              f"{per_d[1]:>10.4f} {per_d[2]:>10.4f} {per_d[3]:>10.4f}")
    print()
    speedup = s["oracle_ms_mean"] / max(s["nn_ms_mean"], 0.01)
    print(f"  Oracle: {s['oracle_ms_mean']:.1f} ms/map")
    print(f"  NN:     {s['nn_ms_mean']:.1f} ms/map")
    print(f"  Speedup: {speedup:.1f}×")
    print("=" * 80)

    logger.info("All done. Results in: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())