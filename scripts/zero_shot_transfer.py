#!/usr/bin/env python3
"""
Zero-Shot Transfer Evaluation — Direction 3.

Train a viability model on HouseExpo (apartments). Evaluate it on
procedurally generated environments it has never seen during training:

    1. Rectilinear corridors / aisles  ("warehouse"-style maps)
    2. Recursive-DFS mazes              (dense narrow passages)
    3. Recursive-division mazes         (block-structured layouts)

For each procedural map we:
    - run the model to obtain predicted (4, H, W) viability,
    - run the *same* Oracle used at training time to get GT,
    - compute per-direction IoU + Dice + pixel accuracy.

We aggregate over many maps to produce a transfer report.

Notes on scope:
  * No fine-tuning is performed.
  * Maps are generated at the same resolution the model was trained at
    (default 512×512 — read from config or checkpoint).
  * Procedural map RNG is seeded so runs are reproducible.

Usage:
    python scripts/zero_shot_transfer.py \
        --model_path outputs/<exp>/checkpoints/best_iou.pth \
        --train_dataset houseexpo \
        --test_map_type maze \
        --num-maps 50 \
        --robot-length 30 --robot-width 20

The script writes a JSON summary and a small PNG grid for visual sanity.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make the project root importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import MultiRobotViabilityUNet
from src.oracle.directional_viability import generate_labels_for_map
from src.oracle.extended_oracles import (
    angle_to_sincos,
    continuous_angle_viability,
    escape_cost_map,
    escape_cost_map_for_angle,
    normalise_cost_map,
    COST_MAX_VALUE,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Procedural map generators
# ===========================================================================

def make_corridors(
    size: int = 512,
    n_corridors_h: int = 3,
    n_corridors_v: int = 3,
    corridor_width: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Rectilinear corridor / warehouse-aisle map.

    Generates a solid grid of obstacles, then carves horizontal and
    vertical corridors of fixed width at random positions to form a
    cross-grid that resembles a warehouse aisle layout.

    Args:
        size:           Map side length (square).
        n_corridors_h:  Number of horizontal aisles.
        n_corridors_v:  Number of vertical aisles.
        corridor_width: Aisle width in pixels.
        rng:            Optional numpy Generator (for reproducibility).

    Returns:
        (size, size) uint8 occupancy grid (1=free, 0=obstacle).
    """
    rng = rng or np.random.default_rng()
    grid = np.zeros((size, size), dtype=np.uint8)  # all obstacle

    # Carve horizontal corridors
    margin = corridor_width
    h_centres = rng.integers(margin, size - margin, size=n_corridors_h)
    for cy in h_centres:
        y0 = max(0, cy - corridor_width // 2)
        y1 = min(size, cy + corridor_width // 2)
        grid[y0:y1, :] = 1

    # Carve vertical corridors
    v_centres = rng.integers(margin, size - margin, size=n_corridors_v)
    for cx in v_centres:
        x0 = max(0, cx - corridor_width // 2)
        x1 = min(size, cx + corridor_width // 2)
        grid[:, x0:x1] = 1

    # Add a 1-pixel border of obstacles for parity with HouseExpo maps
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0
    return grid


def make_dfs_maze(
    size: int = 512,
    cell: int = 24,
    wall: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a maze via randomised depth-first search ("recursive backtracker").

    Each cell of size `cell` is a corridor; walls of thickness `wall`
    separate adjacent cells. DFS knocks down walls between visited and
    unvisited neighbours until every cell is reachable.

    Args:
        size: Output map side length.
        cell: Corridor cell size in pixels.
        wall: Wall thickness in pixels.
        rng:  Optional numpy Generator.

    Returns:
        (size, size) uint8 occupancy grid.
    """
    rng = rng or np.random.default_rng()
    step = cell + wall
    n = size // step  # number of cells per side
    if n < 4:
        raise ValueError(
            f"Map too small for cell={cell}, wall={wall}, size={size}. "
            f"Need size > 4*(cell+wall)."
        )

    # Initially: everything is wall (0). Carve open cells (1) at grid centres.
    grid = np.zeros((size, size), dtype=np.uint8)

    def cell_origin(i: int, j: int) -> Tuple[int, int]:
        """Top-left corner of cell (i, j)."""
        return wall + i * step, wall + j * step

    # Open all cell interiors first
    for i in range(n):
        for j in range(n):
            y, x = cell_origin(i, j)
            grid[y:y + cell, x:x + cell] = 1

    # DFS: knock walls between cells
    visited = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        nbrs = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and not visited[ni, nj]:
                nbrs.append((ni, nj, di, dj))
        if not nbrs:
            stack.pop()
            continue
        ni, nj, di, dj = nbrs[int(rng.integers(0, len(nbrs)))]
        # Knock down the wall between (i, j) and (ni, nj)
        y0, x0 = cell_origin(i, j)
        if di != 0:    # vertical neighbour, horizontal wall to remove
            y_wall = y0 + cell if di > 0 else y0 - wall
            grid[y_wall:y_wall + wall, x0:x0 + cell] = 1
        else:          # horizontal neighbour, vertical wall to remove
            x_wall = x0 + cell if dj > 0 else x0 - wall
            grid[y0:y0 + cell, x_wall:x_wall + wall] = 1

        visited[ni, nj] = True
        stack.append((ni, nj))

    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0
    return grid


def make_recursive_division_maze(
    size: int = 512,
    min_room: int = 60,
    door_width: int = 24,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Recursive-division maze. Easier topology than DFS — produces room-like
    structure with single doorways between sub-regions.

    Args:
        size:       Map side length.
        min_room:   Recursion stops when a sub-region is smaller than this.
        door_width: Width of the carved doorway in pixels.
        rng:        Optional numpy Generator.

    Returns:
        (size, size) uint8 occupancy grid.
    """
    rng = rng or np.random.default_rng()
    grid = np.ones((size, size), dtype=np.uint8)

    def divide(y0: int, y1: int, x0: int, x1: int) -> None:
        h, w = y1 - y0, x1 - x0
        if h < min_room or w < min_room:
            return

        if w >= h:
            # Vertical wall
            _lo, _hi = x0 + min_room // 2, x1 - min_room // 2
            if _lo >= _hi:
                return
            wx = int(rng.integers(_lo, _hi))
            grid[y0:y1, wx] = 0
            door_y = int(rng.integers(y0, y1 - door_width))
            grid[door_y:door_y + door_width, wx] = 1
            divide(y0, y1, x0, wx)
            divide(y0, y1, wx + 1, x1)
        else:
            # Horizontal wall
            _lo, _hi = y0 + min_room // 2, y1 - min_room // 2
            if _lo >= _hi:
                return
            wy = int(rng.integers(_lo, _hi))
            grid[wy, x0:x1] = 0
            door_x = int(rng.integers(x0, x1 - door_width))
            grid[wy, door_x:door_x + door_width] = 1
            divide(y0, wy, x0, x1)
            divide(wy + 1, y1, x0, x1)

    divide(1, size - 1, 1, size - 1)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0
    return grid


MAP_GENERATORS = {
    "corridors": make_corridors,
    "maze": make_dfs_maze,
    "rooms": make_recursive_division_maze,
}


# ===========================================================================
# Inference helpers
# ===========================================================================

def _build_input_basic(occ: np.ndarray, L: int, W: int, resolution: int) -> torch.Tensor:
    H, Wd = occ.shape
    x = np.zeros((1, 3, H, Wd), dtype=np.float32)
    x[0, 0] = occ.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    return torch.from_numpy(x)


def _build_input_angle(
    occ: np.ndarray, L: int, W: int, angle_deg: float, resolution: int
) -> torch.Tensor:
    H, Wd = occ.shape
    s, c = angle_to_sincos(angle_deg)
    x = np.zeros((1, 5, H, Wd), dtype=np.float32)
    x[0, 0] = occ.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    x[0, 3] = s
    x[0, 4] = c
    return torch.from_numpy(x)


def predict_basic(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    resolution: int,
    device: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference for the basic 4-direction model.

    Returns:
        (4, H, W) uint8 binary predictions.
    """
    inp = _build_input_basic(occ, L, W, resolution).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)
    pred = (probs > threshold).float().cpu().numpy()[0].astype(np.uint8)
    return pred


def predict_continuous_angle(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    angle_deg: float,
    resolution: int,
    device: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """Single-channel binary prediction for angle-conditioned model."""
    inp = _build_input_angle(occ, L, W, angle_deg, resolution).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)
    pred = (probs > threshold).float().cpu().numpy()[0, 0].astype(np.uint8)
    return pred


def predict_cost_map(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    resolution: int,
    device: str,
) -> np.ndarray:
    """4-channel regression cost prediction (denormalised)."""
    inp = _build_input_basic(occ, L, W, resolution).to(device)
    with torch.no_grad():
        logits = model(inp)
        # Cost-map model uses linear output trained against costs in [0, 1]
        pred_norm = torch.clamp(logits, 0.0, 1.0).cpu().numpy()[0]
    return (pred_norm * float(COST_MAX_VALUE)).astype(np.float32)


# ===========================================================================
# Metric helpers
# ===========================================================================

def _per_direction_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute IoU, Dice, accuracy for each of the 4 channels and overall.

    Args:
        pred: (4, H, W) uint8 binary prediction.
        gt:   (4, H, W) uint8 binary ground truth.

    Returns:
        Flat dict of metrics.
    """
    out: Dict[str, float] = {}
    names = ["N", "S", "E", "W"]
    iou_sum = dice_sum = acc_sum = 0.0
    for i, n in enumerate(names):
        p, t = pred[i].astype(bool), gt[i].astype(bool)
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        psum, tsum = p.sum(), t.sum()
        iou = inter / union if union > 0 else 1.0
        dice = (2 * inter) / (psum + tsum) if (psum + tsum) > 0 else 1.0
        acc = (p == t).mean()
        out[f"iou_{n}"] = float(iou)
        out[f"dice_{n}"] = float(dice)
        out[f"acc_{n}"] = float(acc)
        iou_sum += iou
        dice_sum += dice
        acc_sum += acc
    out["iou_mean"] = iou_sum / 4.0
    out["dice_mean"] = dice_sum / 4.0
    out["acc_mean"] = acc_sum / 4.0
    return out


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """IoU/Dice/acc for a single (H, W) binary pair."""
    p = pred.astype(bool)
    t = gt.astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    psum, tsum = p.sum(), t.sum()
    return {
        "iou": float(inter / union) if union > 0 else 1.0,
        "dice": float(2 * inter / (psum + tsum)) if (psum + tsum) > 0 else 1.0,
        "acc": float((p == t).mean()),
    }


def _regression_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / Pearson r for cost-map evaluation."""
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    pf, gf = pred.flatten(), gt.flatten()
    pf -= pf.mean()
    gf -= gf.mean()
    denom = np.sqrt((pf * pf).sum() * (gf * gf).sum())
    pearson = float((pf * gf).sum() / denom) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson_r": pearson}


# ===========================================================================
# Evaluation drivers
# ===========================================================================

def evaluate_basic(
    model: torch.nn.Module,
    map_type: str,
    num_maps: int,
    L: int,
    W: int,
    resolution: int,
    device: str,
    seed: int,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate the 3→4 binary basic model on procedural maps."""
    gen = MAP_GENERATORS[map_type]
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    times_model: List[float] = []
    times_oracle: List[float] = []

    for i in range(num_maps):
        sub_rng = np.random.default_rng(seed + i + 1)
        occ = gen(size=resolution, rng=sub_rng) if map_type == "corridors" \
              else gen(size=resolution, rng=sub_rng)

        # Oracle ground truth
        t0 = time.perf_counter()
        gt = generate_labels_for_map(occ, L, W).astype(np.uint8)  # (4, H, W)
        times_oracle.append(time.perf_counter() - t0)

        # Model prediction
        t0 = time.perf_counter()
        pred = predict_basic(model, occ, L, W, resolution, device, threshold)
        times_model.append(time.perf_counter() - t0)

        m = _per_direction_metrics(pred, gt)
        m["map_idx"] = i
        rows.append(m)

    summary: Dict[str, float] = {}
    keys = [k for k in rows[0].keys() if k != "map_idx"]
    for k in keys:
        vals = [r[k] for r in rows]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    summary["model_time_ms_mean"] = float(np.mean(times_model)) * 1000.0
    summary["oracle_time_ms_mean"] = float(np.mean(times_oracle)) * 1000.0
    summary["speedup"] = (np.mean(times_oracle) / np.mean(times_model)) \
        if np.mean(times_model) > 0 else float("nan")

    return {"per_map": rows, "summary": summary}


def evaluate_continuous_angle(
    model: torch.nn.Module,
    map_type: str,
    num_maps: int,
    L: int,
    W: int,
    resolution: int,
    device: str,
    seed: int,
    angles_deg: List[float],
    threshold: float = 0.5,
) -> Dict:
    """Evaluate the 5→1 angle-conditioned binary model."""
    gen = MAP_GENERATORS[map_type]
    rows: List[Dict] = []
    for i in range(num_maps):
        sub_rng = np.random.default_rng(seed + i + 1)
        occ = gen(size=resolution, rng=sub_rng)
        for ang in angles_deg:
            gt = continuous_angle_viability(occ, L, W, ang).astype(np.uint8)
            pred = predict_continuous_angle(
                model, occ, L, W, ang, resolution, device, threshold
            )
            m = _binary_metrics(pred, gt)
            m["map_idx"] = i
            m["angle_deg"] = float(ang)
            rows.append(m)

    # Summarise
    summary: Dict[str, float] = {}
    for k in ("iou", "dice", "acc"):
        vals = [r[k] for r in rows]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    # Per-angle breakdown
    by_angle: Dict[float, List[float]] = defaultdict(list)
    for r in rows:
        by_angle[r["angle_deg"]].append(r["iou"])
    summary["per_angle_iou_mean"] = {
        f"{ang:.0f}": float(np.mean(v)) for ang, v in sorted(by_angle.items())
    }
    return {"per_query": rows, "summary": summary}


def evaluate_cost_map(
    model: torch.nn.Module,
    map_type: str,
    num_maps: int,
    L: int,
    W: int,
    resolution: int,
    device: str,
    seed: int,
) -> Dict:
    """Evaluate the 3→4 cost-map regression model."""
    gen = MAP_GENERATORS[map_type]
    rows: List[Dict] = []
    for i in range(num_maps):
        sub_rng = np.random.default_rng(seed + i + 1)
        occ = gen(size=resolution, rng=sub_rng)
        gt4 = escape_cost_map(occ, L, W, direction=None)  # (4, H, W) float32
        pred4 = predict_cost_map(model, occ, L, W, resolution, device)
        # Per-direction metrics
        per_dir: Dict[str, float] = {}
        names = ["N", "S", "E", "W"]
        for c, n in enumerate(names):
            mm = _regression_metrics(pred4[c], gt4[c])
            for kk, vv in mm.items():
                per_dir[f"{kk}_{n}"] = vv
        per_dir["map_idx"] = i
        rows.append(per_dir)

    summary: Dict[str, float] = {}
    keys = [k for k in rows[0].keys() if k != "map_idx"]
    for k in keys:
        vals = [r[k] for r in rows]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    return {"per_map": rows, "summary": summary}


# ===========================================================================
# Visualisation
# ===========================================================================

def save_qualitative_grid(
    model: torch.nn.Module,
    map_type: str,
    L: int,
    W: int,
    resolution: int,
    device: str,
    out_path: Path,
    seed: int = 0,
    n_examples: int = 4,
) -> None:
    """Save a 4×3 qualitative grid (occupancy, GT, prediction)."""
    import matplotlib.pyplot as plt

    gen = MAP_GENERATORS[map_type]
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3.2 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for r in range(n_examples):
        sub_rng = np.random.default_rng(seed + 100 + r)
        occ = gen(size=resolution, rng=sub_rng)
        gt = generate_labels_for_map(occ, L, W).astype(np.uint8)
        pred = predict_basic(model, occ, L, W, resolution, device)

        # Show channel 0 (North) for compactness — same channel for both
        axes[r, 0].imshow(occ, cmap="gray")
        axes[r, 0].set_title(f"{map_type} #{r}")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(gt[0], cmap="viridis", vmin=0, vmax=1)
        axes[r, 1].set_title("Oracle (North)")
        axes[r, 1].axis("off")
        axes[r, 2].imshow(pred[0], cmap="viridis", vmin=0, vmax=1)
        axes[r, 2].set_title("Model (North)")
        axes[r, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot transfer evaluation.")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to trained checkpoint (.pth).")
    p.add_argument("--train_dataset", type=str, default="houseexpo",
                   help="Name of training dataset (informational only).")
    p.add_argument("--test_map_type", type=str, default="maze",
                   choices=list(MAP_GENERATORS.keys()),
                   help="Procedural map family to test on.")
    p.add_argument("--oracle_type", type=str, default="basic",
                   choices=["basic", "continuous_angle", "cost_map"],
                   help="Must match the training mode of the checkpoint.")
    p.add_argument("--num-maps", type=int, default=30,
                   help="Number of procedural maps to generate.")
    p.add_argument("--robot-length", type=int, default=30)
    p.add_argument("--robot-width", type=int, default=20)
    p.add_argument("--resolution", type=int, default=None,
                   help="Override resolution (default: read from checkpoint).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold for binary modes.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default="outputs/zero_shot_transfer")
    p.add_argument("--angles-deg", type=float, nargs="+",
                   default=[0.0, 30.0, 60.0, 90.0, 135.0, 210.0],
                   help="Test angles for continuous_angle mode.")
    p.add_argument("--save-grid", action="store_true",
                   help="Save a qualitative figure of N example predictions.")
    return p.parse_args()


def _load_model(checkpoint_path: str, device: str, oracle_type: str
                ) -> Tuple[torch.nn.Module, int]:
    """Load model + recover resolution. Returns (model, resolution)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        # Fallback configuration
        if oracle_type == "basic":
            cfg = {"in_channels": 3, "classes": 4}
        elif oracle_type == "continuous_angle":
            cfg = {"in_channels": 5, "classes": 1}
        elif oracle_type == "cost_map":
            cfg = {"in_channels": 3, "classes": 4}
        logger.warning("No 'config' in checkpoint; using fallback %s", cfg)

    model = MultiRobotViabilityUNet(**cfg).to(device)
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_dict_key])
    model.eval()

    resolution = ckpt.get("resolution", 512)
    return model, int(resolution)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    model, default_res = _load_model(args.model_path, device, args.oracle_type)
    resolution = args.resolution or default_res
    logger.info("Loaded checkpoint: %s  resolution=%d  oracle_type=%s",
                args.model_path, resolution, args.oracle_type)

    L, W_robot = args.robot_length, args.robot_width
    logger.info("Robot: L=%d  W=%d", L, W_robot)
    logger.info("Test maps: %d × %s", args.num_maps, args.test_map_type)

    # ---- Run the right evaluator ----
    if args.oracle_type == "basic":
        result = evaluate_basic(
            model, args.test_map_type, args.num_maps, L, W_robot,
            resolution, device, args.seed, args.threshold,
        )
    elif args.oracle_type == "continuous_angle":
        result = evaluate_continuous_angle(
            model, args.test_map_type, args.num_maps, L, W_robot,
            resolution, device, args.seed, args.angles_deg, args.threshold,
        )
    elif args.oracle_type == "cost_map":
        result = evaluate_cost_map(
            model, args.test_map_type, args.num_maps, L, W_robot,
            resolution, device, args.seed,
        )
    else:
        raise ValueError(f"Unknown oracle_type: {args.oracle_type}")

    # ---- Print summary ----
    summary = result["summary"]
    logger.info("==== TRANSFER SUMMARY ====")
    for k, v in summary.items():
        if isinstance(v, dict):
            logger.info("  %s:", k)
            for kk, vv in v.items():
                logger.info("    %s = %s", kk, vv)
        else:
            logger.info("  %-25s = %s", k, v)

    # ---- Persist results ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.train_dataset}_to_{args.test_map_type}_{args.oracle_type}"
    json_path = out_dir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "per_item": result.get("per_map") or result.get("per_query"),
        }, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    if args.save_grid and args.oracle_type == "basic":
        grid_path = out_dir / f"{tag}_qualitative.png"
        save_qualitative_grid(
            model, args.test_map_type, L, W_robot, resolution, device,
            grid_path, seed=args.seed, n_examples=4,
        )
        logger.info("Saved grid: %s", grid_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())