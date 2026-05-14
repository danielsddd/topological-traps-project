#!/usr/bin/env python3
"""
scripts/demo_dwa_viability.py

Full experiment: Vanilla DWA vs DWA + Viability Cost
on ALL test-split HouseExpo maps x ALL training robot sizes.

EXPERIMENTAL DESIGN
-------------------
  Maps:        150 test-split HouseExpo maps  (data/manifest.csv, split=test)
  Robot sizes: 3 training sizes from checkpoint  (e.g. 20x15, 30x20, 40x25)
  Planners:    Vanilla DWA (w_viability=0) vs DWA+Viability (w_viability=8)
  Total:       150 maps x 3 sizes x 2 planners = 900 episodes

  Start placement: pixel where oracle east-label = 0  (confirmed east trap)
  Goal placement:  pixel where oracle east-label = 1  (confirmed east clear)
  Guarantee: EVERY episode starts the robot in a genuine east-facing trap
             identified by the oracle — the exact pattern the model learned.

RESULTS TABLE
-------------
  Per robot size:  success%, deadlock%, timeout%, N
  Aggregate:       across all 3 sizes
  Figures:         trajectory_grid.png  (3 best-contrast maps)
                   metrics_summary.png  (bar chart per size)
                   timing_comparison.png (NN batch vs Oracle)

USAGE
-----
  # Full run on GPU (~10 min for 150 maps x 3 sizes)
  python scripts/demo_dwa_viability.py \\
      --checkpoint outputs/viability_continuous_angle_*/checkpoints/best_iou.pth \\
      --device cuda --output-dir outputs/dwa_experiment

  # Quick sanity check — 10 maps, CPU
  python scripts/demo_dwa_viability.py \\
      --checkpoint <path> --max-maps 10 --device cpu --skip-timing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import MultiRobotViabilityUNet
from src.planning.dwa_planner import DWAConfig, DWAPlanner, DWAState

logger = logging.getLogger(__name__)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

VANILLA_COLOR   = "#E53935"
VIABILITY_COLOR = "#43A047"
ORACLE_COLOR    = "#C62828"
NN_COLOR        = "#1565C0"


# ===========================================================================
# Data loading
# ===========================================================================

def load_test_maps(
    processed_dir: Path,
    manifest_path: Path,
    max_maps: Optional[int],
    seed: int,
) -> List[Path]:
    """Return paths of test-split maps. Falls back to all maps if no manifest."""
    if manifest_path.exists():
        import pandas as pd
        df    = pd.read_csv(manifest_path)
        files = df[df["split"] == "test"]["filename"].tolist()
        paths = [processed_dir / f for f in files if (processed_dir / f).exists()]
        logger.info("Manifest: %d test-split maps.", len(paths))
    else:
        paths = sorted(processed_dir.glob("*.npy"))
        logger.warning("No manifest.csv — using all %d maps.", len(paths))

    if max_maps and len(paths) > max_maps:
        rng   = np.random.default_rng(seed)
        idx   = rng.choice(len(paths), size=max_maps, replace=False)
        paths = [paths[i] for i in sorted(idx)]
        logger.info("Subsampled to %d maps.", max_maps)

    return paths


def _bfs_dist_to_targets(
    occ: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    BFS distance (in pixels) from every free pixel to the nearest target pixel,
    travelling only through free (occ==1) pixels.

    This correctly respects walls — unlike binary_dilation which uses
    Euclidean distance and can place the robot 40px from an exit that is
    actually 300px of navigable path away.

    Args:
        occ:     (H, W) uint8 occupancy grid.
        targets: (H, W) bool mask of target pixels (east_label=1 & free).

    Returns:
        (H, W) float32 array. np.inf where unreachable.
    """
    from collections import deque
    H, W  = occ.shape
    dist  = np.full((H, W), np.inf, dtype=np.float32)
    queue: deque = deque()

    for y, x in zip(*np.where(targets)):
        dist[y, x] = 0.0
        queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        d    = dist[y, x]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and occ[ny, nx] == 1 \
                    and np.isinf(dist[ny, nx]):
                dist[ny, nx] = d + 1.0
                queue.append((ny, nx))

    return dist


def find_trap_scenario(
    occ: np.ndarray,
    east_label: np.ndarray,
    dist_tf: np.ndarray,
    robot_L: int,
    rng: np.random.Generator,
    min_dist_px: float = 60.0,     # goal must be at least this far
    max_dist_px: float = 120.0,    # goal at most this far — just past the exit
    max_bfs_to_exit: int = 50,
) -> Optional[Tuple[DWAState, Tuple[float, float]]]:
    """
    Find (start, goal) where:
      - start: east_label=0, BFS distance to exit = 50 steps (deep enough that
               vanilla DWA heading east gets stuck, but exit IS reachable)
      - goal:  east_label=1, 80–200px from start, just past the exit

    WHY this produces a clean viability gap
    ----------------------------------------
    With goal just outside the exit, success ↔ escaped the dead-end.
    Viability signal directly measures trap-escape skill, not map traversal.

    BFS=50: exit is ~50 navigable steps away. DWA horizon = 14×8=112px.
    A straight exit-directed trajectory at max speed covers 112px >  50px BFS
    → some trajectories DO reach east_label=1 (via≈1) → viability planner
    strongly prefers them (cost difference = w_viability × 1.0 = 8 pts).
    Vanilla still goes east (0° heading error) → stays in east_label=0 (via≈0)
    → deeper into trap → never reaches goal 80-200px away.

    max_dist_px=200: goal is close enough that once escaped the robot reaches it
    quickly. No long cross-map navigation that drowns the escape signal.
    """
    from scipy.ndimage import label as scipy_label

    H, W    = occ.shape
    min_clr = max(robot_L / 2.0, 8.0)

    # Connected components — start and goal must share a component (goal reachable)
    labeled_occ, _ = scipy_label(occ)

    # BFS distance to nearest exit pixel
    exit_targets = (east_label == 1) & (occ == 1)
    bfs_dist     = _bfs_dist_to_targets(occ, exit_targets)

    # Start: east_label=0, BFS 5–50 steps to exit, left 60%
    start_mask = (
        (east_label == 0)
        & (occ == 1)
        & (dist_tf >= min_clr)
        & (bfs_dist >= 5)
        & (bfs_dist <= max_bfs_to_exit)
        & np.isfinite(bfs_dist)
    )
    start_mask[:, int(W * 0.6):] = False

    sY, sX = np.where(start_mask)
    if len(sY) < 3:
        return None

    # Goal: east_label=1, right 50%, free, clearance
    goal_mask = (east_label == 1) & (occ == 1) & (dist_tf >= min_clr)
    goal_mask[:, :W // 2] = False
    gY, gX = np.where(goal_mask)
    if len(gY) < 3:
        return None

    # Pre-compute connected-component IDs for fast reachability check
    start_comps = labeled_occ[sY, sX]
    goal_comps  = labeled_occ[gY, gX]

    for _ in range(800):
        si = rng.integers(0, len(sY))
        gi = rng.integers(0, len(gY))

        # Reject if goal not in same connected component as start
        if start_comps[si] != goal_comps[gi]:
            continue

        sx, sy = float(sX[si]), float(sY[si])
        gx, gy = float(gX[gi]), float(gY[gi])
        dist   = np.hypot(gx - sx, gy - sy)

        if (
            min_dist_px <= dist <= max_dist_px
            and abs(gy - sy) <= 80
            and gx > sx
        ):
            return DWAState(x=sx, y=sy, theta=0.0), (gx, gy)

    return None


# ===========================================================================
# Episode
# ===========================================================================

@dataclass
class EpisodeResult:
    reached_goal: bool
    deadlocked: bool
    n_steps: int
    path_length_px: float
    straight_line_dist: float
    path_efficiency: float
    final_dist_to_goal: float
    viability_precompute_ms: float


def _path_len(path: List[Tuple[float, float]]) -> float:
    return float(sum(
        np.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        for i in range(1, len(path))
    ))


def run_episode(
    occ: np.ndarray,
    start: DWAState,
    goal: Tuple[float, float],
    planner: DWAPlanner,
    dist_tf: np.ndarray,
    max_steps: int = 500,
    stuck_window: int = 80,
    stuck_spatial_px: float = 45.0,  # robot circling >45px area is navigating, not stuck
) -> Tuple[EpisodeResult, List[Tuple[float, float]]]:
    """
    stuck_spatial_px: if the robot stays within a 20×20 px box for stuck_window
    steps it is genuinely oscillating/trapped.  A robot on a legitimate detour
    covers >> 20 px in 80 steps (min_speed=0.5 → 40 px minimum).
    Distance-to-goal comparison is intentionally removed: it fires falsely
    whenever the robot moves away from the goal during a beneficial detour,
    which systematically penalises the viability planner.
    """
    cfg  = planner.config
    gx, gy = float(goal[0]), float(goal[1])
    state  = start
    path_xy: List[Tuple[float, float]] = [(state.x, state.y)]
    straight = float(np.hypot(start.x - gx, start.y - gy))
    H, W     = occ.shape

    via_ms = planner.precompute_viability(occ)

    for step in range(max_steps):
        d = float(np.hypot(state.x - gx, state.y - gy))

        if d <= cfg.goal_radius:
            pl = _path_len(path_xy)
            return EpisodeResult(
                True, False, step, pl, straight,
                min(straight / max(pl, 1.0), 1.0), d, via_ms,
            ), path_xy

        # Spatial coverage stuck detection:
        # Only fires if robot is oscillating in a tiny area — not during detours.
        if step >= stuck_window and len(path_xy) >= stuck_window:
            recent_pos = np.array(path_xy[-stuck_window:])
            x_cov = float(recent_pos[:, 0].max() - recent_pos[:, 0].min())
            y_cov = float(recent_pos[:, 1].max() - recent_pos[:, 1].min())
            if max(x_cov, y_cov) < stuck_spatial_px:
                pl = _path_len(path_xy)
                return EpisodeResult(
                    False, True, step, pl, straight, 0.0, d, via_ms
                ), path_xy

        try:
            bv, bw, _ = planner.plan(state, goal, occ, dist_tf)
        except Exception:
            break

        ns = DWAPlanner.apply_motion(state, bv, bw, cfg.dt)
        nx = int(np.clip(round(ns.x), 0, W - 1))
        ny = int(np.clip(round(ns.y), 0, H - 1))
        if occ[ny, nx] == 0:
            break

        path_xy.append((ns.x, ns.y))
        state = ns

    d   = float(np.hypot(state.x - gx, state.y - gy))
    pl  = _path_len(path_xy)
    return EpisodeResult(False, False, max_steps, pl, straight, 0.0, d, via_ms), path_xy


# ===========================================================================
# Timing benchmark
# ===========================================================================

def benchmark_timing(
    model: torch.nn.Module,
    oracle_type: str,
    sizes: List[Tuple[int, int]],
    resolution: int,
    device: str,
    occ: np.ndarray,
    n_heading_bins: int,
    n_repeats: int,
) -> Dict:
    L, W_r = sizes[len(sizes) // 2]
    planner = DWAPlanner(
        model=model, oracle_type=oracle_type,
        robot_L=L, robot_W=W_r, resolution=resolution, device=device,
        config=DWAConfig(n_heading_bins=n_heading_bins, w_viability=8.0),
    )
    for _ in range(3):
        planner.precompute_viability(occ)

    nn_times   = [planner.precompute_viability(occ) for _ in range(n_repeats)]
    nn_ms      = float(np.median(nn_times))
    oracle_ms  = DWAPlanner.time_oracle_N_headings(occ, L, W_r, n_headings=n_heading_bins)
    speedup    = oracle_ms / max(nn_ms, 0.1)

    return {
        "n_heading_bins":          n_heading_bins,
        "nn_batch_ms":             round(nn_ms, 2),
        "oracle_seq_ms":           round(oracle_ms, 2),
        "speedup":                 round(speedup, 1),
        "nn_feasible_for_dwa":     nn_ms < 20.0,
        "oracle_feasible_for_dwa": oracle_ms < 20.0,
        "dwa_cycle_budget_ms":     20.0,
    }


# ===========================================================================
# Figures
# ===========================================================================

def plot_timing(timing: Dict, out: Path) -> None:
    if not timing.get("oracle_seq_ms"):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals   = [timing["oracle_seq_ms"], timing["nn_batch_ms"]]
    labels = [
        f"Oracle\n({timing['n_heading_bins']}x BFS, sequential)",
        f"NN batch\n({timing['n_heading_bins']} headings, 1 GPU call)",
    ]
    bars = ax.bar(labels, vals, color=[ORACLE_COLOR, NN_COLOR],
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax.axhline(20.0, color="#FF8F00", linestyle="--", linewidth=2,
               label="DWA 50 Hz budget (20 ms)")
    ax.text(0.97, 0.97, f"Speedup: {timing['speedup']}x",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, fontweight="bold", color=NN_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=NN_COLOR, linewidth=1.5))
    ax.set_ylabel("Precomputation time (ms)")
    ax.set_title(f"Heading-viability precomputation\n"
                 f"({timing['n_heading_bins']} bins, 512x512 map)")
    ax.legend(); ax.set_ylim(0, max(vals)*1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved timing figure -> %s", out)


def plot_trajectory_grid(examples: List[Dict], out: Path) -> None:
    n = min(len(examples), 3)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(13, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, ex in enumerate(examples[:n]):
        occ  = ex["occ"]
        via_map = ex.get("via_map")
        start = ex["start"]
        gx, gy = ex["goal"]

        for col, (path, result, color, label) in enumerate([
            (ex["vanilla_path"],   ex["vanilla_result"],   VANILLA_COLOR,   "Vanilla DWA"),
            (ex["viability_path"], ex["viability_result"], VIABILITY_COLOR, "DWA+Viability"),
        ]):
            ax = axes[row, col]
            if via_map is not None:
                ax.imshow(np.where(occ == 1, via_map, np.nan),
                          cmap="RdYlGn", vmin=0, vmax=1,
                          origin="upper", interpolation="nearest", alpha=0.8)
                ax.imshow(np.where(occ == 0, 0.1, np.nan), cmap="gray",
                          vmin=0, vmax=1, origin="upper",
                          interpolation="nearest", alpha=0.9)
            else:
                ax.imshow(occ, cmap="gray", origin="upper")

            if len(path) > 1:
                xs = [p[0] for p in path]; ys = [p[1] for p in path]
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, zorder=5)
                ax.plot(xs[-1], ys[-1], "o", markersize=7, color=color, zorder=6)

            ax.plot(start.x, start.y, "s", markersize=10, color="#1565C0", zorder=7)
            ax.plot(gx, gy, "*", markersize=14, color="#FF8F00", zorder=7)

            if result.reached_goal:
                status, scol = f"[GOAL] ({result.n_steps} steps)", VIABILITY_COLOR
            elif result.deadlocked:
                status, scol = f"[DEADLOCK] ({result.n_steps} steps)", ORACLE_COLOR
            else:
                status, scol = f"[TIMEOUT] ({result.n_steps} steps)", "grey"

            ax.text(0.03, 0.97, status, transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold", color=scol,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(f"Robot {ex.get('size_str','')}", fontsize=10)

    fig.suptitle(
        "DWA Trajectory Comparison on HouseExpo Test Maps\n"
        "(RdYlGn = east-viability from NN; red = confirmed trap; blue sq = start; star = goal)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved trajectory grid -> %s", out)


def plot_metrics_summary(
    results_by_size: Dict[str, Dict[str, List[EpisodeResult]]],
    out: Path,
) -> None:
    sizes = list(results_by_size.keys())
    if not sizes:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x  = np.arange(len(sizes))
    bw = 0.35

    for ax_idx, (title, fn, ylabel) in enumerate([
        ("Success Rate (%)",  lambda r: r.reached_goal, "% episodes reaching goal"),
        ("Deadlock Rate (%)", lambda r: r.deadlocked,   "% episodes deadlocked"),
    ]):
        ax = axes[ax_idx]
        van_v, via_v = [], []
        for sz in sizes:
            vn = results_by_size[sz]["vanilla"]
            vi = results_by_size[sz]["viability"]
            van_v.append(100.0 * sum(fn(r) for r in vn) / max(len(vn), 1))
            via_v.append(100.0 * sum(fn(r) for r in vi) / max(len(vi), 1))

        bv = ax.bar(x - bw/2, van_v, bw, color=VANILLA_COLOR,
                    label="Vanilla DWA", alpha=0.85, edgecolor="white")
        ba = ax.bar(x + bw/2, via_v, bw, color=VIABILITY_COLOR,
                    label="DWA+Viability", alpha=0.85, edgecolor="white")
        for bars, vals in [(bv, van_v), (ba, via_v)]:
            for bar, val in zip(bars, vals):
                if val > 3:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 1.5,
                            f"{val:.0f}%", ha="center", va="bottom",
                            fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Robot\n{s}" for s in sizes])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    n = len(results_by_size[sizes[0]]["vanilla"]) if sizes else 0
    fig.suptitle(
        f"DWA Trap Avoidance on HouseExpo Test Maps\n"
        f"({n} maps per robot size — oracle-guided trap scenarios)",
        fontsize=12,
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved metrics summary -> %s", out)


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(
    checkpoint_path: str,
    oracle_type_override: Optional[str],
    device: str,
) -> Tuple[torch.nn.Module, str, List[Tuple[int, int]], int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    oracle_type = oracle_type_override or ckpt.get("oracle_type", "basic")

    fallback = {
        "basic":            {"in_channels": 3, "classes": 4},
        "continuous_angle": {"in_channels": 5, "classes": 1},
        "cost_map":         {"in_channels": 3, "classes": 4},
        "angle_cost_map":   {"in_channels": 5, "classes": 1},
    }
    cfg    = ckpt.get("config") or fallback.get(oracle_type, {"in_channels": 3, "classes": 4})
    model  = MultiRobotViabilityUNet(**cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()

    resolution = int(ckpt.get("resolution", 512))
    raw_sizes  = ckpt.get("robot_sizes", [[20,15],[30,20],[40,25]])
    sizes      = [(int(s[0]), int(s[1])) for s in raw_sizes if len(s) == 2]
    if not sizes:
        sizes = [(20,15),(30,20),(40,25)]

    logger.info("oracle_type=%s  sizes=%s  resolution=%d", oracle_type, sizes, resolution)
    return model, oracle_type, sizes, resolution


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vanilla DWA vs DWA+Viability — full HouseExpo test experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pth).")
    p.add_argument("--oracle-type", default=None,
                   choices=["basic", "continuous_angle", "cost_map", "angle_cost_map"])
    p.add_argument("--max-maps", type=int, default=20,
                   help="Max test maps per robot size. None = all in test split.")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/dwa_experiment")
    p.add_argument("--w-viability", type=float, default=8.0)
    p.add_argument("--n-heading-bins", type=int, default=16)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--skip-timing", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model -------------------------------------------------------
    model, oracle_type, sizes, resolution = load_model(
        args.checkpoint, args.oracle_type, device
    )

    # ---- Paths ------------------------------------------------------------
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        processed_dir = PROJECT_ROOT / yaml_cfg["paths"]["processed_maps"]
        labels_root   = PROJECT_ROOT / yaml_cfg.get("paths", {}).get(
                            "labels_dir", "data/labels")
    else:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        labels_root   = PROJECT_ROOT / "data" / "labels"

    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    # Load ALL test maps — the loop will stop after max_maps successful episodes
    all_map_paths = load_test_maps(processed_dir, manifest_path,
                                   max_maps=None, seed=args.seed)
    if not all_map_paths:
        logger.error("No maps found in %s", processed_dir)
        return 1
    logger.info(
        "Pool: %d test maps. Target: %d successful episodes per robot size.",
        len(all_map_paths), args.max_maps,
    )

    # ---- Shared DWA config ------------------------------------------------
    shared_cfg = dict(
        max_speed=8.0, min_speed=0.5, max_omega=0.6,
        predict_steps=14,   # was 18 — 14×8=112px horizon covers BFS=20 exit
        n_v=7,              # was 10 — 7 speed samples, enough coverage
        n_omega=11,         # was 21 — 11 odd samples includes ω=0
        n_heading_bins=args.n_heading_bins,
        w_goal_heading=1.0, w_goal_dist=0.6,
        w_clearance=2.0,    w_speed=1.5,
        goal_radius=35.0,   min_clearance=3.0,
    )
    vanilla_cfg   = DWAConfig(**shared_cfg, w_viability=0.0)
    viability_cfg = DWAConfig(**shared_cfg, w_viability=args.w_viability)

    # ---- Timing benchmark (once, on first map) ----------------------------
    timing_results: Dict = {}
    if not args.skip_timing:
        logger.info("Running timing benchmark...")
        occ0 = np.load(str(all_map_paths[0])).astype(np.uint8)
        timing_results = benchmark_timing(
            model, oracle_type, sizes, resolution, device, occ0,
            n_heading_bins=args.n_heading_bins,
            n_repeats=args.timing_repeats,
        )
        logger.info(
            "NN batch: %.1f ms | Oracle: %.1f ms | Speedup: %sx",
            timing_results["nn_batch_ms"],
            timing_results["oracle_seq_ms"],
            timing_results.get("speedup", "N/A"),
        )
        plot_timing(timing_results, out_dir / "timing_comparison.png")

    # ---- Main experiment: all maps x all sizes ----------------------------
    results_by_size: Dict[str, Dict[str, List[EpisodeResult]]] = {}
    best_examples:   List[Dict] = []

    for robot_L, robot_W in sizes:
        size_str   = f"{robot_L}x{robot_W}"
        labels_dir = labels_root / f"robot_{robot_L}x{robot_W}"

        if not labels_dir.exists():
            logger.warning("Labels not found: %s — skipping.", labels_dir)
            continue

        logger.info("=" * 60)
        logger.info("Robot %s  |  labels: %s", size_str, labels_dir)
        logger.info("=" * 60)

        vanilla_planner = DWAPlanner(
            model=model, oracle_type=oracle_type,
            robot_L=robot_L, robot_W=robot_W,
            resolution=resolution, device=device,
            config=vanilla_cfg,
        )
        viability_planner = DWAPlanner(
            model=model, oracle_type=oracle_type,
            robot_L=robot_L, robot_W=robot_W,
            resolution=resolution, device=device,
            config=viability_cfg,
        )

        van_results: List[EpisodeResult] = []
        via_results: List[EpisodeResult] = []
        rng     = np.random.default_rng(args.seed)
        skipped = 0

        # Run through all available maps until we have exactly max_maps episodes.
        # Maps that don't yield a valid trap scenario are skipped transparently.
        pbar = tqdm(total=args.max_maps, desc=size_str, ncols=80)

        for map_path in all_map_paths:
            if len(van_results) >= args.max_maps:
                break

            label_path = labels_dir / map_path.name
            if not label_path.exists():
                skipped += 1
                continue

            try:
                occ   = np.load(str(map_path)).astype(np.uint8)
                label = np.load(str(label_path))   # (4, H, W) uint8
            except Exception as e:
                logger.debug("Load error %s: %s", map_path.name, e)
                skipped += 1
                continue

            # East channel index 2  (label order: N=0, S=1, E=2, W=3)
            if label.ndim == 3 and label.shape[0] == 4:
                east_label = label[2].astype(np.uint8)
            elif label.ndim == 3 and label.shape[2] == 4:
                east_label = label[:, :, 2].astype(np.uint8)
            else:
                skipped += 1
                continue

            dist_tf  = distance_transform_edt(occ).astype(np.float32)
            scenario = find_trap_scenario(
                occ, east_label, dist_tf, robot_L, rng=rng,
                max_bfs_to_exit=30 + robot_L,   # 20→50, 30→60, 40→70
            )
            if scenario is None:
                skipped += 1
                continue

            start, goal = scenario

            van_res, van_path = run_episode(
                occ, start, goal, vanilla_planner, dist_tf,
                max_steps=args.max_steps,
            )
            via_res, via_path = run_episode(
                occ, start, goal, viability_planner, dist_tf,
                max_steps=args.max_steps,
            )

            van_results.append(van_res)
            via_results.append(via_res)
            pbar.update(1)

            # Collect best-contrast examples (vanilla deadlocks, viability succeeds)
            if len(best_examples) < 3 and (
                van_res.deadlocked and via_res.reached_goal
                or van_res.deadlocked and not via_res.deadlocked
                or len(best_examples) == 0
            ):
                viability_planner.precompute_viability(occ)
                via_overlay = (
                    np.stack(list(viability_planner._via_cache.values())).min(axis=0)
                    if viability_planner._via_cache else None
                )
                best_examples.append({
                    "occ": occ, "start": start, "goal": goal,
                    "vanilla_path": van_path, "viability_path": via_path,
                    "vanilla_result": van_res, "viability_result": via_res,
                    "via_map": via_overlay, "size_str": size_str,
                    "map_name": map_path.name,
                })

        pbar.close()

        results_by_size[size_str] = {"vanilla": van_results, "viability": via_results}
        n = len(van_results)
        logger.info(
            "Size %s: %d episodes (skipped %d)  "
            "vanilla=%.1f%% success  viability=%.1f%% success",
            size_str, n, skipped,
            100.0 * sum(r.reached_goal for r in van_results) / max(n, 1),
            100.0 * sum(r.reached_goal for r in via_results) / max(n, 1),
        )

    # ---- Summary table ----------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  {'ROBOT':12} {'PLANNER':18} {'SUCCESS%':>9} "
          f"{'DEADLOCK%':>10} {'TIMEOUT%':>9} {'N':>5}")
    print("  " + "-" * 70)

    summary_rows = []
    for size_str, res in results_by_size.items():
        for pname in ["vanilla", "viability"]:
            rl = res[pname]
            n  = len(rl)
            if n == 0:
                continue
            suc  = 100.0 * sum(r.reached_goal for r in rl) / n
            dead = 100.0 * sum(r.deadlocked   for r in rl) / n
            tout = 100.0 * sum(not r.reached_goal and not r.deadlocked for r in rl) / n
            ms   = float(np.mean([r.viability_precompute_ms for r in rl])) \
                   if pname == "viability" else 0.0
            print(f"  {size_str:12} {pname:18} "
                  f"{suc:>8.1f}% {dead:>9.1f}% {tout:>8.1f}% {n:>5}")
            summary_rows.append({
                "robot_size": size_str, "planner": pname, "n": n,
                "success_rate": round(suc, 2),
                "deadlock_rate": round(dead, 2),
                "timeout_rate": round(tout, 2),
                "mean_via_precompute_ms": round(ms, 2),
            })

    if results_by_size:
        all_van = [r for v in results_by_size.values() for r in v["vanilla"]]
        all_via = [r for v in results_by_size.values() for r in v["viability"]]
        print("  " + "-" * 70)
        for pname, rl in [("vanilla  (aggregate)", all_van),
                          ("viability (aggregate)", all_via)]:
            n    = len(rl)
            suc  = 100.0 * sum(r.reached_goal for r in rl) / max(n, 1)
            dead = 100.0 * sum(r.deadlocked   for r in rl) / max(n, 1)
            tout = 100.0 * sum(not r.reached_goal and not r.deadlocked for r in rl) / max(n, 1)
            print(f"  {'AGGREGATE':12} {pname:18} "
                  f"{suc:>8.1f}% {dead:>9.1f}% {tout:>8.1f}% {n:>5}")
    print("=" * 74 + "\n")

    # ---- Figures ----------------------------------------------------------
    plot_trajectory_grid(best_examples, out_dir / "trajectory_grid.png")
    plot_metrics_summary(results_by_size, out_dir / "metrics_summary.png")

    # ---- Save JSON --------------------------------------------------------
    out = {
        "args": {
            "checkpoint":  args.checkpoint,
            "oracle_type": oracle_type,
            "n_maps":      len(all_map_paths),
            "sizes":       [list(s) for s in sizes],
            "max_steps":   args.max_steps,
            "w_viability": args.w_viability,
            "device":      device,
            "resolution":  resolution,
        },
        "timing":      timing_results,
        "summary":     summary_rows,
        "per_episode": {
            sz: {pn: [asdict(r) for r in rl] for pn, rl in planners.items()}
            for sz, planners in results_by_size.items()
        },
    }
    jpath = out_dir / "results.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results -> %s", jpath)
    logger.info("All outputs -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())