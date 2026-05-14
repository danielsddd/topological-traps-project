#!/usr/bin/env python3
"""
scripts/benchmark_prm_hard.py

Hard-map PRM benchmark on a synthetic warehouse with narrow aisles and
dead-end alcoves. Compares three planners:

  1. Standard PRM          — uniform sampling, no viability
  2. TrapAwarePRM (Oracle) — exact ground-truth viability (slow pipeline)
  3. TrapAwarePRM (NN)     — predicted viability (fast pipeline)

Key metrics:
  - Trap sample rate
  - Path-found rate
  - TOTAL pipeline time  (prediction + roadmap build + query)
  - Roadmap build time   (just the PRM phase)
  - Path query time
  - Path length

The Oracle PRM establishes an upper bound on trap avoidance quality.
The NN PRM should match it closely while being ~15× faster overall.

On extremely trap-dense maps (≥50% trap density), the default hybrid-sampling
parameters may need tuning.  Key levers:

  --uniform-ratio   0.3    Increase unconditional fill for better connectivity.
  --vicinity-nodes  30     More nodes near start/goal to guarantee endpoint coverage.
  --threshold       0.4    Lower NN threshold to accept marginal corridor pixels.

Run from the project root:
    python scripts/benchmark_prm_hard.py

    # Tune for connectivity on a trap-dense map
    python scripts/benchmark_prm_hard.py --uniform-ratio 0.3 --vicinity-nodes 30

    # Lower NN threshold (accept pixels with viability ≥ 0.4 instead of 0.5)
    python scripts/benchmark_prm_hard.py --threshold 0.4

    # Larger, harder warehouse
    python scripts/benchmark_prm_hard.py --size 512 --shelves 8 --aisles 12 --dead-ends 12
"""

import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.models.unet                   import MultiRobotViabilityUNet
from src.utils.helpers                 import get_device
from src.integration.prm               import StandardPRM, TrapAwarePRM
from src.oracle.directional_viability  import generate_labels_for_map

_eval_spec = importlib.util.spec_from_file_location(
    "evaluate", project_root / "scripts" / "evaluate.py"
)
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
build_batch_input = _eval_mod.build_batch_input
load_config       = _eval_mod.load_config


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Warehouse PRM benchmark: Standard vs Oracle vs NN TrapAwarePRM"
    )
    p.add_argument("--size",        type=int,   default=512)
    p.add_argument("--shelves",     type=int,   default=6,
                   help="Shelf rows (default 6)")
    p.add_argument("--aisles",      type=int,   default=10,
                   help="Shelf columns / aisle count (default 10)")
    p.add_argument("--aisle-w",     type=int,   default=18,
                   help="Aisle width in pixels (default 18)")
    p.add_argument("--shelf-d",     type=int,   default=28,
                   help="Shelf depth in pixels (default 28)")
    p.add_argument("--dead-ends",   type=int,   default=8,
                   help="Dead-end alcoves injected into shelves (default 8)")
    p.add_argument("--runs",        type=int,   default=10,
                   help="Repeated runs per planner (default 10)")
    p.add_argument("--num-samples", type=int,   default=600)
    p.add_argument("--k-nn",        type=int,   default=12)
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Viability acceptance threshold (default 0.5). "
                        "Lower to 0.3–0.4 if NN is too conservative.")
    p.add_argument("--uniform-ratio", type=float, default=0.15,
                   help="Fraction of PRM nodes placed unconditionally (no viability "
                        "filter). Increase to 0.3–0.5 on very trap-dense maps to "
                        "restore roadmap connectivity. Default 0.15.")
    p.add_argument("--vicinity-nodes", type=int, default=20,
                   help="Extra unconditional nodes placed near start/goal. "
                        "Increase to 30–50 if endpoints are poorly connected. "
                        "Default 20.")
    p.add_argument("--checkpoint",  type=str,   default=None)
    p.add_argument("--config",      type=str,   default="configs/config.yaml")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--output-dir",  type=str,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Warehouse map generator
# ---------------------------------------------------------------------------

def make_warehouse(
    size: int = 512,
    n_shelf_rows: int = 6,
    n_aisles: int = 10,
    aisle_w: int = 18,
    shelf_depth: int = 28,
    margin: int = 20,
    dead_ends: int = 8,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a warehouse-style occupancy grid.

    Layout:
      - Outer obstacle walls
      - Horizontal through-corridors at top and bottom
      - Left-spine vertical main aisle
      - Shelf rows: obstacle blocks separated by narrow aisles
      - Dead-end alcoves: short obstacle pockets extending into the aisle,
        creating regions where the robot can enter but cannot turn around

    Returns:
        (size, size) uint8 — 1=free, 0=obstacle.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    occ  = np.zeros((size, size), dtype=np.uint8)
    wall = 8

    # Interior free space
    occ[wall:size - wall, wall:size - wall] = 1

    usable_h = size - 2 * margin  # noqa: F841 (kept for readability)

    # Horizontal through-corridors
    corridor_h = aisle_w
    occ[margin:margin + corridor_h,               margin:size - margin] = 1
    occ[size - margin - corridor_h:size - margin, margin:size - margin] = 1

    # Vertical left-spine corridor
    occ[margin:size - margin, margin:margin + aisle_w] = 1

    # Shelf rows
    inner_top    = margin + corridor_h
    inner_bottom = size - margin - corridor_h
    inner_height = inner_bottom - inner_top
    row_pitch    = inner_height // (n_shelf_rows + 1)

    shelf_zone_w = (size - 2 * margin) - aisle_w
    seg_pitch    = shelf_zone_w // n_aisles

    # Random dead-end slot selection
    all_slots = [(r, a) for r in range(n_shelf_rows) for a in range(n_aisles)]
    chosen    = rng.choice(
        len(all_slots),
        size=min(dead_ends, len(all_slots)),
        replace=False,
    )
    dead_end_slots = {all_slots[i] for i in chosen}

    for row_i in range(n_shelf_rows):
        row_center   = inner_top + (row_i + 1) * row_pitch
        shelf_top    = row_center - shelf_depth // 2
        shelf_bottom = row_center + shelf_depth // 2

        for col_i in range(n_aisles):
            col_start   = margin + aisle_w + col_i * seg_pitch
            shelf_right = col_start + seg_pitch - aisle_w

            if shelf_right <= col_start:
                continue

            r0 = max(0, shelf_top)
            r1 = min(size, shelf_bottom)
            c0 = max(0, col_start)
            c1 = min(size, shelf_right)
            occ[r0:r1, c0:c1] = 0  # paint shelf obstacle

            # Dead-end alcove: extend shelf downward into the aisle below,
            # creating a pocket the robot can enter but not turn around in.
            if (row_i, col_i) in dead_end_slots:
                alcove_d = aisle_w + 4
                ext_top  = r1
                ext_bot  = min(size, r1 + alcove_d)
                ext_l    = c0 + (c1 - c0) // 4
                ext_r    = c1 - (c1 - c0) // 4
                if ext_r > ext_l and ext_bot > ext_top:
                    occ[ext_top:ext_bot, ext_l:ext_r] = 0

        # Re-open horizontal strip between this row and the next
        strip_top = shelf_bottom
        strip_bot = shelf_bottom + aisle_w
        occ[strip_top:strip_bot, margin:size - margin] = 1

    # Restore outer obstacle border
    occ[:wall, :]  = 0
    occ[-wall:, :] = 0
    occ[:, :wall]  = 0
    occ[:, -wall:] = 0

    return occ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_exp(base: Path) -> Path:
    cands = sorted(base.glob("viability_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        sys.exit("No viability_* experiments found.")
    return cands[-1]


def find_checkpoint(exp_dir: Path) -> Path:
    for name in ("best_iou.pth", "last.pth"):
        p = exp_dir / "checkpoints" / name
        if p.exists():
            return p
    sys.exit(f"No checkpoint in {exp_dir / 'checkpoints'}")


def find_start_goal(occ: np.ndarray, seed: int = 1) -> tuple:
    """Pick start (top-left quadrant) and goal (bottom-right quadrant)."""
    H, W   = occ.shape
    rng    = np.random.default_rng(seed)
    margin = max(H, W) // 16

    def rand_free(r0, r1, c0, c1):
        region = occ[r0:r1, c0:c1]
        free   = np.argwhere(region > 0)
        if not len(free):
            return None
        i = rng.integers(len(free))
        return (int(free[i, 0]) + r0, int(free[i, 1]) + c0)

    start = rand_free(margin, H // 3,        margin,        W // 3)
    goal  = rand_free(2 * H // 3, H - margin, 2 * W // 3, W - margin)

    if start is None or goal is None:
        free = np.argwhere(occ > 0)
        idx  = rng.choice(len(free), 2, replace=False)
        start, goal = tuple(free[idx[0]]), tuple(free[idx[1]])

    return start, goal


def path_length(path) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])))
        for i in range(len(path) - 1)
    )


def planner_trap_rate(planner, trap_mask: np.ndarray) -> float:
    nodes = planner.nodes.astype(int)
    H, W  = trap_mask.shape
    count = sum(
        1 for r, c in nodes
        if 0 <= r < H and 0 <= c < W and trap_mask[r, c]
    )
    return count / max(1, len(nodes))


def make_trap_aware(occ: np.ndarray, viability: np.ndarray, args) -> TrapAwarePRM:
    """Construct a TrapAwarePRM from CLI args."""
    return TrapAwarePRM(
        occupancy           = occ,
        viability_map       = viability,
        viability_threshold = args.threshold,
        num_samples         = args.num_samples,
        k_nn                = args.k_nn,
        uniform_ratio       = args.uniform_ratio,
        vicinity_nodes      = args.vicinity_nodes,
    )


# ---------------------------------------------------------------------------
# Single timed run
# ---------------------------------------------------------------------------

def run_one(planner, start: tuple, goal: tuple, seed: int) -> dict:
    """Build roadmap + query path, return timing and result dict."""
    np.random.seed(seed)

    t0 = time.perf_counter()
    planner.build_roadmap(start, goal)
    build_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    path = planner.query(start, goal)
    query_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "build_ms":  build_ms,
        "query_ms":  query_ms,
        "path":      path,
        "path_len":  path_length(path),
        "found":     path is not None,
        "planner":   planner,
    }


# ---------------------------------------------------------------------------
# Diagnostic: viability map overlay
# ---------------------------------------------------------------------------

def save_viability_diagnostic(
    occ: np.ndarray,
    oracle_labels: np.ndarray,
    viability_nn: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save a 3-panel diagnostic figure showing Oracle vs NN viability.

    Panels:
      1. Oracle max viability (max over 4 dirs): ground truth safe/trap map.
      2. NN max viability: the model's prediction.
      3. Difference (Oracle − NN): positive = NN is overly conservative
         (blue = NN labelled safe pixels as traps → main cause of path failures).

    This figure is the first thing to inspect when the NN path-found rate
    is much lower than the Oracle's.
    """
    oracle_max = oracle_labels.max(axis=0).astype(np.float32)  # (H, W)
    nn_max     = viability_nn.max(axis=0)                       # (H, W)
    diff       = oracle_max - nn_max                            # positive = NN too strict

    # Build obstacle overlay
    obstacle = (occ == 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=130)

    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], np.where(obstacle, np.nan, oracle_max),
         "Oracle max viability\n(ground truth)",       "RdYlGn", 0, 1),
        (axes[1], np.where(obstacle, np.nan, nn_max),
         "NN max viability\n(model prediction)",        "RdYlGn", 0, 1),
        (axes[2], np.where(obstacle, np.nan, diff),
         "Difference  Oracle − NN\n(blue = NN too conservative)",
         "RdBu", -1, 1),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Viability Map Diagnostic — Oracle vs NN\n"
                 "Inspect the difference panel: large blue regions mean "
                 "the NN is over-conservative and may disconnect the roadmap.",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = out_dir / "viability_diagnostic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved viability diagnostic → {out.name}")

    # Also save raw arrays for further analysis
    np.save(out_dir / "viability_oracle_max.npy", oracle_max)
    np.save(out_dir / "viability_nn_max.npy", nn_max)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_overlay(
    occ: np.ndarray,
    trap_mask: np.ndarray,
    planners_data: list,
    start: tuple,
    goal: tuple,
    out: Path,
    run_i: int,
) -> None:
    """
    Side-by-side roadmap overlay for all three planners.

    planners_data: list of (label, planner_obj, path)
    """
    def make_bg(o, t):
        rgb = np.ones((*o.shape, 3))
        rgb[o == 0] = [0.15, 0.15, 0.15]
        rgb[t]      = [1.00, 0.82, 0.82]
        return rgb

    bg = make_bg(occ, trap_mask)
    n  = len(planners_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=130)
    if n == 1:
        axes = [axes]

    colors_planner = ["#C00000", "#4472C4", "#70AD47"]

    for ax, (label, planner, path), col in zip(axes, planners_data, colors_planner):
        ax.imshow(bg, origin="upper", interpolation="nearest")

        nodes = planner.nodes.astype(int)
        H, W  = occ.shape
        for r, c in nodes:
            if 0 <= r < H and 0 <= c < W:
                color = "red" if trap_mask[r, c] else "limegreen"
                ax.plot(c, r, "o", color=color, markersize=2.5,
                        alpha=0.55, markeredgewidth=0)

        if path:
            pa = np.array(path)
            ax.plot(pa[:, 1], pa[:, 0], "-", color=col,
                    linewidth=2.0, alpha=0.85, label="Path")

        ax.plot(start[1], start[0], "g*", ms=13,
                markeredgecolor="k", markeredgewidth=0.5)
        ax.plot(goal[1],  goal[0],  "r*", ms=13,
                markeredgecolor="k", markeredgewidth=0.5)

        trap_rt = planner_trap_rate(planner, trap_mask)
        ax.set_title(
            f"{label}\ntrap rate: {trap_rt:.3f}  path: {'✓' if path else '✗'}",
            fontsize=10,
        )
        ax.axis("off")

    handles = [
        mpatches.Patch(color="limegreen", label="Safe node"),
        mpatches.Patch(color="red",       label="Trap node"),
        mpatches.Patch(color="#ffcccc",   label="Trap region"),
        mpatches.Patch(color="#262626",   label="Obstacle / shelf"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Warehouse Benchmark — Run {run_i + 1}", fontsize=13, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay → {out.name}")


def plot_timing_bars(stats: list, out: Path) -> None:
    """
    Four-panel bar chart: trap rate / total pipeline time / query time / path length.
    """
    labels     = [s["name"] for s in stats]
    trap_rates = [s["avg_trap_rate"]  for s in stats]
    total_ms   = [s["avg_total_ms"]   for s in stats]
    query_ms   = [s["avg_query_ms"]   for s in stats]
    path_lens  = [s["avg_path_len"]   for s in stats]

    colors = ["#C00000", "#4472C4", "#70AD47"]
    x      = np.arange(len(labels))
    width  = 0.55

    def bar_panel(ax, values, title, unit, fmt=".3f"):
        bars = ax.bar(x, values, color=colors, edgecolor="black",
                      linewidth=0.8, width=width)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:{fmt}} {unit}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=130)

    bar_panel(axes[0], trap_rates, "Mean trap sample rate\n(lower = better trap avoidance)",
              "", fmt=".3f")
    bar_panel(axes[1], total_ms,  "Total pipeline time\n(prediction + build)",
              "ms", fmt=".0f")
    bar_panel(axes[2], query_ms,  "Path query time\n(Dijkstra on roadmap)",
              "ms", fmt=".2f")
    bar_panel(axes[3], path_lens, "Mean path length\n(pixels — lower = more direct)",
              "pixels", fmt=".0f")

    plt.suptitle(
        "Warehouse Benchmark — Standard PRM vs Oracle TrapAwarePRM vs NN TrapAwarePRM",
        fontsize=11, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved timing chart → {out.name}")

def _sweep_dir(exp_dir: Path, args) -> Path:
    """
    Return a unique output directory for this parameter combination.

    Default (baseline):   evaluation/prm_benchmark_hard/
    Non-default params:   evaluation/prm_benchmark_hard/ur0.30_vn30_thr0.40/

    This means every sweep variant saves independently without needing
    --output-dir, and the baseline results are never overwritten.
    """
    baseline = (
        args.uniform_ratio  == 0.15 and
        args.vicinity_nodes == 20   and
        args.threshold      == 0.5
    )
    base = exp_dir / "evaluation" / "prm_benchmark_hard"
    if baseline:
        return base
    tag = (
        f"ur{args.uniform_ratio:.2f}"
        f"_vn{args.vicinity_nodes}"
        f"_thr{args.threshold:.2f}"
    )
    return base / tag

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    dev    = torch.device(get_device())
    rng    = np.random.default_rng(args.seed)

    exp_dir    = find_latest_exp(project_root / "outputs")
    ckpt_path  = Path(args.checkpoint) if args.checkpoint else find_checkpoint(exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else \
                 _sweep_dir(exp_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution       = config.get("data", {}).get("resolution", 512)
    robot_l, robot_w = 30, 20

    print(f"Device        : {dev}")
    print(f"Checkpoint    : {ckpt_path.name}")
    print(f"Map           : {args.size}×{args.size}  "
          f"shelves={args.shelves}  aisles={args.aisles}  dead-ends={args.dead_ends}")
    print(f"PRM           : {args.num_samples} nodes  k={args.k_nn}  runs={args.runs}")
    print(f"Sampling      : threshold={args.threshold}  "
          f"uniform_ratio={args.uniform_ratio}  vicinity_nodes={args.vicinity_nodes}")

    # ---- Generate warehouse map ------------------------------------------
    print("\nGenerating warehouse map...")
    occ = make_warehouse(
        size         = args.size,
        n_shelf_rows = args.shelves,
        n_aisles     = args.aisles,
        aisle_w      = args.aisle_w,
        shelf_depth  = args.shelf_d,
        dead_ends    = args.dead_ends,
        rng          = rng,
    )
    np.save(output_dir / "warehouse_map.npy", occ)
    free_pct = occ.mean() * 100
    print(f"Free space    : {free_pct:.1f}%")

    # ---- Oracle labels (timed — expensive step) --------------------------
    print("\nRunning Oracle (generates ground-truth viability labels)...")
    t0            = time.perf_counter()
    oracle_labels = generate_labels_for_map(occ, robot_l, robot_w)   # (4,H,W) uint8
    oracle_ms     = (time.perf_counter() - t0) * 1000.0
    print(f"Oracle time   : {oracle_ms:.1f} ms")

    # Trap mask from Oracle ground truth
    trap_mask       = (occ == 1) & (oracle_labels.max(axis=0) == 0)
    oracle_trap_pct = trap_mask.sum() / max(1, occ.sum()) * 100
    print(f"Trap pixels   : {oracle_trap_pct:.1f}% of free space")

    viability_oracle = oracle_labels.astype(np.float32)  # 0.0 / 1.0

    # ---- NN viability (timed) --------------------------------------------
    print("Running NN inference...")
    model = MultiRobotViabilityUNet.from_checkpoint(str(ckpt_path), device=str(dev))
    model.eval().to(dev)

    # Detect model input channels from first encoder weight
    _enc = {k: v for k, v in model.state_dict().items()
            if 'encoder' in k and 'weight' in k}
    _model_in_ch = next(iter(_enc.values())).shape[1]

    def _make_input(occ_, robot_l_, robot_w_, velocity_norm=0.0):
        """Build model input, adding velocity channel if model expects 4 ch."""
        inp_ = build_batch_input(occ_, [(robot_l_, robot_w_)], resolution)
        if _model_in_ch == 4:
            H_, W_ = occ_.shape
            v_ch = torch.full((1, 1, H_, W_), fill_value=velocity_norm)
            inp_ = torch.cat([inp_, v_ch], dim=1)
        return inp_

    # warm-up
    inp_w = _make_input(occ, robot_l, robot_w).to(dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(inp_w)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    t0  = time.perf_counter()
    inp = _make_input(occ, robot_l, robot_w).to(dev)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.sigmoid(logits)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    nn_ms = (time.perf_counter() - t0) * 1000.0

    # Convention depends on oracle_type stored in checkpoint:
    #   cost_map  → high output = deep trap  → invert to get viability
    #   binary / velocity → high output = viable → use directly
    import torch as _torch
    _raw_ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    _oracle_type = _raw_ckpt.get("oracle_type", "basic")
    del _raw_ckpt

    _probs_np = probs[0].cpu().numpy()  # (4, H, W) — high = viable for binary/velocity
    if _oracle_type == "cost_map":
        # cost_map: invert + normalise so high = viable
        _v = 1.0 - _probs_np
        _lo, _hi = float(_v.min()), float(_v.max())
        viability_nn = (_v - _lo) / (_hi - _lo + 1e-8)
    else:
        # binary / velocity: already high = viable, no inversion needed
        viability_nn = _probs_np  # (4, H, W) in [0, 1]
    print(f"NN time       : {nn_ms:.1f} ms")

    # ---- Viability diagnostic (new) --------------------------------------
    # Always saved — useful for understanding NN vs Oracle disagreement.
    # On a trap-dense map, inspect viability_diagnostic.png first if
    # the NN path-found rate is much lower than the Oracle's.
    print("\nSaving viability diagnostic...")
    save_viability_diagnostic(occ, oracle_labels, viability_nn, output_dir)

    # ---- Start / goal ----------------------------------------------------
    start, goal = find_start_goal(occ, seed=args.seed)
    print(f"\nStart: {start}   Goal: {goal}")

    # ---- Benchmark loop --------------------------------------------------
    # Each entry: (display_label, prediction_ms_for_this_planner, viability_or_None)
    planner_specs = [
        ("Standard PRM",          0.0,       None),
        ("TrapAwarePRM (Oracle)", oracle_ms, viability_oracle),
        ("TrapAwarePRM (NN)",     nn_ms,     viability_nn),
    ]

    # Accumulators: per-planner lists of per-run measurements
    accumulator = {label: {
        "build_ms": [], "query_ms": [], "path_len": [], "trap_rate": [], "found": [],
    } for label, _, _ in planner_specs}

    print(f"\nRunning {args.runs} benchmark runs...\n")

    for run_i in tqdm(range(args.runs), desc="Runs"):
        seed = args.seed + run_i

        run_planners = []  # for overlay figure

        for label, pred_ms, viability in planner_specs:
            if viability is None:
                planner = StandardPRM(occ, args.num_samples, args.k_nn)
            else:
                planner = make_trap_aware(occ, viability, args)

            result = run_one(planner, start, goal, seed)
            acc    = accumulator[label]
            acc["build_ms"].append(result["build_ms"])
            acc["query_ms"].append(result["query_ms"])
            acc["path_len"].append(result["path_len"])
            acc["trap_rate"].append(planner_trap_rate(planner, trap_mask))
            acc["found"].append(float(result["found"]))

            run_planners.append((label, planner, result["path"]))

        # Overlay figure for first 2 runs
        if run_i < 2:
            plot_overlay(
                occ, trap_mask, run_planners, start, goal,
                out   = output_dir / f"warehouse_overlay_run{run_i + 1:02d}.png",
                run_i = run_i,
            )

    # ---- Aggregate results -----------------------------------------------
    def avga(lst):
        return float(np.mean(lst)) if lst else 0.0

    def avg_nonzero(lst):
        pos = [v for v in lst if v is not None and v > 0]
        return float(np.mean(pos)) if pos else 0.0

    def std_(lst):
        return float(np.std(lst)) if lst else 0.0

    stats = []
    for label, pred_ms, _ in planner_specs:
        acc = accumulator[label]
        avg_build  = avg_nonzero(acc["build_ms"])
        avg_total  = avg_build + pred_ms
        stats.append({
            "name":            label,
            "prediction_ms":   round(pred_ms, 2),
            "avg_trap_rate":   avga(acc["trap_rate"]),
            "std_trap_rate":   std_(acc["trap_rate"]),
            "avg_build_ms":    avg_build,
            "avg_query_ms":    avg_nonzero(acc["query_ms"]),
            "avg_path_len":    avg_nonzero(acc["path_len"]),
            "avg_total_ms":    avg_total,
            "path_found_pct":  avga(acc["found"]) * 100.0,
        })

    # ---- Print results table ---------------------------------------------
    std_s = next(s for s in stats if "Standard" in s["name"])
    orc_s = next(s for s in stats if "Oracle"   in s["name"])
    nn_s  = next(s for s in stats if "(NN)"     in s["name"])

    col_w = 25
    header = (
        f"{'Planner':{col_w}} {'Trap':>6} {'Pred':>8} {'Build':>8}"
        f" {'Total':>8} {'Query':>7} {'PathLen':>8} {'Found':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for s in stats:
        print(
            f"{s['name']:{col_w}} "
            f"{s['avg_trap_rate']:>6.3f} "
            f"{s['prediction_ms']:>8.1f} "
            f"{s['avg_build_ms']:>8.1f} "
            f"{s['avg_total_ms']:>8.1f} "
            f"{s['avg_query_ms']:>7.3f} "
            f"{s['avg_path_len']:>8.1f} "
            f"{s['path_found_pct']:>6.1f}%"
        )
    print(sep)
    print(f"*{args.runs} runs · {args.num_samples} nodes/planner · k={args.k_nn} · "
          f"threshold={args.threshold} · uniform_ratio={args.uniform_ratio} · "
          f"vicinity_nodes={args.vicinity_nodes}*")

    # ---- Key findings ----------------------------------------------------
    def trap_reduction(planner_s):
        if std_s["avg_trap_rate"] > 0:
            return (1 - planner_s["avg_trap_rate"] / std_s["avg_trap_rate"]) * 100
        return 0.0

    trap_red_oracle = trap_reduction(orc_s)
    trap_red_nn     = trap_reduction(nn_s)
    speedup         = (orc_s["avg_total_ms"] / nn_s["avg_total_ms"]
                       if nn_s["avg_total_ms"] > 0 else 0.0)

    print(f"\n  Trap reduction  Oracle vs Standard : {trap_red_oracle:.1f}%")
    print(f"  Trap reduction  NN vs Standard     : {trap_red_nn:.1f}%")
    print(f"  Trap rate gap   Oracle vs NN       : "
          f"{abs(orc_s['avg_trap_rate'] - nn_s['avg_trap_rate']):.4f} "
          f"({'NN better' if nn_s['avg_trap_rate'] < orc_s['avg_trap_rate'] else 'Oracle better'})")
    print()
    print(f"  Total pipeline  Oracle : {orc_s['avg_total_ms']:.1f} ms")
    print(f"  Total pipeline  NN     : {nn_s['avg_total_ms']:.1f} ms")
    print(f"  NN pipeline speedup vs Oracle      : {speedup:.1f}×")
    print()
    print(f"  Query time      Oracle : {orc_s['avg_query_ms']:.3f} ms")
    print(f"  Query time      NN     : {nn_s['avg_query_ms']:.3f} ms")
    print(f"  Query time      Std    : {std_s['avg_query_ms']:.3f} ms")
    print()
    print(f"  Path found      Oracle : {orc_s['path_found_pct']:.1f}%")
    print(f"  Path found      NN     : {nn_s['path_found_pct']:.1f}%")
    print(f"  Path found      Std    : {std_s['path_found_pct']:.1f}%")

    # Connectivity health check
    if nn_s["path_found_pct"] < orc_s["path_found_pct"] - 20:
        print(
            "\n  ⚠  NN path-found rate is >20% below Oracle. "
            "The NN viability map may be over-conservative.\n"
            "  Suggested fixes (re-run with one or more of):\n"
            f"    --uniform-ratio {min(args.uniform_ratio + 0.15, 0.5):.2f}   "
            "(increase unconditional fill)\n"
            f"    --vicinity-nodes {args.vicinity_nodes + 10}   "
            "(more endpoint coverage)\n"
            f"    --threshold {max(args.threshold - 0.1, 0.3):.2f}              "
            "(accept more marginal pixels)\n"
            "  Then inspect: viability_diagnostic.png in the output dir."
        )

    # ---- Figures ---------------------------------------------------------
    print("\nGenerating figures...")
    plot_timing_bars(stats, output_dir / "warehouse_timing.png")

    # ---- Save JSON -------------------------------------------------------
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_):    return bool(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            return super().default(o)

    result_path = output_dir / "warehouse_benchmark_results.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "map_config": {k: v for k, v in vars(args).items()},
                "free_pct":              free_pct,
                "trap_pct":              oracle_trap_pct,
                "oracle_prediction_ms":  oracle_ms,
                "nn_prediction_ms":      nn_ms,
                "results":               stats,
            },
            f, indent=2, cls=NumpyEncoder,
        )

    print(f"\nResults → {result_path}")
    print(f"Figures → {output_dir}")
    print("\nNext:")
    print("  python scripts/local/update_results_md_warehouse.py")
    print("  git add outputs/ RESULTS.md && git commit -m 'warehouse benchmark' && git push")


if __name__ == "__main__":
    main()