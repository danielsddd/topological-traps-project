#!/usr/bin/env python3
"""
scripts/benchmark_prm.py

TrapAwarePRM vs Standard PRM benchmark.

Runs both planners on N randomly-selected test maps and compares:
  - Trap sample rate  : fraction of roadmap nodes in topological traps
  - Path trap exposure: fraction of path nodes in trap regions
  - Build time        : ms to construct the roadmap

No DiscoPyGal required — uses the pure-NumPy implementation in
src/integration/prm.py.

Run from the project root:
    python scripts/benchmark_prm.py

    # More maps, custom checkpoint
    python scripts/benchmark_prm.py --num-maps 10 --checkpoint outputs/.../best_iou.pth

    # Quick sanity check
    python scripts/benchmark_prm.py --num-maps 3 --num-samples 300
"""

import sys
import json
import random
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.models.unet   import MultiRobotViabilityUNet
from src.utils.helpers import get_device
from src.integration.prm import StandardPRM, TrapAwarePRM, PRMComparison, PRMResult

# Load build_batch_input from evaluate.py via importlib
_eval_spec = importlib.util.spec_from_file_location(
    "evaluate", project_root / "scripts" / "evaluate.py"
)
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
build_batch_input = _eval_mod.build_batch_input


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TrapAwarePRM vs StandardPRM benchmark"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint path (default: auto-detect latest)")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--num-maps", type=int, default=8,
                   help="Number of test maps to benchmark (default: 8)")
    p.add_argument("--num-samples", type=int, default=500,
                   help="PRM roadmap nodes per planner (default: 500)")
    p.add_argument("--k-nn", type=int, default=10,
                   help="K nearest neighbours in roadmap (default: 10)")
    p.add_argument("--viability-threshold", type=float, default=0.5,
                   help="Minimum viability to accept a sample (default: 0.5)")
    p.add_argument("--trap-penalty", type=float, default=5.0,
                   help="Edge weight penalty factor for trap regions (default: 5.0)")
    p.add_argument("--uniform-ratio", type=float, default=0.15,
                   help="Fraction of PRM nodes placed unconditionally, bypassing "
                        "viability filter. Increase to restore connectivity on "
                        "trap-dense maps (default: 0.15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to save results (default: latest exp evaluation dir)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_exp(base: Path) -> Path:
    candidates = sorted(base.glob("viability_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit("ERROR: No viability_* directories found. Run training first.")
    return candidates[-1]


def find_checkpoint(exp_dir: Path) -> Path:
    for name in ("best_iou.pth", "last.pth"):
        p = exp_dir / "checkpoints" / name
        if p.exists():
            return p
    sys.exit(f"ERROR: No checkpoint found in {exp_dir / 'checkpoints'}")


def predict_viability(
    model: torch.nn.Module,
    occupancy: np.ndarray,
    robot_length: int,
    robot_width: int,
    resolution: int,
    dev: torch.device,
) -> np.ndarray:
    """Run the model on one map, return (4, H, W) float32 viability."""
    inp = build_batch_input(occupancy, [(robot_length, robot_width)], resolution).to(dev)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.sigmoid(logits)
    return probs[0].cpu().numpy()  # (4, H, W)


def find_start_goal(
    occupancy: np.ndarray,
    margin: int = 20,
    seed: int = 0,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Pick two free pixels that are far apart.

    Strategy: split the map into 4 quadrants, pick one free pixel
    in the top-left quadrant as start and one in the bottom-right as goal.
    Falls back to random free pixels if quadrants are empty.
    """
    H, W = occupancy.shape
    rng  = np.random.default_rng(seed)

    def random_free_in_region(r0, r1, c0, c1):
        region = occupancy[r0:r1, c0:c1]
        free   = np.argwhere(region > 0)
        if len(free) == 0:
            return None
        idx = rng.integers(len(free))
        return (int(free[idx, 0]) + r0, int(free[idx, 1]) + c0)

    start = random_free_in_region(margin, H // 2,     margin, W // 2)
    goal  = random_free_in_region(H // 2, H - margin, W // 2, W - margin)

    if start is None or goal is None:
        # Fall back: any two free pixels
        free = np.argwhere(occupancy > 0)
        if len(free) < 2:
            sys.exit("ERROR: Map has fewer than 2 free pixels.")
        idx   = rng.choice(len(free), size=2, replace=False)
        start = tuple(free[idx[0]])
        goal  = tuple(free[idx[1]])

    return start, goal


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_trap_rate_comparison(
    all_results: List[List[PRMResult]],
    out: Path,
    dpi: int = 180,
):
    """
    Bar chart: mean trap sample rate for Standard PRM vs TrapAwarePRM (NN).
    One group per map, with a summary group.
    """
    setup_style()

    std_rates  = [r[0].trap_rate for r in all_results]
    trap_rates = [r[1].trap_rate for r in all_results if len(r) > 1]

    n = len(std_rates)
    x = np.arange(n + 1)   # +1 for the mean summary bar
    w = 0.35

    std_vals  = std_rates  + [float(np.mean(std_rates))]
    trap_vals = trap_rates + [float(np.mean(trap_rates))]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 5), dpi=dpi)

    bars_std  = ax.bar(x - w/2, std_vals,  w, label="Standard PRM",
                       color="#C00000", edgecolor="black", linewidth=0.7)
    bars_trap = ax.bar(x + w/2, trap_vals, w, label="TrapAwarePRM (NN)",
                       color="#4472C4", edgecolor="black", linewidth=0.7)

    # Value labels
    for bar in list(bars_std) + list(bars_trap):
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Highlight mean bars
    for bars in [bars_std, bars_trap]:
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2)
        bars[-1].set_hatch("//")

    xlabels = [f"Map {i+1}" for i in range(n)] + ["Mean"]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Trap sample rate (fraction of roadmap nodes)")
    ax.set_title("Trap Encounter Rate: Standard PRM vs TrapAwarePRM")
    ax.legend()
    ax.set_ylim(0, min(1.0, max(std_vals + trap_vals) * 1.25))
    ax.grid(axis="y", alpha=0.3)

    # Annotate mean reduction
    mean_std  = np.mean(std_rates)
    mean_trap = np.mean(trap_rates)
    reduction = (mean_std - mean_trap) / mean_std * 100 if mean_std > 0 else 0
    ax.text(0.98, 0.95,
            f"Trap reduction: {reduction:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


def plot_roadmap_overlay(
    occupancy: np.ndarray,
    oracle_labels: np.ndarray,
    std_result:  PRMResult,
    trap_result: PRMResult,
    start: Tuple[int, int],
    goal:  Tuple[int, int],
    out: Path,
    map_idx: int,
    dpi: int = 180,
):
    """
    Side-by-side visual: map + trap mask + roadmap nodes coloured by trap/safe.
    Left = Standard PRM (many red nodes in traps).
    Right = TrapAwarePRM (few red nodes).
    """
    # Trap mask: free pixel where Oracle says no direction is viable
    trap_mask = (occupancy == 1) & (oracle_labels.max(axis=0) == 0)

    def make_bg(occ, traps):
        H, W = occ.shape
        rgb  = np.ones((H, W, 3), dtype=np.float32)  # white
        rgb[occ == 0]  = [0.2, 0.2, 0.2]            # dark = obstacle
        rgb[traps]     = [1.0, 0.85, 0.85]           # pink = trap region
        return rgb

    bg = make_bg(occupancy, trap_mask)

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=dpi)

    for ax, result, title in [
        (axes[0], std_result,  f"Standard PRM  (trap rate: {std_result.trap_rate:.2f})"),
        (axes[1], trap_result, f"TrapAwarePRM  (trap rate: {trap_result.trap_rate:.2f})"),
    ]:
        ax.imshow(bg, origin="upper")

        # Colour nodes: red if in trap, green if safe
        nodes = result.nodes.astype(int)
        for row, col in nodes:
            if 0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]:
                color = "red" if trap_mask[row, col] else "limegreen"
                ax.plot(col, row, "o", color=color, markersize=2.5,
                        alpha=0.6, markeredgewidth=0)

        # Draw path
        if result.path:
            path_arr = np.array(result.path)
            ax.plot(path_arr[:, 1], path_arr[:, 0],
                    "b-", linewidth=1.8, alpha=0.8, label="Path")

        # Start / goal markers
        ax.plot(start[1], start[0], "g*", markersize=12, label="Start",
                markeredgecolor="black", markeredgewidth=0.5)
        ax.plot(goal[1],  goal[0],  "r*", markersize=12, label="Goal",
                markeredgecolor="black", markeredgewidth=0.5)

        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=8, markerscale=1.5)

    # Shared legend for node colours
    red_patch   = mpatches.Patch(color="red",       label="Trap node")
    green_patch = mpatches.Patch(color="limegreen", label="Safe node")
    pink_patch  = mpatches.Patch(color="#ffdddd",   label="Trap region")
    gray_patch  = mpatches.Patch(color="#333333",   label="Obstacle")
    fig.legend(handles=[green_patch, red_patch, pink_patch, gray_patch],
               loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Roadmap Comparison — Map {map_idx + 1}", fontsize=13, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")

# Add this class right before main()
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    dev    = torch.device(get_device())

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Paths -----------------------------------------------------------
    exp_dir     = find_latest_exp(project_root / "outputs")
    ckpt_path   = Path(args.checkpoint) if args.checkpoint else find_checkpoint(exp_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else \
                  exp_dir / "evaluation" / "prm_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    map_dir       = Path(config["paths"]["processed_maps"])
    label_base    = Path(config["paths"]["labels_dir"])
    manifest_path = Path(config["paths"]["manifest"])
    resolution    = config.get("data", {}).get("resolution", 512)

    # Use the medium training robot size for the benchmark
    robot_length, robot_width = 30, 20

    print(f"Device     : {dev}")
    print(f"Checkpoint : {ckpt_path.name}")
    print(f"Robot size : {robot_length}×{robot_width}")
    print(f"PRM nodes  : {args.num_samples}  k-NN: {args.k_nn}")
    print(f"Maps       : {args.num_maps} (seed={args.seed})")

    # ---- Load model ------------------------------------------------------
    model = MultiRobotViabilityUNet.from_checkpoint(str(ckpt_path), device=str(dev))
    model.eval().to(dev)

    # ---- Select test maps from manifest ----------------------------------
    import pandas as pd
    manifest  = pd.read_csv(manifest_path)
    test_files = manifest[manifest["split"] == "test"]["filename"].tolist()
    selected   = random.sample(test_files, min(args.num_maps, len(test_files)))

    label_dir = label_base / f"robot_{robot_length}x{robot_width}"

    # ---- Run benchmark ---------------------------------------------------
    all_results:   List[List[PRMResult]] = []
    per_map_stats: List[dict]            = []
    vis_maps: List[int] = list(range(args.num_maps))  # visualise first 3

    print(f"\nRunning benchmark on {len(selected)} maps...\n")

    for map_idx, filename in enumerate(tqdm(selected, desc="Maps")):
        map_path   = map_dir / filename
        label_path = label_dir / filename

        if not map_path.exists() or not label_path.exists():
            tqdm.write(f"  SKIP {filename}: map or label not found")
            continue

        occupancy     = np.load(map_path).astype(np.uint8)
        oracle_labels = np.load(label_path).astype(np.uint8)   # (4, H, W)

        # Neural network viability prediction
        viability_nn = predict_viability(
            model, occupancy, robot_length, robot_width, resolution, dev
        )

        # Start / goal (deterministic per map index)
        start, goal = find_start_goal(occupancy, seed=map_idx)

        # Run comparison
        comp = PRMComparison(
            occupancy           = occupancy,
            oracle_labels       = oracle_labels,
            model_viability     = viability_nn,
            num_samples         = args.num_samples,
            k_nn                = args.k_nn,
            viability_threshold = args.viability_threshold,
            trap_penalty        = args.trap_penalty,
            uniform_ratio       = args.uniform_ratio,
        )
        results = comp.run(start=start, goal=goal, seed=args.seed)
        all_results.append(results)

        # Collect per-map stats
        row = {"map": filename, "start": start, "goal": goal}
        for r in results:
            key = r.planner_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            row[f"{key}_trap_rate"]    = r.trap_rate
            row[f"{key}_path_found"]   = r.path_found
            row[f"{key}_path_length"]  = r.path_length
            row[f"{key}_build_ms"]     = r.build_time_ms
        per_map_stats.append(row)

        # Console summary
        std  = results[0]
        trap = results[1] if len(results) > 1 else None
        line = f"  {filename[:30]:30s} | Std trap: {std.trap_rate:.3f}"
        if trap:
            reduction = (std.trap_rate - trap.trap_rate) / std.trap_rate * 100 \
                        if std.trap_rate > 0 else 0
            line += f" | TrapAware: {trap.trap_rate:.3f}  (↓{reduction:.1f}%)"
        tqdm.write(line)

        # Visualisation for first 3 maps
        if map_idx in vis_maps and len(results) >= 2:
            plot_roadmap_overlay(
                occupancy     = occupancy,
                oracle_labels = oracle_labels,
                std_result    = results[0],
                trap_result   = results[1],
                start         = start,
                goal          = goal,
                out           = output_dir / f"roadmap_map{map_idx+1:02d}.png",
                map_idx       = map_idx,
            )

    # ---- Summary figures -------------------------------------------------
    print("\nGenerating figures...")
    plot_trap_rate_comparison(
        all_results = all_results,
        out         = output_dir / "trap_rate_comparison.png",
    )

    # ---- Aggregate stats -------------------------------------------------
    std_rates  = [r[0].trap_rate for r in all_results]
    trap_rates = [r[1].trap_rate for r in all_results if len(r) > 1]

    summary = {
        "num_maps":                  len(all_results),
        "robot_size":                [robot_length, robot_width],
        "num_samples":               args.num_samples,
        "k_nn":                      args.k_nn,
        "viability_threshold":       args.viability_threshold,
        "trap_penalty":              args.trap_penalty,
        "standard_prm": {
            "mean_trap_rate": float(np.mean(std_rates)),
            "std_trap_rate":  float(np.std(std_rates)),
            "path_found_rate": float(np.mean([
                r[0].path_found for r in all_results
            ])),
        },
        "trap_aware_prm_nn": {
            "mean_trap_rate": float(np.mean(trap_rates)),
            "std_trap_rate":  float(np.std(trap_rates)),
            "path_found_rate": float(np.mean([
                r[1].path_found for r in all_results if len(r) > 1
            ])),
            "trap_reduction_pct": float(
                (np.mean(std_rates) - np.mean(trap_rates))
                / np.mean(std_rates) * 100
            ) if np.mean(std_rates) > 0 else 0.0,
        },
        "per_map": per_map_stats,
    }

    results_path = output_dir / "prm_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)


    # ---- Print summary ---------------------------------------------------
    print(f"\n{'='*55}")
    print("PRM BENCHMARK SUMMARY")
    print(f"{'='*55}")
    print(f"  Maps benchmarked : {summary['num_maps']}")
    print(f"  PRM nodes/planner: {args.num_samples}")
    print()
    print(f"  {'Planner':30s} | {'Trap rate':>10} | {'Path found':>10}")
    print(f"  {'-'*55}")
    s = summary["standard_prm"]
    t = summary["trap_aware_prm_nn"]
    print(f"  {'Standard PRM':30s} | {s['mean_trap_rate']:>10.3f} | "
          f"{s['path_found_rate']:>10.1%}")
    print(f"  {'TrapAwarePRM (NN)':30s} | {t['mean_trap_rate']:>10.3f} | "
          f"{t['path_found_rate']:>10.1%}")
    print()
    print(f"  Trap reduction: {t['trap_reduction_pct']:.1f}%")
    print(f"\n  Results → {results_path}")
    print(f"  Figures → {output_dir}")
    print("\nNext: git add + push, then view on GitHub.")


if __name__ == "__main__":
    main()