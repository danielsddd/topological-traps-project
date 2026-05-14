#!/usr/bin/env python3
"""
scripts/demo_dwa_size_sweep.py

Experiment: "Maximum Safe Robot Size" — push robot size until failure.

MOTIVATION
----------
A warehouse operator wants robots as large as possible (to carry more cargo)
while still being able to navigate. This experiment answers:
  "What is the largest robot size where DWA+Viability still achieves >50%
   success, and how does that compare to Vanilla DWA?"

The gap between those two crossover sizes is the "viability margin" —
the practical benefit of the learned viability signal.

EXPERIMENTAL DESIGN
-------------------
  Maps:   150 HouseExpo test-split maps (same oracle trap scenarios)
  Sizes:  Sweep from training sizes through extrapolation, e.g.
          (15,10) → (20,15) → (25,18) → (30,20) → (35,23) → (40,25)
          → (45,28) → (50,30) → (55,33) → (60,35) → (70,40) → (80,45)
  Model:  The continuous-angle U-Net takes (L/res, W/res) as input channels
          — it can, in principle, extrapolate to unseen sizes. Performance
          will degrade for sizes far beyond training range [20x15, 40x25],
          but that graceful degradation is exactly what we want to measure.
  Metric: success%, deadlock% per size. Plot: success vs robot area (L×W).

KEY RESULT EXPECTED
-------------------
  Vanilla DWA: success drops steeply once corridors become tight for the
               robot — it has no foresight and enters dead-ends it cannot
               exit. Curve drops off a cliff.

  DWA+Viability: the learned viability signal marks dead-end corridors as
                 dangerous even for large robots. The planner routes around
                 them for slightly longer before eventually failing when no
                 detour exists. Curve drops off later and more gradually.

  → Area between curves = "viability margin" = practical benefit.

USAGE
-----
  # Full sweep on GPU (~1-2 hours for 12 sizes × 150 maps)
  python scripts/demo_dwa_size_sweep.py \\
      --checkpoint outputs/viability_continuous_angle_*/checkpoints/best_iou.pth \\
      --device cuda --output-dir outputs/dwa_size_sweep

  # Quick test: fewer maps, fewer sizes
  python scripts/demo_dwa_size_sweep.py \\
      --checkpoint <path> --max-maps 20 --device cpu \\
      --size-sweep "15,10 20,15 30,20 40,25 50,30 60,35"

  # After run — update RESULTS.md
  python scripts/local/update_results_md_dwa.py \\
      --dwa-json outputs/dwa_size_sweep/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import MultiRobotViabilityUNet
from src.planning.dwa_planner import DWAConfig, DWAPlanner, DWAState

# Re-use helpers from demo_dwa_viability
from scripts.demo_dwa_viability import (
    EpisodeResult,
    find_trap_scenario,
    load_test_maps,
    run_episode,
)

logger = logging.getLogger(__name__)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

VANILLA_COLOR   = "#E53935"
VIABILITY_COLOR = "#43A047"


# ===========================================================================
# Default size sweep
# ===========================================================================

DEFAULT_SIZES: List[Tuple[int, int]] = [
    (30, 20),   # training size — baseline
    (32, 21),
    (34, 23),
    (36, 24),
    (38, 25),
    (40, 25),   # training size — largest
    (42, 27),
    (44, 29),
    (46, 30),
    (48, 32),
    (50, 33),
]
# Maintain L/W ratio ≈ 1.5 (matches training sizes: 20/15=1.33, 30/20=1.5, 40/25=1.6)


def parse_size_sweep(s: str) -> List[Tuple[int, int]]:
    """Parse '15,10 20,15 30,20' into [(15,10),(20,15),(30,20)]."""
    sizes = []
    for pair in s.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            sizes.append((int(parts[0]), int(parts[1])))
    return sizes


# ===========================================================================
# Figures
# ===========================================================================

def plot_size_sweep(
    results: Dict[str, Dict],   # size_str → {vanilla: [...], viability: [...]}
    training_sizes: List[Tuple[int, int]],
    out_path: Path,
    success_threshold: float = 50.0,
) -> None:
    """
    Plot success rate vs robot area for vanilla and viability planners.
    Mark training range, crossover points, and viability margin.
    """
    def area_of(s: str) -> int:
        L, W = (int(x) for x in s.split("x"))
        return L * W

    size_strs = sorted(results.keys(), key=area_of)
    areas      = []
    van_suc    = []
    via_suc    = []
    van_dead   = []
    via_dead   = []

    for sz in size_strs:
        L, W   = (int(x) for x in sz.split("x"))
        areas.append(L * W)
        vn = results[sz]["vanilla"]
        vi = results[sz]["viability"]
        n  = max(len(vn), 1)
        van_suc.append(100.0 * sum(r.reached_goal for r in vn) / n)
        via_suc.append(100.0 * sum(r.reached_goal for r in vi) / max(len(vi), 1))
        van_dead.append(100.0 * sum(r.deadlocked for r in vn) / n)
        via_dead.append(100.0 * sum(r.deadlocked for r in vi) / max(len(vi), 1))

    training_areas = {L * W for L, W in training_sizes}
    x = np.array(areas)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (metric_vals_v, metric_vals_a, ylabel, title) in enumerate([
        (van_suc, via_suc, "Success rate (%)", "Success Rate vs Robot Size"),
        (van_dead, via_dead, "Deadlock rate (%)", "Deadlock Rate vs Robot Size"),
    ]):
        ax = axes[ax_idx]

        ax.plot(x, metric_vals_v, "o-", color=VANILLA_COLOR,
                linewidth=2.5, markersize=8, label="Vanilla DWA", zorder=4)
        ax.plot(x, metric_vals_a, "s-", color=VIABILITY_COLOR,
                linewidth=2.5, markersize=8, label="DWA+Viability", zorder=4)

        # Shade training range
        if training_areas:
            t_min, t_max = min(training_areas), max(training_areas)
            ax.axvspan(t_min, t_max, alpha=0.08, color="blue",
                       label=f"Training range ({t_min}–{t_max} px²)")

        # Crossover lines (where each planner drops below threshold)
        for vals, color, name in [
            (metric_vals_v, VANILLA_COLOR,   "Vanilla"),
            (metric_vals_a, VIABILITY_COLOR, "Viability"),
        ]:
            if ax_idx == 0:   # success rate
                crossovers = [
                    x[i] for i in range(len(vals))
                    if vals[i] < success_threshold
                ]
                if crossovers:
                    ax.axvline(crossovers[0], color=color, linestyle=":",
                               linewidth=1.5, alpha=0.7)
                    ax.text(crossovers[0], 5, f"{name}\nfails\n>{crossovers[0]:.0f}px²",
                            color=color, fontsize=8, ha="center", va="bottom")

        # Viability margin annotation (success only)
        if ax_idx == 0:
            van_cross = next(
                (x[i] for i in range(len(van_suc)) if van_suc[i] < success_threshold),
                None,
            )
            via_cross = next(
                (x[i] for i in range(len(via_suc)) if via_suc[i] < success_threshold),
                None,
            )
            if van_cross and via_cross and via_cross > van_cross:
                margin = via_cross - van_cross
                ax.annotate(
                    f"Viability margin\n+{margin:.0f} px²",
                    xy=((van_cross + via_cross) / 2, success_threshold + 5),
                    ha="center", fontsize=9, fontweight="bold", color="#1565C0",
                    arrowprops=None,
                )

        ax.axhline(success_threshold, color="grey", linestyle="--",
                   linewidth=1.2, alpha=0.6,
                   label=f"{success_threshold:.0f}% threshold")
        ax.set_xlabel("Robot area (L × W, pixels²)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(-5, 110)
        ax.grid(linestyle=":", alpha=0.4)

        # X-tick labels: "20x15\n300px²"
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{sz}\n{a}px²" for sz, a in zip(size_strs, areas)],
            fontsize=7, rotation=45, ha="right",
        )

    fig.suptitle(
        "DWA Planning: Maximum Safe Robot Size on HouseExpo Test Maps\n"
        "(oracle trap scenarios, continuous-angle viability model)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved size sweep figure -> %s", out_path)


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DWA size sweep: maximum safe robot size experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pth). "
                        "Use continuous_angle checkpoint for best results.")
    p.add_argument("--oracle-type", default=None,
                   choices=["basic", "continuous_angle", "cost_map", "angle_cost_map"])
    p.add_argument("--size-sweep", type=str, default=None,
                   help="Space-separated L,W pairs, e.g. '15,10 20,15 30,20'. "
                        "Defaults to a 12-size sweep from 15x10 to 80x45.")
    p.add_argument("--max-maps", type=int, default=10,
                   help="Max successful episodes per size.")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/dwa_size_sweep")
    p.add_argument("--w-viability", type=float, default=8.0)
    p.add_argument("--n-heading-bins", type=int, default=16)
    p.add_argument("--success-threshold", type=float, default=50.0,
                   help="Success rate below which a size is considered 'failed'.")
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

    # ---- Size sweep definition --------------------------------------------
    if args.size_sweep:
        sweep_sizes = parse_size_sweep(args.size_sweep)
    else:
        sweep_sizes = DEFAULT_SIZES
    logger.info("Size sweep: %d sizes from %s to %s",
                len(sweep_sizes), sweep_sizes[0], sweep_sizes[-1])

    # ---- Load model -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    oracle_type = args.oracle_type or ckpt.get("oracle_type", "continuous_angle")

    fallback = {
        "basic":            {"in_channels": 3, "classes": 4},
        "continuous_angle": {"in_channels": 5, "classes": 1},
        "cost_map":         {"in_channels": 3, "classes": 4},
        "angle_cost_map":   {"in_channels": 5, "classes": 1},
    }
    model_cfg = ckpt.get("config") or fallback.get(oracle_type, {"in_channels": 3, "classes": 4})
    model = MultiRobotViabilityUNet(**model_cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()

    resolution      = int(ckpt.get("resolution", 512))
    raw_sizes       = ckpt.get("robot_sizes", [])
    training_sizes  = [(int(s[0]), int(s[1])) for s in raw_sizes if len(s) == 2]

    # Fallback: if checkpoint doesn't store robot_sizes, use known training sizes.
    # The continuous_angle model was trained on these three sizes.
    if not training_sizes:
        training_sizes = [(20, 15), (30, 20), (40, 25)]
        logger.info("robot_sizes not in checkpoint — using default training sizes %s",
                    training_sizes)

    logger.info("Loaded model: oracle_type=%s  resolution=%d  training_sizes=%s",
                oracle_type, resolution, training_sizes)

    # ---- Paths ------------------------------------------------------------
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)
        processed_dir = PROJECT_ROOT / yaml_cfg["paths"]["processed_maps"]
        labels_root   = PROJECT_ROOT / yaml_cfg.get("paths", {}).get("labels_dir", "data/labels")
    else:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        labels_root   = PROJECT_ROOT / "data" / "labels"

    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    map_paths = load_test_maps(processed_dir, manifest_path,
                               max_maps=None, seed=args.seed)
    if not map_paths:
        logger.error("No maps found in %s", processed_dir)
        return 1
    logger.info("Pool: %d maps · target: %d episodes per size.", len(map_paths), args.max_maps)

    # ---- Shared DWA config ------------------------------------------------
    shared_cfg = dict(
        max_speed=8.0, min_speed=0.5, max_omega=0.6,
        predict_steps=14, n_v=7, n_omega=11,
        n_heading_bins=args.n_heading_bins,
        w_goal_heading=1.0, w_goal_dist=0.6,
        w_clearance=2.0,    w_speed=1.5,
        goal_radius=35.0,   min_clearance=3.0,
    )
    vanilla_cfg   = DWAConfig(**shared_cfg, w_viability=0.0)
    viability_cfg = DWAConfig(**shared_cfg, w_viability=args.w_viability)

    # ---- Main sweep -------------------------------------------------------
    results_by_size: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []

    for robot_L, robot_W in sweep_sizes:
        size_str   = f"{robot_L}x{robot_W}"
        area       = robot_L * robot_W
        in_training = (robot_L, robot_W) in training_sizes
        tag         = " [trained]" if in_training else " [extrap]"
        labels_dir  = labels_root / f"robot_{robot_L}x{robot_W}"
        use_disk    = labels_dir.exists()

        if not use_disk:
            # Use 40×25 labels as proxy for all sizes — 40×25 dead-ends are
            # confirmed traps valid for any robot ≤40px, and conservative
            # (even stricter) for larger robots. The NN model still runs with
            # the actual robot dimensions; only scenario placement uses the proxy.
            proxy_dir = labels_root / "robot_40x25"
            if not proxy_dir.exists():
                proxy_dir = labels_root / f"robot_{training_sizes[-1][0]}x{training_sizes[-1][1]}"
            if proxy_dir.exists():
                labels_dir = proxy_dir
                logger.info("Size %s: using 40×25 proxy labels for scenario placement", size_str)
            else:
                logger.warning("Size %s: no labels found — skipping.", size_str)
                continue

        logger.info("=" * 60)
        logger.info("Size %s  area=%d px²%s", size_str, area, tag)
        logger.info("=" * 60)

        # Adjust goal_radius and min_clearance for robot size
        size_cfg = dict(shared_cfg)
        size_cfg["goal_radius"]   = max(35.0, robot_L * 1.2)
        size_cfg["min_clearance"] = max(3.0, robot_L / 3.0)
        van_cfg_sz = DWAConfig(**size_cfg, w_viability=0.0)
        via_cfg_sz = DWAConfig(**size_cfg, w_viability=args.w_viability)

        vanilla_planner = DWAPlanner(
            model=model, oracle_type=oracle_type,
            robot_L=robot_L, robot_W=robot_W,
            resolution=resolution, device=device,
            config=van_cfg_sz,
        )
        viability_planner = DWAPlanner(
            model=model, oracle_type=oracle_type,
            robot_L=robot_L, robot_W=robot_W,
            resolution=resolution, device=device,
            config=via_cfg_sz,
        )

        van_results: List[EpisodeResult] = []
        via_results: List[EpisodeResult] = []
        rng       = np.random.default_rng(args.seed)
        skipped   = 0
        bfs_depth = 30 + robot_L   # scale trap depth: 30→60, 40→70, 50→80

        pbar = tqdm(total=args.max_maps, desc=size_str, ncols=80)

        for map_path in map_paths:
            if len(van_results) >= args.max_maps:
                break

            label_path = labels_dir / map_path.name
            if not label_path.exists():
                skipped += 1
                continue
            try:
                occ   = np.load(str(map_path)).astype(np.uint8)
                label = np.load(str(label_path))
            except Exception as e:
                logger.debug("Load error %s: %s", map_path.name, e)
                skipped += 1
                continue

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
                max_bfs_to_exit=bfs_depth,
            )
            if scenario is None:
                skipped += 1
                continue

            start, goal = scenario

            van_res, _ = run_episode(
                occ, start, goal, vanilla_planner, dist_tf,
                max_steps=args.max_steps,
            )
            via_res, _ = run_episode(
                occ, start, goal, viability_planner, dist_tf,
                max_steps=args.max_steps,
            )

            van_results.append(van_res)
            via_results.append(via_res)
            pbar.update(1)

        pbar.close()

        n = len(van_results)
        if n == 0:
            logger.warning("Size %s: all maps skipped.", size_str)
            continue

        van_suc  = 100.0 * sum(r.reached_goal for r in van_results) / n
        via_suc  = 100.0 * sum(r.reached_goal for r in via_results) / max(len(via_results), 1)
        van_dead = 100.0 * sum(r.deadlocked   for r in van_results) / n
        via_dead = 100.0 * sum(r.deadlocked   for r in via_results) / max(len(via_results), 1)

        results_by_size[size_str] = {
            "vanilla":   van_results,
            "viability": via_results,
        }

        logger.info(
            "Size %s: n=%d  vanilla=%.1f%% suc  viability=%.1f%% suc  "
            "(deadlock: van=%.1f%% via=%.1f%%)",
            size_str, n, van_suc, via_suc, van_dead, via_dead,
        )

        for pname, rl, suc, dead in [
            ("vanilla",   van_results, van_suc, van_dead),
            ("viability", via_results, via_suc, via_dead),
        ]:
            summary_rows.append({
                "robot_size":    size_str,
                "area_px2":      area,
                "in_training":   in_training,
                "planner":       pname,
                "n":             len(rl),
                "success_rate":  round(suc, 2),
                "deadlock_rate": round(dead, 2),
                "timeout_rate":  round(
                    100.0 * sum(not r.reached_goal and not r.deadlocked for r in rl)
                    / max(len(rl), 1), 2
                ),
            })

    # ---- Summary table ----------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  {'SIZE':12} {'AREA':8} {'PLANNER':18} "
          f"{'SUCCESS%':>9} {'DEADLOCK%':>10} {'N':>5}")
    print("  " + "-" * 68)
    for row in summary_rows:
        tag = "*" if row["in_training"] else " "
        print(f"  {row['robot_size']:12} {row['area_px2']:8} {row['planner']:18} "
              f"{row['success_rate']:>8.1f}% {row['deadlock_rate']:>9.1f}% "
              f"{row['n']:>5} {tag}")
    print("  * = in training set")
    print("=" * 72 + "\n")

    # ---- Figure -----------------------------------------------------------
    plot_size_sweep(
        results_by_size, training_sizes,
        out_dir / "size_sweep.png",
        success_threshold=args.success_threshold,
    )

    # ---- Save JSON --------------------------------------------------------
    json_out = {
        "args": {
            "checkpoint":    args.checkpoint,
            "oracle_type":   oracle_type,
            "sweep_sizes":   [list(s) for s in sweep_sizes],
            "training_sizes": [list(s) for s in training_sizes],
            "n_maps":        len(map_paths),
            "max_steps":     args.max_steps,
            "w_viability":   args.w_viability,
            "device":        device,
            "resolution":    resolution,
        },
        "summary": summary_rows,
        "per_episode": {
            sz: {
                pn: [asdict(r) for r in rl]
                for pn, rl in planners.items()
            }
            for sz, planners in results_by_size.items()
        },
    }
    jpath = out_dir / "results.json"
    with open(jpath, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info("Results -> %s", jpath)
    logger.info("All outputs -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())