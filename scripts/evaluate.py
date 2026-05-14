#!/usr/bin/env python3
"""
Model Evaluation Script for Directional Topological Traps.

This script evaluates a trained model:
1. Overall metrics on test set
2. Per-robot-size metrics
3. Generalization evaluation (seen vs unseen sizes)
4. Speed benchmarking (Oracle vs Neural Network)
5. Fleet scaling benchmark (Oracle vs sequential NN vs batched NN)
6. Generate visualizations

Usage:
    # Full evaluation
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_iou.pth

    # Quick evaluation (subset)
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_iou.pth --quick

    # Generalization test only
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_iou.pth --generalization-only

    # Fleet benchmark only
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_iou.pth --fleet-benchmark-only
"""

import argparse
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.data.dataset import MultiRobotViabilityDataset, RobotSpecificDataset
from src.models.unet import MultiRobotViabilityUNet
from src.models.metrics import MetricTracker, compute_per_channel_metrics
from src.evaluation.evaluator import Evaluator, benchmark_speed
from src.visualization.plotting import (
    plot_training_curves,
    plot_predictions,
    plot_per_direction_viability,
    plot_robot_size_comparison,
    setup_plotting_style,
)
from src.oracle.directional_viability import generate_labels_for_map
from src.utils.helpers import get_device, format_time, Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fleet benchmark: the full set of robot sizes to sweep over (N = 1..len).
# The first 4 match the project's actual train/test sizes; the rest extend
# the fleet to demonstrate the batched model's constant-time advantage.
# ---------------------------------------------------------------------------
FLEET_SIZES: List[Tuple[int, int]] = [
    (20, 15),   # small   – train
    (30, 20),   # medium  – train
    (40, 25),   # large   – train
    (25, 18),   # unseen  – test
    (35, 22),   # extra unseen
    (15, 10),   # extra small
    (45, 30),   # extra large
    (28, 18),   # intermediate
    (32, 21),   # intermediate
    (38, 24),   # near-large
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained viability prediction model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: next to checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string accepted by PyTorch: 'cuda', 'cuda:0', 'cpu', etc.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation (reduced samples)",
    )
    parser.add_argument(
        "--generalization-only",
        action="store_true",
        help="Only run generalization evaluation",
    )
    parser.add_argument(
        "--fleet-benchmark-only",
        action="store_true",
        help="Only run the fleet scaling benchmark (skip all other evaluations)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate prediction visualizations",
    )
    parser.add_argument(
        "--num-vis-samples",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--benchmark-speed",
        action="store_true",
        default=True,
        help="Run single-size speed benchmark (Oracle vs NN)",
    )
    parser.add_argument(
        "--fleet-benchmark",
        action="store_true",
        default=True,
        help="Run multi-size fleet scaling benchmark",
    )
    parser.add_argument(
        "--fleet-repeats",
        type=int,
        default=5,
        help="Number of timed repeats per N for fleet benchmark (for stable averages)",
    )
    parser.add_argument(
        "--fleet-oracle-maps",
        type=int,
        default=3,
        help="Number of maps to average Oracle fleet timing over (Oracle is slow)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _sync(dev: torch.device) -> None:
    """
    Synchronise the CUDA device so that all pending GPU kernels finish before
    the caller reads the wall clock.

    Works correctly for any legal CUDA device specifier — "cuda", "cuda:0",
    "cuda:1", etc. — because we check ``dev.type`` instead of comparing the
    raw string.  On CPU this is a no-op.

    Args:
        dev: A ``torch.device`` object (NOT a raw string).
    """
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ---------------------------------------------------------------------------
# Fleet benchmark helpers
# ---------------------------------------------------------------------------

def build_batch_input(
    occupancy: np.ndarray,
    sizes: List[Tuple[int, int]],
    resolution: int = 512,
) -> torch.Tensor:
    """
    Build a batched model input for multiple robot sizes from a single map.

    The model expects 3-channel input:
      - Channel 0: occupancy grid (float, 0/1)
      - Channel 1: robot_length / resolution  (constant spatial map)
      - Channel 2: robot_width  / resolution  (constant spatial map)

    Args:
        occupancy: (H, W) binary occupancy grid (1=free, 0=obstacle).
        sizes: List of (robot_length, robot_width) tuples, length N.
        resolution: Grid resolution used for channel normalisation.

    Returns:
        Float32 tensor of shape (N, 3, H, W).  Not yet on any device.
    """
    H, W = occupancy.shape
    occ_tensor = torch.from_numpy(occupancy.astype(np.float32))  # (H, W)

    samples: List[torch.Tensor] = []
    for robot_l, robot_w in sizes:
        ch_l = torch.full((H, W), fill_value=robot_l / resolution, dtype=torch.float32)
        ch_w = torch.full((H, W), fill_value=robot_w / resolution, dtype=torch.float32)
        sample = torch.stack([occ_tensor, ch_l, ch_w], dim=0)  # (3, H, W)
        samples.append(sample)

    return torch.stack(samples, dim=0)  # (N, 3, H, W)


def _time_oracle_fleet(
    occupancy: np.ndarray,
    sizes: List[Tuple[int, int]],
) -> float:
    """
    Run the Oracle sequentially for every size in ``sizes`` on one map.

    Returns wall time in milliseconds.  This is the baseline cost: every new
    robot size requires a full morphological-erosion + BFS pipeline run.
    """
    t0 = time.perf_counter()
    for robot_l, robot_w in sizes:
        generate_labels_for_map(occupancy, robot_l, robot_w)
    return (time.perf_counter() - t0) * 1000.0


def _time_model_sequential(
    model: torch.nn.Module,
    occupancy: np.ndarray,
    sizes: List[Tuple[int, int]],
    dev: torch.device,
    resolution: int = 512,
) -> float:
    """
    Run the model with one forward pass per robot size (sequential).

    Returns the *total* wall time in ms for all N passes.
    Warm-up must be performed *before* calling this function.

    Args:
        model:      Trained model, already on ``dev`` and in eval mode.
        occupancy:  (H, W) map used for timing.
        sizes:      List of (L, W) pairs; one forward pass per pair.
        dev:        ``torch.device`` object.
        resolution: Normalisation denominator for robot dimensions.
    """
    model.eval()
    total_ms = 0.0
    with torch.no_grad():
        for robot_l, robot_w in sizes:
            inp = build_batch_input(occupancy, [(robot_l, robot_w)], resolution).to(dev)
            _sync(dev)
            t0 = time.perf_counter()
            _ = model(inp)
            _sync(dev)
            total_ms += (time.perf_counter() - t0) * 1000.0
    return total_ms


def _time_model_batched(
    model: torch.nn.Module,
    occupancy: np.ndarray,
    sizes: List[Tuple[int, int]],
    dev: torch.device,
    resolution: int = 512,
) -> float:
    """
    Run the model with ALL robot sizes in a single forward pass.

    Returns wall time in ms for the one pass.  The GPU processes an N-sample
    batch in roughly the same time as a 1-sample batch (up to memory limits),
    so per-size cost approaches zero as N grows.
    Warm-up must be performed *before* calling this function.

    Args:
        model:      Trained model, already on ``dev`` and in eval mode.
        occupancy:  (H, W) map used for timing.
        sizes:      List of (L, W) pairs, all batched in one pass.
        dev:        ``torch.device`` object.
        resolution: Normalisation denominator for robot dimensions.
    """
    model.eval()
    inp = build_batch_input(occupancy, sizes, resolution).to(dev)
    with torch.no_grad():
        _sync(dev)
        t0 = time.perf_counter()
        _ = model(inp)
        _sync(dev)
    return (time.perf_counter() - t0) * 1000.0


def measure_fleet_scaling(
    model: torch.nn.Module,
    config: dict,
    dev: torch.device,
    fleet_sizes: List[Tuple[int, int]] = FLEET_SIZES,
    num_repeats: int = 5,
    num_oracle_maps: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    Benchmark Oracle / Model-sequential / Model-batched as fleet size N grows.

    For each N in 1 .. len(fleet_sizes):
      - Oracle:     N separate pipeline calls sequentially (time = N × T_oracle).
      - Model seq:  N separate single-sample forward passes (time = N × T_nn).
      - Model bat:  ONE forward pass with batch size N      (time ≈ T_nn, flat).

    Model timing is averaged over ``num_repeats`` runs per N.
    Oracle timing is averaged over up to ``num_oracle_maps`` different maps.

    Args:
        model:           Trained U-Net, on ``dev`` and in eval mode.
        config:          Parsed config.yaml dict.
        dev:             ``torch.device`` object.
        fleet_sizes:     Ordered list of (L, W) pairs; sweep runs N=1..len.
        num_repeats:     Timed repetitions per N for the model.
        num_oracle_maps: Maps to average Oracle time over (Oracle is slow).
        verbose:         Print progress table.

    Returns:
        Dict with lists keyed by n_sizes, oracle_ms, sequential_nn_ms,
        batched_nn_ms, speedup_*, plus metadata.  Ready for JSON + plot.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("FLEET SCALING BENCHMARK")
        print("=" * 60)
        print(f"  Fleet sizes tested ({len(fleet_sizes)} total):")
        for i, s in enumerate(fleet_sizes, 1):
            print(f"    N={i}: {s[0]}×{s[1]}")
        print(f"  Model timing repeats per N : {num_repeats}")
        print(f"  Oracle averaged over       : up to {num_oracle_maps} maps")
        print(f"  Device                     : {dev}")

    map_dir = Path(config["paths"]["processed_maps"])
    resolution: int = config.get("data", {}).get("resolution", 512)

    # ---- Load representative maps ----------------------------------------
    map_files = sorted(map_dir.glob("*.npy"))
    if not map_files:
        raise FileNotFoundError(f"No .npy maps found in {map_dir}")

    # Guard: reduce num_oracle_maps if the directory has fewer files.
    actual_oracle_maps = min(num_oracle_maps, len(map_files))
    if actual_oracle_maps < num_oracle_maps:
        logger.warning(
            "Requested %d oracle maps but only %d available; using %d.",
            num_oracle_maps, len(map_files), actual_oracle_maps,
        )
    oracle_maps = [np.load(f) for f in map_files[:actual_oracle_maps]]

    # One fixed map for all model timing calls (eliminates map-to-map variance).
    # Index 0 is safe because we confirmed map_files is non-empty above.
    model_map = oracle_maps[0]

    # ---- Warm-up the model -----------------------------------------------
    # Run several dummy forward passes so that cuDNN auto-tuner, JIT, and
    # CUDA kernel caches are primed *before* any timed section.
    # _sync after the loop ensures every warm-up kernel has retired.
    if verbose:
        print("\nWarming up model (all batch sizes)...")
    with torch.no_grad():
        for n in range(1, len(fleet_sizes) + 1):
            warmup_inp = build_batch_input(
                model_map, fleet_sizes[:n], resolution
            ).to(dev)
            for _ in range(2):           # 2 passes each is enough to cache the algo
                _ = model(warmup_inp)
    _sync(dev)

    # ---- Sweep N = 1 .. len(fleet_sizes) ---------------------------------
    n_sizes_list: List[int] = []
    oracle_ms_list: List[float] = []
    seq_ms_list: List[float] = []
    bat_ms_list: List[float] = []

    if verbose:
        print(
            f"\n{'N':>4} | {'Oracle (ms)':>12} | {'Seq NN (ms)':>12} | "
            f"{'Bat NN (ms)':>12} | {'Speedup seq':>11} | {'Speedup bat':>11}"
        )
        print("  " + "-" * 72)

    for n in range(1, len(fleet_sizes) + 1):
        current_sizes = fleet_sizes[:n]

        # Oracle: average across oracle_maps (deterministic; no repeat needed).
        oracle_times = [_time_oracle_fleet(occ, current_sizes) for occ in oracle_maps]
        avg_oracle_ms = float(np.mean(oracle_times))

        # Model sequential: average across num_repeats timed runs.
        seq_times = [
            _time_model_sequential(model, model_map, current_sizes, dev, resolution)
            for _ in range(num_repeats)
        ]
        avg_seq_ms = float(np.mean(seq_times))

        # Model batched: average across num_repeats timed runs.
        bat_times = [
            _time_model_batched(model, model_map, current_sizes, dev, resolution)
            for _ in range(num_repeats)
        ]
        avg_bat_ms = float(np.mean(bat_times))

        speedup_seq = avg_oracle_ms / avg_seq_ms if avg_seq_ms > 0 else float("inf")
        speedup_bat = avg_oracle_ms / avg_bat_ms if avg_bat_ms > 0 else float("inf")

        n_sizes_list.append(n)
        oracle_ms_list.append(avg_oracle_ms)
        seq_ms_list.append(avg_seq_ms)
        bat_ms_list.append(avg_bat_ms)

        if verbose:
            print(
                f"  {n:>2} | {avg_oracle_ms:>12.1f} | {avg_seq_ms:>12.1f} | "
                f"{avg_bat_ms:>12.1f} | {speedup_seq:>10.1f}x | {speedup_bat:>10.1f}x"
            )

    results = {
        "fleet_sizes": [list(s) for s in fleet_sizes],
        "n_sizes": n_sizes_list,
        "oracle_ms": oracle_ms_list,
        "sequential_nn_ms": seq_ms_list,
        "batched_nn_ms": bat_ms_list,
        "speedup_batched_vs_oracle": [
            o / b if b > 0 else None
            for o, b in zip(oracle_ms_list, bat_ms_list)
        ],
        "speedup_sequential_vs_oracle": [
            o / s if s > 0 else None
            for o, s in zip(oracle_ms_list, seq_ms_list)
        ],
        "num_repeats": num_repeats,
        "num_oracle_maps": actual_oracle_maps,
    }

    if verbose:
        n_max = len(fleet_sizes)
        sp_bat_final = results["speedup_batched_vs_oracle"][-1]
        print(
            f"\n  At N={n_max}: "
            f"Oracle={oracle_ms_list[-1]:.1f} ms, "
            f"Seq NN={seq_ms_list[-1]:.1f} ms, "
            f"Bat NN={bat_ms_list[-1]:.1f} ms  "
            f"→ batched speedup: {sp_bat_final:.1f}×"
        )

    return results


# ---------------------------------------------------------------------------
# Fleet benchmark plot
# ---------------------------------------------------------------------------

def plot_fleet_scaling(
    fleet_results: Dict,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """
    Produce two publication-quality subplots:
      Left  – Total processing time (ms) vs fleet size N.
      Right – Speedup over Oracle vs fleet size N.

    The left panel shows that Oracle and sequential NN grow linearly while
    batched NN stays essentially flat.  The right panel makes the growing
    advantage immediately obvious.

    Args:
        fleet_results: Dict returned by measure_fleet_scaling().
        output_path:   Where to save the PNG figure.
        verbose:       Print save path.
    """
    setup_plotting_style()

    ns = fleet_results["n_sizes"]
    oracle_ms = fleet_results["oracle_ms"]
    seq_ms = fleet_results["sequential_nn_ms"]
    bat_ms = fleet_results["batched_nn_ms"]
    # Filter out any None entries (shouldn't occur, but be defensive)
    speedup_seq = [(n, v) for n, v in zip(ns, fleet_results["speedup_sequential_vs_oracle"]) if v is not None]
    speedup_bat = [(n, v) for n, v in zip(ns, fleet_results["speedup_batched_vs_oracle"]) if v is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left: absolute timing -------------------------------------------
    ax = axes[0]
    ax.plot(ns, oracle_ms, "r-o", linewidth=2.2, markersize=6, label="Oracle (sequential)")
    ax.plot(ns, seq_ms,    "b-s", linewidth=2.2, markersize=6, label="NN sequential (1 pass / size)")
    ax.plot(ns, bat_ms,    "g-^", linewidth=2.2, markersize=6, label="NN batched (1 pass total)")

    ax.set_xlabel("Number of robot sizes queried (N)", fontsize=12)
    ax.set_ylabel("Total processing time (ms)", fontsize=12)
    ax.set_title("Fleet Query Time vs Number of Robot Sizes", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    # Annotate the final batched point to highlight near-constant time
    ax.annotate(
        f"Batched NN\n≈{bat_ms[-1]:.0f} ms (flat)",
        xy=(ns[-1], bat_ms[-1]),
        xytext=(ns[-1] - len(ns) * 0.38, bat_ms[-1] + max(oracle_ms) * 0.07),
        fontsize=9,
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.3),
    )

    # ---- Right: speedup --------------------------------------------------
    ax2 = axes[1]
    if speedup_seq:
        ns_seq, vals_seq = zip(*speedup_seq)
        ax2.plot(ns_seq, vals_seq, "b-s", linewidth=2.2, markersize=6, label="Sequential NN vs Oracle")
    if speedup_bat:
        ns_bat, vals_bat = zip(*speedup_bat)
        ax2.plot(ns_bat, vals_bat, "g-^", linewidth=2.2, markersize=6, label="Batched NN vs Oracle")

    ax2.axhline(y=1.0, color="r", linestyle="--", linewidth=1.3, label="Oracle baseline (1×)")
    ax2.set_xlabel("Number of robot sizes queried (N)", fontsize=12)
    ax2.set_ylabel("Speedup over Oracle (×)", fontsize=12)
    ax2.set_title("Speedup vs Oracle as Fleet Size Grows", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ns)

    # Annotate peak batched speedup
    if speedup_bat:
        n_peak, sp_peak = speedup_bat[-1]
        ax2.annotate(
            f"{sp_peak:.0f}× at N={n_peak}",
            xy=(n_peak, sp_peak),
            xytext=(n_peak - len(ns) * 0.32, sp_peak * 0.84),
            fontsize=9,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.3),
        )

    plt.suptitle(
        "Directional Topological Traps – Fleet Query Scalability\n"
        "(single map, varying number of robot sizes)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"  Fleet scaling plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Standard evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_overall(model, test_loader, dev: torch.device, verbose=True):
    """Compute overall metrics on the test set."""
    if verbose:
        print("\n" + "=" * 60)
        print("OVERALL TEST SET EVALUATION")
        print("=" * 60)

    evaluator = Evaluator(model, device=str(dev))
    metrics = evaluator.evaluate_dataset(test_loader, verbose=verbose)

    if verbose:
        print(f"\nResults:")
        print(f"  IoU:       {metrics.get('iou', 0):.4f}")
        print(f"  Dice:      {metrics.get('dice', 0):.4f}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")

        print(f"\nPer-direction IoU:")
        for dir_name in ["N", "S", "E", "W"]:
            key = f"iou_{dir_name}"
            if key in metrics:
                print(f"  {dir_name}: {metrics[key]:.4f}")

    return metrics


def evaluate_per_robot_size(model, config, dev: torch.device, quick=False, verbose=True):
    """Evaluate the model separately for each robot size."""
    if verbose:
        print("\n" + "=" * 60)
        print("PER-ROBOT-SIZE EVALUATION")
        print("=" * 60)

    map_dir = config["paths"]["processed_maps"]
    label_base_dir = config["paths"]["labels_dir"]
    manifest_path = config["paths"]["manifest"]

    train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]
    test_sizes = [tuple(s) for s in config["robot_sizes"]["test_only"]]
    all_sizes = train_sizes + test_sizes

    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    test_files = manifest[manifest["split"] == "test"]["filename"].tolist()
    if quick:
        test_files = test_files[:100]

    evaluator = Evaluator(model, device=str(dev))
    results = {}

    for length, width in tqdm(all_sizes, desc="Robot sizes"):
        # label_dir convention matches evaluator.py: robot_{L}x{W}
        label_dir = Path(label_base_dir) / f"robot_{length}x{width}"

        metrics = evaluator.evaluate_robot_size(
            map_dir=map_dir,
            label_dir=str(label_dir),
            robot_length=length,
            robot_width=width,
            file_list=test_files,
            batch_size=16,
        )
        metrics["is_train_size"] = (length, width) in train_sizes
        results[(length, width)] = metrics

    if verbose:
        print(f"\nResults by robot size:")
        print(f"  {'Size':>8} | {'IoU':>8} | {'Dice':>8} | {'Type':>8}")
        print("  " + "-" * 45)
        for size, m in results.items():
            size_type = "Train" if m["is_train_size"] else "Test"
            print(
                f"  {size[0]:>2}x{size[1]:>3} | {m.get('iou', 0):>8.4f} | "
                f"{m.get('dice', 0):>8.4f} | {size_type:>8}"
            )

    return results


def evaluate_generalization(model, config, dev: torch.device, quick=False, verbose=True):
    """Compare seen vs unseen robot size performance."""
    if verbose:
        print("\n" + "=" * 60)
        print("GENERALIZATION EVALUATION")
        print("=" * 60)

    map_dir = config["paths"]["processed_maps"]
    label_base_dir = config["paths"]["labels_dir"]
    manifest_path = config["paths"]["manifest"]

    train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]
    test_sizes = [tuple(s) for s in config["robot_sizes"]["test_only"]]

    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    test_files = manifest[manifest["split"] == "test"]["filename"].tolist()
    if quick:
        test_files = test_files[:100]

    evaluator = Evaluator(model, device=str(dev))
    results = evaluator.evaluate_generalization(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        train_sizes=train_sizes,
        test_only_sizes=test_sizes,
        file_list=test_files,
    )

    if verbose:
        summary = results["summary"]
        print(f"\nGeneralization Summary:")
        print(f"  Seen sizes (train):")
        print(f"    Avg IoU:  {summary['seen_avg_iou']:.4f}")
        print(f"    Avg Dice: {summary['seen_avg_dice']:.4f}")
        print(f"  Unseen sizes (test):")
        print(f"    Avg IoU:  {summary['unseen_avg_iou']:.4f}")
        print(f"    Avg Dice: {summary['unseen_avg_dice']:.4f}")
        print(f"  Generalization gap (IoU): {summary['generalization_gap_iou']:.4f}")

    return results


def run_speed_benchmark(model, config, dev: torch.device, verbose=True):
    """
    Benchmark single-size Oracle vs Neural Network inference speed.

    Both Oracle and NN are timed on the same set of maps so the averages
    are directly comparable.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SPEED BENCHMARK")
        print("=" * 60)

    map_dir = config["paths"]["processed_maps"]
    resolution: int = config.get("data", {}).get("resolution", 512)

    # Use the medium training robot size as the representative
    robot_length, robot_width = 30, 20

    # Load the same N maps for both Oracle and NN so the averages are comparable
    num_benchmark_maps = 10
    map_files = sorted(Path(map_dir).glob("*.npy"))
    actual_maps = min(num_benchmark_maps, len(map_files))
    if actual_maps < num_benchmark_maps:
        logger.warning(
            "Requested %d benchmark maps but only %d available; using %d.",
            num_benchmark_maps, len(map_files), actual_maps,
        )
    occupancies = [np.load(f) for f in map_files[:actual_maps]]

    # --- Oracle ---
    print(f"\nBenchmarking Oracle ({actual_maps} maps)...")
    oracle_times = []
    for occ in tqdm(occupancies, desc="Oracle"):
        t0 = time.perf_counter()
        _ = generate_labels_for_map(occ, robot_length, robot_width)
        oracle_times.append((time.perf_counter() - t0) * 1000.0)
    avg_oracle_ms = float(np.mean(oracle_times))

    # --- Neural Network: warm-up then timed on the exact same maps ---
    print(f"Benchmarking Neural Network ({actual_maps} maps)...")
    model.eval()

    warmup_inp = build_batch_input(
        occupancies[0], [(robot_length, robot_width)], resolution
    ).to(dev)
    with torch.no_grad():
        for _ in range(10):
            _ = model(warmup_inp)
    _sync(dev)

    nn_times = []
    with torch.no_grad():
        for occ in tqdm(occupancies, desc="Neural Network"):
            inp = build_batch_input(
                occ, [(robot_length, robot_width)], resolution
            ).to(dev)
            _sync(dev)
            t0 = time.perf_counter()
            _ = model(inp)
            _sync(dev)
            nn_times.append((time.perf_counter() - t0) * 1000.0)

    avg_nn_ms = float(np.mean(nn_times))

    results = {
        "oracle_avg_ms": avg_oracle_ms,
        "nn_avg_ms": avg_nn_ms,
        "speedup": avg_oracle_ms / avg_nn_ms if avg_nn_ms > 0 else float("inf"),
        "num_maps": actual_maps,
        "robot_size": [robot_length, robot_width],
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Oracle average:  {avg_oracle_ms:.2f} ms/map")
        print(f"  NN average:      {avg_nn_ms:.2f} ms/map")
        print(f"  Speedup:         {results['speedup']:.1f}x")

    return results


def generate_visualizations(
    model, config, dev: torch.device, output_dir, num_samples=10, verbose=True
):
    """Generate per-sample prediction visualisation figures."""
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

    map_dir = config["paths"]["processed_maps"]
    label_base_dir = config["paths"]["labels_dir"]
    manifest_path = config["paths"]["manifest"]
    resolution: int = config.get("data", {}).get("resolution", 512)

    train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]
    robot_length, robot_width = train_sizes[1]  # medium training size

    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    test_files = manifest[manifest["split"] == "test"]["filename"].tolist()[:num_samples]

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # label_dir convention matches evaluator.py: robot_{L}x{W}
    label_dir = Path(label_base_dir) / f"robot_{robot_length}x{robot_width}"

    model.eval()

    for i, filename in enumerate(tqdm(test_files, desc="Visualizing")):
        map_path = Path(map_dir) / filename
        label_path = label_dir / filename

        if not map_path.exists() or not label_path.exists():
            continue

        occupancy = np.load(map_path)
        labels = np.load(label_path)

        inp = build_batch_input(
            occupancy, [(robot_length, robot_width)], resolution
        ).to(dev)
        with torch.no_grad():
            logits = model(inp)
            predictions = torch.sigmoid(logits)[0].cpu().numpy()  # (4, H, W)

        all_dir_path = figures_dir / f"all_directions_{i:03d}.png"
        plot_per_direction_viability(
            occupancy=occupancy,
            labels=labels,
            predictions=predictions,
            output_path=str(all_dir_path),
            title=f"All Directions – {filename}",
        )

    if verbose:
        print(f"\nSaved {num_samples} visualizations to: {figures_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Config ---
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # --- Device ---
    # Convert the device string to a torch.device ONCE here.
    # Every function receives the torch.device object, never the raw string.
    # This makes "cuda", "cuda:0", "cuda:1", "cpu" all work correctly.
    raw_device = args.device or get_device()
    dev = torch.device(raw_device)
    print(f"Device: {dev}")

    # --- Output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DIRECTIONAL TOPOLOGICAL TRAPS - EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")

    # --- Model ---
    print("\nLoading model...")
    model = MultiRobotViabilityUNet.from_checkpoint(args.checkpoint, device=str(dev))
    model = model.to(dev)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    all_results: Dict = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat(),
        "device": str(dev),
    }

    # ==========================================================
    # --fleet-benchmark-only short-circuit
    # ==========================================================
    if args.fleet_benchmark_only:
        fleet_results = measure_fleet_scaling(
            model=model,
            config=config,
            dev=dev,
            fleet_sizes=FLEET_SIZES,
            num_repeats=args.fleet_repeats,
            num_oracle_maps=args.fleet_oracle_maps,
            verbose=True,
        )
        all_results["fleet_scaling"] = fleet_results

        plot_fleet_scaling(
            fleet_results=fleet_results,
            output_path=output_dir / "figures" / "fleet_scaling.png",
        )

        results_path = output_dir / "fleet_scaling_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("FLEET BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}")
        return

    # ==========================================================
    # --generalization-only short-circuit
    # ==========================================================
    if args.generalization_only:
        gen_results = evaluate_generalization(model, config, dev, quick=args.quick)
        all_results["generalization"] = gen_results
    else:
        # --- Full evaluation path ---

        from torch.utils.data import DataLoader
        from src.data.dataset import create_dataloaders

        train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]

        _, _, test_loader = create_dataloaders(
            map_dir=config["paths"]["processed_maps"],
            label_base_dir=config["paths"]["labels_dir"],
            manifest_path=config["paths"]["manifest"],
            robot_sizes=train_sizes,
            batch_size=16,
            num_workers=4,
        )

        overall_metrics = evaluate_overall(model, test_loader, dev)
        all_results["overall"] = overall_metrics

        per_size_results = evaluate_per_robot_size(model, config, dev, quick=args.quick)
        all_results["per_robot_size"] = {
            f"{k[0]}x{k[1]}": v for k, v in per_size_results.items()
        }

        gen_results = evaluate_generalization(model, config, dev, quick=args.quick)
        all_results["generalization"] = gen_results

        if args.benchmark_speed:
            speed_results = run_speed_benchmark(model, config, dev)
            all_results["speed_benchmark"] = speed_results

        if args.fleet_benchmark:
            fleet_results = measure_fleet_scaling(
                model=model,
                config=config,
                dev=dev,
                fleet_sizes=FLEET_SIZES,
                num_repeats=args.fleet_repeats,
                num_oracle_maps=args.fleet_oracle_maps,
                verbose=True,
            )
            all_results["fleet_scaling"] = fleet_results

            plot_fleet_scaling(
                fleet_results=fleet_results,
                output_path=output_dir / "figures" / "fleet_scaling.png",
            )

        if args.visualize:
            generate_visualizations(
                model, config, dev, output_dir,
                num_samples=args.num_vis_samples,
            )

    # --- Persist JSON results ---
    results_path = output_dir / "evaluation_results.json"

    def _convert_keys(obj):
        """Recursively convert non-string dict keys to strings for JSON."""
        if isinstance(obj, dict):
            return {str(k): _convert_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_keys(item) for item in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(_convert_keys(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()