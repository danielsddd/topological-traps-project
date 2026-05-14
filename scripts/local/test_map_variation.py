"""
scripts/local/test_map_variation.py

Fleet benchmark averaged over N randomly-sampled maps.

Run from the project root:
    python scripts/local/test_map_variation.py

    # More maps, more repeats
    python scripts/local/test_map_variation.py --num-maps 20 --repeats 5

    # Reproducible random selection
    python scripts/local/test_map_variation.py --seed 123

    # Specific checkpoint
    python scripts/local/test_map_variation.py --checkpoint outputs/viability_XYZ/checkpoints/best_iou.pth
"""

import sys
import argparse
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import importlib.util
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Import helpers from scripts/evaluate.py via importlib
# (scripts/ has no __init__.py so package-style imports always fail)
# ---------------------------------------------------------------------------
_eval_spec = importlib.util.spec_from_file_location(
    "evaluate",
    project_root / "scripts" / "evaluate.py",
)
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)

build_batch_input      = _eval_mod.build_batch_input
_time_model_sequential = _eval_mod._time_model_sequential
_time_model_batched    = _eval_mod._time_model_batched
_sync                  = _eval_mod._sync
load_config            = _eval_mod.load_config

from src.models.unet   import MultiRobotViabilityUNet
from src.utils.helpers import get_device


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fleet benchmark averaged over randomly-sampled maps"
    )
    parser.add_argument(
        "--num-maps", type=int, default=10,
        help="Number of maps to sample randomly (default: 10)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed repetitions per (N, map) pair (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for map selection (default: 42)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: auto-detect latest)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config("configs/config.yaml")
    dev    = torch.device(get_device())
    res    = config.get("data", {}).get("resolution", 512)

    FLEET_SIZES = [
        (20, 15), (30, 20), (40, 25), (25, 18), (35, 22),
        (15, 10), (45, 30), (28, 18), (32, 21), (38, 24),
    ]
    NS = [1, 5, 10]

    # ---- Checkpoint -------------------------------------------------------
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint
    else:
        candidates = sorted(
            (project_root / "outputs").glob("viability_*/checkpoints/best_iou.pth")
        )
        if not candidates:
            sys.exit("ERROR: No checkpoint found. Pass --checkpoint explicitly.")
        checkpoint = candidates[-1]

    if not checkpoint.exists():
        sys.exit(f"ERROR: Checkpoint not found: {checkpoint}")

    # ---- Random map sample ------------------------------------------------
    processed_dir = Path(config["paths"]["processed_maps"])
    all_maps = sorted(processed_dir.glob("*.npy"))
    if not all_maps:
        sys.exit(f"ERROR: No .npy files found in {processed_dir}")

    num_maps = min(args.num_maps, len(all_maps))
    if num_maps < args.num_maps:
        print(f"Warning: only {len(all_maps)} maps available, using all of them.")

    random.seed(args.seed)
    selected = random.sample(all_maps, num_maps)

    # ---- Load model -------------------------------------------------------
    model = MultiRobotViabilityUNet.from_checkpoint(str(checkpoint), device=str(dev))
    model.eval()
    model.to(dev)

    # ---- Warm up every batch size that will be timed ---------------------
    # Use the first selected map for warm-up (any map works; shape is the same).
    warmup_occ = np.load(selected[0])
    print(f"Warming up {len(NS)} batch sizes × 3 passes...")
    with torch.no_grad():
        for n in NS:
            warmup_inp = build_batch_input(warmup_occ, FLEET_SIZES[:n], res).to(dev)
            for _ in range(3):
                _ = model(warmup_inp)
    _sync(dev)

    # ---- Print header -----------------------------------------------------
    print(f"\nDevice     : {dev}")
    print(f"Checkpoint : {checkpoint.name}")
    print(f"Maps       : {num_maps} random (seed={args.seed})")
    print(f"Repeats    : {args.repeats} per (N, map)")
    print(f"Maps selected:")
    for m in selected:
        print(f"  • {m.name}")

    # ---- Benchmark loop ---------------------------------------------------
    # For each N, collect timing across all maps × repeats, then print stats.

    print(f"\n{'N':>4} | {'Seq mean (ms)':>14} | {'Seq std':>8} | "
          f"{'Bat mean (ms)':>14} | {'Bat std':>8} | {'Ratio':>8}")
    print("  " + "-" * 68)

    for n in NS:
        sizes = FLEET_SIZES[:n]
        all_seq, all_bat = [], []

        for map_path in selected:
            occupancy = np.load(map_path)

            for _ in range(args.repeats):
                all_seq.append(
                    _time_model_sequential(model, occupancy, sizes, dev, res)
                )
                all_bat.append(
                    _time_model_batched(model, occupancy, sizes, dev, res)
                )

        mean_seq = float(np.mean(all_seq))
        std_seq  = float(np.std(all_seq))
        mean_bat = float(np.mean(all_bat))
        std_bat  = float(np.std(all_bat))
        ratio    = mean_seq / mean_bat if mean_bat > 0 else float("inf")

        print(
            f"  {n:>2} | {mean_seq:>14.1f} | {std_seq:>8.1f} | "
            f"{mean_bat:>14.1f} | {std_bat:>8.1f} | {ratio:>7.2f}x"
        )

    print(f"\nTotal measurements per N: {num_maps * args.repeats} "
          f"({num_maps} maps × {args.repeats} repeats)")


if __name__ == "__main__":
    main()