#!/usr/bin/env python3
"""
Phase 9 — PRM Comparison: Standard vs TrapAwarePRM.

Runs both planners on 5 HouseExpo maps that contain traps.
Generates comparison figures and saves stats to JSON.

Usage:
    python scripts/local/09_run_prm_comparison.py --checkpoint outputs/best.pth
    python scripts/local/09_run_prm_comparison.py --oracle-only  # no model needed
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.data.map_loader import load_map
from src.oracle import generate_labels_for_map
from src.integration.prm import PRMComparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--oracle-only", action="store_true",
                   help="Use Oracle viability only (no model needed)")
    p.add_argument("--num-maps",    type=int, default=5)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--show",        action="store_true")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str):
    from src.models.unet import MultiRobotViabilityUNet
    model = MultiRobotViabilityUNet()
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_viability(model, occupancy, robot_length, robot_width, device):
    """Run model forward pass → (4, H, W) float32 viability."""
    H, W = occupancy.shape
    inp  = np.zeros((1, 3, H, W), dtype=np.float32)
    inp[0, 0] = occupancy.astype(np.float32)
    inp[0, 1] = robot_length / 512
    inp[0, 2] = robot_width  / 512
    with torch.no_grad():
        logits = model(torch.from_numpy(inp).to(device))
        probs  = torch.sigmoid(logits)
    return probs[0].cpu().numpy()


def find_start_goal(occupancy, margin=30):
    """Find start (top-left free) and goal (bottom-right free)."""
    H, W = occupancy.shape
    for r in range(margin, H - margin):
        for c in range(margin, W - margin):
            if occupancy[r, c] == 1:
                start = (r, c)
                break
        else:
            continue
        break
    else:
        start = (margin, margin)

    for r in range(H - margin, margin, -1):
        for c in range(W - margin, margin, -1):
            if occupancy[r, c] == 1:
                goal = (r, c)
                break
        else:
            continue
        break
    else:
        goal = (H - margin, W - margin)

    return start, goal


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = None
    if args.checkpoint and not args.oracle_only:
        print(f"Loading model from: {args.checkpoint}")
        model = load_model(args.checkpoint, device)
        print("  [OK] model loaded")

    # Robot size for demo
    robot_length, robot_width = 30, 20

    # Select maps with interesting traps (use test split)
    import pandas as pd
    df       = pd.read_csv("data/manifest.csv")
    test_ids = df[df["split"] == "test"]["filename"].tolist()
    map_files = [Path("data/raw_maps") / f.replace(".npy", ".json")
                 for f in test_ids
                 if (Path("data/raw_maps") / f.replace(".npy", ".json")).exists()]
    map_files = map_files[:args.num_maps]

    print(f"Running comparison on {len(map_files)} maps...")
    print(f"  Robot: {robot_length}×{robot_width}  "
          f"samples: {args.num_samples}  threshold: {args.threshold}")

    all_results = []
    out_dir     = Path("outputs/figures/discopygal")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, json_path in enumerate(map_files):
        print(f"\n[{i+1}/{len(map_files)}] {json_path.stem[:24]}")

        occupancy = load_map(json_path)
        free_pct  = occupancy.mean() * 100
        print(f"  free={free_pct:.1f}%")

        # Oracle labels
        oracle = generate_labels_for_map(occupancy, robot_length, robot_width)

        # Model viability
        model_v = None
        if model is not None:
            model_v = predict_viability(
                model, occupancy, robot_length, robot_width, device
            )

        start, goal = find_start_goal(occupancy)
        print(f"  start={start} goal={goal}")

        comp = PRMComparison(
            occupancy           = occupancy,
            oracle_labels       = oracle,
            model_viability     = model_v,
            num_samples         = args.num_samples,
            k_nn                = 10,
            viability_threshold = args.threshold,
            trap_penalty        = 5.0,
        )

        results = comp.run(start, goal, seed=42 + i)

        # Print per-map results
        for r in results:
            print(f"  {r.planner_name:30s}  "
                  f"trap={r.trap_rate:.1%}  "
                  f"path={'OK' if r.path_found else 'FAIL'}  "
                  f"time={r.build_time_ms:.0f}ms")

        comp.plot(
            results,
            save_path=str(out_dir / f"comparison_{json_path.stem[:16]}.png"),
            show=args.show,
        )

        # Collect stats
        all_results.append({
            "map": json_path.stem,
            "free_pct": free_pct,
            "planners": [
                {
                    "name":          r.planner_name,
                    "trap_rate":     r.trap_rate,
                    "trap_count":    r.trap_count,
                    "n_nodes":       r.n_nodes,
                    "path_found":    r.path_found,
                    "path_length":   r.path_length,
                    "build_time_ms": r.build_time_ms,
                }
                for r in results
            ],
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    planner_names = [r["name"] for r in all_results[0]["planners"]]
    for name in planner_names:
        rates = [
            next(p["trap_rate"] for p in m["planners"] if p["name"] == name)
            for m in all_results
        ]
        print(f"  {name:35s}: mean trap rate = {np.mean(rates):.1%}")

    # Save JSON
    out_json = Path("outputs/results/prm_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_json}")
    print(f"Figures in {out_dir}/")


if __name__ == "__main__":
    main()