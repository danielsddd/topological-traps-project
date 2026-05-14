#!/usr/bin/env python3
"""
Phase 2 — Preprocess HouseExpo JSON maps to .npy occupancy grids.

Reads all JSON files from data/raw_maps/
Saves binary occupancy grids to data/processed/<map_id>.npy
Skips files that are already processed (safe to re-run).

Usage:
    python scripts/02_preprocess_maps.py
    python scripts/02_preprocess_maps.py --num-workers 8
    python scripts/02_preprocess_maps.py --resolution 512
"""

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.map_loader import load_map
from configs.config_schema import CorruptedMapError, MapLoadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess HouseExpo maps to .npy grids")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if .npy already exists")
    return p.parse_args()


def process_one(args: tuple) -> dict:
    """
    Worker function — processes a single JSON file.
    Returns a result dict (picklable, safe for multiprocessing).
    """
    json_path, output_dir, resolution, force = args
    map_id = json_path.stem
    out_path = output_dir / f"{map_id}.npy"

    # Skip if already done
    if out_path.exists() and not force:
        return {"map_id": map_id, "status": "skipped"}

    try:
        grid = load_map(json_path, resolution=resolution)
        np.save(out_path, grid)
        return {
            "map_id": map_id,
            "status": "ok",
            "free_ratio": float(grid.mean()),
            "shape": grid.shape,
        }
    except (MapLoadError, CorruptedMapError) as e:
        return {"map_id": map_id, "status": "error", "error": str(e)}
    except Exception as e:
        return {"map_id": map_id, "status": "error", "error": f"Unexpected: {e}"}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_dir = Path("data/raw_maps")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {raw_dir}")
        logger.error("Run: python scripts/local/01_download_houseexpo.py --num-maps 1000")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} JSON files")
    logger.info(f"Output dir : {out_dir.resolve()}")
    logger.info(f"Resolution : {args.resolution}x{args.resolution}")
    logger.info(f"Workers    : {args.num_workers}")

    # Build work items
    work_items = [
        (path, out_dir, args.resolution, args.force)
        for path in json_files
    ]

    # Count already processed
    already_done = sum(
        1 for path in json_files
        if (out_dir / f"{path.stem}.npy").exists() and not args.force
    )
    if already_done > 0:
        logger.info(f"Already processed: {already_done} (skipping, use --force to redo)")

    # Process
    t_start = time.time()
    results = []

    if args.num_workers <= 1:
        # Single-process — easier to debug
        for item in tqdm(work_items, desc="Processing"):
            results.append(process_one(item))
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_one, work_items),
                total=len(work_items),
                desc="Processing",
            ):
                results.append(result)

    elapsed = time.time() - t_start

    # Tally results
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for r in results if r["status"] == "error"]

    # Save stats
    stats = {
        "total": len(results),
        "ok": len(ok),
        "skipped": len(skipped),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "resolution": args.resolution,
        "error_list": [{"map_id": r["map_id"], "error": r["error"]} for r in errors],
    }

    if ok:
        free_ratios = [r["free_ratio"] for r in ok]
        stats["free_ratio_mean"] = round(float(np.mean(free_ratios)), 3)
        stats["free_ratio_std"]  = round(float(np.std(free_ratios)), 3)

    stats_path = out_dir / "preprocessing_stats.json"
    import json as json_mod
    with open(stats_path, "w") as f:
        json_mod.dump(stats, f, indent=2)

    # Report
    print("\n" + "=" * 55)
    print("PREPROCESSING COMPLETE")
    print("=" * 55)
    print(f"  Total     : {stats['total']}")
    print(f"  Processed : {stats['ok']}")
    print(f"  Skipped   : {stats['skipped']} (already existed)")
    print(f"  Errors    : {stats['errors']}")
    if ok:
        print(f"  Free ratio: {stats['free_ratio_mean']:.1%} ± {stats['free_ratio_std']:.1%}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"  Stats     : {stats_path}")

    if errors:
        print(f"\nFailed maps:")
        for e in errors[:10]:
            print(f"  {e['map_id']}: {e['error']}")

    if stats["errors"] > stats["total"] * 0.05:
        logger.warning(f"More than 5% of maps failed — check the error list")
        sys.exit(1)

    print(f"\n✓ Ready for Phase 3: python scripts/03_create_manifest.py")


if __name__ == "__main__":
    main()