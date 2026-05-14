#!/usr/bin/env python3
"""
Phase 4 — Generate Oracle viability labels for all maps × all robot sizes.

Reads occupancy grids from data/processed/
Writes (4, 512, 512) label arrays to data/labels/robot_LxW/

Safe to re-run: skips already-generated labels by default.
Use --force to regenerate everything.

Usage:
    python scripts/04_generate_labels.py                  # all sizes, all maps
    python scripts/04_generate_labels.py --max-maps 10    # quick test
    python scripts/04_generate_labels.py --force          # regenerate all
    python scripts/04_generate_labels.py --verify-only    # check existing labels
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.oracle.generator import LabelGenerator, DatasetVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Generate Oracle viability labels")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--num-workers",  type=int, default=None,
                   help="CPU workers (default: cpu_count - 2)")
    p.add_argument("--max-maps",     type=int, default=None,
                   help="Limit number of maps (for testing)")
    p.add_argument("--force",        action="store_true",
                   help="Recompute even if label already exists")
    p.add_argument("--verify-only",  action="store_true",
                   help="Only verify existing labels, do not generate")
    p.add_argument("--robot-sizes",  nargs="+", default=None,
                   help="Specific sizes e.g. 20x15 30x20 (default: all from config)")
    return p.parse_args()


def parse_size(s: str) -> tuple[int, int]:
    """Parse '30x20' → (30, 20)."""
    parts = s.split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid robot size '{s}' — expected format: LxW e.g. 30x20")
    return int(parts[0]), int(parts[1])


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    processed_dir = Path("data/processed")
    labels_dir    = Path("data/labels")

    # Resolve robot sizes
    if args.robot_sizes:
        robot_sizes = [parse_size(s) for s in args.robot_sizes]
        logger.info(f"Using specified sizes: {robot_sizes}")
    else:
        robot_sizes = cfg.robot.all_sizes
        logger.info(f"Using all sizes from config: {robot_sizes}")

    num_workers = args.num_workers or cfg.oracle.num_workers

    # Sanity checks
    if not processed_dir.exists():
        logger.error(f"Processed dir not found: {processed_dir.resolve()}")
        sys.exit(1)

    map_files = sorted(processed_dir.glob("*.npy"))
    if not map_files:
        logger.error("No .npy files in data/processed/ — run preprocess.py first")
        sys.exit(1)

    n_maps = min(len(map_files), args.max_maps) if args.max_maps else len(map_files)
    logger.info(f"Maps available: {len(map_files)}  |  Will process: {n_maps}")
    logger.info(f"Robot sizes   : {robot_sizes}")
    logger.info(f"Workers       : {num_workers}")
    logger.info(f"Labels dir    : {labels_dir.resolve()}")
    logger.info(f"Force         : {args.force}")

    # ---- Verify only -------------------------------------------------------
    if args.verify_only:
        logger.info("Verify-only mode — checking existing labels...")
        verifier = DatasetVerifier(
            processed_dir=processed_dir,
            labels_dir=labels_dir,
            robot_sizes=robot_sizes,
        )
        results = verifier.run_all_checks(sample_size=100)

        print("\n=== VERIFICATION RESULTS ===")
        all_passed = True
        for check, info in results.items():
            if check == "all_passed":
                continue
            status = "✓" if info["passed"] else "✗"
            print(f"  {status} {check}: ", end="")
            if info["passed"]:
                print("PASSED")
            else:
                print(f"FAILED — {info['error_count']} errors")
                for ex in info.get("examples", [])[:3]:
                    print(f"      {ex}")
                all_passed = False

        print(f"\n{'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
        sys.exit(0 if all_passed else 1)

    # ---- Generate labels ---------------------------------------------------
    gen = LabelGenerator(
        processed_dir=processed_dir,
        labels_dir=labels_dir,
        robot_sizes=robot_sizes,
        num_workers=num_workers,
        skip_existing=not args.force,
    )

    stats = gen.generate_all(max_maps=args.max_maps)

    # Save stats
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    stats_path = Path("outputs/results/oracle_stats.json")
    gen.save_stats(stats_path)

    # Print summary
    print("\n" + "=" * 55)
    print("ORACLE LABEL GENERATION COMPLETE")
    print("=" * 55)
    print(f"  Total tasks  : {stats['total_tasks']}")
    print(f"  Generated    : {stats['completed']}")
    print(f"  Skipped      : {stats['skipped']} (already existed)")
    print(f"  Failed       : {stats['failed']}")
    print(f"  Time         : {stats['elapsed_seconds']:.1f}s")
    print(f"  Speed        : {stats['tasks_per_second']:.1f} maps/s")
    print(f"  Stats saved  : {stats_path}")

    if stats["failed"] > 0:
        logger.warning(f"{stats['failed']} tasks failed — check oracle_stats.json")

    # Count files per size
    print("\nLabel counts per robot size:")
    for L, W in robot_sizes:
        tag = f"robot_{L}x{W}"
        n   = len(list((labels_dir / tag).glob("*.npy")))
        print(f"  {tag}: {n} labels")

    # ---- Quick verification after generation --------------------------------
    if stats["completed"] > 0 and stats["failed"] == 0:
        if args.max_maps:
            # Test run — skip completeness check, only verify generated files
            logger.info("Test run — skipping completeness check (only partial labels)")
            print("\n✓ Test run complete — labels generated and verified individually")
        else:
            logger.info("Running full verification...")
            verifier = DatasetVerifier(
                processed_dir=processed_dir,
                labels_dir=labels_dir,
                robot_sizes=robot_sizes,
            )
            results = verifier.run_all_checks(sample_size=50)
            if results["all_passed"]:
                print("\n✓ Full verification passed")
            else:
                print("\n✗ Verification found issues — check oracle_stats.json")
                sys.exit(1)

    print(f"\n✓ Ready for Phase 5: python scripts/05_eye_test.py")


if __name__ == "__main__":
    main()