"""
Oracle batch label generator.

Generates (4, H, W) viability labels for every (map, robot_size) pair.
Parallelized across maps using multiprocessing.Pool.
Safe to re-run — skips already-generated labels by default.

Classes:
    LabelGenerator  — main generator with resume support
    DatasetVerifier — post-generation integrity checks
"""
from __future__ import annotations

import json
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .directional_viability import generate_labels_for_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level — required for pickling)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    """
    Process one (map_path, robot_size, label_dir, force) job.
    Must be at module level for multiprocessing pickling.
    """
    map_path, robot_length, robot_width, label_dir, force = args
    label_path = Path(label_dir) / Path(map_path).name

    if label_path.exists() and not force:
        return {"status": "skipped", "map": Path(map_path).name}

    try:
        occupancy = np.load(map_path)
        labels    = generate_labels_for_map(occupancy, robot_length, robot_width)

        # Validate before saving
        if labels.shape[0] != 4:
            return {"status": "error", "map": Path(map_path).name,
                    "error": f"Bad shape: {labels.shape}"}
        if labels.max() > 1 or labels.min() < 0:
            return {"status": "error", "map": Path(map_path).name,
                    "error": f"Non-binary values: [{labels.min()},{labels.max()}]"}

        np.save(label_path, labels)
        return {
            "status": "ok",
            "map": Path(map_path).name,
            "viable_ratio": float(labels.mean()),
        }
    except Exception as e:
        return {"status": "error", "map": Path(map_path).name, "error": str(e)}


# ---------------------------------------------------------------------------
# LabelGenerator
# ---------------------------------------------------------------------------

class LabelGenerator:
    """
    Batch label generator for the viability dataset.

    Generates (4, H, W) viability labels for all maps × all robot sizes.
    Supports parallel processing and resuming from interruption.

    Usage:
        gen = LabelGenerator(
            processed_dir="data/processed",
            labels_dir="data/labels",
            robot_sizes=[(20,15), (30,20), (40,25), (25,18)],
            num_workers=16,
        )
        stats = gen.generate_all()
    """

    def __init__(
        self,
        processed_dir: str | Path,
        labels_dir: str | Path,
        robot_sizes: List[Tuple[int, int]],
        num_workers: Optional[int] = None,
        skip_existing: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.labels_dir    = Path(labels_dir)
        self.robot_sizes   = robot_sizes
        self.num_workers   = num_workers or max(1, cpu_count() - 2)
        self.skip_existing = skip_existing

        self.stats: Dict = {
            "total_tasks": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "elapsed_seconds": 0.0,
            "tasks_per_second": 0.0,
        }

    # ------------------------------------------------------------------
    def label_path(self, map_stem: str, L: int, W: int) -> Path:
        return self.labels_dir / f"robot_{L}x{W}" / f"{map_stem}.npy"

    # ------------------------------------------------------------------
    def _build_work_items(
        self,
        max_maps: Optional[int] = None,
        robot_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[tuple]:
        """Build list of (map_path, L, W, label_dir, force) tuples."""
        sizes     = robot_sizes or self.robot_sizes
        map_files = sorted(self.processed_dir.glob("*.npy"))
        if max_maps:
            map_files = map_files[:max_maps]

        items = []
        n_skipped = 0
        for map_path in map_files:
            for L, W in sizes:
                lp = self.label_path(map_path.stem, L, W)
                if self.skip_existing and lp.exists():
                    n_skipped += 1
                    continue
                items.append((str(map_path), L, W, str(lp.parent), False))

        if n_skipped:
            logger.info(f"Skipping {n_skipped} already-generated labels")
        return items

    # ------------------------------------------------------------------
    def _create_label_dirs(self) -> None:
        for L, W in self.robot_sizes:
            (self.labels_dir / f"robot_{L}x{W}").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _run_items(self, items: List[tuple], desc: str = "Generating") -> List[dict]:
        """Execute work items, parallel or serial."""
        if not items:
            return []

        if self.num_workers <= 1:
            results = [_worker(it) for it in tqdm(items, desc=desc)]
        else:
            with Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_worker, items),
                    total=len(items),
                    desc=desc,
                ))
        return results

    # ------------------------------------------------------------------
    def _tally(self, results: List[dict]) -> None:
        for r in results:
            if r["status"] == "ok":
                self.stats["completed"] += 1
            elif r["status"] == "skipped":
                self.stats["skipped"] += 1
            else:
                self.stats["failed"] += 1
                self.stats["errors"].append(
                    {"map": r["map"], "error": r.get("error", "unknown")}
                )

    # ------------------------------------------------------------------
    def generate_all(self, max_maps: Optional[int] = None) -> Dict:
        """
        Generate labels for all maps × all robot sizes.

        Args:
            max_maps: Limit number of maps (useful for testing).

        Returns:
            Statistics dictionary.
        """
        self._create_label_dirs()
        items = self._build_work_items(max_maps=max_maps)
        self.stats["total_tasks"] = len(items) + self.stats["skipped"]

        if not items:
            logger.info("Nothing to generate — all labels already exist.")
            return self.stats

        logger.info(
            f"Generating {len(items)} labels | "
            f"workers={self.num_workers} | "
            f"sizes={self.robot_sizes}"
        )

        t0      = time.time()
        results = self._run_items(items, desc="Oracle labels")
        elapsed = time.time() - t0

        self._tally(results)
        self.stats["elapsed_seconds"]  = round(elapsed, 1)
        self.stats["tasks_per_second"] = round(len(items) / elapsed, 2) if elapsed else 0

        logger.info(
            f"Done in {elapsed:.1f}s  "
            f"({self.stats['tasks_per_second']} tasks/s)  "
            f"ok={self.stats['completed']}  "
            f"skipped={self.stats['skipped']}  "
            f"failed={self.stats['failed']}"
        )
        if self.stats["errors"]:
            logger.warning("First 5 errors:")
            for e in self.stats["errors"][:5]:
                logger.warning(f"  {e['map']}: {e['error']}")

        return self.stats

    # ------------------------------------------------------------------
    def generate_for_robot_size(
        self,
        robot_length: int,
        robot_width: int,
        max_maps: Optional[int] = None,
    ) -> Dict:
        """Generate labels for a single robot size only."""
        self._create_label_dirs()
        items = self._build_work_items(
            max_maps=max_maps, robot_sizes=[(robot_length, robot_width)]
        )
        logger.info(f"Generating {len(items)} labels for {robot_length}×{robot_width}")
        results = self._run_items(items, desc=f"robot_{robot_length}x{robot_width}")
        ok      = sum(1 for r in results if r["status"] == "ok")
        failed  = sum(1 for r in results if r["status"] == "error")
        return {"completed": ok, "failed": failed}

    # ------------------------------------------------------------------
    def verify_completeness(self) -> Tuple[bool, List[str]]:
        """Return (all_complete, list_of_missing_label_paths)."""
        missing = []
        for mp in sorted(self.processed_dir.glob("*.npy")):
            for L, W in self.robot_sizes:
                lp = self.label_path(mp.stem, L, W)
                if not lp.exists():
                    missing.append(str(lp))
        return len(missing) == 0, missing

    # ------------------------------------------------------------------
    def save_stats(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Stats saved to {path}")


# ---------------------------------------------------------------------------
# DatasetVerifier
# ---------------------------------------------------------------------------

class DatasetVerifier:
    """
    Post-generation integrity checks for the label dataset.

    Checks:
        completeness  — all maps have labels for all robot sizes
        shapes        — every label is (4, H, W)
        binary        — all values in {0, 1}
        monotonicity  — larger robots have ≤ viable pixels than smaller ones
    """

    def __init__(
        self,
        processed_dir: str | Path,
        labels_dir: str | Path,
        robot_sizes: List[Tuple[int, int]],
    ):
        self.processed_dir = Path(processed_dir)
        self.labels_dir    = Path(labels_dir)
        # Sort ascending by area so monotonicity check is meaningful
        self.robot_sizes   = sorted(robot_sizes, key=lambda s: s[0] * s[1])

    def _label_path(self, map_stem: str, L: int, W: int) -> Path:
        return self.labels_dir / f"robot_{L}x{W}" / f"{map_stem}.npy"

    # ------------------------------------------------------------------
    def verify_completeness(self) -> Tuple[bool, List[str]]:
        missing = []
        for mp in sorted(self.processed_dir.glob("*.npy")):
            for L, W in self.robot_sizes:
                lp = self._label_path(mp.stem, L, W)
                if not lp.exists():
                    missing.append(str(lp))
        return len(missing) == 0, missing

    # ------------------------------------------------------------------
    def verify_shapes(self, sample_size: int = 100) -> Tuple[bool, List[str]]:
        errors = []
        map_files = list(self.processed_dir.glob("*.npy"))[:sample_size]
        for mp in map_files:
            occ = np.load(mp)
            H, W = occ.shape
            for L, Wr in self.robot_sizes:
                lp = self._label_path(mp.stem, L, Wr)
                if lp.exists():
                    labels = np.load(lp)
                    if labels.shape != (4, H, W):
                        errors.append(f"{lp}: expected (4,{H},{W}), got {labels.shape}")
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    def verify_binary(self, sample_size: int = 100) -> Tuple[bool, List[str]]:
        errors = []
        for L, W in self.robot_sizes:
            label_dir = self.labels_dir / f"robot_{L}x{W}"
            for lp in list(label_dir.glob("*.npy"))[:sample_size]:
                labels = np.load(lp)
                unique = np.unique(labels)
                if not np.all(np.isin(unique, [0, 1])):
                    errors.append(f"{lp.name}: non-binary values {unique}")
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    def verify_monotonicity(self, sample_size: int = 50) -> Tuple[bool, List[str]]:
        """Larger robots must have ≤ viable pixels than smaller robots."""
        errors = []
        map_files = list(self.processed_dir.glob("*.npy"))[:sample_size]
        for mp in map_files:
            counts = []
            for L, W in self.robot_sizes:
                lp = self._label_path(mp.stem, L, W)
                if lp.exists():
                    counts.append(((L, W), int(np.load(lp).sum())))
            for i in range(len(counts) - 1):
                size_a, cnt_a = counts[i]
                size_b, cnt_b = counts[i + 1]
                if cnt_a < cnt_b:
                    errors.append(
                        f"{mp.name}: {size_a} has {cnt_a} viable < "
                        f"{size_b} has {cnt_b} (larger robot MORE viable?)"
                    )
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    def run_all_checks(self, sample_size: int = 100) -> Dict:
        """Run all checks and return summary dict."""
        results = {}

        logger.info("Checking completeness...")
        ok, missing = self.verify_completeness()
        results["completeness"] = {"passed": ok, "missing_count": len(missing),
                                   "examples": missing[:5]}

        logger.info("Checking shapes...")
        ok, errs = self.verify_shapes(sample_size)
        results["shapes"] = {"passed": ok, "error_count": len(errs), "examples": errs[:5]}

        logger.info("Checking binary values...")
        ok, errs = self.verify_binary(sample_size)
        results["binary"] = {"passed": ok, "error_count": len(errs), "examples": errs[:5]}

        logger.info("Checking monotonicity...")
        ok, errs = self.verify_monotonicity(sample_size)
        results["monotonicity"] = {"passed": ok, "error_count": len(errs),
                                   "examples": errs[:5]}

        results["all_passed"] = all(v["passed"] for v in results.values()
                                    if isinstance(v, dict) and "passed" in v)
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_oracle_dataset(
    processed_dir: str | Path,
    labels_dir: str | Path,
    robot_sizes: List[Tuple[int, int]],
    num_workers: int = 8,
    force: bool = False,
    max_maps: Optional[int] = None,
) -> Dict:
    """
    Generate Oracle labels for all maps × all robot sizes.

    Args:
        processed_dir: Directory of .npy occupancy grids.
        labels_dir:    Base output directory.
        robot_sizes:   List of (length, width) tuples.
        num_workers:   CPU parallelism.
        force:         Recompute even if label already exists.
        max_maps:      Limit maps processed (for testing).

    Returns:
        Statistics dict from LabelGenerator.
    """
    gen = LabelGenerator(
        processed_dir=processed_dir,
        labels_dir=labels_dir,
        robot_sizes=robot_sizes,
        num_workers=num_workers,
        skip_existing=not force,
    )
    return gen.generate_all(max_maps=max_maps)