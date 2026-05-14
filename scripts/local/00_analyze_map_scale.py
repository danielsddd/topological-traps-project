# scripts/local/00_analyze_map_scale.py
"""
CRITICAL calibration step — run this before choosing robot sizes.

Measures corridor widths in rasterized maps using distance transform.
The distance transform at each free pixel = distance to nearest wall.
So a passage of width W has pixels with value W/2 at its center.

We want robot sizes where:
  - Small robot: fits through most corridors (interesting, not trivial)
  - Large robot: blocked by many corridors (interesting, not impossible)
  - The "trap zone": robot fits through but can't rotate inside → topological trap
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.map_loader import load_map


def analyze_maps(map_dir: Path, n_maps: int = 10):
    json_files = sorted(map_dir.glob("*.json"))[:n_maps]
    if not json_files:
        print("No maps found. Run the download script first.")
        return

    all_passage_widths = []   # half-widths × 2 = full passage widths
    map_stats = []

    for path in json_files:
        grid = load_map(path)  # 1=free, 0=obstacle

        # Distance transform: each free pixel gets distance to nearest obstacle
        # This tells us the local "radius" of free space at each point
        dist = cv2.distanceTransform(grid, cv2.DIST_L2, 5)

        # Passage width at a point = 2 × dist (diameter, not radius)
        passage_widths = dist[grid == 1] * 2

        all_passage_widths.extend(passage_widths.tolist())

        stats = {
            "file": path.name[:20],
            "p10": np.percentile(passage_widths, 10),
            "p25": np.percentile(passage_widths, 25),
            "p50": np.percentile(passage_widths, 50),
            "p75": np.percentile(passage_widths, 75),
            "p90": np.percentile(passage_widths, 90),
            "max": passage_widths.max(),
        }
        map_stats.append(stats)
        print(f"  {stats['file']}: p25={stats['p25']:.1f}px  p50={stats['p50']:.1f}px  "
              f"p75={stats['p75']:.1f}px  max={stats['max']:.1f}px")

    all_w = np.array(all_passage_widths)
    print(f"\n=== ACROSS ALL {len(json_files)} MAPS ===")
    print(f"  p10  = {np.percentile(all_w, 10):.1f} px")
    print(f"  p25  = {np.percentile(all_w, 25):.1f} px")
    print(f"  p50  = {np.percentile(all_w, 50):.1f} px  ← median corridor width")
    print(f"  p75  = {np.percentile(all_w, 75):.1f} px")
    print(f"  p90  = {np.percentile(all_w, 90):.1f} px")
    print(f"  max  = {all_w.max():.1f} px  ← widest open room")

    # --- Recommendation ---
    p25 = np.percentile(all_w, 25)
    p50 = np.percentile(all_w, 50)
    p75 = np.percentile(all_w, 75)

    print(f"\n=== ROBOT SIZE RECOMMENDATION ===")
    print(f"For topological traps to be interesting:")
    print(f"  Small  robot diagonal should be ~p25 = {p25:.0f}px  → fits most corridors")
    print(f"  Medium robot diagonal should be ~p50 = {p50:.0f}px  → fits half the corridors")
    print(f"  Large  robot diagonal should be ~p75 = {p75:.0f}px  → blocked by many corridors")
    print(f"")
    print(f"  Diagonal = sqrt(L² + W²).  For L/W ratio ~4:3:")

    for label, target_diag in [("Small", p25), ("Medium", p50), ("Large", p75)]:
        # L/W = 4/3 → L = 4k, W = 3k → diag = 5k → k = diag/5
        k = target_diag / 5.0
        L, W = round(4 * k), round(3 * k)
        print(f"  {label:6s}: L={L}px, W={W}px  (diagonal={np.sqrt(L**2+W**2):.1f}px)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of passage widths
    axes[0].hist(all_w, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    for pct, color, label in [(25, "orange", "p25"), (50, "red", "p50"), (75, "green", "p75")]:
        v = np.percentile(all_w, pct)
        axes[0].axvline(v, color=color, linewidth=2, label=f"{label}={v:.0f}px")
    axes[0].set_xlabel("Passage width (pixels)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of corridor widths across all maps")
    axes[0].legend()

    # Distance transform visualization of first map
    grid0 = load_map(json_files[0])
    dist0 = cv2.distanceTransform(grid0, cv2.DIST_L2, 5)
    im = axes[1].imshow(dist0, cmap="hot")
    plt.colorbar(im, ax=axes[1], label="Distance to wall (px)")
    axes[1].set_title(f"Distance transform: {json_files[0].stem[:20]}")
    axes[1].axis("off")

    plt.tight_layout()
    out = Path("outputs/figures/eye_test/map_scale_analysis.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved to {out}")
    plt.show()


if __name__ == "__main__":
    analyze_maps(Path("data/raw_maps"))