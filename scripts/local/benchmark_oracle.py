#!/usr/bin/env python3
"""
Benchmark: Naive O((H×W)²) vs Reverse BFS O(H×W) Oracle.

Tests on:
  1. HouseExpo floor plan maps (~40% free) — realistic indoor navigation
  2. Hand-crafted corridor trap map       — worst case, analytically verifiable

The corridor map is designed specifically for the 30×20 robot:
  - Corridor width = 22px  → robot_width (20) fits, diagonal (36) does NOT
  - Corridor length = 150px → robot_length (30) fits easily
  - Open rooms at each end → robot can rotate there (70px > diagonal 36px)

Expected Oracle output:
  - Inside corridor: viable-E=YES, viable-W=YES (escape to open rooms)
  - Inside corridor: viable-N=NO,  viable-S=NO  (robot_length=30 > corridor_width=22)
  - Open rooms: viable in ALL directions

Run: python scripts/local/benchmark_oracle.py
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.map_loader import load_map
from src.oracle.rotation_check import compute_rotation_safe_mask
from src.oracle.translation_check import compute_translation_safe_mask
from src.oracle.directional_viability import (
    compute_viability_single_direction,
    compute_viability_naive,
)
from src.oracle import generate_labels_for_map

ROBOT_L, ROBOT_W  = 30, 20
ROBOT_DIAG        = int((ROBOT_L**2 + ROBOT_W**2) ** 0.5)
DIRECTION         = "N"
CROP_SIZES        = [64, 128, 192, 256]
NAIVE_MAX_SIZE    = 128
TARGET_FREE_MIN   = 0.30
TARGET_FREE_MAX   = 0.55
REVBFS_FULL_512   = 231.0


# ---------------------------------------------------------------------------
# Map generators
# ---------------------------------------------------------------------------

def generate_corridor_trap(size: int = 256) -> np.ndarray:
    """
    Hand-crafted directional trap. Fits within `size`×`size`.

    For robot 30×20 (diagonal=36):
        Room size 60px: robot CAN rotate (60 > 36)
        Corridor width 22px: robot fits E/W (20 < 22), cannot rotate (36 > 22)
        Total width: 60 + 100 + 60 = 220px  ← fits in 256
    """
    occ = np.zeros((size, size), dtype=np.uint8)

    room_size  = 60    # open area, robot can rotate here
    corr_width = 22    # > robot_width=20, < diagonal=36 → TRAP
    corr_length = 100  # long enough for BFS to be meaningful

    total_w = room_size + corr_length + room_size  # = 220px
    total_h = room_size                             # = 60px

    x0 = (size - total_w)  // 2   # = 18
    y0 = (size - total_h)  // 2   # = 98

    # Left open room
    occ[y0:y0+room_size, x0:x0+room_size] = 1

    # Horizontal corridor (E/W travel)
    cy = y0 + (room_size - corr_width) // 2
    occ[cy:cy+corr_width, x0+room_size:x0+room_size+corr_length] = 1

    # Right open room
    occ[y0:y0+room_size, x0+room_size+corr_length:x0+total_w] = 1

    return occ

def generate_t_junction_trap(size: int = 256) -> np.ndarray:
    """
    T-junction map: one main corridor + one dead-end branch.

    Main corridor (horizontal, 22px wide): robot can escape E or W
    Dead-end branch (vertical, 22px wide): robot enters heading N
        but cannot exit — true topological trap for N direction.
    """
    occ = np.zeros((size, size), dtype=np.uint8)

    room_size  = 60
    corr_width = 22

    cx = size // 2
    cy = size // 2

    # Open room at center
    occ[cy-room_size//2:cy+room_size//2,
        cx-room_size//2:cx+room_size//2] = 1

    # Left arm (horizontal)
    occ[cy-corr_width//2:cy+corr_width//2, 10:cx-room_size//2] = 1

    # Right arm (horizontal)
    occ[cy-corr_width//2:cy+corr_width//2, cx+room_size//2:size-10] = 1

    # Dead-end top arm (vertical) — robot can enter heading S (downward)
    # but cannot exit heading N (back out) because it cannot rotate at the top
    occ[10:cy-room_size//2, cx-corr_width//2:cx+corr_width//2] = 1

    return occ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_crop(occ: np.ndarray, size: int,
               free_min: float = TARGET_FREE_MIN,
               free_max: float = TARGET_FREE_MAX) -> tuple[np.ndarray, float]:
    H, W  = occ.shape
    step  = max(1, size // 8)
    best_crop, best_free, best_dist = None, 0.0, float("inf")
    target = (free_min + free_max) / 2

    for ty in range(0, H - size + 1, step):
        for tx in range(0, W - size + 1, step):
            crop = occ[ty:ty+size, tx:tx+size]
            free = float(crop.mean())
            dist = abs(free - target)
            if free_min <= free <= free_max and dist < best_dist:
                best_crop, best_free, best_dist = crop.copy(), free, dist

    if best_crop is None:
        for ty in range(0, H - size + 1, step):
            for tx in range(0, W - size + 1, step):
                crop = occ[ty:ty+size, tx:tx+size]
                dist = abs(float(crop.mean()) - target)
                if dist < best_dist:
                    best_crop, best_free, best_dist = crop.copy(), float(crop.mean()), dist

    return best_crop, best_free


def run_benchmark_section(occ_full: np.ndarray, crop_sizes: list[int],
                           naive_max: int) -> list[tuple]:
    print(f"\n{'Size':>6} | {'Free%':>6} | {'TransSafe%':>11} | "
          f"{'Naive(ms)':>12} | {'RevBFS(ms)':>12} | {'Speedup':>8} | {'Match':>5}")
    print("-" * 80)

    rows = []
    for size in crop_sizes:
        if size > occ_full.shape[0]:
            continue
        crop, free = find_crop(occ_full, size)
        rot   = compute_rotation_safe_mask(crop, ROBOT_L, ROBOT_W)
        trans = compute_translation_safe_mask(crop, ROBOT_L, ROBOT_W, DIRECTION)
        trans_pct = float(trans.mean()) * 100

        if trans.sum() == 0:
            print(f"{size:>6} | {free*100:>5.1f}% | {'0% (no space)':>11} | "
                  f"{'skip':>12} | {'skip':>12} | {'—':>8} | {'—':>5}")
            continue

        t0     = time.perf_counter()
        v_fast = compute_viability_single_direction(trans, rot, DIRECTION)
        t_fast = (time.perf_counter() - t0) * 1000

        if size <= naive_max:
            t0      = time.perf_counter()
            v_naive = compute_viability_naive(trans, rot, DIRECTION)
            t_naive = (time.perf_counter() - t0) * 1000
            match   = np.array_equal(v_fast, v_naive)
            speedup = f"{t_naive/t_fast:.1f}×"
            rows.append((size, free, t_naive, t_fast))
            print(f"{size:>6} | {free*100:>5.1f}% | {trans_pct:>10.1f}% | "
                  f"{t_naive:>10.1f}ms | {t_fast:>10.1f}ms | {speedup:>8} | {match!s:>5}")
        else:
            rows.append((size, free, None, t_fast))
            print(f"{size:>6} | {free*100:>5.1f}% | {trans_pct:>10.1f}% | "
                  f"{'skipped':>12} | {t_fast:>10.1f}ms | {'?':>8} | {'—':>5}")
    return rows


def extrapolate(rows: list[tuple]) -> dict:
    valid = [(s*s, t) for s, _, t, _ in rows if t is not None]
    if len(valid) < 2:
        print("  Not enough data points.")
        return {}
    log_n    = np.log([n for n, _ in valid])
    log_t    = np.log([t for _, t in valid])
    coeffs   = np.polyfit(log_n, log_t, 1)
    exponent = coeffs[0]
    naive_512 = np.exp(np.polyval(coeffs, np.log(512*512)))
    print(f"  Naive scaling exponent : O(N^{exponent:.2f})")
    print(f"  Naive at 512×512       : {naive_512/1000:.2f}s  (extrapolated)")
    print(f"  Reverse BFS at 512×512 : {REVBFS_FULL_512:.0f}ms  (measured)")
    print(f"  Oracle speedup         : {naive_512/REVBFS_FULL_512:.0f}×")
    print(f"  Model vs Oracle        : ~{REVBFS_FULL_512/5:.0f}×")
    print(f"  Model vs Naive         : ~{naive_512/5:.0f}×")
    return {"exponent": exponent, "naive_512_ms": naive_512}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Robot: {ROBOT_L}×{ROBOT_W}px  diagonal={ROBOT_DIAG}px")
    print(f"Trap condition: corridor width between {ROBOT_W}px and {ROBOT_DIAG}px")

    # ---- Section 1: HouseExpo ---------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 1: HouseExpo Indoor Floor Plans")
    print("=" * 80)

    all_maps = sorted(Path("data/raw_maps").glob("*.json"))
    if not all_maps:
        print("ERROR: No maps in data/raw_maps/")
        return

    map_stats = [(abs(load_map(f).mean() - 0.40), f) for f in all_maps[:30]]
    map_stats.sort()
    house_file = map_stats[0][1]
    house_occ  = load_map(house_file)
    print(f"Map: {house_file.stem[:24]}  free={house_occ.mean()*100:.1f}%")

    house_rows = run_benchmark_section(house_occ, CROP_SIZES, NAIVE_MAX_SIZE)
    print("\n--- HouseExpo Extrapolation ---")
    house_ext  = extrapolate(house_rows)

    # ---- Section 2: Corridor trap map -------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 2: Hand-Crafted Corridor Trap Map (analytically verified)")
    print("=" * 80)

    corr_occ = generate_corridor_trap(size=256)
    free_pct = corr_occ.mean() * 100
    print(f"Corridor map 256×256  free={free_pct:.1f}%")
    print(f"Corridor width=22px: robot_width(20) fits, diagonal(36) does NOT → TRAP")

    # Save for reference
    Path("data/samples").mkdir(parents=True, exist_ok=True)
    np.save("data/samples/corridor_trap_256.npy", corr_occ)

    corr_rows = run_benchmark_section(corr_occ, CROP_SIZES, NAIVE_MAX_SIZE)
    print("\n--- Corridor Trap Extrapolation ---")
    corr_ext  = extrapolate(corr_rows)

    # ---- Section 3: Oracle on both full maps ------------------------------
    print("\n" + "=" * 80)
    print("SECTION 3: Full Oracle — timing + correctness check")
    print("=" * 80)

    for name, occ in [("HouseExpo crop (256×256)", find_crop(house_occ, 256)[0]),
                       ("Corridor trap (256×256)",   corr_occ)]:
        t0     = time.perf_counter()
        labels = generate_labels_for_map(occ, ROBOT_L, ROBOT_W)
        t_ms   = (time.perf_counter() - t0) * 1000
        viable = [f"{labels[i].mean()*100:.1f}%" for i in range(4)]
        print(f"\n{name}")
        print(f"  Oracle time : {t_ms:.1f}ms")
        print(f"  Viable N/S/E/W: {viable[0]} / {viable[1]} / {viable[2]} / {viable[3]}")

    # ---- Visualize --------------------------------------------------------
    house_crop, _ = find_crop(house_occ, 256)
    h_labels      = generate_labels_for_map(house_crop, ROBOT_L, ROBOT_W)
    c_labels      = generate_labels_for_map(corr_occ,   ROBOT_L, ROBOT_W)
    t_occ         = generate_t_junction_trap(256)
    t_labels      = generate_labels_for_map(t_occ,      ROBOT_L, ROBOT_W)

    fig, axes = plt.subplots(3, 5, figsize=(24, 15))
    dirs = ["N", "S", "E", "W"]

    for row_idx, (occ, labels, title) in enumerate([
        (house_crop, h_labels, f"HouseExpo  free={house_crop.mean()*100:.1f}%"),
        (corr_occ,   c_labels, "Corridor trap  (N/S=trapped, E/W=viable)"),
        (t_occ,      t_labels, "T-junction trap  (dead-end branch at top)"),
    ]):
        axes[row_idx, 0].imshow(occ, cmap="gray")
        axes[row_idx, 0].set_title(title, fontsize=9)
        axes[row_idx, 0].axis("off")
        for col_idx, d in enumerate(dirs):
            axes[row_idx, col_idx+1].imshow(
                labels[col_idx], cmap="RdYlGn", vmin=0, vmax=1)
            pct = labels[col_idx].mean() * 100
            axes[row_idx, col_idx+1].set_title(f"Viable-{d}  {pct:.0f}%", fontsize=9)
            axes[row_idx, col_idx+1].axis("off")

    plt.suptitle(
        f"Oracle labels: {ROBOT_L}×{ROBOT_W} robot (diagonal={ROBOT_DIAG}px)  |  "
        f"green=viable, red=trapped",
        fontsize=13,
    )
    plt.tight_layout()
    out = Path("outputs/figures/eye_test/benchmark_oracle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"\nFigure saved: {out}")
    plt.show()

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY FOR REPORT")
    print("=" * 80)
    naive_h = house_ext.get("naive_512_ms", float("nan"))
    naive_c = corr_ext.get("naive_512_ms", float("nan"))
    print(f"  Naive Oracle (brute-force) at 512×512:")
    print(f"    HouseExpo maps : ~{naive_h/1000:.1f}s  (extrapolated)")
    print(f"    Corridor maps  : ~{naive_c/1000:.1f}s  (extrapolated)")
    print(f"  Reverse BFS Oracle at 512×512 : {REVBFS_FULL_512:.0f}ms  (measured)")
    print(f"  Neural network (after training): ~5ms  (target)")
    print(f"  Model vs Oracle speedup        : ~{REVBFS_FULL_512/5:.0f}×")


if __name__ == "__main__":
    main()