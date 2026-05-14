# scripts/local/generate_synthetic_maps.py
"""
Generate synthetic test maps for Oracle benchmarking.
No download needed — pure NumPy.

Map types:
  maze       - grid maze, all corridors exactly 1 passage wide
  warehouse  - open grid with pillar obstacles
  corridor   - single long corridor with dead ends
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def generate_maze(size: int = 128, corridor_width: int = 3) -> np.ndarray:
    """
    Recursive backtracking maze. Every corridor is exactly
    `corridor_width` pixels wide — worst case for naive BFS.
    """
    # Work in cell units, then scale up
    cells = size // (corridor_width + 1)
    grid  = np.zeros((cells, cells), dtype=bool)

    def carve(y, x):
        grid[y, x] = True
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        random.shuffle(dirs)
        for dy, dx in dirs:
            ny, nx = y + dy*2, x + dx*2
            if 0 <= ny < cells and 0 <= nx < cells and not grid[ny, nx]:
                grid[y+dy, x+dx] = True
                grid[ny, nx]      = True
                carve(ny, nx)

    random.seed(42)
    carve(0, 0)

    # Scale up to pixel map
    occ = np.zeros((size, size), dtype=np.uint8)
    for cy in range(cells):
        for cx in range(cells):
            if grid[cy, cx]:
                py = cy * (corridor_width + 1)
                px = cx * (corridor_width + 1)
                occ[py:py+corridor_width, px:px+corridor_width] = 1

    return occ


def generate_warehouse(size: int = 256, pillar_spacing: int = 30,
                        pillar_size: int = 8) -> np.ndarray:
    """Open warehouse floor with regular pillar obstacles."""
    occ = np.ones((size, size), dtype=np.uint8)
    for py in range(pillar_spacing, size - pillar_size, pillar_spacing):
        for px in range(pillar_spacing, size - pillar_size, pillar_spacing):
            occ[py:py+pillar_size, px:px+pillar_size] = 0
    # Outer walls
    occ[0,  :] = 0; occ[-1, :] = 0
    occ[:,  0] = 0; occ[:, -1] = 0
    return occ


def generate_dead_end_corridor(size: int = 256, corridor_width: int = 25) -> np.ndarray:
    """
    Long corridor with a dead end — the canonical topological trap.
    A robot entering heading East cannot exit heading East.
    """
    occ = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    # Main corridor
    occ[mid-corridor_width//2 : mid+corridor_width//2, 10:size-10] = 1
    # Dead-end pocket at right end
    occ[mid-corridor_width : mid+corridor_width,
        size-60 : size-10] = 1
    return occ


if __name__ == "__main__":
    out = Path("data/samples")
    out.mkdir(exist_ok=True)

    maps = {
        "maze_128":         generate_maze(128, corridor_width=3),
        "maze_256":         generate_maze(256, corridor_width=3),
        "warehouse_256":    generate_warehouse(256),
        "dead_end_256":     generate_dead_end_corridor(256, corridor_width=25),
    }

    for name, occ in maps.items():
        np.save(out / f"{name}.npy", occ)
        print(f"{name}: shape={occ.shape}  free={occ.mean():.1%}")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, occ) in zip(axes, maps.items()):
        ax.imshow(occ, cmap="gray")
        ax.set_title(f"{name}\nfree={occ.mean():.1%}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/figures/eye_test/synthetic_maps.png", dpi=100)
    plt.show()
    print("Saved to outputs/figures/eye_test/synthetic_maps.png")