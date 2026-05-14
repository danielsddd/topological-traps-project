"""
Map Loader — HouseExpo JSON → binary occupancy grid.

HouseExpo format:
    verts: list of [x, y] world coordinates (meters) forming the floor polygon
    bbox:  {"min": [x, y], "max": [x, y]}

Output: uint8 numpy array (H, W)
    1 = free space (inside floor polygon)
    0 = obstacle  (walls / exterior)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from configs.config_schema import CorruptedMapError, MapLoadError

logger = logging.getLogger(__name__)


def load_map(
    json_path: str | Path,
    resolution: int = 512,
    margin: int = 8,
    wall_thickness: int = 2,
) -> np.ndarray:
    """
    Load a HouseExpo JSON file and convert to a binary occupancy grid.

    Algorithm:
        1. Parse JSON → get polygon vertices in world coords (meters)
        2. Normalize vertices to fit within [margin, resolution-margin]
        3. Fill polygon interior → free space (1)
        4. Erode boundary slightly to create wall thickness (0 border)

    Args:
        json_path:       Path to HouseExpo .json file.
        resolution:      Output grid size (resolution × resolution). Default 512.
        margin:          Pixel margin around the floor plan. Default 8.
        wall_thickness:  Width of walls drawn at polygon boundary. Default 2.

    Returns:
        np.ndarray, shape (resolution, resolution), dtype uint8.
        Values: 1 = free, 0 = obstacle.

    Raises:
        MapLoadError:      If the file cannot be read.
        CorruptedMapError: If the JSON is malformed or has no usable polygon.
    """
    json_path = Path(json_path)

    # ------------------------------------------------------------------
    # 1. Load JSON
    # ------------------------------------------------------------------
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CorruptedMapError(f"Invalid JSON in {json_path}: {e}")
    except OSError as e:
        raise MapLoadError(f"Cannot read {json_path}: {e}")

    if "verts" not in data:
        raise CorruptedMapError(f"No 'verts' key in {json_path}")

    verts_world = np.array(data["verts"], dtype=np.float32)  # (N, 2) — [x, y]

    if len(verts_world) < 3:
        raise CorruptedMapError(f"Too few vertices ({len(verts_world)}) in {json_path}")

    # ------------------------------------------------------------------
    # 2. Compute bounding box (prefer stored bbox, fallback to verts)
    # ------------------------------------------------------------------
    if "bbox" in data and data["bbox"]:
        bbox = data["bbox"]
        x_min = float(bbox["min"][0])
        y_min = float(bbox["min"][1])
        x_max = float(bbox["max"][0])
        y_max = float(bbox["max"][1])
    else:
        x_min, y_min = verts_world.min(axis=0)
        x_max, y_max = verts_world.max(axis=0)

    world_w = x_max - x_min
    world_h = y_max - y_min

    if world_w <= 0 or world_h <= 0:
        raise CorruptedMapError(f"Degenerate bounding box in {json_path}: w={world_w}, h={world_h}")

    # ------------------------------------------------------------------
    # 3. Scale world coords → pixel coords
    #    Keep aspect ratio, fit within [margin, resolution-margin]
    # ------------------------------------------------------------------
    drawable = resolution - 2 * margin
    scale = drawable / max(world_w, world_h)   # uniform scale to preserve aspect

    # World → pixel: note image y-axis is flipped (y increases downward)
    px = (verts_world[:, 0] - x_min) * scale + margin          # x → col
    py = (y_max - verts_world[:, 1]) * scale + margin           # y → row (flipped)

    pixel_verts = np.stack([px, py], axis=1).astype(np.int32)  # (N, 2) int

    # ------------------------------------------------------------------
    # 4. Draw: fill polygon interior = free (255), rest = obstacle (0)
    # ------------------------------------------------------------------
    canvas = np.zeros((resolution, resolution), dtype=np.uint8)

    # Fill interior
    cv2.fillPoly(canvas, [pixel_verts], color=255)

    # Draw wall outline to ensure clean boundary (not ragged anti-aliasing)
    cv2.polylines(
        canvas,
        [pixel_verts],
        isClosed=True,
        color=0,
        thickness=wall_thickness,
    )

    # ------------------------------------------------------------------
    # 5. Convert to binary: 1 = free, 0 = obstacle
    # ------------------------------------------------------------------
    grid = (canvas > 0).astype(np.uint8)

    # Sanity check: there should be meaningful free space
    free_ratio = grid.mean()
    if free_ratio < 0.01:
        raise CorruptedMapError(
            f"Almost no free space in {json_path.name} (free_ratio={free_ratio:.3f})"
        )
    if free_ratio > 0.99:
        raise CorruptedMapError(
            f"Almost all free space in {json_path.name} — polygon likely degenerate"
        )

    return grid


def load_map_batch(
    json_paths: list[Path],
    resolution: int = 512,
    skip_errors: bool = True,
) -> dict[str, np.ndarray]:
    """
    Load multiple maps. Skips (and logs) any that fail.

    Args:
        json_paths:   List of JSON file paths.
        resolution:   Output grid size.
        skip_errors:  If True, log and skip bad files. If False, raise.

    Returns:
        Dict mapping map_id (stem of filename) → occupancy grid.
    """
    results = {}
    for path in json_paths:
        try:
            grid = load_map(path, resolution=resolution)
            results[path.stem] = grid
        except (MapLoadError, CorruptedMapError) as e:
            if skip_errors:
                logger.warning(f"Skipping {path.name}: {e}")
            else:
                raise
    return results


if __name__ == "__main__":
    """Quick visual test — run from repo root:
       python -m src.data.map_loader
    """
    import sys
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    json_files = sorted(Path("data/raw_maps").glob("*.json"))
    if not json_files:
        print("No JSON files found in data/raw_maps/")
        print("Run: python scripts/local/01_download_houseexpo.py --num-maps 10")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files. Loading first 4...")

    fig, axes = plt.subplots(1, min(4, len(json_files)), figsize=(16, 4))
    if len(json_files) == 1:
        axes = [axes]

    for i, (ax, path) in enumerate(zip(axes, json_files[:4])):
        try:
            grid = load_map(path)
            free_pct = grid.mean() * 100
            ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{path.stem[:12]}\nfree={free_pct:.1f}%", fontsize=8)
            ax.axis("off")
            print(f"  ✓ {path.name}: shape={grid.shape}, free={free_pct:.1f}%")
        except Exception as e:
            ax.set_title(f"ERROR\n{path.stem[:12]}", fontsize=8, color="red")
            print(f"  ✗ {path.name}: {e}")

    plt.tight_layout()
    out = Path("outputs/figures/eye_test/map_loader_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"\nSaved figure to {out}")
    plt.show()