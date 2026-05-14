#!/usr/bin/env python3
"""
scripts/demo_angle_sweep.py

Animates viability as robot heading sweeps 0°→360°.

Visualisation:
  Left  — occupancy map with robot shown as a rotated rectangle
           (actual dimensions: 30×20 px) at the reference pixel,
           with a yellow heading arrow.
  Right — NN viability map for that heading.
           Coloured border + label = VIABLE / TRAPPED.
  Bar   — Oracle vs NN timing comparison (normalised bars).

Layout is set ONCE before the rendering loop — no tight_layout inside frames.
"""

import sys
import math
import logging
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from PIL import Image
import io

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.unet   import MultiRobotViabilityUNet
from src.utils.helpers import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROBOT_L    = 30    # robot length in pixels (along heading direction)
ROBOT_W    = 20    # robot width  in pixels (perpendicular to heading)
RESOLUTION = 512

MS_FAST     = 180
MS_CARDINAL = 700
MS_LOOP_END = 1200


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Continuous-angle viability heading sweep demo"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--map",       default=None)
    p.add_argument("--n-angles",  type=int, default=24,
                   help="Angle steps for 0→360° sweep (default 24 = every 15°)")
    p.add_argument("--out",       default="outputs/closed_loop_demo/demo_angle_sweep.gif")
    p.add_argument("--seed",      type=int, default=3)
    p.add_argument("--device",    default=None)
    p.add_argument("--oracle-ms", type=float, default=189.0,
                   help="Oracle inference time in ms (for timing bar)")
    p.add_argument("--nn-ms",     type=float, default=10.0,
                   help="NN inference time in ms (for timing bar)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(map_path, seed):
    if map_path:
        occ = np.load(map_path).astype(np.uint8)
        logger.info("Map: %s  free=%.1f%%", map_path, occ.mean() * 100)
        return occ
    candidates = sorted(Path("data/processed").glob("*.npy"))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    for f in candidates:
        occ = np.load(f).astype(np.uint8)
        if 0.45 <= occ.mean() <= 0.65:
            logger.info("Auto-selected: %s  free=%.1f%%", f.name, occ.mean() * 100)
            return occ
    occ = np.load(candidates[0]).astype(np.uint8)
    logger.info("Fallback: %s", candidates[0].name)
    return occ


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, occ: np.ndarray, angle_deg: float, dev) -> np.ndarray:
    H, W  = occ.shape
    rad   = math.radians(angle_deg)
    inp   = np.zeros((1, 5, H, W), dtype=np.float32)
    inp[0, 0] = occ.astype(np.float32)
    inp[0, 1] = ROBOT_L / RESOLUTION
    inp[0, 2] = ROBOT_W / RESOLUTION
    inp[0, 3] = math.sin(rad)
    inp[0, 4] = math.cos(rad)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(inp).to(dev)))
    return probs[0, 0].cpu().numpy()   # (H, W)


# ---------------------------------------------------------------------------
# Reference pixel — the free pixel whose viability varies most across angles
# ---------------------------------------------------------------------------

def find_ref_pixel(occ: np.ndarray, via_cache: dict) -> tuple:
    from scipy.ndimage import binary_erosion
    clearance = max(ROBOT_L, ROBOT_W) // 2 + 4
    struct    = np.ones((clearance*2+1, clearance*2+1), dtype=bool)
    valid     = binary_erosion(occ.astype(bool), structure=struct)
    stack     = np.stack(list(via_cache.values()), axis=0)
    std_map   = stack.std(axis=0)
    std_map[~valid] = 0.0
    margin = int(min(occ.shape) * 0.15)
    std_map[:margin,:]=std_map[-margin:,:]=std_map[:,:margin]=std_map[:,-margin:]=0.0
    r, c = np.unravel_index(np.argmax(std_map), std_map.shape)
    logger.info("Ref pixel (%d,%d) std=%.3f clearance=%dpx", r, c, std_map[r,c], clearance)
    return int(r), int(c)


# ---------------------------------------------------------------------------
# Rotated robot rectangle helper
# ---------------------------------------------------------------------------

def add_robot_rect(ax, cx: float, cy: float, angle_deg: float,
                   robot_l: int, robot_w: int, color: str, alpha: float = 0.85):
    """
    Draw a rotated rectangle representing the robot.

    The robot is `robot_l` pixels long (along heading) and `robot_w` pixels wide.
    It is drawn centred at (cx, cy) and rotated so the long axis points in the
    heading direction.

    Coordinate system: column=x (right), row=y (down in image coords).
    Heading angle 0° = North = up = -y direction.
    """
    rad = math.radians(angle_deg)

    # Half-dimensions
    hl = robot_l / 2.0
    hw = robot_w / 2.0

    # Corners of unrotated rect (centred at origin), then rotate + translate.
    # Unrotated: long axis along column (x), width along row (y)
    corners_local = np.array([
        [-hw, -hl],
        [ hw, -hl],
        [ hw,  hl],
        [-hw,  hl],
    ], dtype=float)

    # Rotation by angle_deg (heading = 0° points up = -row direction)
    # In image coords: col-axis = x-right, row-axis = y-down
    # Heading arrow: dx = sin(θ), dy = -cos(θ)  →  rotation matrix:
    #   [ cos(θ)  -sin(θ) ]   applied to (col, row) local offsets
    #   [ sin(θ)   cos(θ) ]
    # But "heading up" = angle 0 means the rectangle's long axis is vertical
    # in image space (dy < 0), so we rotate local coords by angle_deg in
    # standard CCW sense.
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    corners_world = (R @ corners_local.T).T + np.array([cx, cy])

    poly = plt.Polygon(corners_world, closed=True,
                       facecolor=color, edgecolor="white",
                       linewidth=1.2, alpha=alpha, zorder=6)
    ax.add_patch(poly)


# ---------------------------------------------------------------------------
# Single frame renderer
# ---------------------------------------------------------------------------


def render_frame(
    ax_map:  plt.Axes,
    ax_via:  plt.Axes,
    ax_bar:  plt.Axes,
    occ:     np.ndarray,
    via_map: np.ndarray,
    angle_deg: float,
    ref_r: int, ref_c: int,
    oracle_ms: float, nn_ms: float, n_angles: int,
) -> None:
    """
    Draw one sweep frame into the provided axes.
    Does NOT call tight_layout or savefig — the caller handles those.
    """
    H, W = occ.shape

    ref_v     = float(via_map[ref_r, ref_c])
    ref_color = "#00FF00" if ref_v > 0.5 else "#FF4444"
    ref_label = "VIABLE" if ref_v > 0.5 else "TRAPPED"

    # ---- Left: occupancy + rotated robot + heading arrow ----------------
    ax_map.cla()
    ax_map.axis("off")
    rgb = np.ones((H, W, 3))
    rgb[occ == 0] = [0.09, 0.09, 0.09]
    ax_map.imshow(rgb, origin="upper", interpolation="nearest",
                  extent=[0, W, H, 0])
    ax_map.set_xlim(0, W)
    ax_map.set_ylim(H, 0)
    ax_map.set_aspect("equal", adjustable="box")

    # Robot rectangle at reference pixel
    add_robot_rect(ax_map, cx=ref_c, cy=ref_r,
                   angle_deg=angle_deg,
                   robot_l=ROBOT_L, robot_w=ROBOT_W,
                   color=ref_color, alpha=0.80)

    # Heading arrow: starts at front of robot, points outward
    rad     = math.radians(angle_deg)
    # "North" (0°) = up = decreasing row = (dx=0, dy=-1) in image coords
    dir_col =  math.sin(rad)   # column direction
    dir_row = -math.cos(rad)   # row direction (negative = up)

    arrow_start_col = ref_c + dir_col * (ROBOT_L / 2.0)
    arrow_start_row = ref_r + dir_row * (ROBOT_L / 2.0)
    arrow_end_col   = ref_c + dir_col * (ROBOT_L / 2.0 + 30)
    arrow_end_row   = ref_r + dir_row * (ROBOT_L / 2.0 + 30)

    # White outline
    ax_map.annotate(
        "", xy=(arrow_end_col, arrow_end_row),
        xytext=(arrow_start_col, arrow_start_row),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=4.5, mutation_scale=20),
        annotation_clip=False, zorder=7,
    )
    # Gold arrow
    ax_map.annotate(
        "", xy=(arrow_end_col, arrow_end_row),
        xytext=(arrow_start_col, arrow_start_row),
        arrowprops=dict(arrowstyle="-|>", color="#FFD700", lw=2.5, mutation_scale=18),
        annotation_clip=False, zorder=8,
    )

    cardinal = {0: "North ↑", 90: "East →", 180: "South ↓", 270: "West ←"}
    lbl = cardinal.get(round(angle_deg) % 360, "")
    ax_map.set_title(
        f"Robot heading: {angle_deg:.0f}°" + (f"  [{lbl}]" if lbl else ""),
        color=ref_color, fontsize=11, pad=4, fontweight="bold",
    )

    # ---- Right: viability heatmap + robot outline -----------------------
    ax_via.cla()
    ax_via.axis("off")
    free = occ == 1
    ax_via.imshow(
        np.where(free, via_map, np.nan),
        cmap="RdYlGn", vmin=0, vmax=1,
        origin="upper", interpolation="nearest",
        extent=[0, W, H, 0],
    )
    ax_via.set_xlim(0, W)
    ax_via.set_ylim(H, 0)
    ax_via.set_aspect("equal", adjustable="box")

    # Robot outline on viability map (no fill — don't hide the heatmap)
    add_robot_rect(ax_via, cx=ref_c, cy=ref_r,
                   angle_deg=angle_deg,
                   robot_l=ROBOT_L, robot_w=ROBOT_W,
                   color="none", alpha=1.0)
    # Redraw as edge-only
    poly_edge = plt.Polygon(
        _rect_corners(ref_c, ref_r, angle_deg, ROBOT_L, ROBOT_W),
        closed=True, facecolor="none",
        edgecolor=ref_color, linewidth=2.5, zorder=7,
    )
    ax_via.add_patch(poly_edge)

    # Status label — never overflows: shift to whichever side has more room
    tx  = ref_c - 18 if ref_c > W // 2 else ref_c + 18
    txa = "right"    if ref_c > W // 2 else "left"
    ax_via.text(
        tx, ref_r,
        f"{ref_label}\n({ref_v:.2f})",
        color=ref_color, fontsize=9.5, fontweight="bold",
        va="center", ha=txa,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.88),
        zorder=9,
        clip_on=True,
    )

    mean_v   = float(via_map[free].mean()) if free.any() else 0.0
    viable_p = float((via_map[free] > 0.5).mean() * 100) if free.any() else 0.0
    ax_via.set_title(
        f"Viability at {angle_deg:.0f}°  |  mean={mean_v:.2f}  viable={viable_p:.0f}%",
        color="white", fontsize=10, pad=4,
    )

    # ---- Bottom: timing bar (normalised 0..1) ---------------------------
    ax_bar.cla()
    ax_bar.set_facecolor("#111111")
    ax_bar.axis("off")

    total_o = oracle_ms * n_angles
    total_n = nn_ms     * n_angles
    frac_n  = total_n / total_o          # e.g. 0.053 for 10ms/189ms

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.65, 1.65)

    ax_bar.barh(1, 1.0,    color="#1a3a6e", height=0.42, alpha=0.85)
    ax_bar.barh(0, frac_n, color="#00BFFF", height=0.42, alpha=0.85)

    # Oracle label — centred inside the (full-width) bar
    ax_bar.text(0.5, 1,
                f"Oracle: {total_o:.0f} ms for {n_angles} angles",
                va="center", ha="center",
                color="white", fontsize=9, fontweight="bold")

    # NN label — placed to the RIGHT of the short bar, but never < 0.10
    nn_text_x = max(frac_n + 0.03, 0.10)
    ax_bar.text(nn_text_x, 0,
                f"NN: {total_n:.0f} ms  —  {total_o / total_n:.0f}× faster",
                va="center", ha="left",
                color="white", fontsize=9, fontweight="bold")

    ax_bar.text(0.5, -0.50,
                "Querying all heading angles — NN wins by constant factor",
                ha="center", color="#aaaaaa", fontsize=7.5, style="italic")


def _rect_corners(cx, cy, angle_deg, robot_l, robot_w):
    """Return (4,2) array of rectangle corners for a polygon patch."""
    rad  = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    R    = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    hl, hw = robot_l / 2.0, robot_w / 2.0
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=float)
    return (R @ local.T).T + np.array([cx, cy])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    dev   = torch.device(args.device or get_device())

    model = MultiRobotViabilityUNet.from_checkpoint(args.checkpoint, device=str(dev))
    model.eval().to(dev)

    # Warm up
    dummy = torch.zeros(1, 5, 64, 64, device=dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    logger.info("Model warmed up (5-ch)")

    occ    = load_map(args.map, args.seed)
    angles = [i * 360.0 / args.n_angles for i in range(args.n_angles)] + [0.0]

    logger.info("Running inference for %d angles...", len(set(angles)))
    via_cache = {a: predict(model, occ, a, dev) for a in set(angles)}
    ref_r, ref_c = find_ref_pixel(occ, via_cache)

    # ---- Build figure ONCE — layout set here, not inside render_frame ---
    fig = plt.figure(figsize=(13, 7.2))
    fig.patch.set_facecolor("#111111")

    # Reserve top space for suptitle, bottom for timing bar
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[4.0, 1.0],
        hspace=0.28,
        wspace=0.06,
        left=0.02,
        right=0.98,
        top=0.90,     # leaves room for suptitle
        bottom=0.04,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_via = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax in (ax_map, ax_via, ax_bar):
        ax.set_facecolor("#111111")

    fig.suptitle(
        "Direction 1a: Continuous-Angle Viability — Heading Sweep 0°→360°\n"
        "Rectangle = robot (30×20 px)  ·  Arrow = heading  ·  "
        "Green = can escape  ·  Red = trapped at this heading",
        color="white", fontsize=10, fontweight="bold",
    )

    # ---- Rendering loop -------------------------------------------------
    pil_frames: list = []
    durations:  list = []

    logger.info("Rendering %d frames...", len(angles))

    for i, angle in enumerate(angles):
        render_frame(
            ax_map, ax_via, ax_bar,
            occ, via_cache[angle], angle,
            ref_r, ref_c,
            args.oracle_ms, args.nn_ms, args.n_angles,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#111111", dpi=120)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())

        deg_int = round(angle) % 360
        durations.append(
            MS_LOOP_END if i == len(angles) - 1 else
            MS_CARDINAL if deg_int in {0, 90, 180, 270} else
            MS_FAST
        )

    plt.close(fig)

    # ---- Save GIF -------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving → %s  (%d frames)", out_path, len(pil_frames))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    logger.info("Done ✓  —  %.1f s loop", sum(durations) / 1000)


if __name__ == "__main__":
    main()