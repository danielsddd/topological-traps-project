#!/usr/bin/env python3
"""
scripts/demo_trap_avoidance.py

Choreographed closed-loop trap-avoidance demo.

Story:
  1. Robot moves quickly along a clear initial path.
  2. Three obstacles appear one by one, each blocking the current path.
  3. Around each obstacle the animation slows — the viability map
     updates visibly and the right panel shows the ACTUAL NN inference
     time vs what the Oracle would have taken.
  4. Robot reaches the goal, demo ends with "GOAL REACHED".

The timing panel makes the NN's speed advantage concrete:
  "NN re-queried in 9 ms  (Oracle would take ~189 ms — 21× slower)"

Usage:
    python scripts/demo_trap_avoidance.py \\
        --checkpoint outputs/viability_20260507_141829/checkpoints/best_iou.pth \\
        --out outputs/closed_loop_demo/demo_trap_story.gif

    python scripts/demo_trap_avoidance.py \\
        --checkpoint outputs/viability_20260507_141829/checkpoints/best_iou.pth \\
        --map data/processed/002ae037be8b7b7a8605866296c2d0a1.npy \\
        --out outputs/closed_loop_demo/demo_trap_story.gif
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
import io

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.unet import MultiRobotViabilityUNet
from src.utils.helpers import get_device
from src.oracle.directional_viability import generate_labels_for_map

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Timing / movement constants
# ---------------------------------------------------------------------------
FAST_MS     = 70
SLOW_MS     = 320
GOAL_MS     = 900
SLOW_WINDOW = 7

FAST_STEP   = 6
SLOW_STEP   = 2

PATH_HISTORY_LEN = 3

OBSTACLE_SCHEDULE = [
    (20, 0.35, 26),
    (44, 0.55, 24),
    (64, 0.72, 22),
]

GOAL_RADIUS   = 18


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--map",    default=None)
    p.add_argument("--out",    default="outputs/closed_loop_demo/demo_trap_story.gif")
    p.add_argument("--frames", type=int, default=160)
    p.add_argument("--seed",   type=int, default=7)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def load_or_pick_map(map_path, seed):
    if map_path:
        occ = np.load(map_path).astype(np.uint8)
        logger.info("Map: %s  free=%.1f%%", map_path, occ.mean() * 100)
        return occ
    candidates = sorted(Path("data/processed").glob("*.npy"))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    for f in candidates:
        occ = np.load(f).astype(np.uint8)
        if 0.42 <= occ.mean() <= 0.68:
            logger.info("Auto-selected: %s  free=%.1f%%", f.name, occ.mean()*100)
            return occ
    occ = np.load(candidates[0]).astype(np.uint8)
    logger.info("Fallback: %s", candidates[0].name)
    return occ


def find_start_goal(occ, seed):
    H, W   = occ.shape
    margin = 45
    rng    = np.random.default_rng(seed)

    def rand_free(r0, r1, c0, c1):
        free = np.argwhere(occ[r0:r1, c0:c1] > 0)
        if not len(free):
            return None
        i = rng.integers(len(free))
        return (int(free[i,0]) + r0, int(free[i,1]) + c0)

    for _ in range(200):
        s = rand_free(margin, H//3,      margin, W//3)
        g = rand_free(2*H//3, H-margin,  2*W//3, W-margin)
        if s and g:
            return s, g
    return (margin, margin), (H-margin, W-margin)


# ---------------------------------------------------------------------------
# Path planning
# ---------------------------------------------------------------------------

def bfs_path(occ, via_min, start, goal, via_thresh=0.25):
    from collections import deque as _dq
    H, W = occ.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]),  int(goal[1])
    if not (0<=sr<H and 0<=sc<W and occ[sr,sc]): return None
    if not (0<=gr<H and 0<=gc<W and occ[gr,gc]): return None

    visited = np.zeros((H,W), dtype=bool)
    parent  = {}
    queue   = _dq([(sr,sc)])
    visited[sr,sc] = True

    while queue:
        r, c = queue.popleft()
        if (r,c) == (gr,gc):
            path, cur = [], (r,c)
            while cur in parent:
                path.append(cur); cur = parent[cur]
            path.append((sr,sc))
            return list(reversed(path))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0<=nr<H and 0<=nc<W and occ[nr,nc] and not visited[nr,nc]):
                if via_min is not None and via_min[nr,nc] < via_thresh:
                    continue
                visited[nr,nc] = True
                parent[(nr,nc)] = (r,c)
                queue.append((nr,nc))
    return None


def subsample(path, step):
    if not path: return []
    pts = [path[i] for i in range(0, len(path), step)]
    if path[-1] not in pts: pts.append(path[-1])
    return pts


# ---------------------------------------------------------------------------
# Model inference — timed
# ---------------------------------------------------------------------------

def predict_viability_timed(model, occ, dev):
    """Run model inference and return (via4, elapsed_ms)."""
    H, W = occ.shape
    inp = np.zeros((1, 3, H, W), dtype=np.float32)
    inp[0, 0] = occ.astype(np.float32)
    inp[0, 1] = 30 / 512
    inp[0, 2] = 20 / 512
    tensor = torch.from_numpy(inp).to(dev)

    # Warm CUDA sync before timing
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return probs[0].cpu().numpy(), elapsed_ms   # (4,H,W), float


def time_oracle(occ, robot_l=30, robot_w=20, dev=None):
    """Run Oracle on current map and return wall time in ms."""
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    generate_labels_for_map(occ, robot_l, robot_w)
    return (time.perf_counter() - t0) * 1000.0


def via_min_map(via4):
    return via4.min(axis=0)


# ---------------------------------------------------------------------------
# Obstacle injection
# ---------------------------------------------------------------------------

def inject_obstacle(occ, path, frac, size):
    occ_new = occ.copy()
    if not path: return occ_new, None
    idx = max(size, min(len(path)-size-1, int(len(path)*frac)))
    r, c = path[idx]
    r0=max(0,r-size//2); r1=min(occ.shape[0],r+size//2)
    c0=max(0,c-size//2); c1=min(occ.shape[1],c+size//2)
    occ_new[r0:r1, c0:c1] = 0
    return occ_new, (r0,r1,c0,c1)


# ---------------------------------------------------------------------------
# Timing bar helper
# ---------------------------------------------------------------------------

def draw_timing_bar(ax, nn_ms, oracle_ms):
    """
    Draw a horizontal bar chart comparing NN inference vs Oracle time.
    Shown on the right panel during and after replan events.
    """
    ax.set_xlim(0, max(oracle_ms * 1.15, 1))
    ax.set_ylim(-0.5, 1.5)
    ax.set_facecolor("#111111")
    ax.axis("off")

    speedup = oracle_ms / nn_ms if nn_ms > 0 else 0

    # Oracle bar
    ax.barh(1, oracle_ms, color="#C00000", height=0.4, alpha=0.85)
    ax.text(oracle_ms + oracle_ms * 0.02, 1,
            f"Oracle: {oracle_ms:.0f} ms",
            va="center", color="white", fontsize=8.5, fontweight="bold")

    # NN bar
    ax.barh(0, nn_ms, color="#00BFFF", height=0.4, alpha=0.85)
    ax.text(nn_ms + oracle_ms * 0.02, 0,
            f"NN: {nn_ms:.1f} ms  ({speedup:.0f}× faster)",
            va="center", color="white", fontsize=8.5, fontweight="bold")

    ax.text(oracle_ms * 0.5, -0.4,
            "Inference time per map",
            ha="center", va="top", color="#aaaaaa", fontsize=7.5, style="italic")


# ---------------------------------------------------------------------------
# Render one frame → PIL Image
# ---------------------------------------------------------------------------

PATH_COLOURS = ["#00BFFF", "#00FF88", "#FFD700", "#FF8C00"]
OBS_COLOURS  = ["#FF4444", "#FF8800", "#FF44FF"]


def render_frame(fig, ax1, ax2_via, ax2_bar,
                 occ, via_min,
                 robot_pos, current_path, path_history,
                 obstacles, replan_frames, frame_num,
                 start, goal, goal_reached,
                 last_nn_ms, oracle_ms,
                 show_timing):
    H, W = occ.shape
    for ax in [ax1, ax2_via, ax2_bar]:
        ax.cla()
        ax.axis("off")

    # ---- Left: map + paths + robot --------------------------------------
    rgb = np.ones((H, W, 3))
    rgb[occ == 0] = [0.09, 0.09, 0.09]
    ax1.imshow(rgb, origin="upper", interpolation="nearest")

    n_hist = len(path_history)
    for hi, old_path in enumerate(path_history):
        alpha = 0.15 + 0.3 * (hi / max(1, n_hist-1))
        if len(old_path) >= 2:
            ax1.plot([p[1] for p in old_path], [p[0] for p in old_path],
                     "--", color="#888888", alpha=alpha, linewidth=1.1, zorder=2)

    if len(current_path) >= 2:
        cidx = min(len(replan_frames), len(PATH_COLOURS)-1)
        ax1.plot([p[1] for p in current_path], [p[0] for p in current_path],
                 "-", color=PATH_COLOURS[cidx], linewidth=2.3, alpha=0.95, zorder=3)

    for bi, box in enumerate(obstacles):
        r0,r1,c0,c1 = box
        col = OBS_COLOURS[min(bi, len(OBS_COLOURS)-1)]
        ax1.add_patch(plt.Rectangle((c0,r0), c1-c0, r1-r0,
                                    lw=1.8, edgecolor=col, facecolor=col,
                                    alpha=0.55, zorder=4))

    ax1.plot(start[1], start[0], "g*", ms=14,
             markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax1.plot(goal[1],  goal[0],  "r*", ms=14,
             markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    rr, rc = robot_pos
    ax1.plot(rc, rr, "o",
             color="#00FF00" if goal_reached else "#FFD700",
             ms=10, markeredgecolor="white", markeredgewidth=0.9, zorder=7)

    if goal_reached:
        status, scol = "✓  GOAL REACHED", "#00FF00"
    elif replan_frames and frame_num == replan_frames[-1]:
        status, scol = f"⚠  OBSTACLE {len(replan_frames)} — RE-PLANNING", "#FF4444"
    elif replan_frames and frame_num <= replan_frames[-1] + SLOW_WINDOW:
        status, scol = "following new path...", "#00FF88"
    else:
        status, scol = "navigating →", "white"

    ax1.set_title(f"frame {frame_num:03d}  |  {status}",
                  color=scol, fontsize=10, pad=5, fontweight="bold")

    handles = [
        mpatches.Patch(color=PATH_COLOURS[0], label="Initial path"),
        mpatches.Patch(color=PATH_COLOURS[1], label="Re-planned (1)"),
        mpatches.Patch(color=PATH_COLOURS[2], label="Re-planned (2)"),
        mpatches.Patch(color=PATH_COLOURS[3], label="Re-planned (3)"),
        mpatches.Patch(color="#888888",        label="History"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=6.5,
               facecolor="#1a1a1a", edgecolor="#555555",
               labelcolor="white", framealpha=0.85)

    # ---- Right top: NN viability heatmap --------------------------------
    free_mask   = occ == 1
    via_display = np.where(free_mask, via_min, np.nan)
    ax2_via.axis("on")
    ax2_via.imshow(via_display, cmap="RdYlGn", vmin=0, vmax=1,
                   origin="upper", interpolation="nearest")
    ax2_via.set_xticks([]); ax2_via.set_yticks([])
    mean_v = float(via_min[free_mask].mean()) if free_mask.any() else 0.0
    title_col = "#FF4444" if show_timing else "white"
    ax2_via.set_title(
        ("NN RE-QUERIED  ↓  see timing" if show_timing
         else f"NN viability map  |  mean={mean_v:.2f}"),
        color=title_col, fontsize=9.5, pad=4, fontweight="bold"
    )

    # ---- Right bottom: timing bar (shown during/after replan) ----------
    ax2_bar.axis("on")
    if show_timing and last_nn_ms is not None:
        draw_timing_bar(ax2_bar, last_nn_ms, oracle_ms)
        ax2_bar.set_title("Inference speed comparison",
                           color="#aaaaaa", fontsize=8, pad=3)
    else:
        ax2_bar.set_facecolor("#111111")
        ax2_bar.axis("off")
        ax2_bar.set_title("", color="white", fontsize=8)

    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#111111", dpi=110)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def make_demo(occ_orig, model, dev, start, goal,
              total_frames, out_path):

    occ = occ_orig.copy()

    # Initial inference (timed)
    via4, init_nn_ms = predict_viability_timed(model, occ, dev)
    via_min = via_min_map(via4)
    oracle_ms = time_oracle(occ, dev=dev)
    logger.info("Initial NN inference: %.1f ms  |  Oracle: %.1f ms  |  speedup: %.1fx",
                init_nn_ms, oracle_ms, oracle_ms / init_nn_ms)

    path_dense = (bfs_path(occ, via_min, start, goal)
               or bfs_path(occ, None,    start, goal))
    if path_dense is None:
        logger.error("No path found — try a different map/seed"); return

    path         = subsample(path_dense, FAST_STEP)
    path_history = deque(maxlen=PATH_HISTORY_LEN)
    obstacles    = []
    replan_frames = []
    last_nn_ms   = init_nn_ms   # shown in timing bar

    robot_pos    = list(start)
    waypoint_idx = 0
    step_size    = FAST_STEP

    obs_schedule = list(OBSTACLE_SCHEDULE)

    slow_frame_set = set()
    for (fi,_,_) in obs_schedule:
        for d in range(-SLOW_WINDOW, SLOW_WINDOW+1):
            slow_frame_set.add(fi+d)

    # Figure layout: left panel (map) + right col split into via (top) + bar (bottom)
    fig = plt.figure(figsize=(13, 6.5))
    fig.patch.set_facecolor("#111111")
    gs  = fig.add_gridspec(2, 2,
                           height_ratios=[3.5, 1],
                           width_ratios=[1, 1],
                           hspace=0.35, wspace=0.08)

    ax1     = fig.add_subplot(gs[:, 0])   # left: full height
    ax2_via = fig.add_subplot(gs[0, 1])   # right top: viability map
    ax2_bar = fig.add_subplot(gs[1, 1])   # right bottom: timing bar

    for ax in [ax1, ax2_via, ax2_bar]:
        ax.set_facecolor("#111111")

    pil_frames = []
    durations  = []
    show_timing = False

    logger.info("Rendering up to %d frames...", total_frames)

    for frame_i in range(total_frames):

        # ---- Goal check -------------------------------------------------
        dist = np.linalg.norm(np.array(robot_pos) - np.array(goal))
        goal_reached = dist < GOAL_RADIUS

        # ---- Obstacle injection -----------------------------------------
        if not goal_reached and obs_schedule and frame_i == obs_schedule[0][0]:
            _, frac, size = obs_schedule.pop(0)
            occ, box = inject_obstacle(occ, path_dense, frac, size)
            obstacles.append(box)

            via4, nn_ms = predict_viability_timed(model, occ, dev)
            via_min  = via_min_map(via4)
            oracle_ms = time_oracle(occ, dev=dev)   # measure on this exact map
            last_nn_ms = nn_ms
            show_timing = True

            logger.info("[frame %d] Obstacle %d | NN: %.1f ms | Oracle: %.1f ms | speedup: %.1fx",
                        frame_i, len(obstacles), nn_ms, oracle_ms, oracle_ms/nn_ms)

            path_history.append(path[:])
            new_dense = (bfs_path(occ, via_min, tuple(robot_pos), goal)
                      or bfs_path(occ, None,    tuple(robot_pos), goal))
            if new_dense:
                path_dense   = new_dense
                path         = subsample(path_dense, SLOW_STEP)
                waypoint_idx = 0
                replan_frames.append(frame_i)
                step_size = SLOW_STEP

        # ---- Speed recovery & re-subsample ------------------------------
        if (replan_frames and step_size == SLOW_STEP
                and frame_i == replan_frames[-1] + SLOW_WINDOW):
            step_size  = FAST_STEP
            show_timing = False
            remaining  = path_dense[max(0, waypoint_idx*SLOW_STEP):]
            if len(remaining) > 1:
                path = path[:waypoint_idx+1] + subsample(remaining, FAST_STEP)

        # ---- Render ------------------------------------------------------
        pil_img = render_frame(
            fig, ax1, ax2_via, ax2_bar,
            occ, via_min,
            robot_pos, path, list(path_history),
            obstacles, replan_frames, frame_i,
            start, goal, goal_reached,
            last_nn_ms, oracle_ms,
            show_timing,
        )
        pil_frames.append(pil_img)

        if goal_reached:
            durations.append(GOAL_MS)
            for _ in range(2):
                pil_frames.append(pil_img)
                durations.append(GOAL_MS)
            logger.info("[frame %d] Goal reached!", frame_i)
            break

        durations.append(SLOW_MS if frame_i in slow_frame_set else FAST_MS)

        # ---- Advance robot ----------------------------------------------
        if waypoint_idx < len(path) - 1:
            waypoint_idx += 1
            robot_pos = list(path[waypoint_idx])

    plt.close(fig)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving GIF → %s  (%d frames)", out_path, len(pil_frames))
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    logger.info("Done ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    dev   = torch.device(args.device or get_device())
    logger.info("Device: %s", dev)

    model = MultiRobotViabilityUNet.from_checkpoint(args.checkpoint, device=str(dev))
    model.eval().to(dev)
    logger.info("Model loaded")

    # Warm up so first timing is accurate
    dummy = torch.zeros(1, 3, 64, 64, device=dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    logger.info("Model warmed up")

    occ         = load_or_pick_map(args.map, args.seed)
    start, goal = find_start_goal(occ, args.seed)
    logger.info("Start=%s  Goal=%s", start, goal)

    make_demo(
        occ_orig    = occ,
        model       = model,
        dev         = dev,
        start       = start,
        goal        = goal,
        total_frames= args.frames,
        out_path    = args.out,
    )


if __name__ == "__main__":
    main()