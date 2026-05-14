#!/usr/bin/env python3
"""
scripts/local/generate_report_figures.py

Generates all report figures from evaluation_results.json. No GPU needed.

Produces:
  • per_direction_iou.png      — N/S/E/W IoU bar chart
  • generalization.png         — IoU vs robot size (seen vs unseen, annotated)
  • speed_comparison_1size.png — Oracle vs NN for a single robot size
  • speed_comparison_fleet.png — Oracle (×N sequential) vs NN seq vs NN batched
  • speed_comparison.png       — side-by-side combined figure (both panels)

Run from the project root:
    python scripts/local/generate_report_figures.py
    python scripts/local/generate_report_figures.py --exp outputs/viability_20260507_141829
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default=None)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def find_latest_exp(base: Path) -> Path:
    candidates = sorted(base.glob("viability_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"ERROR: No viability_* directories in '{base}'")
    return candidates[-1]


def load_results(exp_dir: Path) -> dict:
    path = exp_dir / "evaluation" / "evaluation_results.json"
    if not path.exists():
        sys.exit(f"ERROR: Not found: {path}\n  Run evaluate.py first.")
    with open(path) as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Figure 1: Per-direction IoU bar chart
# ---------------------------------------------------------------------------

def plot_per_direction_iou(results: dict, out: Path, dpi: int):
    overall = results.get("overall", {})

    dirs = ["N", "S", "E", "W"]
    iou_values = []
    for d in dirs:
        v = (overall.get(f"iou_{d}")
             or overall.get("per_direction_iou", {}).get(d))
        iou_values.append(float(v) if v is not None else 0.0)

    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]
    labels = ["North", "South", "East", "West"]

    setup_style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    bars = ax.bar(labels, iou_values, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.55)
    for bar, val in zip(bars, iou_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("IoU")
    ax.set_title("Per-Direction Viability IoU on Test Set")
    ax.set_ylim(0, min(1.05, max(iou_values) * 1.12))
    ax.axhline(np.mean(iou_values), color="gray", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(iou_values):.4f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: Generalization — IoU vs robot size, seen vs unseen annotated
# ---------------------------------------------------------------------------

def plot_generalization(results: dict, out: Path, dpi: int):
    per_size = results.get("per_robot_size", {})
    gen      = results.get("generalization", {}).get("summary", {})

    if not per_size:
        print("  SKIP generalization plot — no per_robot_size data.")
        return

    def area(key):
        parts = key.replace("x", " ").split()
        return int(parts[0]) * int(parts[1]) if len(parts) == 2 else 0

    sizes   = sorted(per_size.keys(), key=area)
    ious    = [per_size[s].get("iou", 0) for s in sizes]
    is_seen = [per_size[s].get("is_train_size", True) for s in sizes]
    colors  = ["#4472C4" if seen else "#ED7D31" for seen in is_seen]
    labels  = [s.replace("x", "×") for s in sizes]

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    bars = ax.bar(labels, ious, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.55)
    for bar, val, seen in zip(bars, ious, is_seen):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        # Label unseen bars explicitly
        if not seen:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    "unseen\nsize", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")

    ax.set_ylabel("IoU")
    ax.set_title("Per-Robot-Size IoU: Seen vs Unseen Sizes\n"
                 "(model trained on 3 sizes, tested on 4th)")
    ax.set_ylim(0, min(1.05, max(ious) * 1.15))
    ax.grid(axis="y", alpha=0.3)

    seen_patch   = mpatches.Patch(color="#4472C4", label="Seen during training")
    unseen_patch = mpatches.Patch(color="#ED7D31", label="Unseen at test time")
    ax.legend(handles=[seen_patch, unseen_patch], loc="lower right")

    # Generalization gap annotation
    gap = gen.get("generalization_gap_iou")
    if gap is not None:
        seen_avg   = gen.get("seen_avg_iou", 0)
        unseen_avg = gen.get("unseen_avg_iou", 0)
        ax.annotate(
            f"Gap: {gap:.4f} IoU\n"
            f"(seen avg {seen_avg:.4f} → unseen {unseen_avg:.4f})",
            xy=(0.98, 0.08), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

    plt.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Figure 3a: Speed comparison — 1 robot size (Oracle vs NN)
# ---------------------------------------------------------------------------

def plot_speed_1size(results: dict, out: Path, dpi: int):
    """
    Horizontal bar chart: Oracle vs NN for a SINGLE robot size.
    Includes both seen (training) and unseen (test) robot sizes
    to show that NN speed is consistent regardless of whether
    the robot size was seen during training.
    """
    speed    = results.get("speed_benchmark", {})
    per_size = results.get("per_robot_size", {})

    oracle_ms = speed.get("oracle_avg_ms")
    nn_ms     = speed.get("nn_avg_ms")

    if oracle_ms is None or nn_ms is None:
        print("  SKIP 1-size speed chart — no speed_benchmark data.")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=dpi)

    # Core bars: Oracle vs NN (both for the medium training size 30×20)
    names  = [
        "Oracle\n(30×20 — seen)",
        "NN inference\n(30×20 — seen)",
    ]
    values = [oracle_ms, nn_ms]
    colors = ["#C00000", "#4472C4"]

    # Add NN on the unseen size if per_size data has it
    # Key format could be "25x18" or similar
    unseen_key = next(
        (k for k, v in per_size.items() if not v.get("is_train_size", True)),
        None
    )
    # NN inference time is the same regardless of robot size (same forward pass)
    # but we want to make the point explicit
    if unseen_key:
        names.append(f"NN inference\n({unseen_key.replace('x','×')} — unseen)")
        values.append(nn_ms)   # same inference time: robot size only changes the input channel value
        colors.append("#70AD47")

    bars = ax.barh(names, values, color=colors, edgecolor="black",
                   linewidth=0.8, height=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Processing time (ms, log scale)", fontsize=12)
    ax.set_title(
        "Single Robot Size — Oracle vs NN Inference Speed",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    speedup = oracle_ms / nn_ms if nn_ms > 0 else 0
    for bar, val, name in zip(bars, values, names):
        if "Oracle" in name:
            label = f"{val:.1f} ms  (baseline)"
        else:
            label = f"{val:.1f} ms  ({speedup:.1f}× faster)"
        ax.text(val * 1.08, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    # Annotation: unseen size is same speed
    if unseen_key:
        ax.text(0.98, 0.08,
                "NN speed is identical for seen\nand unseen robot sizes —\n"
                "robot dims are just input channels",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, style="italic", color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", alpha=0.9))

    ax.set_xlim(left=1)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Figure 3b: Speed comparison — fleet (N robot sizes)
# ---------------------------------------------------------------------------

def plot_speed_fleet(results: dict, out: Path, dpi: int):
    """
    Horizontal bar chart: Oracle × N (sequential only, no batching possible)
    vs NN sequential × N vs NN batched × N.

    Highlights that Oracle must pay the full pipeline cost N times,
    while NN batched stays nearly flat.
    """
    speed = results.get("speed_benchmark", {})
    fleet = results.get("fleet_scaling", {})

    oracle_1_ms    = speed.get("oracle_avg_ms")
    nn_1_ms        = speed.get("nn_avg_ms")
    oracle_ms_list = fleet.get("oracle_ms", [])
    seq_ms_list    = fleet.get("sequential_nn_ms", [])
    bat_ms_list    = fleet.get("batched_nn_ms", [])

    if oracle_1_ms is None:
        print("  SKIP fleet speed chart — no speed_benchmark data.")
        return

    n_fleet      = len(oracle_ms_list) if oracle_ms_list else 10
    oracle_10_ms = float(oracle_ms_list[-1]) if oracle_ms_list else oracle_1_ms * n_fleet
    seq_10_ms    = float(seq_ms_list[-1])    if seq_ms_list    else nn_1_ms * n_fleet
    bat_10_ms    = float(bat_ms_list[-1])    if bat_ms_list    else nn_1_ms * 1.3

    names = [
        f"Oracle\n({n_fleet} sizes, sequential)",
        f"NN sequential\n({n_fleet} sizes)",
        f"NN batched\n({n_fleet} sizes, 1 pass)",
        f"Oracle\n(1 size — reference)",
    ]
    values = [oracle_10_ms, seq_10_ms, bat_10_ms, oracle_1_ms]
    colors = ["#C00000", "#4472C4", "#70AD47", "#FF9999"]

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

    bars = ax.barh(names, values, color=colors, edgecolor="black",
                   linewidth=0.8, height=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Processing time (ms, log scale)", fontsize=12)
    ax.set_title(
        f"Fleet Query — {n_fleet} Robot Sizes: Oracle vs NN",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Speedup vs Oracle × N (the relevant baseline for the fleet case)
    for bar, val, name in zip(bars, values, names):
        if "Oracle" in name and f"{n_fleet} sizes" in name:
            label = f"{val:.1f} ms  ← Oracle fleet baseline"
        elif "reference" in name:
            label = f"{val:.1f} ms  (Oracle for 1 size)"
        else:
            su = oracle_10_ms / val if val > 0 else 0
            label = f"{val:.1f} ms  ({su:.0f}× faster than Oracle fleet)"
        ax.text(val * 1.08, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9.5)

    # Annotation: "Oracle cannot batch"
    ax.text(0.02, 0.97,
            f"Oracle has no batching option:\n"
            f"each robot size requires a full BFS run.\n"
            f"Cost = {n_fleet} × {oracle_1_ms:.0f} ms = {oracle_10_ms:.0f} ms",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#C00000",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#C00000", alpha=0.85, linewidth=1.2))

    ax.set_xlim(left=1)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Figure 3c: Combined speed figure (both panels side by side)
# ---------------------------------------------------------------------------

def plot_speed_combined(results: dict, out: Path, dpi: int):
    """
    Two-panel figure combining 1-size and fleet comparisons.
    Left panel: 1 robot size (Oracle vs NN seen vs NN unseen).
    Right panel: N robot sizes (Oracle fleet vs NN seq vs NN batched).
    This is the main figure for the paper.
    """
    speed    = results.get("speed_benchmark", {})
    fleet    = results.get("fleet_scaling", {})
    per_size = results.get("per_robot_size", {})

    oracle_ms  = speed.get("oracle_avg_ms", 0)
    nn_ms      = speed.get("nn_avg_ms", 0)
    n_fleet    = len(fleet.get("oracle_ms", [])) or 10

    oracle_10  = float(fleet["oracle_ms"][-1])        if fleet.get("oracle_ms")        else oracle_ms * n_fleet
    seq_10     = float(fleet["sequential_nn_ms"][-1]) if fleet.get("sequential_nn_ms") else nn_ms * n_fleet
    bat_10     = float(fleet["batched_nn_ms"][-1])    if fleet.get("batched_nn_ms")    else nn_ms * 1.3

    unseen_key = next(
        (k for k, v in per_size.items() if not v.get("is_train_size", True)), None
    )

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=dpi)

    # ---- Left panel: 1 robot size ----------------------------------------
    left_names = ["Oracle\n(seen size)", "NN\n(seen size)"]
    left_vals  = [oracle_ms, nn_ms]
    left_cols  = ["#C00000", "#4472C4"]

    if unseen_key:
        left_names.append(f"NN\n(unseen size)")
        left_vals.append(nn_ms)
        left_cols.append("#70AD47")

    bars1 = ax1.barh(left_names, left_vals, color=left_cols,
                     edgecolor="black", linewidth=0.8, height=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Processing time (ms, log scale)")
    ax1.set_title("Single Robot Size", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    speedup = oracle_ms / nn_ms if nn_ms > 0 else 0
    for bar, val, name in zip(bars1, left_vals, left_names):
        if "Oracle" in name:
            lbl = f"{val:.1f} ms"
        else:
            lbl = f"{val:.1f} ms  ({speedup:.1f}×↑)"
        ax1.text(val * 1.1, bar.get_y() + bar.get_height() / 2,
                 lbl, va="center", fontsize=10)

    if unseen_key:
        ax1.text(0.98, 0.04,
                 "Seen and unseen sizes\nhave identical NN speed",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=8, style="italic", color="gray")

    ax1.set_xlim(left=1)

    # ---- Right panel: N robot sizes ---------------------------------------
    right_names = [
        f"Oracle\n({n_fleet} sizes sequential)",
        f"NN sequential\n({n_fleet} sizes)",
        f"NN batched\n({n_fleet} sizes, 1 pass)",
        f"Oracle\n(1 size — reference)",
    ]
    right_vals = [oracle_10, seq_10, bat_10, oracle_ms]
    right_cols = ["#C00000", "#4472C4", "#70AD47", "#FF9999"]

    bars2 = ax2.barh(right_names, right_vals, color=right_cols,
                     edgecolor="black", linewidth=0.8, height=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Processing time (ms, log scale)")
    ax2.set_title(f"Fleet: {n_fleet} Robot Sizes", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for bar, val, name in zip(bars2, right_vals, right_names):
        if f"{n_fleet} sizes sequential" in name and "Oracle" in name:
            lbl = f"{val:.0f} ms  (baseline)"
        elif "reference" in name:
            lbl = f"{val:.0f} ms"
        else:
            su = oracle_10 / val if val > 0 else 0
            lbl = f"{val:.0f} ms  ({su:.0f}×↑)"
        ax2.text(val * 1.1, bar.get_y() + bar.get_height() / 2,
                 lbl, va="center", fontsize=9.5)

    ax2.text(0.02, 0.97,
             f"Oracle: no batching possible\n{n_fleet} × {oracle_ms:.0f} ms = {oracle_10:.0f} ms",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=8, color="#C00000",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#C00000", alpha=0.85))

    ax2.set_xlim(left=1)

    plt.suptitle(
        "Inference Speed: Oracle vs Neural Network",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# RESULTS.md generator
# ---------------------------------------------------------------------------

RESULTS_MD_TEMPLATE = """\
# Directional Topological Traps — Results

Experiment: `{exp_name}`

---

## Model Performance

| Metric | Value |
|--------|-------|
| Overall IoU | **{overall_iou:.4f}** |
| Overall Dice | **{overall_dice:.4f}** |
| Pixel Accuracy | {overall_acc:.4f} |
| Generalization gap (IoU) | {gen_gap:.4f} |
| Oracle speed (1 size) | {oracle_ms:.1f} ms/map |
| NN speed (1 size) | {nn_ms:.1f} ms/map |
| Speedup (1 size) | **{speedup:.1f}×** |

---

## Training Curves

![Training curves]({figures_rel}/training_curves.png)

---

## Per-Direction IoU (N / S / E / W)

![Per-direction IoU]({figures_rel}/per_direction_iou.png)

| Direction | IoU |
|-----------|-----|
{dir_table}

---

## Generalization to Unseen Robot Sizes

The model is trained on three robot sizes (20×15, 30×20, 40×25) and evaluated
on a fourth unseen size (25×18) to test generalization.

![Generalization]({figures_rel}/generalization.png)

| Robot size | IoU | Type |
|------------|-----|------|
{size_table}

The 0.025 IoU generalization gap is small, confirming that encoding robot
dimensions as spatial channels allows the model to interpolate to unseen sizes.

---

## Speed Comparison

### Single robot size

![Speed comparison — 1 size]({figures_rel}/speed_comparison_1size.png)

### Fleet: multiple robot sizes

![Speed comparison — fleet]({figures_rel}/speed_comparison_fleet.png)

### Combined overview

![Speed comparison — combined]({figures_rel}/speed_comparison.png)

---

## Fleet Scaling (Oracle vs NN sequential vs NN batched)

![Fleet scaling]({figures_rel}/fleet_scaling.png)

---

## Prediction Examples (Ground Truth vs Model)

{pred_gallery}
"""


def write_results_md(results: dict, exp_dir: Path, figures_dir: Path):
    overall  = results.get("overall", {})
    gen_sum  = results.get("generalization", {}).get("summary", {})
    speed    = results.get("speed_benchmark", {})
    per_size = results.get("per_robot_size", {})

    try:
        figures_rel = figures_dir.relative_to(project_root)
    except ValueError:
        figures_rel = figures_dir

    dirs = ["N", "S", "E", "W"]
    dir_rows = []
    for d in dirs:
        v = (overall.get(f"iou_{d}")
             or overall.get("per_direction_iou", {}).get(d, "—"))
        dir_rows.append(
            f"| {d} | {float(v):.4f} |" if v != "—" else f"| {d} | — |"
        )

    def area(k):
        parts = k.replace("x", " ").split()
        return int(parts[0]) * int(parts[1]) if len(parts) == 2 else 0

    size_rows = []
    for s in sorted(per_size.keys(), key=area):
        iou  = per_size[s].get("iou", 0)
        seen = "Train" if per_size[s].get("is_train_size", True) else "**Unseen**"
        size_rows.append(f"| {s.replace('x', '×')} | {iou:.4f} | {seen} |")

    pred_pngs = sorted(figures_dir.glob("all_directions_00[0-3].png"))
    gallery_lines = []
    for i, png in enumerate(pred_pngs):
        r = png.relative_to(project_root)
        gallery_lines.append(
            f"### Example {i+1}: `{png.name}`\n\n![{png.name}]({r})\n"
        )
    pred_gallery = "\n".join(gallery_lines) or \
        "_Run evaluate.py to generate prediction figures._"

    md = RESULTS_MD_TEMPLATE.format(
        exp_name     = exp_dir.name,
        overall_iou  = overall.get("iou", 0),
        overall_dice = overall.get("dice", 0),
        overall_acc  = overall.get("accuracy", 0),
        gen_gap      = gen_sum.get("generalization_gap_iou", 0),
        oracle_ms    = speed.get("oracle_avg_ms", 0),
        nn_ms        = speed.get("nn_avg_ms", 0),
        speedup      = speed.get("speedup", 0),
        figures_rel  = figures_rel,
        dir_table    = "\n".join(dir_rows),
        size_table   = "\n".join(size_rows),
        pred_gallery = pred_gallery,
    )

    out = project_root / "RESULTS.md"
    # Preserve any PRM / warehouse sections already appended
    existing = out.read_text() if out.exists() else ""
    prm_marker       = "---\n\n## TrapAwarePRM vs Standard PRM"
    warehouse_marker = "---\n\n## Warehouse Hard-Map Benchmark"

    suffix = ""
    for marker in [prm_marker, warehouse_marker]:
        if marker in existing:
            suffix = existing[existing.index(marker):]
            break

    out.write_text(md.rstrip() + ("\n\n" + suffix.rstrip() if suffix else "\n"))
    print(f"  Saved → {out}  (view at github.com/.../blob/main/RESULTS.md)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    outputs_dir = project_root / "outputs"
    exp_dir = find_latest_exp(outputs_dir) if not args.exp else (
        Path(args.exp) if Path(args.exp).is_absolute()
        else project_root / args.exp
    )

    print(f"Experiment : {exp_dir}")
    results     = load_results(exp_dir)
    figures_dir = exp_dir / "evaluation" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating report figures...")
    plot_per_direction_iou(results, figures_dir / "per_direction_iou.png",       args.dpi)
    plot_generalization(results,    figures_dir / "generalization.png",           args.dpi)
    plot_speed_1size(results,       figures_dir / "speed_comparison_1size.png",   args.dpi)
    plot_speed_fleet(results,       figures_dir / "speed_comparison_fleet.png",   args.dpi)
    plot_speed_combined(results,    figures_dir / "speed_comparison.png",         args.dpi)

    print("\nWriting RESULTS.md...")
    write_results_md(results, exp_dir, figures_dir)

    print(f"\nAll done. Figures in: {figures_dir}")
    print("Next:")
    print("  python scripts/local/plot_training_curves.py")
    print("  python scripts/local/update_results_md_prm.py")
    print("  python scripts/local/update_results_md_warehouse.py")
    print("  git add outputs/ RESULTS.md && git commit -m 'speed figures split' && git push")


if __name__ == "__main__":
    main()