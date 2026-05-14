#!/usr/bin/env python3
"""
scripts/local/plot_training_curves.py

Generate the training loss + IoU curves figure from a saved checkpoint.
The history (train_loss, val_loss, val_iou, val_dice, learning_rate) is
stored inside the checkpoint under the "history" key by trainer.py.

Run from the project root:
    python scripts/local/plot_training_curves.py

    # Specific experiment
    python scripts/local/plot_training_curves.py --exp outputs/viability_20260507_141829

    # Specific checkpoint
    python scripts/local/plot_training_curves.py --checkpoint outputs/viability_20260507_141829/checkpoints/best_iou.pth
"""

import sys
import argparse
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe; works on cluster and locally
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training curves from checkpoint history"
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment directory (default: latest outputs/viability_*/)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint path (default: <exp>/checkpoints/best_iou.pth)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output PNG path (default: <exp>/evaluation/figures/training_curves.png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output DPI (default: 200)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment / checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_exp(base: Path) -> Path:
    candidates = sorted(base.glob("viability_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"ERROR: No viability_* directories found under '{base}'.")
    return candidates[-1]


def load_history(checkpoint_path: Path) -> dict:
    """
    Extract training history from a checkpoint.

    Tries (in order):
      1. training_history.json next to the checkpoint directory
      2. "history" key inside the .pth checkpoint
    """
    # 1. JSON sidecar — trainer.py saves this after training completes
    json_path = checkpoint_path.parent.parent / "training_history.json"
    if json_path.exists():
        print(f"Loading history from: {json_path}")
        with open(json_path) as f:
            return json.load(f)

    # 2. Embedded in checkpoint
    print(f"Loading history from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history")
    if not history:
        sys.exit(
            "ERROR: No 'history' key found in checkpoint and no training_history.json.\n"
            "       The checkpoint may have been saved before training finished, "
            "or history was not recorded."
        )
    return history


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(history: dict, out_path: Path, dpi: int) -> None:
    """
    Four-panel figure: train/val loss, val IoU, val Dice, learning rate.
    Also overlays per-direction IoU (N/S/E/W) on a fifth panel if present.
    """
    epochs = range(1, len(history.get("train_loss", history.get("val_loss", []))) + 1)
    if not list(epochs):
        sys.exit("ERROR: History is empty — no epochs recorded.")

    # Detect per-direction keys  (val_iou_N, val_iou_S, …)
    dir_keys = [k for k in history if k.startswith("val_iou_") and
                len(history[k]) == len(list(epochs))]

    has_dirs = bool(dir_keys)
    ncols = 3 if has_dirs else 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=dpi)
    flat = axes.flatten()

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })

    # ---- Panel 0: Loss curves ----------------------------------------
    ax = flat[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 1: Val IoU --------------------------------------------
    ax = flat[1]
    if "val_iou" in history:
        ax.plot(epochs, history["val_iou"], "g-", linewidth=2, label="Val IoU")
        best_epoch = int(np.argmax(history["val_iou"])) + 1
        best_iou   = max(history["val_iou"])
        ax.axvline(best_epoch, color="g", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Best: {best_iou:.4f}\n(epoch {best_epoch})",
            xy=(best_epoch, best_iou),
            xytext=(best_epoch + max(1, len(list(epochs)) * 0.05), best_iou - 0.02),
            fontsize=9,
            color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Validation IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(history.get("val_iou", [0])) - 0.05))

    # ---- Panel 2: Val Dice -------------------------------------------
    ax = flat[2]
    if "val_dice" in history:
        ax.plot(epochs, history["val_dice"], "m-", linewidth=2, label="Val Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Learning rate --------------------------------------
    ax = flat[3]
    if "learning_rate" in history:
        ax.plot(epochs, history["learning_rate"], "k-", linewidth=2)
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    # ---- Panel 4 (optional): Per-direction IoU -----------------------
    if has_dirs and len(flat) > 4:
        ax = flat[4]
        colors = {"val_iou_N": "royalblue", "val_iou_S": "tomato",
                  "val_iou_E": "seagreen",  "val_iou_W": "darkorange"}
        labels = {"val_iou_N": "North", "val_iou_S": "South",
                  "val_iou_E": "East",  "val_iou_W": "West"}
        for k in sorted(dir_keys):
            ax.plot(epochs, history[k], linewidth=2,
                    color=colors.get(k, "gray"),
                    label=labels.get(k, k))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title("Per-Direction Validation IoU")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused panels
    for ax in flat[5 if has_dirs else 4:]:
        ax.set_visible(False)

    num_epochs = len(list(epochs))
    best_iou   = max(history.get("val_iou", [0]))
    plt.suptitle(
        f"Directional Topological Traps — Training History\n"
        f"({num_epochs} epochs, best val IoU = {best_iou:.4f})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    outputs_dir = project_root / "outputs"

    # Resolve experiment
    if args.exp:
        exp_dir = Path(args.exp) if Path(args.exp).is_absolute() \
                  else project_root / args.exp
    else:
        exp_dir = find_latest_exp(outputs_dir)

    print(f"Experiment : {exp_dir}")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() \
                    else project_root / args.checkpoint
    else:
        ckpt_path = exp_dir / "checkpoints" / "best_iou.pth"
        if not ckpt_path.exists():
            # fall back to last.pth
            ckpt_path = exp_dir / "checkpoints" / "last.pth"

    if not ckpt_path.exists():
        sys.exit(f"ERROR: Checkpoint not found: {ckpt_path}")

    # Load history
    history = load_history(ckpt_path)

    num_epochs = len(history.get("train_loss", history.get("val_loss", [])))
    best_iou   = max(history.get("val_iou", [0]))
    print(f"Epochs recorded : {num_epochs}")
    print(f"Best val IoU    : {best_iou:.4f}")
    print(f"History keys    : {list(history.keys())}")

    # Resolve output path
    if args.out:
        out_path = Path(args.out) if Path(args.out).is_absolute() \
                   else project_root / args.out
    else:
        out_path = exp_dir / "evaluation" / "figures" / "training_curves.png"

    plot(history, out_path, dpi=args.dpi)


if __name__ == "__main__":
    main()