#!/usr/bin/env python3
"""
Main Training Script for Directional Topological Traps Model.

This is the project's *single* training entry point. It supports
five distinct training modes, selected by --oracle_type:

  basic            (default — preserves the original pipeline)
                   3-channel input  (occupancy + L + W) → 4-channel binary
                   viability for [N, S, E, W]. BCE+Dice loss.

  continuous_angle (Direction 1 — Any-Heading Oracle)
                   5-channel input  (occupancy + L + W + sin θ + cos θ)
                   → 1-channel binary viability for the chosen angle.
                   At each __getitem__ a random angle is sampled and the
                   ground truth is computed *on the fly* via the
                   rotate-East-rotate Oracle in extended_oracles.py.

  cost_map         (Time-to-Escape regression — 4 directions)
                   3-channel input → 4-channel float (Huber loss).
                   Targets are normalised escape costs for [N, S, E, W].

  angle_cost_map   (Direction 1 + cost-map combined)
                   5-channel input → 1-channel float, escape cost for
                   the sampled angle.

Plus a separate runtime mode:

  --demo_closed_loop
                   Loads a trained checkpoint and runs an interactive /
                   matplotlib-animation demo where the environment
                   changes in real-time and the planner re-queries the
                   model to avoid newly-formed traps. (Direction 2.)

The original Phase-1 behaviour is fully preserved when
--oracle_type basic and --demo_closed_loop is not set.

Usage:
    # Original behaviour (unchanged)
    python scripts/train.py --config configs/config.yaml

    # Continuous-angle training
    python scripts/train.py --config configs/config.yaml \
        --oracle_type continuous_angle --epochs 30

    # Time-to-Escape regression
    python scripts/train.py --config configs/config.yaml \
        --oracle_type cost_map --epochs 30

    # Closed-loop demo
    python scripts/train.py --demo_closed_loop \
        --checkpoint outputs/<exp>/checkpoints/best_iou.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Existing project modules
from src.data.dataset import MultiRobotViabilityDataset, create_dataloaders
from src.data.augmentations import MultiRobotAugmentation, DeterministicAugmentation
from src.models.unet import MultiRobotViabilityUNet, create_model
from src.models.losses import create_loss
from src.training.trainer import Trainer, create_optimizer, create_scheduler
from src.utils.helpers import set_seed, get_device, print_model_summary

# at the top of train.py, add to the existing velocity_oracle imports:
from src.oracle.velocity_oracle import (
    DEFAULT_VELOCITIES,
    V_MAX,
    braking_distance_px,
    normalise_velocity,
    velocity_viability,
)

# Phase-2 extensions
from src.oracle.extended_oracles import (
    angle_to_sincos,
    continuous_angle_viability,
    escape_cost_map,
    escape_cost_map_for_angle,
    normalise_cost_map,
    COST_MAX_VALUE,
)
from src.experiments import (
    VelocityViabilityDataset,
    VelocityCostMapDataset,
)

logger = logging.getLogger(__name__)

# Valid --oracle_type values
ORACLE_TYPES = ("basic", "continuous_angle", "cost_map", "angle_cost_map", "velocity", "velocity_cost")


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Directional Topological Traps Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- existing flags (unchanged) ---
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (reduced data)")

    # --- Phase-2 flags ---
    parser.add_argument("--oracle_type", type=str, default="basic",
                        choices=ORACLE_TYPES + ("auto",),
                        help="Which Oracle / model variant to train. "
                             "Use 'auto' only with --demo_closed_loop to read "
                             "the mode from the checkpoint metadata.")
    parser.add_argument("--num-angles-per-map", type=int, default=8,
                        help="(continuous_angle / angle_cost_map) virtual "
                             "angle samples per epoch per map.")
    parser.add_argument("--demo_closed_loop", action="store_true",
                        help="Run the closed-loop trap-aware planning demo "
                             "instead of training. Requires --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --demo_closed_loop.")
    parser.add_argument("--demo-map", type=str, default=None,
                        help="Optional .npy occupancy grid for closed-loop demo. "
                             "Default: pick a random validation map.")
    parser.add_argument("--demo-output", type=str,
                        default="outputs/closed_loop_demo/demo.gif",
                        help="Where to save the demo animation.")
    parser.add_argument("--demo-frames", type=int, default=120,
                        help="Total number of simulation frames.")
    parser.add_argument("--demo-change-every", type=int, default=20,
                        help="Re-modify the environment every N frames.")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict, args) -> dict:
    """Set up experiment directory and logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"viability_{args.oracle_type}_{timestamp}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["paths"]["output_dir"]) / exp_name

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "tensorboard").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return {
        "output_dir": output_dir,
        "checkpoint_dir": output_dir / "checkpoints",
        "tensorboard_dir": output_dir / "tensorboard",
        "exp_name": exp_name,
    }


# ===========================================================================
# Phase-2 datasets — used only when --oracle_type != basic
# ===========================================================================

class _BaseExtendedDataset(Dataset):
    """
    Shared parent for the Phase-2 datasets.

    Reads the same manifest.csv and processed map directory as the
    Phase-1 dataset so the train/val/test split stays identical and
    every label is computed on-the-fly per __getitem__.
    """

    def __init__(
        self,
        map_dir: Union[str, Path],
        manifest_path: Optional[str],
        robot_sizes: List[Tuple[int, int]],
        split: str,
        resolution: int = 512,
        transform: Optional[Callable] = None,
    ):
        self.map_dir = Path(map_dir)
        self.robot_sizes = list(robot_sizes)
        self.split = split
        self.resolution = resolution
        self.transform = transform

        if manifest_path and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            split_files = df[df["split"] == split]["filename"].tolist()
        else:
            split_files = [f for f in os.listdir(map_dir) if f.endswith(".npy")]

        self.map_files = []
        for fn in sorted(split_files):
            p = self.map_dir / fn
            if p.exists():
                self.map_files.append(p)

        logger.info(
            "ExtendedDataset[%s]: %d maps × %d robot sizes  (%s)",
            split, len(self.map_files), len(self.robot_sizes), type(self).__name__,
        )

    def __len__(self) -> int:
        return len(self.map_files)

    def _select_robot(self, idx: int) -> Tuple[int, int]:
        """Random robot size for training, deterministic for val/test."""
        if self.split == "train":
            return self.robot_sizes[np.random.randint(0, len(self.robot_sizes))]
        return self.robot_sizes[idx % len(self.robot_sizes)]

    def _load_map(self, idx: int) -> np.ndarray:
        return np.load(self.map_files[idx]).astype(np.uint8)


class ContinuousAngleViabilityDataset(_BaseExtendedDataset):
    """
    Dataset for --oracle_type continuous_angle.

    Inputs : 5-channel  [occupancy, L_norm, W_norm, sin θ, cos θ]
    Targets: 1-channel  binary viability for heading θ
    """

    def __init__(self, *args, num_angles_per_map: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_angles_per_map = max(1, num_angles_per_map)

    def __len__(self) -> int:
        # Virtual length: each map appears num_angles_per_map times per epoch.
        return len(self.map_files) * self.num_angles_per_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        map_idx = idx % len(self.map_files)
        occ = self._load_map(map_idx)
        L, W = self._select_robot(map_idx)

        # Sample angle: random for train, deterministic for val/test
        if self.split == "train":
            angle = float(np.random.uniform(0.0, 360.0))
        else:
            # 8 evenly spaced angles per map for stable validation
            virtual = idx // len(self.map_files)
            angle = (virtual * (360.0 / self.num_angles_per_map)) % 360.0

        # On-the-fly oracle (cardinals are exact, others use NEAREST rotate)
        label = continuous_angle_viability(occ, L, W, angle).astype(np.float32)

        if self.transform is not None:
            # Pack label as (1,H,W) so existing transforms can index axis 0
            occ_f, lab4 = self.transform(
                occ.astype(np.float32),
                np.broadcast_to(label, (4, *label.shape)).astype(np.float32),
            )
            occ = (occ_f > 0.5).astype(np.uint8)
            label = lab4[0]

        H, Wd = occ.shape
        s, c = angle_to_sincos(angle)
        x = np.zeros((5, H, Wd), dtype=np.float32)
        x[0] = occ.astype(np.float32)
        x[1] = float(L) / float(self.resolution)
        x[2] = float(W) / float(self.resolution)
        x[3] = s
        x[4] = c

        y = label.astype(np.float32)[None, :, :]  # (1, H, W)

        meta = {
            "robot_length": L, "robot_width": W,
            "map_name": self.map_files[map_idx].stem,
            "angle_deg": float(angle),
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


class CostMapDataset(_BaseExtendedDataset):
    """
    Dataset for --oracle_type cost_map.

    Inputs : 3-channel  [occupancy, L_norm, W_norm]
    Targets: 4-channel  float (normalised escape cost for [N, S, E, W])
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        occ = self._load_map(idx)
        L, W = self._select_robot(idx)

        cost4 = escape_cost_map(occ, L, W, direction=None)  # (4, H, Wd) float
        cost4_norm = normalise_cost_map(cost4)              # → [0, 1]

        if self.transform is not None:
            occ_f, lab4 = self.transform(
                occ.astype(np.float32), cost4_norm.astype(np.float32),
            )
            occ = (occ_f > 0.5).astype(np.uint8)
            cost4_norm = lab4

        H, Wd = occ.shape
        x = np.zeros((3, H, Wd), dtype=np.float32)
        x[0] = occ.astype(np.float32)
        x[1] = float(L) / float(self.resolution)
        x[2] = float(W) / float(self.resolution)

        y = cost4_norm.astype(np.float32)  # (4, H, W)
        meta = {"robot_length": L, "robot_width": W,
                "map_name": self.map_files[idx].stem}
        return torch.from_numpy(x), torch.from_numpy(y), meta


class AngleCostMapDataset(_BaseExtendedDataset):
    """
    Dataset for --oracle_type angle_cost_map.

    Inputs : 5-channel  [occupancy, L_norm, W_norm, sin θ, cos θ]
    Targets: 1-channel  float (normalised escape cost for heading θ)
    """

    def __init__(self, *args, num_angles_per_map: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_angles_per_map = max(1, num_angles_per_map)

    def __len__(self) -> int:
        return len(self.map_files) * self.num_angles_per_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        map_idx = idx % len(self.map_files)
        occ = self._load_map(map_idx)
        L, W = self._select_robot(map_idx)

        if self.split == "train":
            angle = float(np.random.uniform(0.0, 360.0))
        else:
            virtual = idx // len(self.map_files)
            angle = (virtual * (360.0 / self.num_angles_per_map)) % 360.0

        cost = escape_cost_map_for_angle(occ, L, W, angle)
        cost_norm = normalise_cost_map(cost)  # (H, W) in [0, 1]

        if self.transform is not None:
            # Same packing trick as ContinuousAngleViabilityDataset
            occ_f, lab4 = self.transform(
                occ.astype(np.float32),
                np.broadcast_to(cost_norm, (4, *cost_norm.shape)).astype(np.float32),
            )
            occ = (occ_f > 0.5).astype(np.uint8)
            cost_norm = lab4[0]

        H, Wd = occ.shape
        s, c = angle_to_sincos(angle)
        x = np.zeros((5, H, Wd), dtype=np.float32)
        x[0] = occ.astype(np.float32)
        x[1] = float(L) / float(self.resolution)
        x[2] = float(W) / float(self.resolution)
        x[3] = s
        x[4] = c

        y = cost_norm.astype(np.float32)[None, :, :]  # (1, H, W)
        meta = {"robot_length": L, "robot_width": W,
                "map_name": self.map_files[map_idx].stem,
                "angle_deg": float(angle)}
        return torch.from_numpy(x), torch.from_numpy(y), meta


def _make_extended_loaders(
    config: dict,
    args: argparse.Namespace,
    train_sizes: List[Tuple[int, int]],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build extended-mode train/val/test loaders."""
    map_dir = config["paths"]["processed_maps"]
    manifest_path = config["paths"]["manifest"]
    resolution = config["data"].get("resolution", 512)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)

    common = dict(
        map_dir=map_dir,
        manifest_path=manifest_path,
        robot_sizes=train_sizes,
        resolution=resolution,
    )

    # Augmentations: training only; same direction-aware swap rule as basic.
    train_aug = MultiRobotAugmentation()
    val_aug = DeterministicAugmentation()

    if args.oracle_type == "continuous_angle":
        DS = ContinuousAngleViabilityDataset
        kwargs_extra = dict(num_angles_per_map=args.num_angles_per_map)
    elif args.oracle_type == "cost_map":
        DS = CostMapDataset
        kwargs_extra = dict()
    elif args.oracle_type == "angle_cost_map":
        DS = AngleCostMapDataset
        kwargs_extra = dict(num_angles_per_map=args.num_angles_per_map)
    elif args.oracle_type == "velocity":
        DS = VelocityViabilityDataset
        kwargs_extra = dict(
            velocities=DEFAULT_VELOCITIES,   # ← now reads from velocity_oracle.py
            max_decel=2.0, px_per_m=10.0, num_velocities_per_map=4,
        )
    elif args.oracle_type == "velocity_cost":
        DS = VelocityCostMapDataset
        kwargs_extra = dict(
            velocities=DEFAULT_VELOCITIES,   # ← same fix
            max_decel=2.0, px_per_m=10.0, num_velocities_per_map=4,
        )
    else:
        raise ValueError(f"_make_extended_loaders called with oracle_type={args.oracle_type}")

    # IMPORTANT — for angle modes, augmentation interacts with sin/cos in a
    # non-trivial way (rotation flips would change the "true" angle without
    # us also rotating sin/cos). Easiest correct path: disable augmentation
    # for angle modes. This keeps semantics clean; we trade a small amount
    # of regularisation for correctness.
    if args.oracle_type in ("continuous_angle", "angle_cost_map"):
        train_aug = None
        val_aug = None

    ds_train = DS(split="train", transform=train_aug, **common, **kwargs_extra)
    ds_val = DS(split="val", transform=val_aug, **common, **kwargs_extra)
    ds_test = DS(split="test", transform=val_aug, **common, **kwargs_extra)

    pin_memory = bool(config["training"].get("pin_memory", True))
    persistent = num_workers > 0

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


# ===========================================================================
# Loss / Metric for Phase-2 modes
# ===========================================================================

class _RegressionMetricTracker:
    """Minimal per-batch regression tracker: MAE / RMSE / Pearson r."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.pearson_sum = 0.0
        # Per-channel
        self.per_channel_mae: Dict[int, float] = {}
        self.per_channel_count: Dict[int, int] = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = int(pred.shape[0])
        self.count += batch_size

        with torch.no_grad():
            diff = pred - target
            mae = diff.abs().mean().item()
            rmse = torch.sqrt((diff * diff).mean()).item()
            pf = (pred - pred.mean()).flatten()
            tf = (target - target.mean()).flatten()
            denom = torch.sqrt((pf * pf).sum() * (tf * tf).sum()).clamp_min(1e-12)
            pearson = ((pf * tf).sum() / denom).item()

        self.mae_sum += mae * batch_size
        self.rmse_sum += rmse * batch_size
        self.pearson_sum += pearson * batch_size

        for c in range(pred.shape[1]):
            d = (pred[:, c] - target[:, c]).abs().mean().item()
            self.per_channel_mae[c] = self.per_channel_mae.get(c, 0.0) + d * batch_size
            self.per_channel_count[c] = self.per_channel_count.get(c, 0) + batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        out = {
            "mae": self.mae_sum / self.count,
            "rmse": self.rmse_sum / self.count,
            "pearson_r": self.pearson_sum / self.count,
            # NB: trainer keys on "iou" for "best" tracking — we map an
            # increasing "goodness" metric here so existing checkpointing
            # logic still works without modification.
            "iou": (self.pearson_sum / self.count + 1.0) / 2.0,  # ∈ [0, 1]
            "dice": 1.0 - min(1.0, self.mae_sum / self.count),
            "accuracy": 1.0 - min(1.0, self.rmse_sum / self.count),
        }
        names = ["N", "S", "E", "W"]
        for c, n in enumerate(names):
            if c in self.per_channel_count and self.per_channel_count[c] > 0:
                out[f"mae_{n}"] = self.per_channel_mae[c] / self.per_channel_count[c]
                out[f"iou_{n}"] = 1.0 - min(1.0, out[f"mae_{n}"])
        return out

    def get_summary_string(self) -> str:
        m = self.compute()
        if not m:
            return "(no samples)"
        return (f"MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | "
                f"Pearson r={m['pearson_r']:.4f}")


class _BinaryAngleMetricTracker:
    """IoU/Dice tracker for the 1-channel angle-conditioned binary mode.
    API matches the basic-mode tracker so the existing Trainer works
    unchanged."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.count = 0
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.acc_sum = 0.0

    def update(self, pred_logits: torch.Tensor, target: torch.Tensor,
               batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = int(pred_logits.shape[0])
        self.count += batch_size
        with torch.no_grad():
            probs = torch.sigmoid(pred_logits)
            p = (probs > self.threshold).float()
            inter = (p * target).sum()
            psum = p.sum()
            tsum = target.sum()
            union = psum + tsum - inter
            iou = (inter / (union + 1e-6)).item()
            dice = (2 * inter / (psum + tsum + 1e-6)).item()
            acc = (p == target).float().mean().item()
        self.iou_sum += iou * batch_size
        self.dice_sum += dice * batch_size
        self.acc_sum += acc * batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "iou": self.iou_sum / self.count,
            "dice": self.dice_sum / self.count,
            "accuracy": self.acc_sum / self.count,
        }

    def get_summary_string(self) -> str:
        m = self.compute()
        if not m:
            return "(no samples)"
        return f"IoU={m['iou']:.4f} | Dice={m['dice']:.4f} | Acc={m['accuracy']:.4f}"


def _build_loss_for_mode(oracle_type: str, training_cfg: dict) -> nn.Module:
    """Pick the loss appropriate for the chosen oracle_type."""
    if oracle_type == "basic":
        loss_cfg = training_cfg.get("loss", {})
        return create_loss(
            loss_type=loss_cfg.get("type", "dice_bce"),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
        )
    if oracle_type == "continuous_angle":
        # Same Dice+BCE works for 1-channel binary
        loss_cfg = training_cfg.get("loss", {})
        return create_loss(
            loss_type=loss_cfg.get("type", "dice_bce"),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
        )
    if oracle_type in ("cost_map", "angle_cost_map"):
        # Huber: robust regression for cost maps. Targets in [0, 1].
        return nn.SmoothL1Loss(beta=0.1)
    if oracle_type == "velocity":
        # Binary classification — same Dice+BCE as basic
        loss_cfg = training_cfg.get("loss", {})
        return create_loss(
            loss_type=loss_cfg.get("type", "dice_bce"),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
        )
    if oracle_type == "velocity_cost":
        return nn.SmoothL1Loss(beta=0.1)
    raise ValueError(f"Unknown oracle_type: {oracle_type}")


def _make_metric_tracker(oracle_type: str):
    """Tracker compatible with the existing Trainer.validate() loop."""
    if oracle_type == "basic":
        from src.models.metrics import MetricTracker
        return MetricTracker()
    if oracle_type == "continuous_angle":
        return _BinaryAngleMetricTracker()
    if oracle_type == "velocity":
        # 4-channel binary — same tracker as basic
        from src.models.metrics import MetricTracker
        return MetricTracker()
    return _RegressionMetricTracker()  # cost_map, angle_cost_map, velocity_cost


# ===========================================================================
# Closed-loop demo (Direction 2)
# ===========================================================================

def _model_inputs_for_demo(
    occ: np.ndarray, L: int, W: int, resolution: int, oracle_type: str,
    angle_deg: Optional[float] = None,
) -> torch.Tensor:
    """Construct a 1-batch input tensor for inference at demo time."""
    H, Wd = occ.shape
    if oracle_type in ("continuous_angle", "angle_cost_map"):
        if angle_deg is None:
            angle_deg = 0.0
        s, c = angle_to_sincos(angle_deg)
        x = np.zeros((1, 5, H, Wd), dtype=np.float32)
        x[0, 0] = occ.astype(np.float32)
        x[0, 1] = float(L) / float(resolution)
        x[0, 2] = float(W) / float(resolution)
        x[0, 3] = s
        x[0, 4] = c
    else:
        x = np.zeros((1, 3, H, Wd), dtype=np.float32)
        x[0, 0] = occ.astype(np.float32)
        x[0, 1] = float(L) / float(resolution)
        x[0, 2] = float(W) / float(resolution)
    return torch.from_numpy(x)


def _drop_obstacle(
    occ: np.ndarray, rng: np.random.Generator,
    min_size: int = 12, max_size: int = 36,
) -> np.ndarray:
    """
    Drop a random rectangular obstacle in a free area of the map.

    Args:
        occ:      Current occupancy grid (1 = free, 0 = obstacle).
        rng:      numpy Generator.
        min_size: Min side length of the dropped obstacle.
        max_size: Max side length.

    Returns:
        New occupancy grid (modified in-place AND returned).
    """
    H, Wd = occ.shape
    free_ys, free_xs = np.where(occ == 1)
    if len(free_ys) == 0:
        return occ
    idx = int(rng.integers(0, len(free_ys)))
    y0, x0 = int(free_ys[idx]), int(free_xs[idx])
    h = int(rng.integers(min_size, max_size + 1))
    w = int(rng.integers(min_size, max_size + 1))
    y1 = min(H, y0 + h)
    x1 = min(Wd, x0 + w)
    occ[y0:y1, x0:x1] = 0
    return occ


def _greedy_step(
    pos: Tuple[int, int], goal: Tuple[int, int],
    via4: np.ndarray, occ: np.ndarray,
    threshold: float = 0.4,
) -> Optional[Tuple[int, int]]:
    """
    One-cell greedy step toward the goal that prefers high-viability cells.

    Cardinal moves only. Skips a step if every cardinal neighbour is
    either an obstacle or below the viability threshold for the
    direction of motion.

    Args:
        pos:        (y, x) current robot cell.
        goal:       (y, x) target cell.
        via4:       (4, H, W) viability map [N, S, E, W] in [0, 1].
        occ:        (H, W) occupancy.
        threshold:  Min viability to accept a move.

    Returns:
        Next (y, x) — same as pos if no move possible — or None if at goal.
    """
    y, x = pos
    if pos == goal:
        return None

    H, Wd = occ.shape
    candidates = []
    # (dy, dx, direction-index)
    for dy, dx, d_idx in [(-1, 0, 0), (1, 0, 1), (0, 1, 2), (0, -1, 3)]:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < H and 0 <= nx < Wd):
            continue
        if occ[ny, nx] == 0:
            continue
        v = float(via4[d_idx, ny, nx])
        if v < threshold:
            continue
        manhattan = abs(goal[0] - ny) + abs(goal[1] - nx)
        # Prefer high viability AND short distance to goal
        score = v - 0.001 * manhattan
        candidates.append((score, ny, nx))
    if not candidates:
        return pos  # stuck — caller should react
    candidates.sort(reverse=True)
    return (candidates[0][1], candidates[0][2])


def run_closed_loop_demo(args: argparse.Namespace, config: dict) -> int:
    """
    Run the closed-loop trap-aware planning demo (Direction 2).

    Loads a checkpoint, picks (or accepts) a base map, and animates a
    robot stepping toward a goal while obstacles are randomly dropped
    every `--demo-change-every` frames. The model is re-queried at each
    such change and a low-viability "trap" overlay is shown.

    Args:
        args:   Parsed CLI args.
        config: Loaded YAML config.

    Returns:
        Exit code (0 on success).
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if not args.checkpoint:
        logger.error("--demo_closed_loop requires --checkpoint <path>")
        return 2
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 2

    device = args.device or get_device()
    logger.info("Demo device: %s", device)

    # ---- Load model ----
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    ckpt_oracle_type = ckpt.get("oracle_type")

    # Resolve --oracle_type auto from checkpoint metadata
    if args.oracle_type == "auto":
        if ckpt_oracle_type in ORACLE_TYPES:
            args.oracle_type = ckpt_oracle_type
            logger.info("--oracle_type auto resolved to '%s' from checkpoint.",
                        args.oracle_type)
        else:
            logger.error(
                "--oracle_type auto requested but checkpoint has no usable "
                "'oracle_type' metadata. Pass --oracle_type explicitly.")
            return 2

    # Sanity-check user-supplied oracle_type against checkpoint metadata
    if (ckpt_oracle_type is not None
            and ckpt_oracle_type in ORACLE_TYPES
            and ckpt_oracle_type != args.oracle_type):
        logger.warning(
            "Checkpoint oracle_type='%s' differs from CLI --oracle_type='%s'. "
            "Using CLI value, but weights may not load correctly.",
            ckpt_oracle_type, args.oracle_type)

    if cfg is None:
        # Try to infer from oracle_type
        fallback_io = {
            "basic":            {"in_channels": 3, "classes": 4},
            "continuous_angle": {"in_channels": 5, "classes": 1},
            "cost_map":         {"in_channels": 3, "classes": 4},
            "angle_cost_map":   {"in_channels": 5, "classes": 1},
            "velocity":         {"in_channels": 4, "classes": 4},
            "velocity_cost":    {"in_channels": 4, "classes": 4},
        }
        cfg = fallback_io.get(args.oracle_type)
        if cfg is None:
            logger.error("Cannot infer model shape for oracle_type='%s'.",
                         args.oracle_type)
            return 2
        logger.warning("No 'config' in checkpoint; using fallback %s", cfg)
    model = MultiRobotViabilityUNet(**cfg).to(device)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    try:
        model.load_state_dict(ckpt[sd_key])
    except RuntimeError as e:
        logger.error(
            "Failed to load weights — channel mismatch likely. "
            "Built model with in=%d out=%d for oracle_type='%s'. Error: %s",
            cfg["in_channels"], cfg["classes"], args.oracle_type, e)
        return 2
    model.eval()
    logger.info("Loaded model from %s  (in=%d out=%d, mode=%s)",
                ckpt_path, cfg["in_channels"], cfg["classes"], args.oracle_type)

    # ---- Choose base map ----
    resolution = int(ckpt.get("resolution", config["data"].get("resolution", 512)))
    if args.demo_map and Path(args.demo_map).exists():
        occ = np.load(args.demo_map).astype(np.uint8)
    else:
        proc_dir = Path(config["paths"]["processed_maps"])
        candidates = sorted(proc_dir.glob("*.npy"))
        if not candidates:
            logger.error("No .npy maps found in %s", proc_dir)
            return 2
        # Deterministic pick if seed is set
        rng_pick = np.random.default_rng(config["training"].get("seed", 42))
        occ = np.load(candidates[int(rng_pick.integers(0, len(candidates)))]).astype(np.uint8)
    if occ.shape != (resolution, resolution):
        logger.warning("Map shape %s ≠ resolution %d — using as-is", occ.shape, resolution)
    H, Wd = occ.shape
    logger.info("Demo map shape=%s  free=%.1f%%", occ.shape, occ.mean() * 100)

    # ---- Robot size ----
    train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]
    L, W = train_sizes[len(train_sizes) // 2]
    logger.info("Demo robot size: %dx%d", L, W)

    # ---- Pick start and goal in free space ----
    rng = np.random.default_rng(config["training"].get("seed", 42))
    free_ys, free_xs = np.where(occ == 1)
    if len(free_ys) < 2:
        logger.error("Map has fewer than 2 free cells — can't run demo.")
        return 2

    def _pick_distant_pair():
        # Pick pair that are far apart in Manhattan distance
        best_pair, best_d = None, -1
        for _ in range(100):
            i, j = rng.integers(0, len(free_ys), size=2)
            p1 = (int(free_ys[i]), int(free_xs[i]))
            p2 = (int(free_ys[j]), int(free_xs[j]))
            d = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            if d > best_d:
                best_d, best_pair = d, (p1, p2)
        return best_pair

    start, goal = _pick_distant_pair()
    logger.info("Start=%s  Goal=%s", start, goal)

    # ---- Helpers ----
    def _query_via4(curr_occ: np.ndarray) -> np.ndarray:
        """Run the model and return a 4-channel viability map.

        - basic mode  : sigmoid(logits)
        - continuous_angle : query 4 cardinals (0/90/180/270°), stack
        - cost_map    : convert cost → "softness" via 1 - cost_norm
        - angle_cost_map : query 4 cardinals, stack as cost-soft, then 1-cost
        """
        x_inp = _model_inputs_for_demo(curr_occ, L, W, resolution, args.oracle_type)
        x_inp = x_inp.to(device)
        with torch.no_grad():
            if args.oracle_type == "basic":
                probs = torch.sigmoid(model(x_inp))[0].cpu().numpy()  # (4,H,W)
                return probs.astype(np.float32)
            if args.oracle_type == "continuous_angle":
                cardinal_angles = [90.0, 270.0, 0.0, 180.0]  # N, S, E, W
                stacks = []
                for ang in cardinal_angles:
                    xa = _model_inputs_for_demo(curr_occ, L, W, resolution,
                                                args.oracle_type, ang).to(device)
                    p = torch.sigmoid(model(xa))[0, 0].cpu().numpy()
                    stacks.append(p.astype(np.float32))
                return np.stack(stacks, axis=0)
            if args.oracle_type == "cost_map":
                pred = torch.clamp(model(x_inp), 0.0, 1.0)[0].cpu().numpy()  # (4,H,W)
                return (1.0 - pred).astype(np.float32)
            # angle_cost_map
            cardinal_angles = [90.0, 270.0, 0.0, 180.0]
            stacks = []
            for ang in cardinal_angles:
                xa = _model_inputs_for_demo(curr_occ, L, W, resolution,
                                            args.oracle_type, ang).to(device)
                p = torch.clamp(model(xa), 0.0, 1.0)[0, 0].cpu().numpy()
                stacks.append(1.0 - p.astype(np.float32))
            return np.stack(stacks, axis=0)

    # ---- Animation loop ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    frame_state = {
        "occ": occ.copy(),
        "via4": _query_via4(occ),
        "pos": start,
        "trail": [start],
        "frame_idx": 0,
        "stuck_count": 0,
    }

    def _update(frame_i: int):
        s = frame_state
        s["frame_idx"] = frame_i

        # Periodically modify environment + re-query model
        if frame_i > 0 and frame_i % args.demo_change_every == 0:
            s["occ"] = _drop_obstacle(s["occ"], rng)
            s["via4"] = _query_via4(s["occ"])
            logger.info("[frame %d] Environment changed, model re-queried.", frame_i)

        # Step the robot greedily toward goal using current viability map
        nxt = _greedy_step(s["pos"], goal, s["via4"], s["occ"], threshold=0.4)
        if nxt is None:
            # Reached goal — reset to start so we can keep animating
            logger.info("[frame %d] GOAL reached; resetting.", frame_i)
            s["pos"] = start
            s["trail"] = [start]
            s["stuck_count"] = 0
        elif nxt == s["pos"]:
            s["stuck_count"] += 1
            if s["stuck_count"] > 5:
                # Give up and re-route by perturbing position
                logger.info("[frame %d] STUCK — re-querying model.", frame_i)
                s["via4"] = _query_via4(s["occ"])
                s["stuck_count"] = 0
        else:
            s["pos"] = nxt
            s["trail"].append(nxt)
            s["stuck_count"] = 0

        # ---- Render ----
        ax[0].clear()
        ax[1].clear()
        ax[0].imshow(s["occ"], cmap="gray", vmin=0, vmax=1)
        if s["trail"]:
            ts = np.array(s["trail"])
            ax[0].plot(ts[:, 1], ts[:, 0], "-", color="cyan", linewidth=1.5)
        ax[0].plot(start[1], start[0], "go", markersize=10, label="start")
        ax[0].plot(goal[1], goal[0], "r*", markersize=15, label="goal")
        ax[0].plot(s["pos"][1], s["pos"][0], "yo", markersize=8, label="robot")
        ax[0].set_title(f"frame {frame_i} — Closed-loop trap-aware demo")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].axis("off")

        # Trap overlay: pixels where MIN viability across 4 dirs is low.
        min_via = s["via4"].min(axis=0)
        traps = (min_via < 0.4) & (s["occ"] == 1)
        overlay = np.dstack([s["occ"]] * 3).astype(np.float32)
        overlay[traps] = [1.0, 0.4, 0.4]   # pink for traps
        ax[1].imshow(overlay)
        ax[1].plot(s["pos"][1], s["pos"][0], "yo", markersize=8)
        ax[1].set_title(f"min-viability traps (red) | mean min-via={float(min_via[s['occ']==1].mean()):.2f}")
        ax[1].axis("off")

        return []

    anim = animation.FuncAnimation(
        fig, _update, frames=args.demo_frames, interval=120, blit=False,
    )

    # Save the animation
    out_path = Path(args.demo_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if str(out_path).lower().endswith(".gif"):
            anim.save(str(out_path), writer="pillow", fps=8)
        elif str(out_path).lower().endswith((".mp4", ".mov")):
            anim.save(str(out_path), writer="ffmpeg", fps=8)
        else:
            # Fallback to GIF
            out_path = out_path.with_suffix(".gif")
            anim.save(str(out_path), writer="pillow", fps=8)
        logger.info("Demo saved to %s", out_path)
    except Exception as e:
        logger.warning("Could not save animation (%s). Saving last frame instead.", e)
        # Fall back to a static figure
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        logger.info("Saved last frame to %s", png_path)

    plt.close(fig)
    return 0


# ===========================================================================
# Generalised training driver (handles all 4 oracle_types)
# ===========================================================================

def _make_basic_loaders(
    config: dict, args: argparse.Namespace, train_sizes: List[Tuple[int, int]],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Original Phase-1 loaders — preserved verbatim."""
    map_dir = config["paths"]["processed_maps"]
    label_base_dir = config["paths"]["labels_dir"]
    manifest_path = config["paths"]["manifest"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 8)
    resolution = config["data"].get("resolution", 512)

    return create_dataloaders(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        manifest_path=manifest_path,
        robot_sizes=train_sizes,
        batch_size=batch_size,
        num_workers=num_workers,
        resolution=resolution,
        robot_sampling_mode="random",
    )


def _patched_validate(trainer: Trainer, oracle_type: str) -> Dict:
    """Run validation with the right metric tracker for this mode.

    The basic-mode Trainer.validate() hard-codes MetricTracker; for
    Phase-2 modes we monkey-patch the tracker so the existing loop
    continues to work without forking the trainer class.
    """
    from torch.amp import autocast
    from tqdm import tqdm

    trainer.model.eval()
    total_loss = 0.0
    num_batches = 0
    tracker = _make_metric_tracker(oracle_type)

    use_amp = getattr(trainer, "use_amp", False)

    with torch.no_grad():
        for batch in tqdm(trainer.val_loader, desc="Validating", leave=False):
            inputs, labels, _meta = batch
            inputs = inputs.to(trainer.device)
            labels = labels.to(trainer.device)

            if use_amp:
                with autocast("cuda"):
                    outputs = trainer.model(inputs)
                    loss = trainer.criterion(outputs, labels)
            else:
                outputs = trainer.model(inputs)
                loss = trainer.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1
            tracker.update(outputs, labels)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = tracker.compute()
    return {
        "val_loss": avg_loss,
        "val_iou": metrics.get("iou", 0.0),
        "val_dice": metrics.get("dice", 0.0),
        "val_accuracy": metrics.get("accuracy", 0.0),
        **{f"val_{k}": v for k, v in metrics.items() if k.startswith("iou_")},
    }


def run_training(args: argparse.Namespace, config: dict, output_dir: Path) -> int:
    """Phase-1 + Phase-2 training driver."""
    # Random seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    # Device
    device = args.device or get_device()
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    if args.debug:
        logger.warning("DEBUG MODE — reduced data")
        config["data"]["num_maps"] = 100
        config["training"]["num_epochs"] = min(5, config["training"]["num_epochs"])

    train_sizes = [tuple(s) for s in config["robot_sizes"]["train"]]
    test_sizes = [tuple(s) for s in config["robot_sizes"]["test_only"]]
    logger.info("Train sizes : %s", train_sizes)
    logger.info("Test sizes  : %s", test_sizes)

    # ---- Loaders ----
    if args.oracle_type == "basic":
        train_loader, val_loader, test_loader = _make_basic_loaders(config, args, train_sizes)
    else:
        train_loader, val_loader, test_loader = _make_extended_loaders(config, args, train_sizes)
    logger.info("Train batches=%d  Val batches=%d  Test batches=%d",
                len(train_loader), len(val_loader), len(test_loader))

    # ---- Model ----
    in_channels, out_channels = _model_io_for_mode(args.oracle_type)
    model_cfg = config["model"]
    model = MultiRobotViabilityUNet(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=in_channels,
        classes=out_channels,
    )
    print_model_summary(model, input_size=(1, in_channels, 512, 512))
    model = model.to(device)

    # ---- Loss ----
    criterion = _build_loss_for_mode(args.oracle_type, config["training"])
    logger.info("Loss: %s", type(criterion).__name__)

    # ---- Optimiser / scheduler ----
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config["training"].get("optimizer", "adamw"),
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler_cfg = config["training"].get("scheduler", {}) or {}
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_cfg.get("type", "cosine"),
        T_max=config["training"]["num_epochs"],
        eta_min=scheduler_cfg.get("min_lr", 1e-6),
    )

    # ---- Trainer ----
    trainer_config = {
        "use_amp": config["training"].get("use_amp", True),
        "gradient_clip_val": config["training"].get("gradient_clip", 1.0),
        "early_stopping_patience": config["training"].get("early_stopping_patience", 15),
        "num_epochs": config["training"]["num_epochs"],
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        device=device,
        output_dir=str(output_dir),
    )

    # Patch validate() for Phase-2 modes so the right metric tracker is used.
    if args.oracle_type != "basic":
        original_validate = trainer.validate
        def _wrapped():
            return _patched_validate(trainer, args.oracle_type)
        trainer.validate = _wrapped  # type: ignore[assignment]

    # Resume?
    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # Save extra metadata into checkpoints (resolution, oracle_type, etc.)
    # by hooking into save_checkpoint
    original_save = trainer.save_checkpoint
    def _save_with_meta(filename: str):
        result = original_save(filename)
        # Edit the just-written checkpoint to inject metadata
        try:
            ckpt_path = trainer.checkpoint_dir / filename
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ckpt.setdefault("config", {
                "encoder_name": model_cfg.get("encoder_name", "resnet34"),
                "encoder_weights": model_cfg.get("encoder_weights", "imagenet"),
                "in_channels": in_channels,
                "classes": out_channels,
            })
            ckpt["oracle_type"] = args.oracle_type
            ckpt["resolution"] = config["data"].get("resolution", 512)
            torch.save(ckpt, str(ckpt_path))
        except Exception as e:
            logger.warning("Could not augment checkpoint metadata: %s", e)
        return result

    trainer.save_checkpoint = _save_with_meta  # type: ignore[assignment]

    # ---- Train ----
    logger.info("=" * 60)
    logger.info("STARTING TRAINING (oracle_type=%s)", args.oracle_type)
    logger.info("=" * 60)
    history = trainer.train(num_epochs=config["training"]["num_epochs"])

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Best val IoU/score: %.4f", trainer.best_val_iou)
    logger.info("Best val loss:      %.4f", trainer.best_val_loss)

    results = {
        "oracle_type": args.oracle_type,
        "best_val_iou": float(trainer.best_val_iou),
        "best_val_loss": float(trainer.best_val_loss),
        "final_epoch": int(trainer.epoch),
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("Checkpoints in:   %s", output_dir / "checkpoints")
    logger.info("TensorBoard logs: %s", output_dir / "tensorboard")
    return 0


def _model_io_for_mode(oracle_type: str) -> Tuple[int, int]:
    """Map oracle_type → (in_channels, out_channels)."""
    if oracle_type == "basic":
        return 3, 4
    if oracle_type == "continuous_angle":
        return 5, 1
    if oracle_type == "cost_map":
        return 3, 4
    if oracle_type == "angle_cost_map":
        return 5, 1
    if oracle_type in ("velocity", "velocity_cost"):
        return 4, 4
    raise ValueError(f"Unknown oracle_type: {oracle_type}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    # Apply overrides
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.seed:
        config["training"]["seed"] = args.seed

    # Closed-loop demo path
    if args.demo_closed_loop:
        return run_closed_loop_demo(args, config)

    # Standard training path: 'auto' is only valid for the demo
    if args.oracle_type == "auto":
        logger.error("--oracle_type auto is only valid with --demo_closed_loop. "
                     "For training, pass one of: %s", list(ORACLE_TYPES))
        return 2

    # Standard training path
    exp_config = setup_experiment(config, args)
    output_dir = exp_config["output_dir"]
    logger.info("=" * 60)
    logger.info("DIRECTIONAL TOPOLOGICAL TRAPS - TRAINING")
    logger.info("=" * 60)
    logger.info("Experiment   : %s", exp_config["exp_name"])
    logger.info("Oracle type  : %s", args.oracle_type)
    logger.info("Output dir   : %s", output_dir)

    return run_training(args, config, output_dir)


if __name__ == "__main__":
    sys.exit(main())