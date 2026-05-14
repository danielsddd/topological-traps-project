"""
src/experiments/velocity_experiment.py

Self-contained module for the Velocity-Dependent Viability experiment.
Contains the PyTorch Dataset class and all helpers needed by scripts/train.py.

Integration with train.py:
    1. Import this module.
    2. Add "velocity" and "velocity_cost" to ORACLE_TYPES.
    3. Wire into _model_io_for_mode, _build_loss_for_mode, _make_extended_loaders.

See the bottom of this file for the exact 5 edits to train.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.oracle.velocity_oracle import (
    DEFAULT_VELOCITIES,
    V_MAX,
    braking_distance_px,
    normalise_velocity,
    velocity_viability,
    velocity_escape_cost_map,
)
from src.oracle.extended_oracles import normalise_cost_map

logger = logging.getLogger(__name__)


# =========================================================================
# Dataset: Velocity-Dependent Binary Viability
# =========================================================================

class VelocityViabilityDataset(Dataset):
    """
    Dataset for --oracle_type velocity.

    Inputs : 4-channel  [occupancy, L_norm, W_norm, v_norm]
    Targets: 4-channel  uint8  binary viability [N, S, E, W]

    At each __getitem__:
      - A random robot size is sampled (training) or deterministic (val/test).
      - A random velocity is sampled from the configured velocity set.
      - The velocity Oracle computes the ground truth on-the-fly.

    The velocity channel is a constant spatial map: v / V_MAX broadcast
    to (H, W).  This is analogous to how L_norm and W_norm are encoded.
    """

    def __init__(
        self,
        map_dir: Union[str, Path],
        manifest_path: Optional[str],
        robot_sizes: List[Tuple[int, int]],
        split: str,
        resolution: int = 512,
        transform: Optional[Callable] = None,
        velocities: Optional[List[float]] = None,
        max_decel: float = 2.0,
        px_per_m: float = 10.0,
        num_velocities_per_map: int = 4,
    ):
        self.map_dir = Path(map_dir)
        self.robot_sizes = list(robot_sizes)
        self.split = split
        self.resolution = resolution
        self.transform = transform
        self.velocities = velocities or DEFAULT_VELOCITIES
        self.max_decel = max_decel
        self.px_per_m = px_per_m
        self.num_velocities_per_map = max(1, num_velocities_per_map)

        if manifest_path and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            split_files = df[df["split"] == split]["filename"].tolist()
        else:
            split_files = [f for f in os.listdir(map_dir) if f.endswith(".npy")]

        self.map_files: List[Path] = []
        for fn in sorted(split_files):
            p = self.map_dir / fn
            if p.exists():
                self.map_files.append(p)

        logger.info(
            "VelocityViabilityDataset[%s]: %d maps × %d robot sizes × %d velocities",
            split, len(self.map_files), len(self.robot_sizes), len(self.velocities),
        )

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.map_files) * self.num_velocities_per_map
        # Val/test: one sample per map per velocity → deterministic full coverage
        return len(self.map_files) * len(self.velocities)

    def _select_robot(self, idx: int) -> Tuple[int, int]:
        if self.split == "train":
            return self.robot_sizes[np.random.randint(0, len(self.robot_sizes))]
        return self.robot_sizes[idx % len(self.robot_sizes)]

    def _select_velocity(self, idx: int) -> float:
        if self.split == "train":
            return float(np.random.choice(self.velocities))
        # Val/test: deterministic — cycle through every velocity per map
        virtual = idx // len(self.map_files)
        return self.velocities[virtual % len(self.velocities)]

    def _load_map(self, idx: int) -> np.ndarray:
        return np.load(self.map_files[idx]).astype(np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        map_idx = idx % len(self.map_files)
        occ = self._load_map(map_idx)
        L, W = self._select_robot(map_idx)
        vel = self._select_velocity(idx)

        # On-the-fly oracle
        labels = velocity_viability(
            occ, L, W, vel,
            max_decel=self.max_decel,
            px_per_m=self.px_per_m,
        ).astype(np.float32)  # (4, H, W)

        if self.transform is not None:
            occ_f, labels = self.transform(
                occ.astype(np.float32), labels,
            )
            occ = (occ_f > 0.5).astype(np.uint8)

        H, Wd = occ.shape
        v_norm = normalise_velocity(vel)

        x = np.zeros((4, H, Wd), dtype=np.float32)
        x[0] = occ.astype(np.float32)
        x[1] = float(L) / float(self.resolution)
        x[2] = float(W) / float(self.resolution)
        x[3] = v_norm  # broadcast constant

        y = labels.astype(np.float32)  # (4, H, W)

        meta = {
            "robot_length": L,
            "robot_width": W,
            "velocity": vel,
            "velocity_norm": v_norm,
            "d_brake_px": braking_distance_px(vel, self.max_decel, self.px_per_m),
            "map_name": self.map_files[map_idx].stem,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


# =========================================================================
# Dataset: Velocity-Dependent Cost Map (combines Exp 1 + 2)
# =========================================================================

class VelocityCostMapDataset(Dataset):
    """
    Dataset for --oracle_type velocity_cost.

    Inputs : 4-channel  [occupancy, L_norm, W_norm, v_norm]
    Targets: 4-channel  float  escape cost [N, S, E, W]  (normalised to [0,1])

    Combines the velocity-dependent footprint (Exp 1) with continuous
    regression cost maps (Exp 2).
    """

    def __init__(
        self,
        map_dir: Union[str, Path],
        manifest_path: Optional[str],
        robot_sizes: List[Tuple[int, int]],
        split: str,
        resolution: int = 512,
        transform: Optional[Callable] = None,
        velocities: Optional[List[float]] = None,
        max_decel: float = 2.0,
        px_per_m: float = 10.0,
        num_velocities_per_map: int = 4,
    ):
        self.map_dir = Path(map_dir)
        self.robot_sizes = list(robot_sizes)
        self.split = split
        self.resolution = resolution
        self.transform = transform
        self.velocities = velocities or DEFAULT_VELOCITIES
        self.max_decel = max_decel
        self.px_per_m = px_per_m
        self.num_velocities_per_map = max(1, num_velocities_per_map)

        if manifest_path and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            split_files = df[df["split"] == split]["filename"].tolist()
        else:
            split_files = [f for f in os.listdir(map_dir) if f.endswith(".npy")]

        self.map_files: List[Path] = []
        for fn in sorted(split_files):
            p = self.map_dir / fn
            if p.exists():
                self.map_files.append(p)

        logger.info(
            "VelocityCostMapDataset[%s]: %d maps × %d robot sizes × %d velocities",
            split, len(self.map_files), len(self.robot_sizes), len(self.velocities),
        )

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.map_files) * self.num_velocities_per_map
        return len(self.map_files) * len(self.velocities)

    def _select_robot(self, idx: int) -> Tuple[int, int]:
        if self.split == "train":
            return self.robot_sizes[np.random.randint(0, len(self.robot_sizes))]
        return self.robot_sizes[idx % len(self.robot_sizes)]

    def _select_velocity(self, idx: int) -> float:
        if self.split == "train":
            return float(np.random.choice(self.velocities))
        virtual = idx // len(self.map_files)
        return self.velocities[virtual % len(self.velocities)]

    def _load_map(self, idx: int) -> np.ndarray:
        return np.load(self.map_files[idx]).astype(np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        map_idx = idx % len(self.map_files)
        occ = self._load_map(map_idx)
        L, W = self._select_robot(map_idx)
        vel = self._select_velocity(idx)

        cost4 = velocity_escape_cost_map(
            occ, L, W, vel,
            max_decel=self.max_decel,
            px_per_m=self.px_per_m,
        )  # (4, H, W) float32
        cost4_norm = normalise_cost_map(cost4)  # → [0, 1]

        if self.transform is not None:
            occ_f, cost4_norm = self.transform(
                occ.astype(np.float32), cost4_norm.astype(np.float32),
            )
            occ = (occ_f > 0.5).astype(np.uint8)

        H, Wd = occ.shape
        v_norm = normalise_velocity(vel)

        x = np.zeros((4, H, Wd), dtype=np.float32)
        x[0] = occ.astype(np.float32)
        x[1] = float(L) / float(self.resolution)
        x[2] = float(W) / float(self.resolution)
        x[3] = v_norm

        y = cost4_norm.astype(np.float32)  # (4, H, W)

        meta = {
            "robot_length": L,
            "robot_width": W,
            "velocity": vel,
            "velocity_norm": v_norm,
            "d_brake_px": braking_distance_px(vel, self.max_decel, self.px_per_m),
            "map_name": self.map_files[map_idx].stem,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


# =========================================================================
# Inference helpers  (used by evaluation scripts)
# =========================================================================

def predict_velocity_viability(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    velocity: float,
    resolution: int,
    device: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run velocity-aware inference.

    Returns:
        (4, H, W) uint8 binary prediction.
    """
    H, Wd = occ.shape
    v_norm = normalise_velocity(velocity)

    x = np.zeros((1, 4, H, Wd), dtype=np.float32)
    x[0, 0] = occ.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    x[0, 3] = v_norm

    with torch.no_grad():
        inp = torch.from_numpy(x).to(device)
        logits = model(inp)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (4, H, W)

    return (probs > threshold).astype(np.uint8)


def predict_velocity_cost_map(
    model: torch.nn.Module,
    occ: np.ndarray,
    L: int,
    W: int,
    velocity: float,
    resolution: int,
    device: str,
) -> np.ndarray:
    """
    Run velocity cost-map inference.

    Returns:
        (4, H, W) float32 normalised cost prediction in [0, 1].
    """
    H, Wd = occ.shape
    v_norm = normalise_velocity(velocity)

    x = np.zeros((1, 4, H, Wd), dtype=np.float32)
    x[0, 0] = occ.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    x[0, 3] = v_norm

    with torch.no_grad():
        inp = torch.from_numpy(x).to(device)
        out = model(inp).cpu().numpy()[0]  # (4, H, W)

    return np.clip(out, 0.0, 1.0).astype(np.float32)