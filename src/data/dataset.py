"""
PyTorch Dataset for Multi-Robot Viability Prediction.

This module provides the dataset class that loads occupancy grids and
viability labels, constructing 3-channel inputs for the neural network.

Input Format (3 channels):
    Channel 0: Occupancy grid (binary, 0.0 or 1.0)
    Channel 1: Normalized robot length (constant, length/512)
    Channel 2: Normalized robot width (constant, width/512)

Label Format (4 channels):
    Channel 0: North viability (binary)
    Channel 1: South viability (binary)
    Channel 2: East viability (binary)
    Channel 3: West viability (binary)

IMPORTANT: Robot dimensions are normalized by MAP SIZE (512), not by max robot size.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Union, Callable
from pathlib import Path
import random


class MultiRobotViabilityDataset(Dataset):
    """
    PyTorch Dataset for multi-robot viability prediction.
    
    Supports multiple robot sizes with random or fixed size sampling.
    Each sample returns:
        - input: (3, H, W) tensor [occupancy, norm_length, norm_width]
        - labels: (4, H, W) tensor [N, S, E, W viability]
        - metadata: dict with robot dimensions and map name
    
    The key insight is that robot dimensions are encoded as constant
    feature maps (channels 1 and 2), allowing the network to learn
    size-dependent viability predictions.
    """
    
    def __init__(
        self,
        map_dir: Union[str, Path],
        label_base_dir: Union[str, Path],
        robot_sizes: List[Tuple[int, int]],
        resolution: int = 512,
        split: str = "train",
        manifest_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        robot_sampling: str = "random"
    ):
        """
        Initialize the dataset.
        
        Args:
            map_dir: Directory containing .npy occupancy grids
            label_base_dir: Base directory for labels (contains robot_LxW subdirs)
            robot_sizes: List of (length, width) tuples to sample from
            resolution: Image resolution for normalization (default 512)
            split: Data split - "train", "val", or "test"
            manifest_path: Path to CSV file with split assignments
            transform: Optional augmentation function
            robot_sampling: "random" = random size per sample,
                           "fixed" = always use first size,
                           "sequential" = cycle through sizes
        """
        self.map_dir = Path(map_dir)
        self.label_base_dir = Path(label_base_dir)
        self.robot_sizes = robot_sizes
        self.resolution = resolution
        self.split = split
        self.transform = transform
        self.robot_sampling = robot_sampling
        
        # Load manifest and filter by split
        if manifest_path and os.path.exists(manifest_path):
            manifest = pd.read_csv(manifest_path)
            split_files = manifest[manifest["split"] == split]["filename"].tolist()
        else:
            # No manifest - use all files
            split_files = [f for f in os.listdir(map_dir) if f.endswith(".npy")]
        
        # Build list of valid map files
        self.map_files = []
        missing_maps = []
        
        for filename in sorted(split_files):
            path = self.map_dir / filename
            if path.exists():
                self.map_files.append(path)
            else:
                missing_maps.append(filename)
        
        if missing_maps and len(missing_maps) <= 10:
            print(f"Warning: {len(missing_maps)} map files not found")
        
        print(f"Dataset [{split}]: {len(self.map_files)} maps, "
              f"{len(robot_sizes)} robot sizes")
        
        # Validate label directories exist
        self._validate_label_dirs()
        
        # For sequential sampling
        self._current_size_idx = 0
    
    def _validate_label_dirs(self) -> None:
        """Validate that all required label directories exist."""
        missing_dirs = []
        
        for length, width in self.robot_sizes:
            label_dir = self.label_base_dir / f"robot_{length}x{width}"
            if not label_dir.is_dir():
                missing_dirs.append(str(label_dir))
        
        if missing_dirs:
            from configs.config_schema import DataError
            raise DataError(
                f"Missing label directories:\n" + "\n".join(missing_dirs[:5])
            )
    
    def __len__(self) -> int:
        """Return number of samples (= number of maps)."""
        return len(self.map_files)
    
    def _select_robot_size(self, idx: int) -> Tuple[int, int]:
        """
        Select robot size based on sampling strategy.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (length, width)
        """
        if self.robot_sampling == "random":
            return random.choice(self.robot_sizes)
        elif self.robot_sampling == "fixed":
            return self.robot_sizes[0]
        elif self.robot_sampling == "sequential":
            size = self.robot_sizes[self._current_size_idx]
            self._current_size_idx = (self._current_size_idx + 1) % len(self.robot_sizes)
            return size
        else:
            raise ValueError(f"Unknown robot_sampling: {self.robot_sampling}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (input_tensor, label_tensor, metadata)
            - input_tensor: (3, H, W) float32
            - label_tensor: (4, H, W) float32
            - metadata: dict with 'robot_length', 'robot_width', 'map_name'
        """
        map_path = self.map_files[idx]
        robot_length, robot_width = self._select_robot_size(idx)
        
        # Load occupancy grid
        occupancy = np.load(map_path).astype(np.float32)
        
        # Load labels for this robot size
        label_dir = self.label_base_dir / f"robot_{robot_length}x{robot_width}"
        label_path = label_dir / map_path.name
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        labels = np.load(label_path).astype(np.float32)
        
        # Apply augmentation (handles label permutation)
        if self.transform is not None:
            occupancy, labels = self.transform(occupancy, labels)
        
        # Construct 3-channel input
        # CRITICAL: Normalize by MAP SIZE (512), not by robot size
        H, W = occupancy.shape
        input_tensor = np.zeros((3, H, W), dtype=np.float32)
        input_tensor[0] = occupancy                           # Occupancy grid
        input_tensor[1] = robot_length / self.resolution      # Normalized length
        input_tensor[2] = robot_width / self.resolution       # Normalized width
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_tensor)
        label_tensor = torch.from_numpy(labels)
        
        # Metadata
        metadata = {
            "robot_length": robot_length,
            "robot_width": robot_width,
            "map_name": map_path.stem,
            "map_path": str(map_path),
        }
        
        return input_tensor, label_tensor, metadata
    
    def get_sample_for_robot_size(
        self,
        idx: int,
        robot_length: int,
        robot_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample with a specific robot size.
        
        Useful for evaluation when you want to test specific sizes.
        
        Args:
            idx: Map index
            robot_length: Robot length in pixels
            robot_width: Robot width in pixels
        
        Returns:
            Tuple of (input_tensor, label_tensor, metadata)
        """
        map_path = self.map_files[idx]
        
        # Load occupancy grid
        occupancy = np.load(map_path).astype(np.float32)
        
        # Load labels for specified robot size
        label_dir = self.label_base_dir / f"robot_{robot_length}x{robot_width}"
        label_path = label_dir / map_path.name
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        labels = np.load(label_path).astype(np.float32)
        
        # No augmentation for specific size queries
        
        # Construct input
        H, W = occupancy.shape
        input_tensor = np.zeros((3, H, W), dtype=np.float32)
        input_tensor[0] = occupancy
        input_tensor[1] = robot_length / self.resolution
        input_tensor[2] = robot_width / self.resolution
        
        input_tensor = torch.from_numpy(input_tensor)
        label_tensor = torch.from_numpy(labels)
        
        metadata = {
            "robot_length": robot_length,
            "robot_width": robot_width,
            "map_name": map_path.stem,
            "map_path": str(map_path),
        }
        
        return input_tensor, label_tensor, metadata


class RobotSpecificDataset(Dataset):
    """
    Dataset for a single robot size.
    
    Useful for per-robot-size evaluation or when you want to train
    on a single size only.
    """
    
    def __init__(
        self,
        map_dir: Union[str, Path],
        label_dir: Union[str, Path],
        robot_length: int,
        robot_width: int,
        resolution: int = 512,
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize single-robot dataset.
        
        Args:
            map_dir: Directory with occupancy grids
            label_dir: Directory with labels for this robot size
            robot_length: Robot length
            robot_width: Robot width
            resolution: Image resolution for normalization
            file_list: List of filenames to use (None = all)
            transform: Optional augmentation
        """
        self.map_dir = Path(map_dir)
        self.label_dir = Path(label_dir)
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.resolution = resolution
        self.transform = transform
        
        # Get file list
        if file_list is not None:
            self.files = [f for f in file_list if (self.map_dir / f).exists()]
        else:
            self.files = sorted([
                f.name for f in self.map_dir.glob("*.npy")
                if (self.label_dir / f.name).exists()
            ])
        
        print(f"RobotSpecificDataset: {len(self.files)} files, "
              f"robot {robot_length}x{robot_width}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        filename = self.files[idx]
        
        # Load data
        occupancy = np.load(self.map_dir / filename).astype(np.float32)
        labels = np.load(self.label_dir / filename).astype(np.float32)
        
        # Apply augmentation
        if self.transform is not None:
            occupancy, labels = self.transform(occupancy, labels)
        
        # Construct input
        H, W = occupancy.shape
        input_tensor = np.zeros((3, H, W), dtype=np.float32)
        input_tensor[0] = occupancy
        input_tensor[1] = self.robot_length / self.resolution
        input_tensor[2] = self.robot_width / self.resolution
        
        input_tensor = torch.from_numpy(input_tensor)
        label_tensor = torch.from_numpy(labels)
        
        metadata = {
            "robot_length": self.robot_length,
            "robot_width": self.robot_width,
            "map_name": Path(filename).stem,
        }
        
        return input_tensor, label_tensor, metadata


def create_dataloaders(
    map_dir: str,
    label_base_dir: str,
    robot_sizes: List[Tuple[int, int]],
    manifest_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    resolution: int = 512,
    robot_sampling_mode: str = "random",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        map_dir: Directory with occupancy grids
        label_base_dir: Base directory for labels
        robot_sizes: List of robot sizes for training
        manifest_path: Path to split manifest CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        train_transform: Augmentation for training
        val_transform: Augmentation for validation/test
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultiRobotViabilityDataset(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        robot_sizes=robot_sizes,
        resolution=resolution,
        split="train",
        manifest_path=manifest_path,
        transform=train_transform,
        robot_sampling=robot_sampling_mode,
    )
    
    val_dataset = MultiRobotViabilityDataset(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        robot_sizes=robot_sizes,
        resolution=resolution,
        split="val",
        manifest_path=manifest_path,
        transform=val_transform,
        robot_sampling="sequential",
    )

    test_dataset = MultiRobotViabilityDataset(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        robot_sizes=robot_sizes,
        resolution=resolution,
        split="test",
        manifest_path=manifest_path,
        transform=val_transform,
        robot_sampling="sequential",
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function that preserves metadata as list.
    
    Args:
        batch: List of (input, label, metadata) tuples
    
    Returns:
        Tuple of (batched_inputs, batched_labels, metadata_list)
    """
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    
    return inputs, labels, metadata


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()
    
    # Test with default robot sizes
    robot_sizes = [(6, 4), (14, 9), (22, 14)]
    
    dataset = MultiRobotViabilityDataset(
        map_dir=args.map_dir,
        label_base_dir=args.label_dir,
        robot_sizes=robot_sizes,
        split="train",
        manifest_path=args.manifest,
        robot_sampling="random"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        input_tensor, label_tensor, metadata = dataset[0]
        print(f"Input shape: {input_tensor.shape}")
        print(f"Label shape: {label_tensor.shape}")
        print(f"Metadata: {metadata}")
        print(f"Input channel 1 (robot length): {input_tensor[1, 0, 0].item():.4f}")
        print(f"Input channel 2 (robot width): {input_tensor[2, 0, 0].item():.4f}")
