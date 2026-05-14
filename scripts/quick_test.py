#!/usr/bin/env python3
"""
Quick Test Script - End-to-End Pipeline Validation.

This script runs a quick end-to-end test of the entire pipeline:
1. Creates synthetic test maps
2. Generates viability labels
3. Creates a small dataset
4. Trains for a few epochs
5. Evaluates the model

Use this to verify the pipeline works before running on full data.

Usage:
    python scripts/quick_test.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from tqdm import tqdm


def create_synthetic_maps(output_dir: str, num_maps: int = 20):
    """Create synthetic occupancy maps for testing."""
    print("\n=== Creating Synthetic Maps ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_maps):
        # Create random room-like map
        occ = np.ones((128, 128), dtype=np.uint8)
        
        # Add border walls
        wall_width = np.random.randint(5, 15)
        occ[:wall_width, :] = 0
        occ[-wall_width:, :] = 0
        occ[:, :wall_width] = 0
        occ[:, -wall_width:] = 0
        
        # Add some internal obstacles
        num_obstacles = np.random.randint(2, 6)
        for _ in range(num_obstacles):
            x = np.random.randint(20, 100)
            y = np.random.randint(20, 100)
            w = np.random.randint(5, 20)
            h = np.random.randint(5, 20)
            occ[y:y+h, x:x+w] = 0
        
        np.save(output_dir / f"map_{i:04d}.npy", occ)
    
    print(f"  Created {num_maps} synthetic maps in {output_dir}")
    return True


def generate_test_labels(map_dir: str, label_dir: str, robot_sizes: list):
    """Generate viability labels for test maps."""
    print("\n=== Generating Labels ===")
    
    from src.oracle.directional_viability import generate_labels_for_map
    
    map_dir = Path(map_dir)
    label_dir = Path(label_dir)
    
    map_files = sorted(map_dir.glob("*.npy"))
    
    for length, width in robot_sizes:
        size_dir = label_dir / f"robot_{length}x{width}"
        size_dir.mkdir(parents=True, exist_ok=True)
        
        for map_file in tqdm(map_files, desc=f"Labels {length}x{width}"):
            occ = np.load(map_file)
            labels = generate_labels_for_map(occ, length, width)
            np.save(size_dir / map_file.name, labels)
    
    print(f"  Generated labels for {len(robot_sizes)} robot sizes")
    return True


def create_test_manifest(map_dir: str, manifest_path: str):
    """Create train/val/test manifest."""
    print("\n=== Creating Manifest ===")
    
    import pandas as pd
    
    map_dir = Path(map_dir)
    map_files = sorted([f.name for f in map_dir.glob("*.npy")])
    
    # Simple split: 70/15/15
    n = len(map_files)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    splits = ['train'] * train_end + ['val'] * (val_end - train_end) + ['test'] * (n - val_end)
    
    manifest = pd.DataFrame({
        'filename': map_files,
        'split': splits
    })
    
    manifest.to_csv(manifest_path, index=False)
    print(f"  Created manifest with {n} files")
    return True


def run_quick_training(temp_dir: str, robot_sizes: list, num_epochs: int = 3):
    """Run quick training test."""
    print("\n=== Running Quick Training ===")
    
    from src.data.dataset import MultiRobotViabilityDataset
    from src.models.unet import MultiRobotViabilityUNet
    from src.models.losses import create_loss
    from src.models.metrics import compute_iou
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    temp_dir = Path(temp_dir)
    
    # Create dataset
    dataset = MultiRobotViabilityDataset(
        map_dir=str(temp_dir / "maps"),
        label_base_dir=str(temp_dir / "labels"),
        robot_sizes=robot_sizes,
        file_list=None,  # Use all files
        robot_sampling="random",
        use_augmentation=True,
        resolution=128
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # Create model (smaller for testing)
    model = MultiRobotViabilityUNet(
        encoder_name="resnet18",  # Smaller encoder for speed
        encoder_weights=None,     # No pretrained weights for speed
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = create_loss("dice_bce")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    # Quick evaluation
    model.eval()
    total_iou = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            iou = compute_iou(outputs, labels)
            total_iou += iou.item() * inputs.size(0)
            num_samples += inputs.size(0)
    
    avg_iou = total_iou / num_samples
    print(f"\n  Final IoU: {avg_iou:.4f}")
    
    return avg_iou > 0.1  # Very low bar for quick test


def main():
    print("=" * 60)
    print("QUICK PIPELINE TEST")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="traps_test_")
    print(f"\nUsing temp directory: {temp_dir}")
    
    try:
        # Test robot sizes (small for speed)
        robot_sizes = [(4, 3), (6, 4)]
        
        # Create synthetic data
        create_synthetic_maps(f"{temp_dir}/maps", num_maps=20)
        
        # Generate labels
        generate_test_labels(
            f"{temp_dir}/maps",
            f"{temp_dir}/labels",
            robot_sizes
        )
        
        # Create manifest
        create_test_manifest(
            f"{temp_dir}/maps",
            f"{temp_dir}/manifest.csv"
        )
        
        # Run quick training
        success = run_quick_training(temp_dir, robot_sizes, num_epochs=3)
        
        print("\n" + "=" * 60)
        if success:
            print("✓ QUICK TEST PASSED")
            print("=" * 60)
            print("\nThe pipeline is working correctly.")
            print("You can now proceed with full training.")
        else:
            print("✗ QUICK TEST FAILED")
            print("=" * 60)
            print("\nPlease check the error messages above.")
        
        return success
        
    finally:
        # Cleanup
        print(f"\nCleaning up temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
