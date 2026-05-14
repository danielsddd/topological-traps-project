"""
Manifest Management - Train/Validation/Test Split Management.

This module handles creating and loading manifest files that track
which maps belong to which split. Maps are split once and the same
split is used across all robot sizes.

Manifest Format (CSV):
    filename,split
    map_0001.npy,train
    map_0002.npy,val
    map_0003.npy,test
    ...
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random


def create_manifest(
    processed_dir: str,
    output_path: str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create train/val/test split manifest for processed maps.
    
    The manifest is for MAPS only - the same map is always in the same split
    regardless of robot size. This ensures that:
    - The model never sees test maps during training
    - Evaluation is fair across robot sizes
    
    Args:
        processed_dir: Directory containing .npy occupancy files
        output_path: Path to save manifest CSV
        train_split: Fraction for training (default 0.70)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
        verbose: Print summary information
    
    Returns:
        DataFrame with columns ['filename', 'split']
    """
    # Validate split ratios
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all processed files
    processed_dir = Path(processed_dir)
    files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".npy")])
    n_files = len(files)
    
    if n_files == 0:
        raise ValueError(f"No .npy files found in {processed_dir}")
    
    if verbose:
        print(f"Creating manifest for {n_files} files")
    
    # Shuffle indices
    indices = np.random.permutation(n_files)
    
    # Compute split boundaries
    n_train = int(n_files * train_split)
    n_val = int(n_files * val_split)
    # n_test = n_files - n_train - n_val  # Remainder goes to test
    
    # Assign splits
    splits = []
    for i, idx in enumerate(indices):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        splits.append(split)
    
    # Reorder to match original file order
    file_splits = [""] * n_files
    for i, idx in enumerate(indices):
        file_splits[idx] = splits[i]
    
    # Create DataFrame
    manifest = pd.DataFrame({
        "filename": files,
        "split": file_splits
    })
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    manifest.to_csv(output_path, index=False)
    
    # Print summary
    if verbose:
        print(f"\nSplit summary:")
        split_counts = manifest["split"].value_counts()
        for split_name in ["train", "val", "test"]:
            count = split_counts.get(split_name, 0)
            pct = count / n_files * 100
            print(f"  {split_name}: {count} ({pct:.1f}%)")
        print(f"\nManifest saved to: {output_path}")
    
    return manifest


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load manifest from CSV file.
    
    Args:
        manifest_path: Path to manifest CSV
    
    Returns:
        DataFrame with columns ['filename', 'split']
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = pd.read_csv(manifest_path)
    
    # Validate columns
    required_cols = {"filename", "split"}
    if not required_cols.issubset(manifest.columns):
        missing = required_cols - set(manifest.columns)
        raise ValueError(f"Manifest missing columns: {missing}")
    
    return manifest


def get_split_files(
    manifest_path: str,
    split: str
) -> List[str]:
    """
    Get list of filenames for a specific split.
    
    Args:
        manifest_path: Path to manifest CSV
        split: Split name ("train", "val", or "test")
    
    Returns:
        List of filenames in the split
    """
    manifest = load_manifest(manifest_path)
    
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}', must be one of {valid_splits}")
    
    files = manifest[manifest["split"] == split]["filename"].tolist()
    return sorted(files)


def get_manifest_summary(manifest_path: str) -> Dict:
    """
    Get summary statistics for a manifest.
    
    Args:
        manifest_path: Path to manifest CSV
    
    Returns:
        Dictionary with summary statistics
    """
    manifest = load_manifest(manifest_path)
    
    split_counts = manifest["split"].value_counts().to_dict()
    total = len(manifest)
    
    summary = {
        "total_files": total,
        "splits": {},
    }
    
    for split_name in ["train", "val", "test"]:
        count = split_counts.get(split_name, 0)
        summary["splits"][split_name] = {
            "count": count,
            "percentage": count / total * 100 if total > 0 else 0
        }
    
    return summary


def verify_manifest_coverage(
    manifest_path: str,
    processed_dir: str,
    label_base_dir: str,
    robot_sizes: List[Tuple[int, int]]
) -> Tuple[bool, List[str]]:
    """
    Verify that all files in manifest have corresponding maps and labels.
    
    Args:
        manifest_path: Path to manifest CSV
        processed_dir: Directory with occupancy grids
        label_base_dir: Base directory for labels
        robot_sizes: List of robot sizes to check
    
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    manifest = load_manifest(manifest_path)
    processed_dir = Path(processed_dir)
    label_base_dir = Path(label_base_dir)
    
    errors = []
    
    for _, row in manifest.iterrows():
        filename = row["filename"]
        
        # Check map exists
        map_path = processed_dir / filename
        if not map_path.exists():
            errors.append(f"Map not found: {map_path}")
            continue
        
        # Check labels exist for each robot size
        for length, width in robot_sizes:
            label_dir = label_base_dir / f"robot_{length}x{width}"
            label_path = label_dir / filename
            
            if not label_path.exists():
                errors.append(f"Label not found: {label_path}")
    
    all_valid = len(errors) == 0
    return all_valid, errors


def update_manifest_with_new_files(
    manifest_path: str,
    processed_dir: str,
    default_split: str = "train",
    seed: int = 42
) -> pd.DataFrame:
    """
    Update manifest with any new files not already included.
    
    New files are assigned to the default split.
    
    Args:
        manifest_path: Path to existing manifest
        processed_dir: Directory with all processed files
        default_split: Split to assign new files to
        seed: Random seed (for reproducibility if splitting)
    
    Returns:
        Updated manifest DataFrame
    """
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)
    
    # Load existing manifest
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        existing_files = set(manifest["filename"].tolist())
    else:
        manifest = pd.DataFrame(columns=["filename", "split"])
        existing_files = set()
    
    # Find new files
    all_files = set(f for f in os.listdir(processed_dir) if f.endswith(".npy"))
    new_files = sorted(all_files - existing_files)
    
    if new_files:
        print(f"Adding {len(new_files)} new files to manifest")
        
        new_rows = pd.DataFrame({
            "filename": new_files,
            "split": [default_split] * len(new_files)
        })
        
        manifest = pd.concat([manifest, new_rows], ignore_index=True)
        manifest = manifest.sort_values("filename").reset_index(drop=True)
        
        # Save updated manifest
        manifest.to_csv(manifest_path, index=False)
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manifest management utilities")
    subparsers = parser.add_subparsers(dest="command")
    
    # Create manifest
    create_parser = subparsers.add_parser("create", help="Create new manifest")
    create_parser.add_argument("--processed_dir", required=True, help="Directory with .npy files")
    create_parser.add_argument("--output", required=True, help="Output CSV path")
    create_parser.add_argument("--train_split", type=float, default=0.70)
    create_parser.add_argument("--val_split", type=float, default=0.15)
    create_parser.add_argument("--test_split", type=float, default=0.15)
    create_parser.add_argument("--seed", type=int, default=42)
    
    # Show summary
    summary_parser = subparsers.add_parser("summary", help="Show manifest summary")
    summary_parser.add_argument("--manifest", required=True, help="Manifest CSV path")
    
    # Verify coverage
    verify_parser = subparsers.add_parser("verify", help="Verify manifest coverage")
    verify_parser.add_argument("--manifest", required=True, help="Manifest CSV path")
    verify_parser.add_argument("--processed_dir", required=True, help="Directory with maps")
    verify_parser.add_argument("--label_dir", required=True, help="Base label directory")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_manifest(
            processed_dir=args.processed_dir,
            output_path=args.output,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed
        )
    
    elif args.command == "summary":
        summary = get_manifest_summary(args.manifest)
        print(f"Total files: {summary['total_files']}")
        for split, info in summary["splits"].items():
            print(f"  {split}: {info['count']} ({info['percentage']:.1f}%)")
    
    elif args.command == "verify":
        # Use default robot sizes for verification
        robot_sizes = [(6, 4), (10, 6), (14, 9), (18, 11), (22, 14)]
        all_valid, errors = verify_manifest_coverage(
            args.manifest, args.processed_dir, args.label_dir, robot_sizes
        )
        if all_valid:
            print("✓ All files verified")
        else:
            print(f"✗ Found {len(errors)} errors:")
            for err in errors[:20]:
                print(f"  {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
    
    else:
        parser.print_help()
