#!/usr/bin/env python3
"""Phase 3 — Create train/val/test manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.data.manifest import create_manifest, get_manifest_summary

cfg = load_config()

manifest = create_manifest(
    processed_dir="data/processed",
    output_path="data/manifest.csv",
    train_split=cfg.data.train_split,
    val_split=cfg.data.val_split,
    test_split=cfg.data.test_split,
    seed=cfg.data.random_seed,
    verbose=True,
)

summary = get_manifest_summary("data/manifest.csv")
print(f"\nTotal: {summary['total_files']}")
for split, info in summary['splits'].items():
    print(f"  {split}: {info['count']} ({info['percentage']:.1f}%)")

print("\n✓ data/manifest.csv created")